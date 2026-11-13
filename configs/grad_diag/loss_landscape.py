"""Loss-landscape sweep for a trained NequIP model, following Li et al. 2018
("Visualizing the Loss Landscape of Neural Nets", arXiv:1712.09913).

    f(alpha, beta) = L( theta* + alpha * delta + beta * eta )

* theta*  : trained weights (EMA weights of a NequIP checkpoint, eager mode)
* delta, eta : two random Gaussian directions, *filter-wise normalized*: every filter of the
  direction is rescaled to the norm of the matching filter of theta*.  Here a "filter" is
    - one output neuron (column) of each 2D radial-MLP weight matrix (``*edge_mlp*``), as for
      the fully-connected layers in the paper;
    - the whole tensor for every other trainable tensor (the e3nn linear / tensor-product /
      self-connection weights are flat vectors with no neuron axis, the embedding and readout
      are tiny) -- i.e. the paper's "layer-wise" normalization variant for those.
  Non-trainable buffers (per-type scales/shifts, Bessel weights, ZBL) are not perturbed,
  analogous to the paper leaving batch-norm parameters fixed.
* L_E : per-atom-energy MSE,  L_F : force MSE   (the two terms of the NequIP training loss).
  Both come from one forward pass, so the energy and force landscapes cost the same as one.

Optional ``--gradients`` additionally computes, at every grid point, the full-subset gradients
g_E = grad L_E and g_F = grad L_F: their norms, cos(g_E, g_F), and their in-plane components
(g.delta, g.eta) = (dL/dalpha, dL/dbeta), i.e. the gradient vector field on the plotted plane.
This needs a double backward through the force term (~3-4x the cost of a plain evaluation).

Usage (GPU node):
  python loss_landscape.py --ckpt /abs/path/best.ckpt --config ../grad_conflict.yaml \
      --n-structures 1000 --grid 51 --device cuda --out landscape_loss_sweep
Benchmark only:
  python loss_landscape.py --ckpt ... --config ... --bench
"""

import argparse
import os
import time
import warnings

import numpy as np
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

warnings.filterwarnings("ignore")

from nequip.data import AtomicDataDict  # noqa: E402
from nequip.data.modifier import PerAtomModifier  # noqa: E402
from nequip.model import ModelFromCheckpoint  # noqa: E402
from nequip.train.lightning import _SOLE_MODEL_KEY  # noqa: E402
from nequip.utils.global_state import set_global_state  # noqa: E402

E_KEY, F_KEY = AtomicDataDict.TOTAL_ENERGY_KEY, AtomicDataDict.FORCE_KEY
per_atom = PerAtomModifier(E_KEY)


# --------------------------------------------------------------------------- data / model
def load_model(ckpt, device):
    m = ModelFromCheckpoint(ckpt)
    if isinstance(m, torch.nn.ModuleDict):
        m = m[_SOLE_MODEL_KEY]
    return m.to(device).eval()


def load_batches(config, n_structures, batch_size, device):
    """First ``n_structures`` structures of the *training* split, as device-resident batches."""
    cfg = OmegaConf.load(config)
    data = OmegaConf.to_container(cfg.data, resolve=True)
    data["train_dataloader"].update(batch_size=batch_size, shuffle=False, num_workers=0)
    dm = instantiate(data, _recursive_=False)
    dm.setup("fit")
    dl = dm.train_dataloader()
    dl = dl[0] if isinstance(dl, (list, tuple)) else dl
    batches, n = [], 0
    for b in dl:
        batches.append({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()})
        n += int(AtomicDataDict.num_frames(b))
        if n >= n_structures:
            break
    n_atoms = sum(int(AtomicDataDict.num_nodes(b)) for b in batches)
    return batches, n, n_atoms


# --------------------------------------------------------------------------- losses
def _errors(model, b):
    tgt = dict(b)
    out = model(dict(b))
    pe, te = per_atom._func(out), per_atom._func(tgt)
    return (pe - te), (out[F_KEY] - tgt[F_KEY])


def eval_losses(model, batches, n_struct, n_atoms):
    """(L_E, L_F) over the whole subset. Forces need autograd w.r.t. positions, so grad stays on."""
    sseE = sseF = 0.0
    with torch.enable_grad():
        for b in batches:
            dE, dF = _errors(model, b)
            sseE += float((dE.detach().double() ** 2).sum())
            sseF += float((dF.detach().double() ** 2).sum())
    return sseE / n_struct, sseF / (3 * n_atoms)


def eval_gradients(model, params, dirs, batches, n_struct, n_atoms):
    """L_E, L_F, |g_E|, |g_F|, cos(g_E,g_F) and in-plane components (g.delta, g.eta) for E and F."""
    model.train()  # forces keep their graph to the weights only in training mode
    gE = [torch.zeros_like(p) for p in params]
    gF = [torch.zeros_like(p) for p in params]
    LE = LF = 0.0
    for b in batches:
        dE, dF = _errors(model, b)
        cE = (dE**2).sum() / n_struct
        cF = (dF**2).sum() / (3 * n_atoms)
        ge = torch.autograd.grad(cE, params, retain_graph=True, allow_unused=True)
        gf = torch.autograd.grad(cF, params, allow_unused=True)
        for i in range(len(params)):
            if ge[i] is not None:
                gE[i] += ge[i].detach()
            if gf[i] is not None:
                gF[i] += gf[i].detach()
        LE += float(cE.detach())
        LF += float(cF.detach())
    model.eval()
    flat = lambda g: torch.cat([x.flatten() for x in g]).double()  # noqa: E731
    fE, fF = flat(gE), flat(gF)
    out = dict(
        LE=LE, LF=LF, nE=float(fE.norm()), nF=float(fF.norm()),
        cos=float(fE @ fF / (fE.norm() * fF.norm() + 1e-30)),
    )
    for name, g in (("E", fE), ("F", fF)):
        for k, d in enumerate(dirs):
            out[f"g{name}{k}"] = float(g @ flat(d))
    return out


# --------------------------------------------------------------------------- directions
def make_direction(params, seed):
    """Random Gaussian direction with filter-wise normalization (see module docstring)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for name, p in params:
        d = torch.randn(p.shape, generator=gen).to(p.device, p.dtype)
        w = p.detach()
        if "edge_mlp" in name and w.ndim == 2:  # filters = columns = output neurons
            d = d / (d.norm(dim=0, keepdim=True) + 1e-10) * w.norm(dim=0, keepdim=True)
        else:  # layer-wise
            d = d / (d.norm() + 1e-10) * w.norm()
        out.append(d)
    return out


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True, help="yaml with the `data:` section (grad_conflict.yaml)")
    ap.add_argument("--n-structures", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--grid", type=int, default=21, help="points per axis")
    ap.add_argument("--ranges", default="1.0",
                    help="comma-separated half-widths R (alpha, beta in [-R, R]); each is written to "
                         "<out>/range_<R>/ and the same two random directions are used for all of them")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gradients", action="store_true", help="also compute gradient maps (slower)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=16, help="CPU threads (be polite on login nodes)")
    ap.add_argument("--out", default="landscape_out")
    ap.add_argument("--rows", default=None,
                    help="beta-row range 'a:b' to compute (shard the grid over GPUs/jobs; "
                         "plot_loss_landscape.py merges all landscape*.npz in --out)")
    ap.add_argument("--bench", action="store_true", help="time one evaluation and exit")
    a = ap.parse_args()

    set_global_state()  # nequip requires this before any model is built (as nequip-train does)
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    torch.set_num_threads(a.threads)
    print(f"device={dev} threads={a.threads}")
    model = load_model(a.ckpt, dev)
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad and p.numel() > 0]
    plist = [p for _, p in params]
    n_par = sum(p.numel() for p in plist)
    print(f"trainable parameters: {n_par:,}")

    batches, n_struct, n_atoms = load_batches(a.config, a.n_structures, a.batch_size, dev)
    print(f"eval subset: {n_struct} structures, {n_atoms} atoms, {len(batches)} batches of <= {a.batch_size}")

    theta0 = [p.detach().clone() for p in plist]
    d1, d2 = make_direction(params, a.seed), make_direction(params, a.seed + 1)
    dirs = [d1, d2]
    cosd = float(torch.cat([x.flatten() for x in d1]) @ torch.cat([x.flatten() for x in d2])) / (
        float(torch.cat([x.flatten() for x in d1]).norm() * torch.cat([x.flatten() for x in d2]).norm()))
    print(f"cos(delta, eta) = {cosd:+.4f} (random directions are ~orthogonal in high dimension)")

    def set_point(alpha, beta):
        with torch.no_grad():
            for p, t, x, y in zip(plist, theta0, d1, d2):
                p.copy_(t + alpha * x + beta * y)

    # ---- benchmark / center sanity check
    set_point(0.0, 0.0)
    t0 = time.time(); LE0, LF0 = eval_losses(model, batches, n_struct, n_atoms)
    if dev == "cuda":
        torch.cuda.synchronize()
    t_plain = time.time() - t0
    t0 = time.time(); eval_losses(model, batches, n_struct, n_atoms)
    if dev == "cuda":
        torch.cuda.synchronize()
    t_plain = time.time() - t0
    print(f"center: L_E(per-atom MSE)={LE0:.4e}  L_F(MSE)={LF0:.4e}  "
          f"(force RMSE {LF0**0.5:.4f}, per-atom-energy RMSE {LE0**0.5:.4e})")
    print(f"plain eval: {t_plain:.2f}s for {n_struct} structures -> {n_struct/t_plain:.1f} structures/s")
    if a.gradients or a.bench:
        t0 = time.time(); eval_gradients(model, plist, dirs, batches, n_struct, n_atoms)
        if dev == "cuda":
            torch.cuda.synchronize()
        t_grad = time.time() - t0
        print(f"gradient eval: {t_grad:.2f}s ({n_struct/t_grad:.1f} structures/s, {t_grad/t_plain:.1f}x plain)")
    if a.bench:
        return

    # ---- sweep: one pass per zoom range. The same two random directions are used for every range,
    #      so a smaller range is a true zoom of the same plane (same grid size and structures throughout).
    ranges = [float(v) for v in a.ranges.split(",")]
    names = ["LE", "LF"] + (["nE", "nF", "cos", "gE0", "gE1", "gF0", "gF1"] if a.gradients else [])
    r0, r1 = (0, a.grid) if a.rows is None else (int(v) for v in a.rows.split(":"))
    fname = "landscape.npz" if a.rows is None else f"landscape_rows{r0}-{r1}.npz"
    for R in ranges:
        out_dir = os.path.join(a.out, f"range_{R:g}")
        os.makedirs(out_dir, exist_ok=True)
        xs = np.linspace(-R, R, a.grid)
        ys = np.linspace(-R, R, a.grid)
        res = {n: np.full((a.grid, a.grid), np.nan) for n in names}  # [j(beta), i(alpha)]
        t_start = time.time()
        for j in range(r0, r1):
            beta = ys[j]
            for i, alpha in enumerate(xs):
                set_point(alpha, beta)
                if a.gradients:
                    r = eval_gradients(model, plist, dirs, batches, n_struct, n_atoms)
                    for n in names:
                        res[n][j, i] = r[n]
                else:
                    LE, LF = eval_losses(model, batches, n_struct, n_atoms)
                    res["LE"][j, i], res["LF"][j, i] = LE, LF
            el = time.time() - t_start
            done = j + 1 - r0
            print(f"range {R:g}  row {j+1}/{r1}  elapsed {el/60:.1f} min  "
                  f"ETA {el/done*(r1-j-1)/60:.1f} min", flush=True)
            np.savez(os.path.join(out_dir, fname), alphas=xs, betas=ys, n_structures=n_struct,
                     n_atoms=n_atoms, seed=a.seed, ckpt=a.ckpt, n_params=n_par, cos_dirs=cosd,
                     rows_done=j + 1, **res)
        print(f"range {R:g}: done in {(time.time()-t_start)/60:.1f} min -> {out_dir}/{fname}", flush=True)
    set_point(0.0, 0.0)


if __name__ == "__main__":
    main()
