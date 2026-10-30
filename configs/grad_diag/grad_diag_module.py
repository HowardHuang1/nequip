"""Energy-vs-force gradient conflict diagnostics for NequIP (diagnostics only).

``GradDiagModule`` is a drop-in replacement for ``nequip.train.EMALightningModule``.
Training is NOT altered: every step returns the same weighted-sum loss as the original and
Lightning performs its usual single backward. Every ``diag_every`` optimizer steps it
additionally computes the energy-loss gradient ``g_E`` and force-loss gradient ``g_F``
separately (``torch.autograd.grad`` on each loss component, which does not touch ``p.grad``)
and logs, per parameter tensor, per block and globally:

    cos(g_E, g_F), |g_E|, |g_F|, |g_F|/|g_E|, the PCGrad projection coefficients,
    |W| and |W - W_init|, and (optionally) a minibatch-noise floor.

Output (in ``diag_dir``): ``grad_diag.csv`` (long format) and ``snapshots/step_XXXXXXXX.pt``.
Use ``plot_grad_conflicts.py`` to plot them.

Use with ``trainer.devices=1`` and ``model.compile_mode: eager``.
"""

import csv
import math
import os
from typing import Dict, List, Optional

import torch

from nequip.train import EMALightningModule
from nequip.utils import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

EPS = 1e-12
FIELDS = [
    "step",
    "epoch",
    "level",  # global | group | tensor
    "name",
    "n_params",
    "cos",
    "norm_E",
    "norm_F",
    "norm_ratio_F_over_E",
    "proj_coef_E_on_F",  # g_E.g_F / |g_F|^2  (PCGrad: component of g_E along g_F, in units of g_F)
    "proj_coef_F_on_E",  # g_E.g_F / |g_E|^2
    "w_norm",
    "dw_norm",  # |W - W_init|
    "noise_cos_E",  # cos(g_E(batch A), g_E(batch B))  -- same-task minibatch noise floor
    "noise_cos_F",
]


def _short(name: str) -> str:
    """Strip the long ``sole_model.model.func.`` prefix from a parameter name."""
    return name.split("func.", 1)[-1]


def _group(name: str) -> str:
    """Block a parameter tensor belongs to, e.g. ``layer0_convnet.conv.edge_mlp``."""
    parts = _short(name).split(".")
    if parts[0].startswith("layer"):
        return ".".join(parts[:3])
    return parts[0]


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return (torch.dot(a, b) / (a.norm() * b.norm() + EPS)).item()


def _pair_stats(gE: torch.Tensor, gF: torch.Tensor) -> Dict[str, float]:
    nE, nF = gE.norm().item(), gF.norm().item()
    dot = torch.dot(gE, gF).item()
    return {
        "cos": dot / (nE * nF + EPS),
        "norm_E": nE,
        "norm_F": nF,
        "norm_ratio_F_over_E": nF / (nE + EPS),
        "proj_coef_E_on_F": dot / (nF**2 + EPS),
        "proj_coef_F_on_E": dot / (nE**2 + EPS),
    }


class GradDiagModule(EMALightningModule):
    """``EMALightningModule`` plus periodic logging of energy/force gradient conflict.

    Args:
        diag_every (int): run the diagnostic every this many optimizer steps.
        diag_dir (str, optional): output directory (default: current working directory,
            which for ``nequip-train`` is Hydra's output directory when interpolated as
            ``${hydra:runtime.output_dir}``).
        snapshot_every (int): save raw per-tensor weights/gradients every this many steps
            (0 disables).
        noise_floor (bool): also compute g_E and g_F on the previous diagnostic batch and
            log the same-task cosines between the two batches.
        check_first (bool): on the first diagnostic step verify that
            ``c_E g_E + c_F g_F`` equals the gradient of the weighted-sum loss.
    """

    def __init__(
        self,
        *args,
        diag_every: int = 50,
        diag_dir: Optional[str] = None,
        snapshot_every: int = 500,
        noise_floor: bool = True,
        check_first: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert diag_every >= 1
        self.diag_every = diag_every
        self.diag_dir = diag_dir
        self.snapshot_every = snapshot_every
        self.noise_floor = noise_floor
        self.check_first = check_first
        self._w0: Optional[Dict[str, torch.Tensor]] = None
        self._probe: Optional[Dict] = None  # previous diagnostic batch
        self._csv_path: Optional[str] = None
        self._checked = False

    # ------------------------------------------------------------------ helpers
    def _out_dir(self) -> str:
        d = self.diag_dir or os.getcwd()
        os.makedirs(d, exist_ok=True)
        return d

    @staticmethod
    def _clone_batch(batch: Dict) -> Dict:
        return {
            k: (v.detach().clone() if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }

    def _named_params(self) -> List:
        return [
            (n, p)
            for n, p in self.model.named_parameters()
            if p.requires_grad and p.numel() > 0
        ]

    def _component_names(self, loss_dict: Dict, prefix: str):
        """Names of the energy and force loss components in the loss dict."""
        names = [
            k[len(prefix) :]
            for k in loss_dict
            if k.startswith(prefix) and not k.endswith("weighted_sum")
        ]
        e = [n for n in names if "energy" in n]
        f = [n for n in names if "force" in n]
        assert len(e) == 1 and len(f) == 1, (
            f"expected one energy and one force loss component, found {names}"
        )
        return e[0], f[0]

    def _manual_component(self, name: str, out: Dict, tgt: Dict) -> torch.Tensor:
        """MSE of a loss component computed by hand (no MetricsManager state updates)."""
        entry = self.loss.entries[name]
        assert not entry.per_type, "per-type loss components not supported here"
        p, t = entry.field(out, tgt)
        return torch.mean((p - t) ** 2)

    # ------------------------------------------------------------------ training
    def training_step(self, batch, batch_idx, dataloader_idx=0):
        if self.global_step % self.diag_every != 0:
            return super().training_step(batch, batch_idx, dataloader_idx)
        return self._diag_training_step(batch, batch_idx, dataloader_idx)

    def _diag_training_step(self, batch, batch_idx, dataloader_idx):
        # same body as NequIPLightningModule.training_step ...
        target = self.process_target(batch, batch_idx, dataloader_idx)
        this_batch = self._clone_batch(batch) if self.noise_floor else None
        output = self(batch)

        if self.train_metrics is not None:
            with torch.no_grad():
                d = self.train_metrics(
                    output,
                    target,
                    prefix=f"train_metric_step{self.logging_delimiter}",
                )
            self.log_dict(d)

        prefix = f"train_loss_step{self.logging_delimiter}"
        loss_dict = self.loss(output, target, prefix=prefix)
        self.log_dict(loss_dict)
        loss = loss_dict[prefix + "weighted_sum"] * self.world_size

        # ... plus the diagnostic (graph is kept so Lightning's backward still works)
        e_name, f_name = self._component_names(loss_dict, prefix)
        L_E, L_F = loss_dict[prefix + e_name], loss_dict[prefix + f_name]
        named = self._named_params()
        ps = [p for _, p in named]

        gE = torch.autograd.grad(L_E, ps, retain_graph=True, allow_unused=True)
        gF = torch.autograd.grad(L_F, ps, retain_graph=True, allow_unused=True)

        if self.check_first and not self._checked:
            self._check_decomposition(
                loss_dict, prefix, e_name, f_name, ps, gE, gF, output, target
            )
            self._checked = True

        gE_B = gF_B = None
        if self.noise_floor and self._probe is not None:
            gE_B, gF_B = self._probe_grads(batch_idx, dataloader_idx, ps, e_name, f_name)

        if self.trainer.is_global_zero:
            self._write_diag(named, gE, gF, gE_B, gF_B)
        self._probe = this_batch
        return loss

    def _probe_grads(self, batch_idx, dataloader_idx, ps, e_name, f_name):
        b = self._clone_batch(self._probe)
        tgt = self.process_target(b, batch_idx, dataloader_idx)
        out = self(b)
        LE = self._manual_component(e_name, out, tgt)
        LF = self._manual_component(f_name, out, tgt)
        gE = torch.autograd.grad(LE, ps, retain_graph=True, allow_unused=True)
        gF = torch.autograd.grad(LF, ps, allow_unused=True)
        return gE, gF

    def _check_decomposition(
        self, loss_dict, prefix, e_name, f_name, ps, gE, gF, output, target
    ):
        """c_E g_E + c_F g_F must equal the gradient of the weighted-sum loss."""
        cE = self.loss.entries[e_name].coeff
        cF = self.loss.entries[f_name].coeff
        gC = torch.autograd.grad(
            loss_dict[prefix + "weighted_sum"], ps, retain_graph=True, allow_unused=True
        )
        num = den = 0.0
        for a, b, c in zip(gE, gF, gC):
            if a is None or b is None or c is None:
                continue
            num += ((cE * a + cF * b - c).double() ** 2).sum().item()
            den += (c.double() ** 2).sum().item()
        rel = math.sqrt(num) / (math.sqrt(den) + EPS)
        logger.info(
            f"[GradDiag] decomposition check: coeffs (E={cE:.4g}, F={cF:.4g}), "
            f"|cE gE + cF gF - g_total| / |g_total| = {rel:.3e}"
        )
        # the hand-computed MSE used for the noise-floor batch must match the real components
        for n in (e_name, f_name):
            ref = loss_dict[prefix + n].item()
            man = self._manual_component(n, output, target).item()
            logger.info(f"[GradDiag] manual {n} = {man:.6e} vs loss component {ref:.6e}")
            if abs(man - ref) > 1e-4 * max(abs(ref), 1e-12):
                raise RuntimeError(f"manual {n} ({man}) != loss component ({ref})")
        if rel > 1e-3:
            raise RuntimeError(
                f"gradient decomposition check failed (relative error {rel:.3e}); "
                "the separate energy/force gradients do not sum to the training gradient."
            )
        if self.world_size > 1:
            logger.warning(
                "[GradDiag] running with world_size > 1: diagnostics are per-rank. "
                "Use trainer.devices=1 for clean results."
            )

    # ------------------------------------------------------------------ logging
    def _write_diag(self, named, gE, gF, gE_B, gF_B):
        step, epoch = self.global_step, self.current_epoch
        if self._w0 is None:
            self._w0 = {n: p.detach().clone() for n, p in named}

        groups: Dict[str, Dict[str, List]] = {}
        for i, (n, p) in enumerate(named):
            if gE[i] is None or gF[i] is None:
                continue
            g = groups.setdefault(_group(n), {k: [] for k in ("E", "F", "w", "dw", "EB", "FB")})
            g["E"].append(gE[i].detach().flatten())
            g["F"].append(gF[i].detach().flatten())
            g["w"].append(p.detach().flatten())
            g["dw"].append((p.detach() - self._w0[n]).flatten())
            if gE_B is not None and gE_B[i] is not None and gF_B[i] is not None:
                g["EB"].append(gE_B[i].detach().flatten())
                g["FB"].append(gF_B[i].detach().flatten())

        rows = []

        def row(level, name, E, F, w, dw, EB, FB):
            E, F = E.double(), F.double()
            r = {"step": step, "epoch": epoch, "level": level, "name": name,
                 "n_params": E.numel()}
            r.update(_pair_stats(E, F))
            r["w_norm"] = w.double().norm().item()
            r["dw_norm"] = dw.double().norm().item()
            nan = float("nan")
            r["noise_cos_E"] = _cos(E, EB.double()) if EB is not None else nan
            r["noise_cos_F"] = _cos(F, FB.double()) if FB is not None else nan
            rows.append(r)

        def cat(lst):
            return torch.cat(lst) if lst else None

        # per-tensor rows
        for i, (n, p) in enumerate(named):
            if gE[i] is None or gF[i] is None:
                continue
            EB = gE_B[i].flatten() if gE_B is not None and gE_B[i] is not None else None
            FB = gF_B[i].flatten() if gF_B is not None and gF_B[i] is not None else None
            row("tensor", _short(n), gE[i].flatten(), gF[i].flatten(),
                p.detach().flatten(), (p.detach() - self._w0[n]).flatten(), EB, FB)
        # per-group rows
        for gname, g in groups.items():
            row("group", gname, cat(g["E"]), cat(g["F"]), cat(g["w"]), cat(g["dw"]),
                cat(g["EB"]), cat(g["FB"]))
        # global row
        allg = {k: cat([t for g in groups.values() for t in g[k]])
                for k in ("E", "F", "w", "dw", "EB", "FB")}
        row("global", "global", allg["E"], allg["F"], allg["w"], allg["dw"],
            allg["EB"], allg["FB"])

        path = os.path.join(self._out_dir(), "grad_diag.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows)

        glob = rows[-1]
        self.log("gradcos/global", glob["cos"])
        self.log("gradnorm_ratio_F_over_E/global", glob["norm_ratio_F_over_E"])

        if self.snapshot_every and step % self.snapshot_every == 0:
            snap_dir = os.path.join(self._out_dir(), "snapshots")
            os.makedirs(snap_dir, exist_ok=True)
            torch.save(
                {
                    "step": step,
                    "weights": {_short(n): p.detach().cpu().float() for n, p in named},
                    "grad_E": {_short(n): g.detach().cpu().float()
                               for (n, _), g in zip(named, gE) if g is not None},
                    "grad_F": {_short(n): g.detach().cpu().float()
                               for (n, _), g in zip(named, gF) if g is not None},
                },
                os.path.join(snap_dir, f"step_{step:08d}.pt"),
            )
