"""Plot energy-vs-force gradient conflicts from a ``GradDiagModule`` run.

Usage:
    python plot_grad_conflicts.py RUN_DIR [--out OUT_DIR] [--vlim 0.5]

``RUN_DIR`` must contain ``grad_diag.csv`` (and optionally ``snapshots/``). Figures:

  1_cos_heatmap_blocks.png    cos(g_E, g_F) per block vs step  (red = conflict, blue = aligned)
  2_cos_heatmap_tensors.png   the same per individual parameter tensor
  3_cos_lines.png             global + per-layer cos vs step, with the minibatch noise floor
  4_conflict_summary.png      fraction of steps with cos < 0, and the distribution of cos, per block
  5_norm_ratio.png            |g_F| / |g_E| per block vs step (log scale)
  6_projection_coef.png       PCGrad projection coefficients (how much of one gradient lies along the other)
  7_weights.png               |W| and |W - W_init| per block vs step
  8_edge_mlp_weights.png      edge-MLP weight matrices at the first and last snapshot
  summary.txt                 per-block table
"""

import argparse
import csv
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SUB_ORDER = {"linear_1": 0, "edge_mlp": 1, "linear_2": 2, "sc": 3}


def order_key(name):
    m = re.match(r"layer(\d+)_convnet\.conv\.(\w+)(.*)", name)
    if m:
        return (1, int(m.group(1)), SUB_ORDER.get(m.group(2), 9), m.group(3))
    if name.startswith("type_embed"):
        return (0, 0, 0, name)
    if name.startswith("per_atom_energy_readout"):
        return (3, 0, 0, name)
    return (2, 0, 0, name)


def load(path):
    """-> {(level, name): {field: np.array over steps}}, steps array"""
    data = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (r["level"], r["name"])
            data[key]["step"].append(int(r["step"]))
            for k, v in r.items():
                if k not in ("step", "level", "name", "epoch"):
                    data[key][k].append(float(v))
    out = {k: {f: np.array(v) for f, v in d.items()} for k, d in data.items()}
    return out


def names(data, level):
    return sorted([n for (l, n) in data if l == level], key=order_key)


def matrix(data, level, field):
    ns = names(data, level)
    steps = data[(level, ns[0])]["step"]
    return ns, steps, np.stack([data[(level, n)][field] for n in ns])


def heatmap(ax, M, steps, labels, vlim, title, cbar_label):
    im = ax.imshow(
        M, aspect="auto", cmap="RdBu", vmin=-vlim, vmax=vlim, interpolation="nearest",
        extent=[steps[0] - 0.5 * np.median(np.diff(steps)) if len(steps) > 1 else steps[0] - 0.5,
                steps[-1] + 0.5 * np.median(np.diff(steps)) if len(steps) > 1 else steps[-1] + 0.5,
                len(labels), 0],
    )
    ax.set_yticks(np.arange(len(labels)) + 0.5)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("optimizer step")
    ax.set_title(title)
    cb = plt.colorbar(im, ax=ax, pad=0.01)
    cb.set_label(cbar_label)
    return im


def layer_of(name):
    m = re.match(r"layer(\d+)_", name)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--vlim", type=float, default=None,
                    help="symmetric colour limit for cosine heatmaps (default: auto)")
    a = ap.parse_args()
    out = a.out or os.path.join(a.run_dir, "grad_plots")
    os.makedirs(out, exist_ok=True)

    data = load(os.path.join(a.run_dir, "grad_diag.csv"))
    groups = names(data, "group")
    tensors = names(data, "tensor")
    steps = data[("global", "global")]["step"]
    print(f"{len(steps)} diagnostic steps ({steps[0]:.0f}..{steps[-1]:.0f}), "
          f"{len(groups)} blocks, {len(tensors)} tensors")

    gM = np.stack([data[("group", n)]["cos"] for n in groups])
    vlim = a.vlim or float(np.clip(np.nanpercentile(np.abs(gM), 98), 0.2, 1.0))

    # 1 block heatmap
    fig, ax = plt.subplots(figsize=(11, 0.38 * len(groups) + 2))
    _, st, M = matrix(data, "group", "cos")
    heatmap(ax, M, st, groups, vlim, "cos(g_E, g_F) per block   (red = conflict, blue = aligned)",
            "cosine similarity")
    fig.tight_layout(); fig.savefig(os.path.join(out, "1_cos_heatmap_blocks.png"), dpi=150); plt.close(fig)

    # 2 tensor heatmap
    fig, ax = plt.subplots(figsize=(11, 0.28 * len(tensors) + 2))
    _, st, M = matrix(data, "tensor", "cos")
    heatmap(ax, M, st, tensors, vlim, "cos(g_E, g_F) per parameter tensor", "cosine similarity")
    fig.tight_layout(); fig.savefig(os.path.join(out, "2_cos_heatmap_tensors.png"), dpi=150); plt.close(fig)

    # 3 lines: global (with noise floor) + per layer
    glob = data[("global", "global")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharex=True)
    ax = axes[0]
    ax.axhline(0, color="k", lw=0.6)
    ax.plot(steps, glob["cos"], "o-", ms=3, color="C3", label="cos(g_E, g_F)   [conflict signal]")
    ax.plot(steps, glob["noise_cos_E"], ".:", color="C0", alpha=0.8,
            label="cos(g_E(A), g_E(B))   [energy, two batches]")
    ax.plot(steps, glob["noise_cos_F"], ".:", color="C2", alpha=0.8,
            label="cos(g_F(A), g_F(B))   [force, two batches]")
    ax.set_ylim(-1.05, 1.05); ax.set_xlabel("optimizer step"); ax.set_ylabel("cosine")
    ax.set_title("Global: E-vs-F cosine against same-task batch-to-batch cosine")
    ax.legend(fontsize=7, loc="lower right")
    ax = axes[1]
    ax.axhline(0, color="k", lw=0.6)
    layers = sorted({layer_of(g) for g in groups if layer_of(g) is not None})
    cmap = plt.get_cmap("viridis")
    for g in groups:
        L = layer_of(g)
        col = cmap(L / max(1, len(layers) - 1)) if L is not None else "gray"
        ls = {"linear_1": "-", "edge_mlp": "--", "linear_2": ":", "sc": "-."}.get(
            g.split(".")[-1], "-")
        ax.plot(steps, data[("group", g)]["cos"], ls, color=col, lw=1, alpha=0.9)
    ax.set_ylim(-1.05, 1.05); ax.set_xlabel("optimizer step")
    ax.set_title("Per block (colour = layer 0..3, gray = embed/readout)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, "3_cos_lines.png"), dpi=150); plt.close(fig)

    # 4 conflict summary
    frac = np.array([np.mean(data[("group", g)]["cos"] < 0) for g in groups] +
                    [np.mean(glob["cos"] < 0)])
    med = np.array([np.median(data[("group", g)]["cos"]) for g in groups] + [np.median(glob["cos"])])
    labs = groups + ["GLOBAL"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.32 * len(labs) + 2), sharey=True)
    y = np.arange(len(labs))
    axes[0].barh(y, frac, color=["C3" if l != "GLOBAL" else "k" for l in labs])
    axes[0].set_xlim(0, 1); axes[0].set_xlabel("fraction of diagnostic steps with cos(g_E, g_F) < 0")
    axes[0].set_yticks(y); axes[0].set_yticklabels(labs, fontsize=7); axes[0].invert_yaxis()
    axes[0].axvline(0.5, color="gray", lw=0.6, ls="--")
    allc = [data[("group", g)]["cos"] for g in groups] + [glob["cos"]]
    axes[1].boxplot(allc, positions=y, orientation="horizontal", whis=(5, 95), showfliers=False)
    axes[0].set_yticks(y); axes[0].set_yticklabels(labs, fontsize=7)
    axes[1].axvline(0, color="k", lw=0.6); axes[1].set_xlim(-1, 1)
    axes[1].set_xlabel("distribution of cos(g_E, g_F) over steps (box = IQR, whiskers = 5-95%)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "4_conflict_summary.png"), dpi=150); plt.close(fig)

    # 5 norm ratio
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for g in groups:
        L = layer_of(g)
        col = cmap(L / max(1, len(layers) - 1)) if L is not None else "gray"
        ls = {"linear_1": "-", "edge_mlp": "--", "linear_2": ":", "sc": "-."}.get(g.split(".")[-1], "-")
        ax.plot(steps, data[("group", g)]["norm_ratio_F_over_E"], ls, color=col, lw=1)
    ax.plot(steps, glob["norm_ratio_F_over_E"], "k-", lw=2, label="global")
    ax.axhline(1, color="k", lw=0.5)
    ax.set_yscale("log"); ax.set_xlabel("optimizer step"); ax.set_ylabel("|g_F| / |g_E|")
    ax.set_title("Gradient-norm ratio (>1: force gradient larger)"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out, "5_norm_ratio.png"), dpi=150); plt.close(fig)

    # 6 projection coefficients (global + heatmap of E on F)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].plot(steps, glob["proj_coef_E_on_F"], label="g_E.g_F / |g_F|^2  (E along F)")
    axes[0].plot(steps, glob["proj_coef_F_on_E"], label="g_E.g_F / |g_E|^2  (F along E)")
    axes[0].set_yscale("symlog", linthresh=1); axes[0].legend(fontsize=8)
    axes[0].set_xlabel("optimizer step"); axes[0].set_title("Global PCGrad projection coefficients (<0 = conflict)")
    _, st, M = matrix(data, "group", "proj_coef_F_on_E")
    heatmap(axes[1], np.clip(M, -2, 2), st, groups, 2.0,
            "proj coef F on E per block (clipped to +-2)", "g_E.g_F / |g_E|^2")
    fig.tight_layout(); fig.savefig(os.path.join(out, "6_projection_coef.png"), dpi=150); plt.close(fig)

    # 7 weights
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)
    for g in groups:
        L = layer_of(g)
        col = cmap(L / max(1, len(layers) - 1)) if L is not None else "gray"
        ls = {"linear_1": "-", "edge_mlp": "--", "linear_2": ":", "sc": "-."}.get(g.split(".")[-1], "-")
        axes[0].plot(steps, data[("group", g)]["w_norm"], ls, color=col, lw=1)
        axes[1].plot(steps, data[("group", g)]["dw_norm"], ls, color=col, lw=1)
    axes[0].set_title("|W| per block"); axes[1].set_title("|W - W_init| per block")
    for ax in axes:
        ax.set_xlabel("optimizer step"); ax.set_yscale("log")
    fig.tight_layout(); fig.savefig(os.path.join(out, "7_weights.png"), dpi=150); plt.close(fig)

    # 8 edge-MLP weight matrices from snapshots
    snap_dir = os.path.join(a.run_dir, "snapshots")
    snaps = sorted(os.listdir(snap_dir)) if os.path.isdir(snap_dir) else []
    if len(snaps) >= 1:
        import torch
        pick = [snaps[0], snaps[-1]] if len(snaps) > 1 else [snaps[0]]
        loaded = [torch.load(os.path.join(snap_dir, s), map_location="cpu", weights_only=False)
                  for s in pick]
        mats = [n for n in loaded[0]["weights"]
                if ".edge_mlp.mlp." in n and loaded[0]["weights"][n].ndim == 2]
        mats = sorted(mats, key=order_key)
        fig, axes = plt.subplots(len(loaded), len(mats), figsize=(1.9 * len(mats), 2.2 * len(loaded)),
                                 squeeze=False)
        for i, sn in enumerate(loaded):
            for j, n in enumerate(mats):
                w = sn["weights"][n].numpy()
                v = np.abs(w).max()
                axes[i, j].imshow(w, cmap="RdBu", vmin=-v, vmax=v, aspect="auto")
                axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
                if i == 0:
                    axes[i, j].set_title(re.sub(r"_convnet\.conv\.edge_mlp\.mlp\.", ":", n)
                                         .replace(".weight", ""), fontsize=6)
            axes[i, 0].set_ylabel(f"step {sn['step']}", fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(out, "8_edge_mlp_weights.png"), dpi=150); plt.close(fig)

    # summary table
    lines = [f"{'block':44s} {'n':>7s} {'median cos':>10s} {'frac<0':>7s} {'median |gF|/|gE|':>17s}"]
    for g in groups + ["global"]:
        d = data[("group", g)] if g != "global" else glob
        lines.append(f"{g:44s} {int(d['n_params'][0]):7d} {np.median(d['cos']):+10.3f} "
                     f"{np.mean(d['cos'] < 0):7.2f} {np.median(d['norm_ratio_F_over_E']):17.2f}")
    nE, nF = glob["noise_cos_E"], glob["noise_cos_F"]
    lines.append("")
    lines.append(f"noise floor (global): median cos(gE_A,gE_B)={np.nanmedian(nE):+.3f}, "
                 f"median cos(gF_A,gF_B)={np.nanmedian(nF):+.3f}; "
                 f"fraction of steps where gE(A) and gE(B) point opposite = {np.nanmean(nE < 0):.2f}")
    open(os.path.join(out, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nfigures written to {out}")


if __name__ == "__main__":
    main()
