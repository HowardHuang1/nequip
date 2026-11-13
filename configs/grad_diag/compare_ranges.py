"""Compare the landscapes across zoom ranges (see run_landscape.pbs).

    python compare_ranges.py LOSS_SWEEP_DIR GRADIENT_SWEEP_DIR

Reads <dir>/range_<R>/landscape*.npz for every R in both folders and writes into <dir>/zoom_comparison/:
  zoom_metrics.txt             roughness numbers per range (and conflict numbers for the gradient folder)
  zoom_loss_contours.png       (loss folder)     energy and force contours, loss / loss at the trained model
  zoom_roughness_vs_range.png  (loss folder)     roughness metrics against range
  zoom_conflict_maps.png       (gradient folder) cos(g_E, g_F) and in-plane angle per range
  zoom_conflict_vs_range.png   (gradient folder) conflict statistics against range
"""

import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plot_loss_landscape import load, roughness  # noqa: E402


def load_ranges(d):
    """[(R, data)] widest range first; ranges with missing rows are skipped with a warning."""
    out = []
    for p in glob.glob(os.path.join(d, "range_*")):
        try:
            R = float(os.path.basename(p).split("_", 1)[1])
        except ValueError:
            continue
        if not glob.glob(os.path.join(p, "landscape*.npz")):
            continue
        data = load(p)
        if np.isnan(data["LE"]).any():
            print(f"WARNING: {p} has unfinished rows -- skipped")
            continue
        out.append((R, data))
    return sorted(out, key=lambda t: -t[0])


def inplane_angle(d):
    a, b, c, e = d["gE0"], d["gE1"], d["gF0"], d["gF1"]
    cosang = (a * c + b * e) / (np.hypot(a, b) * np.hypot(c, e) + 1e-300)
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))


def metrics(R, d):
    LE, LF = d["LE"], d["LF"]
    n = LE.shape[0]
    c = n // 2
    m = {"R": R, "grid": n, "n_structures": int(d["n_structures"]),
         "L_E at centre": LE[c, c], "L_F at centre": LF[c, c],
         "E max/centre": np.nanmax(LE) / LE[c, c], "F max/centre": np.nanmax(LF) / LF[c, c]}
    for k, v in roughness(LE).items():
        m["E " + k] = v
    for k, v in roughness(LF).items():
        m["F " + k] = v
    if "cos" in d:
        ang = inplane_angle(d)
        m["frac cos(gE,gF)<0"] = float(np.mean(d["cos"] < 0))
        m["median cos"] = float(np.median(d["cos"]))
        m["cos at centre"] = float(d["cos"][c, c])
        m["frac in-plane angle>90deg"] = float(np.mean(ang > 90))
        m["median |gF|/|gE|"] = float(np.median(d["nF"] / d["nE"]))
    return m


def write_table(path, rows):
    keys = [k for k in rows[0] if k != "R"]
    lines = ["metric".ljust(34) + "".join(f"+-{r['R']:g}".rjust(13) for r in rows)]
    for k in keys:
        lines.append(k.ljust(34) + "".join(
            (f"{r[k]:13.4g}" if isinstance(r[k], (int, float, np.floating)) else f"{r[k]:>13}") for r in rows))
    open(path, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def loss_figures(rs, rows, out):
    n = len(rs)
    fig, axes = plt.subplots(2, n, figsize=(3.4 * n, 6.8), squeeze=False)
    for j, (R, d) in enumerate(rs):
        X, Y = np.meshgrid(d["alphas"], d["betas"])
        c = len(d["alphas"]) // 2
        for i, (key, name) in enumerate((("LE", "energy"), ("LF", "force"))):
            ax = axes[i, j]
            Z = np.log10(d[key] / d[key][c, c])
            cs = ax.contourf(X, Y, Z, levels=20, cmap="viridis")
            ax.plot(0, 0, "r*", ms=11)
            ax.set_aspect("equal")
            ax.set_title(f"{name}, range +-{R:g}", fontsize=9)
            fig.colorbar(cs, ax=ax, fraction=0.046).set_label("log10(L / L at trained model)", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "zoom_loss_contours.png"), dpi=140)
    plt.close(fig)

    Rs = [r["R"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key, title in ((axes[0], "local_minima", "strict grid local minima"),
                           (axes[1], "frac_points_with_a_concave_axis", "fraction of points with a concave axis"),
                           (axes[2], "mean_abs_2nd_difference_log10", "mean |2nd difference| of log10 L")):
        ax.plot(Rs, [r["E " + key] for r in rows], "o-", label="energy L_E")
        ax.plot(Rs, [r["F " + key] for r in rows], "s-", label="force L_F")
        ax.set_xscale("log"); ax.set_xlabel("range (alpha, beta in [-R, R])"); ax.set_title(title, fontsize=10)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "zoom_roughness_vs_range.png"), dpi=140)
    plt.close(fig)


def conflict_figures(rs, rows, out):
    n = len(rs)
    fig, axes = plt.subplots(2, n, figsize=(3.4 * n, 6.8), squeeze=False)
    for j, (R, d) in enumerate(rs):
        X, Y = np.meshgrid(d["alphas"], d["betas"])
        cs = axes[0, j].contourf(X, Y, d["cos"], levels=np.linspace(-1, 1, 21), cmap="RdBu")
        axes[0, j].set_title(f"cos(g_E, g_F), range +-{R:g}", fontsize=9)
        fig.colorbar(cs, ax=axes[0, j], fraction=0.046)
        cs = axes[1, j].contourf(X, Y, inplane_angle(d), levels=np.linspace(0, 180, 19), cmap="magma")
        axes[1, j].set_title("in-plane angle g_E vs g_F (deg)", fontsize=9)
        fig.colorbar(cs, ax=axes[1, j], fraction=0.046)
        for ax in axes[:, j]:
            ax.plot(0, 0, "k*", ms=10); ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "zoom_conflict_maps.png"), dpi=140)
    plt.close(fig)

    Rs = [r["R"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(Rs, [r["frac cos(gE,gF)<0"] for r in rows], "o-", label="fraction with cos(g_E, g_F) < 0")
    axes[0].plot(Rs, [r["frac in-plane angle>90deg"] for r in rows], "s-", label="fraction with in-plane angle > 90 deg")
    axes[0].set_ylim(0, 1)
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].plot(Rs, [r["median cos"] for r in rows], "o-", label="median cos over the grid")
    axes[1].plot(Rs, [r["cos at centre"] for r in rows], "s-", label="cos at the trained model")
    axes[1].set_ylim(-1, 1)
    for ax in axes:
        ax.set_xscale("log"); ax.set_xlabel("range (alpha, beta in [-R, R])"); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "zoom_conflict_vs_range.png"), dpi=140)
    plt.close(fig)


def main():
    loss_dir, grad_dir = sys.argv[1], sys.argv[2]
    for d, kind in ((loss_dir, "loss"), (grad_dir, "gradient")):
        rs = load_ranges(d)
        if not rs:
            print(f"no complete ranges in {d}")
            continue
        out = os.path.join(d, "zoom_comparison")
        os.makedirs(out, exist_ok=True)
        rows = [metrics(R, data) for R, data in rs]
        print(f"=== {kind} sweep folder: {d}")
        write_table(os.path.join(out, "zoom_metrics.txt"), rows)
        if kind == "loss":
            loss_figures(rs, rows, out)
        else:
            conflict_figures(rs, rows, out)
    print("done")


if __name__ == "__main__":
    main()
