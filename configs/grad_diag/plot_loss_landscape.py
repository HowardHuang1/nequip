"""Plot the output of ``loss_landscape.py`` (Li et al. 2018 style loss landscapes).

    python plot_loss_landscape.py LANDSCAPE_DIR [--out OUT_DIR]

Merges every ``landscape*.npz`` in LANDSCAPE_DIR (so row-sharded runs combine) and writes:

  energy_landscape_3d.png     3D surface of log10 L_E (per-atom-energy MSE) over (alpha, beta)
  force_landscape_3d.png      3D surface of log10 L_F (force MSE)
  landscape_contours.png      the same two surfaces as contour maps, centre marked
  landscape_lines.png         1D slices through the centre along delta and eta (L / L_centre)
  landscape_metrics.txt       roughness numbers comparing the energy and force surfaces
  gradient_field_inplane.png  (if --gradients was used) the in-plane descent vectors -grad L_E and
                              -grad L_F (projected on the (delta, eta) plane) as arrow fields
  gradient_conflict_maps.png  (if --gradients was used) |g_E|, |g_F|, cos(g_E, g_F) in the full
                              parameter space, and the angle between the in-plane vectors
"""

import argparse
import glob
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401


def load(d):
    files = sorted(glob.glob(os.path.join(d, "landscape*.npz")))
    assert files, f"no landscape*.npz in {d}"
    merged = None
    for f in files:
        z = np.load(f, allow_pickle=True)
        if merged is None:
            merged = {k: z[k] for k in z.files}
        else:
            for k in z.files:
                if z[k].ndim == 2:
                    merged[k] = np.where(np.isnan(merged[k]), z[k], merged[k])
    return merged


def surface3d(X, Y, L, title, path, center):
    Z = np.log10(L)
    fig = plt.figure(figsize=(8.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.95, rstride=1, cstride=1)
    zmin = np.nanmin(Z) - 0.15 * (np.nanmax(Z) - np.nanmin(Z))
    ax.contour(X, Y, Z, zdir="z", offset=zmin, levels=15, cmap="viridis", linewidths=0.6)
    ax.scatter([0], [0], [np.log10(center)], color="red", s=40, zorder=10, label="trained model")
    ax.set_zlim(zmin, np.nanmax(Z))
    ax.set_xlabel(r"$\alpha$ (direction $\delta$)")
    ax.set_ylabel(r"$\beta$ (direction $\eta$)")
    ax.set_zlabel("log10 loss")
    ax.view_init(elev=28, azim=-58)
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def roughness(L):
    """Simple, grid-dependent measures of how rugged a surface is (interior points)."""
    Z = np.log10(L)
    c = Z[1:-1, 1:-1]
    nb = [Z[1 + dj : Z.shape[0] - 1 + dj, 1 + di : Z.shape[1] - 1 + di]
          for dj in (-1, 0, 1) for di in (-1, 0, 1) if (dj, di) != (0, 0)]
    n_min = int(np.sum(np.all([c < n for n in nb], axis=0)))
    n_max = int(np.sum(np.all([c > n for n in nb], axis=0)))
    d2a = L[1:-1, :-2] - 2 * L[1:-1, 1:-1] + L[1:-1, 2:]
    d2b = L[:-2, 1:-1] - 2 * L[1:-1, 1:-1] + L[2:, 1:-1]
    frac_concave = float(np.mean((d2a < 0) | (d2b < 0)))
    rough = float(np.mean(np.abs(np.diff(Z, 2, axis=0))) + np.mean(np.abs(np.diff(Z, 2, axis=1))))
    return dict(local_minima=n_min, local_maxima=n_max,
                frac_points_with_a_concave_axis=frac_concave,
                mean_abs_2nd_difference_log10=rough)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.dir, "plots")
    os.makedirs(out, exist_ok=True)
    d = load(a.dir)
    xs, ys = d["alphas"], d["betas"]
    X, Y = np.meshgrid(xs, ys)
    LE, LF = d["LE"], d["LF"]
    if np.isnan(LE).any():
        print("WARNING: grid has NaN rows (sweep incomplete) -- plots will have gaps")
    ci, cj = int(np.argmin(np.abs(xs))), int(np.argmin(np.abs(ys)))
    cE, cF = LE[cj, ci], LF[cj, ci]
    info = (f"{int(d['n_structures'])} train structures, {int(d['n_params']):,} params, "
            f"grid {len(xs)}x{len(ys)}")

    surface3d(X, Y, LE, f"Energy loss landscape (per-atom-energy MSE)\n{info}",
              os.path.join(out, "energy_landscape_3d.png"), cE)
    surface3d(X, Y, LF, f"Force loss landscape (force MSE)\n{info}",
              os.path.join(out, "force_landscape_3d.png"), cF)

    # contours
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, L, name, c in ((axes[0], LE, "energy L_E", cE), (axes[1], LF, "force L_F", cF)):
        cs = ax.contourf(X, Y, np.log10(L), levels=25, cmap="viridis")
        ax.contour(X, Y, np.log10(L), levels=25, colors="k", linewidths=0.3, alpha=0.5)
        ax.plot(0, 0, "r*", ms=14)
        ax.set_title(f"log10 {name}  (centre = {c:.2e})")
        ax.set_xlabel(r"$\alpha$"); ax.set_ylabel(r"$\beta$"); ax.set_aspect("equal")
        plt.colorbar(cs, ax=ax)
    fig.tight_layout(); fig.savefig(os.path.join(out, "landscape_contours.png"), dpi=150); plt.close(fig)

    # 1D slices through the centre
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, (vec, t) in zip(axes, ((lambda L: L[cj, :], r"along $\delta$ ($\beta=0$)"),
                                    (lambda L: L[:, ci], r"along $\eta$ ($\alpha=0$)"))):
        ax.plot(xs, vec(LE) / cE, label="energy L_E / centre")
        ax.plot(xs, vec(LF) / cF, label="force L_F / centre")
        ax.set_yscale("log"); ax.set_title(t); ax.set_xlabel("step size"); ax.legend()
    axes[0].set_ylabel("loss / loss at trained model")
    fig.tight_layout(); fig.savefig(os.path.join(out, "landscape_lines.png"), dpi=150); plt.close(fig)

    # metrics
    lines = [f"{info}", f"cos(delta, eta) = {float(d['cos_dirs']):+.4f}",
             f"centre: L_E={cE:.4e}  L_F={cF:.4e}", ""]
    for name, L in (("ENERGY L_E", LE), ("FORCE  L_F", LF)):
        r = roughness(L)
        lines.append(f"{name}: max/centre = {np.nanmax(L)/L[cj, ci]:.3g}, "
                     f"min/centre = {np.nanmin(L)/L[cj, ci]:.3g}")
        lines += [f"    {k}: {v:.4g}" if isinstance(v, float) else f"    {k}: {v}" for k, v in r.items()]
    # --- gradient maps
    if "gE0" in d:
        gEa, gEb, gFa, gFb = d["gE0"], d["gE1"], d["gF0"], d["gF1"]
        step = max(1, len(xs) // 15)
        sl = (slice(None, None, step), slice(None, None, step))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
        for ax, (ga, gb, name) in zip(axes, ((gEa, gEb, "energy L_E"), (gFa, gFb, "force L_F"))):
            mag = np.hypot(ga, gb)
            ax.contourf(X, Y, np.log10(LE if name.startswith("energy") else LF), levels=25,
                        cmap="Greys", alpha=0.5)
            q = ax.quiver(X[sl], Y[sl], (-ga / (mag + 1e-30))[sl], (-gb / (mag + 1e-30))[sl],
                          np.log10(mag + 1e-30)[sl], cmap="plasma", angles="xy", pivot="mid")
            ax.plot(0, 0, "c*", ms=14); ax.set_aspect("equal")
            ax.set_title(f"descent direction -grad {name} in the (alpha, beta) plane\n"
                         "(arrow = direction only, colour = log10 magnitude)")
            plt.colorbar(q, ax=ax)
        fig.tight_layout(); fig.savefig(os.path.join(out, "gradient_field_inplane.png"), dpi=150); plt.close(fig)

        ang = np.degrees(np.arccos(np.clip((gEa * gFa + gEb * gFb) /
                                           (np.hypot(gEa, gEb) * np.hypot(gFa, gFb) + 1e-30), -1, 1)))
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        panels = ((np.log10(d["nE"]), "log10 |g_E| (full space)", "viridis", None),
                  (np.log10(d["nF"]), "log10 |g_F| (full space)", "viridis", None),
                  (d["cos"], "cos(g_E, g_F) in full parameter space", "RdBu", (-1, 1)),
                  (ang, "angle between in-plane g_E and g_F (deg)\n(0 = same, 180 = opposite)", "magma", (0, 180)))
        for ax, (Z, t, cm, lim) in zip(axes.ravel(), panels):
            kw = dict(vmin=lim[0], vmax=lim[1]) if lim else {}
            cs = ax.contourf(X, Y, Z, levels=25, cmap=cm, **kw)
            ax.plot(0, 0, "k*", ms=12); ax.set_title(t); ax.set_aspect("equal")
            plt.colorbar(cs, ax=ax)
        fig.tight_layout(); fig.savefig(os.path.join(out, "gradient_conflict_maps.png"), dpi=150); plt.close(fig)
        lines += ["", "GRADIENT MAPS (full parameter space)",
                  f"    fraction of grid points with cos(g_E, g_F) < 0: {np.nanmean(d['cos'] < 0):.3f}",
                  f"    cos at trained model: {d['cos'][cj, ci]:+.3f}",
                  f"    median |g_F|/|g_E| over grid: {np.nanmedian(d['nF'] / d['nE']):.3g}",
                  f"    fraction of grid points with in-plane angle > 90 deg: {np.nanmean(ang > 90):.3f}"]
    open(os.path.join(out, "landscape_metrics.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nfigures written to {out}")


if __name__ == "__main__":
    main()
