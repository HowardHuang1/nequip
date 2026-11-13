# Plot analysis: energy and force loss landscapes of the tutorial NequIP model

What the plots show and why they matter. Companion to `LANDSCAPE_FINDINGS.md` (results, interpretation, caveats).
Six zoom levels (alpha, beta in [-R, R], R = 1, 0.1, 0.03, 0.01, 0.003, 0.001), all with a 31x31 grid and the same
1000 structures, for both the loss sweep and the gradient sweep.

**Layout.** `../landscape_loss_sweep/` and `../landscape_gradient_sweep/` each contain `range_<R>/plots/` (per-range
plots) and `zoom_comparison/` (cross-range figures). The loss sweep makes 4 plots per range; the gradient sweep
makes the same 4 plus 2 gradient plots (it also records the losses).

**How to read them.** Each plot is a 2D slice through the 418K-dimensional weight space around the trained model
(EMA weights of `outputs/2026-10-24/04-09-50/best.ckpt`). alpha and beta are steps along two random,
filter-normalized directions (alpha = 1 shifts every filter by a random vector as large as the filter itself).
L_E = per-atom-energy MSE (centre 1.23e-6), L_F = force MSE (centre 3.42e-3, RMSE 0.0585 eV/A). Heights and
colours are log10 of the loss unless stated. Method: Li et al. 2018 (arXiv:1712.09913).

## Outline: all the plots at a glance

There are **64 plots of 10 types**: 6 plot types per range (4 loss plots in each folder plus 2 gradient plots in
the gradient folder) x 6 ranges = 60, plus 4 cross-range comparison plots. The gradient folder repeats the 4 loss
plots (gradients need the loss) and the numbers match the loss folder exactly.

**Per-range plots** (in each `range_<R>/plots/`, R = 1, 0.1, 0.03, 0.01, 0.003, 0.001):

| Plot | Folder | What it shows | What it tells us |
|---|---|---|---|
| `energy_landscape_3d.png` | both | the energy loss as a 3D surface | At +-1 a needle-like pit with spikes and a ridge; at +-0.03 a straight narrow canyon; at +-0.001 a clean, smooth V-shaped trough. The trained model sits in a very narrow minimum. |
| `force_landscape_3d.png` | both | the force loss as a 3D surface | A smooth bowl with a single minimum at every zoom: the force loss is well behaved at all scales. |
| `landscape_contours.png` | both | energy (left) and force (right) as contour maps | Energy contours are a crescent at +-1 and a straight tilted canyon zoomed in; force contours are a tilted ellipse at every scale. |
| `landscape_lines.png` | both | 1D slices through the centre along the two random directions | Both losses become parabolas zoomed in; the energy curve starts ~1,000x above the force curve near the centre and the gap shrinks with distance. |
| `gradient_conflict_maps.png` | gradient only | gradient sizes, cos(g_E, g_F) and the in-plane angle | The direct conflict measurement: 14.5% of points have negative cos at +-1, 5.7-6.5% zoomed in, as a thin wedge along the canyon. |
| `gradient_field_inplane.png` | gradient only | arrows showing each loss's descent direction | Force arrows converge cleanly on the centre; energy arrows converge on the canyon floor from both sides, and are irregular elsewhere at +-1. |

**Cross-range comparison plots** (in each folder's `zoom_comparison/`):

| Plot | Folder | What it shows | What it tells us |
|---|---|---|---|
| `zoom_loss_contours.png` | loss | all 6 ranges side by side, energy over force | The energy canyon and force ellipse look the same at every zoom from +-0.1 to +-0.001: these are the true local shapes, with principal axes within 3.3 degrees. |
| `zoom_roughness_vs_range.png` | loss | local minima, concave fraction and roughness against range | Roughness falls with zoom except the energy minima count, which stays at 9 (the sampling artifact). |
| `zoom_conflict_maps.png` | gradient | cos map over in-plane-angle map for all 6 ranges | The conflict pattern is identical from +-0.1 to +-0.001, a thin wedge along the canyon: set by the local quadratic structure, not jaggedness. |
| `zoom_conflict_vs_range.png` | gradient | conflict fractions and median cos against range | The negative-cos fraction falls as you zoom in, from 14.5% at +-1 to ~6% at +-0.1, then stays flat (5.7-6.5%) down to +-0.001; median cos ~+0.41 at every zoom. |

Detailed analysis of each plot type follows in sections A-C.

## A. Zoomed out (`range_1/plots/`)

### `energy_landscape_3d.png`
- **Shows:** log10 L_E. The trained model sits in a narrow, needle-like pit below a higher, rippled floor; several
  sharp downward spikes (8 strict grid local minima), a raised ridge right of centre, and a rise of ~18 orders of
  magnitude to the corners (3.9e18 x the centre).
- **Significance:** the energy minimum is extremely narrow: a 6.7% random perturbation (alpha = 0.067) takes the
  per-atom-energy RMSE from ~1.1 meV/atom to ~0.12 eV/atom. The spikes look like ruggedness, but the zoom-ins show
  they are a sampling artifact (grid step far larger than the canyon width).

### `force_landscape_3d.png`
- **Shows:** log10 L_F. A smooth, wide bowl with its low point at the trained model, no spikes (1 local minimum),
  the same raised ridge, ~17 orders of magnitude to the corners (5.9e17).
- **Significance:** the force loss is well behaved at every scale. At alpha = 0.067 the force RMSE only goes from
  0.058 to ~0.21 eV/A. The shared ridge means both losses are hurt by the same weight direction.

### `landscape_contours.png` (energy left, force right)
- **Shows:** force = nested ellipses stretched along a diagonal; energy = a curved crescent-shaped valley with
  island-like contour loops, the trained model at the valley's edge.
- **Significance:** the far-field energy surface is not a simple bowl. This curvature disappears when zooming in
  (section C), where the valley is straight.

### `landscape_lines.png` (1D slices through the centre, L / L at the trained model)
- **Shows:** both curves jump up in a V within one grid step; the energy curve sits ~1000x above the force curve
  near the centre and converges to it near |alpha| = 1; the energy curve has non-monotone bumps at alpha ~ -0.4 to
  -0.6 and +0.6 to +0.85.
- **Significance:** quantifies the relative steepness difference (~930x at alpha = 0.067). The bumps are far-field
  (loss 1e4-1e6 x the minimum), not near the basin.

### `gradient_conflict_maps.png` (four panels; gradient sweep only)
- **Shows:** log10 abs(g_E) and abs(g_F) (smallest at the centre, ~1e14-1e17 at the corners); cos(g_E, g_F) in the
  full 418K-dimensional space (aligned at the edges, scattered negative patches in the middle, down to -0.72); and
  the angle between the in-plane gradients (mostly small, with streaks up to ~168 degrees).
- **Significance:** the direct conflict measurement at +-1: cos(g_E, g_F) < 0 at 14.5% of points, in-plane angle
  above 90 degrees at 6.8%, median abs(g_F)/abs(g_E) = 123 (the force gradient dominates far from the minimum).
  Cos at the trained model itself is +0.40.

### `gradient_field_inplane.png` (arrows = descent direction in the plane, colour = log10 magnitude)
- **Shows:** force = a clean inward field; energy = arrows inward in the outer region but irregular and lopsided
  in the interior.
- **Significance:** the visual counterpart of the contour and conflict plots at +-1. Caveat: the arrows are the
  gradient projected onto only two of 418K directions.

## B. Zoomed in (`range_0.1` ... `range_0.001`, same six plots per range)

Same plot types as section A; what changes as R shrinks:
- **Energy surface:** from a spiky crescent (+-1) to a **straight narrow canyon** at +-0.1 and +-0.03 (steep
  across, nearly flat along the floor, a string of small ripples along the floor), then a clean, smooth V-shaped
  trough at +-0.001 (3D plot) with a flat floor and no spikes.
- **Force surface:** always a smooth tilted elliptical bowl.
- **1D slices:** both curves become parabolas; at +-0.001 the corner loss is 17.8x (energy) and 1.009x (force) the
  centre value.
- **Gradient field:** at +-0.001 the energy arrows converge on the canyon floor from both sides (and are tiny and
  zig-zag along the floor); the force arrows converge toward the centre of the ellipse.
- **Significance:** the near-minimum landscape is smooth and quadratic; the apparent ruggedness was the straight
  canyon sampled by a grid wider than the valley.

## C. Cross-range comparison (`zoom_comparison/`)

Files (all under `/home/howardhuang/nequip/configs/`; note the conflict plots are in the gradient folder):
- `landscape_loss_sweep/zoom_comparison/zoom_loss_contours.png`
- `landscape_loss_sweep/zoom_comparison/zoom_roughness_vs_range.png`
- `landscape_gradient_sweep/zoom_comparison/zoom_conflict_maps.png`
- `landscape_gradient_sweep/zoom_comparison/zoom_conflict_vs_range.png`
Each folder's `zoom_metrics.txt` holds the numbers behind them.

### `zoom_loss_contours.png` (loss folder; energy row over force row, one column per range)
- **Shows:** log10(L / L at trained model) at each zoom. Energy: crescent at +-1, then the same straight tilted
  canyon at every zoom from +-0.1 to +-0.001 (self-similar). Force: the same tilted ellipse at every zoom.
- **Significance:** the picture does not change below +-0.1, so these are the true local (quadratic) shapes, and the
  energy canyon and force ellipse point in nearly the same direction (principal axes within 3.3 degrees).

### `zoom_roughness_vs_range.png` (loss folder)
- **Shows:** energy local minima (8-9 at every range; force 1), fraction of points with a concave axis (energy
  27% -> 10% -> 0; force ~0), and mean abs second difference of log10 L (energy 0.43 -> 0.009; force 0.23 -> 2e-5).
- **Significance:** all roughness measures fall with zoom except the minima count, which stays at 9: the
  signature of the sampling artifact (the straight canyon floor crosses 9 grid columns at every range).

### `zoom_conflict_maps.png` (gradient folder; cos map over angle map, one column per range)
- **Shows:** at +-1 irregular negative patches. From +-0.1 to +-0.001 an identical pattern: positive cos nearly
  everywhere, with a thin negative wedge along the canyon axis; the in-plane angle is large only along the same wedge.
- **Significance:** the conflict is a scale-invariant wedge of directions, set by the local quadratic structure of
  the two losses, not by higher-order jaggedness.

### `zoom_conflict_vs_range.png` (gradient folder)
- **Shows:** fraction of points with cos < 0: it falls from 14.5% at +-1 to 5.7% at +-0.1, then stays flat at
  5.7-6.5% down to +-0.001;
  in-plane angle above 90 degrees: 6.8% -> 1.2-2.6%; median cos and cos at the trained model both ~+0.40-0.43.
- **Significance:** near the trained model, about 6% of directions in this plane conflict and the gradients are
  aligned on average; the conflict does not grow or shrink as the neighbourhood shrinks.

## What this adds up to

- Zoomed out (+-1): non-quadratic, irregular energy surface, 14.5% conflict, force gradient dominating by ~100x.
- Zoomed in (+-0.1 and below): smooth quadratic bowls; the energy loss is an ill-conditioned canyon (stiff/soft
  curvature ~2,000) aligned with the force ellipse; ~6% of directions conflict, scale-invariantly.
- Energy "ruggedness" near the minimum was a sampling artifact. Conflict near the trained model is not explained by
  a jagged landscape, so residual connections (which target jaggedness) are not supported as a fix for it here.

## Caveats

One random plane (seeds 0 and 1) in a 418K-dimensional space, which likely understates the stiffest curvature;
+-1 is mostly far-field; heights are log10; grid-dependent statistics depend on the grid; initialization is not
shown on these plots. See `LANDSCAPE_FINDINGS.md`.
