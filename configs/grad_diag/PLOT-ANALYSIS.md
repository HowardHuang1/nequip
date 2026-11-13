# Plot analysis: energy and force loss landscapes of the tutorial NequIP model

What each of the ten plots shows and why it matters. Companion to `LANDSCAPE_FINDINGS.md` (results, caveats,
planned next run). Status: **preliminary**, one GPU job (Polaris 7841533), range alpha, beta in [-1, 1].

**How to read them.** Each plot is a 2D slice through the 418K-dimensional weight space around the trained model
(EMA weights of `outputs/2026-10-24/04-09-50/best.ckpt`). Axes alpha and beta are steps along two random,
filter-normalized directions: alpha = 1 shifts every filter by a random vector as large as the filter itself.
L_E = per-atom-energy MSE (center 1.23e-6), L_F = force MSE (center 3.42e-3, RMSE 0.0585 eV/A). Heights and
colours are log10 of the loss unless stated. Method: Li et al. 2018 (arXiv:1712.09913).

## Set 1: loss sweep, 31x31 grid, 1000 structures (`../landscape_loss_sweep/plots/`)

### 1. `energy_landscape_3d.png`
- **Shows:** log10 L_E as a 3D surface. The trained model sits in a narrow, needle-like pit below a higher,
  rippled floor. Several sharp downward spikes at the bottom (8 strict grid local minima), a raised ridge to the
  right of center, and a rise of ~18 orders of magnitude to the corners (max/center = 3.9e18).
- **Significance:** the energy minimum is extremely narrow. A 6.7% random perturbation (alpha = 0.067) takes the
  per-atom-energy RMSE from ~1.1 meV/atom to ~0.12 eV/atom, so just outside the pit the model is no longer
  useful. The spikes are candidates for ruggedness but may be undersampling of a narrow basin (grid step 0.067
  vs an estimated energy basin of ~0.002).

### 2. `force_landscape_3d.png`
- **Shows:** log10 L_F. A smooth, wide bowl with its low point at the trained model, no spikes (1 grid local
  minimum), the same raised ridge as the energy plot, and a rise of ~17 orders to the corners (5.9e17).
- **Significance:** at this scale the force loss is well behaved. At alpha = 0.067 the force RMSE only goes from
  0.058 to ~0.21 eV/A, versus a ~100x energy-RMSE jump. The shared ridge means both losses are hurt by the
  same weight direction, i.e. both depend on the same underlying parameters.

### 3. `landscape_contours.png` (energy left, force right)
- **Shows:** force = compact nested ellipses stretched along a diagonal, centered on the trained model.
  Energy = a curved, crescent-shaped valley with island-like contour loops; the trained model sits at the
  valley's edge.
- **Significance:** the energy loss is not a simple bowl at this scale, and its low-loss region does not
  coincide with the force loss's. Moving along the energy valley is not the same as descending the force bowl,
  one possible geometric source of energy/force conflict (interpretation, not tested).

### 4. `landscape_lines.png` (1D slices through the center, L / L at the trained model)
- **Shows:** both curves jump up in a V within one grid step. Near the center the energy curve is ~1000x above
  the force curve (930x at alpha = 0.067) and the two converge near |alpha| = 1 (~2x). Force is nearly
  monotone; energy has non-monotone bumps and dips at alpha ~ -0.4 to -0.6 and +0.6 to +0.85.
- **Significance:** quantifies the steepness difference. Both losses grow x3.85 when alpha doubles
  (0.067 -> 0.133), close to quadratic, so near the minimum both are smooth bowls that differ in width (energy
  basin ~30x narrower, rough estimate). The energy bumps are real irregularity but sit at losses 1e4-1e6 x the
  minimum, far from the basin.

## Set 2: gradient run, 17x17 grid, 200 structures (`../landscape_gradient_sweep/plots/`)

### 5-8. `energy_landscape_3d.png`, `force_landscape_3d.png`, `landscape_contours.png`, `landscape_lines.png`
- **Shows:** the same four views on a coarser grid (step 0.125) and a 5x smaller subset. Same needle pit with
  visible spikes, same ridge, same crescent versus ellipse, same dips in the 1D energy curves. Metrics:
  energy 6 local minima and 26% concave points; force 1 and 0.4%.
- **Significance:** the shapes reproduce across subset size and grid, so they come from the weights and the two
  random directions, not from which structures were used. Otherwise redundant with set 1 (which has the finer
  grid and more data).

### 9. `gradient_conflict_maps.png` (four panels)
- **Shows:**
  - *log10 |g_E|, log10 |g_F|:* both gradients are smallest at the center (~1e-2.4) and grow to ~1e14 (energy)
    and ~1e17 (force) at the corners. The energy map has a horseshoe-shaped low region mirroring the crescent
    valley; the force map is a compact blob.
  - *cos(g_E, g_F) in the full 418K-dimensional space:* aligned (blue, up to +0.96) at the edges, near zero
    around the center, with scattered red conflict patches in the mid-field (down to -0.72). Negative at 13.8%
    of grid points; +0.001 at the trained model.
  - *angle between the in-plane gradients:* mostly 0-24 degrees (same direction) with a few bright streaks up to
    ~168 degrees where the two losses slope in opposite directions; 6.6% of points exceed 90 degrees.
- **Significance:** the direct measurement of conflict. It is patchy and mid-field: far from the minimum both
  gradients just push the exploded weights back (trivial agreement), and conflict appears in separate spots and
  bands rather than everywhere. A cos of ~0 at the trained model is not evidence of orthogonality: at an exact
  stationary point of the weighted loss one expects g_E = -g_F (cos = -1); here the EMA weights are not an exact
  stationary point and the gradients are tiny, so the value is noise-dominated.

### 10. `gradient_field_inplane.png` (arrows = descent direction -grad L in the (alpha, beta) plane; colour = log10 magnitude)
- **Shows:** force = a clean inward field, every arrow pointing toward the trained model. Energy = arrows point
  inward in the outer region but near the center are scattered and lopsided: some run along the crescent, some
  point away from the star.
- **Significance:** the visual counterpart of plot 9 and of the energy-vs-force contrast in plot 3. Caveat: the
  arrows are the gradient projected onto only two of 418K directions, so they are the slope along the plane,
  not the full gradient; far-field arrows are trivially inward.

## What this adds up to

- Both losses share one global bowl and one ridge; they differ in width (energy ~30x narrower) and in how
  bowl-like they look (force smooth, energy crescent-shaped with spikes).
- Gradient conflict is real but patchy (14% of grid points, mid-field), aligned in the far field, ~0 at the
  trained model.
- **Not yet shown:** that the energy landscape is rugged near the trained model. The spikes and extra local
  minima come from a grid mostly in the far field and sampled far more coarsely than the energy basin, so they
  may be artifacts. The planned zoom runs (alpha down to +-0.001, plus the line to the initial weights) test this.
- **Relevance to the residual-connection idea:** the paper's finding that skip connections smooth the landscape
  is for deep networks (20-110 layers); this model has 4 layers and already has a learned, atom-type-
  conditioned self-connection in layers 1-3 (an identity residual, `convnet_resnet`, exists but is off). Unless
  the zoomed energy surface turns out to be genuinely rugged, the data here do not favour smoothness as the
  source of the conflict.

## Caveats (all plots)

One pair of random directions (seeds 0 and 1) in a 418K-dimensional space; +-1 is mostly far-field (the basin is
a few percent wide, ~3 grid points); heights are log10, which exaggerates small differences near the minimum;
the grid-dependent statistics (local-minima counts, concave fractions) differ between the 31x31 and 17x17 grids;
the gradient statistics come from a 200-structure subset; the center is the EMA weights.
