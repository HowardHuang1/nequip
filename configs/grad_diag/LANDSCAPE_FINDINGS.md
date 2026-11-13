# Loss / gradient landscape of the tutorial NequIP model: findings

Status: six zoom levels complete (2026-11-13). Method: Li et al. 2018 (arXiv:1712.09913), filter-wise
normalized random directions around the trained model (EMA weights of `outputs/2026-10-24/04-09-50/best.ckpt`).
Two Polaris jobs, 1 node (4 A100) each: 7843537 (`debug`; ranges +-1, +-0.1, +-0.03; 27.6 min) and 7843804
(`debug-scaling`; ranges +-0.01, +-0.003, +-0.001; 27.7 min). Code: `loss_landscape.py`,
`plot_loss_landscape.py`, `compare_ranges.py`, `run_landscape.pbs`.
Results: `../landscape_loss_sweep/` and `../landscape_gradient_sweep/`, each with one `range_<R>/` folder per
zoom level (raw `.npz` data and `plots/`) and a `zoom_comparison/` folder (cross-range figures and
`zoom_metrics.txt`). Companion: `PLOT-ANALYSIS.md` (what each plot shows).

## Key takeaways

Zoomed out (+-1) versus zoomed in (+-0.1 and smaller), 31x31 grid, 1000 structures:

| | +-1 | +-0.1 and smaller |
|---|---|---|
| Energy surface | crescent-shaped valley, ridge, 8 local minima | straight narrow canyon, smooth V-shaped trough at +-0.001 |
| Force surface | smooth bowl | smooth tilted ellipse |
| Corner loss ~ R^2 | no, far-field | yes, for both losses |
| Negative cos(g_E, g_F) | 14.5% of points | falls to ~6% by +-0.1, then flat: 5.7-6.5% from +-0.1 to +-0.001 |
| Median abs(g_F)/abs(g_E) | 123 | about 0.75-0.79 |

**Implication for your hypothesis:** near the trained model the landscape is smooth, not jagged, so smoothing it with residual connections isn't supported as a fix for this conflict. Two limits apply: this is one random plane, and the early, far-from-minimum phase of training may behave more like the ±1 view.

## How the results were produced

- **Same settings everywhere.** Both sweeps use a 31x31 grid and the same 1000 structures (the first 1000 of the
  training split), and the same two random directions (seeds 0 and 1) at every range, so each smaller range is a
  true zoom of the same plane. alpha, beta in [-R, R] with R = 1, 0.1, 0.03, 0.01, 0.003, 0.001.
- **Four processes per sweep** (one per GPU), each computing a block of grid rows, hence 4 `.npz` files per range.
- **An earlier first run** used a 31x31 / 1000-structure loss sweep but only a 17x17 / 200-structure gradient
  sweep at +-1 (sized to be sure it fit the 1-hour queue limit). The mismatch made the two hard to compare, so it
  was replaced by the matched runs above. The new +-1 loss statistics reproduce the old ones exactly.

### Loss sweep vs gradient sweep

| | Loss sweep (`landscape_loss_sweep/`) | Gradient sweep (`landscape_gradient_sweep/`) |
|---|---|---|
| Computes | the two losses, L_E and L_F | the gradients of L_E and L_F w.r.t. all 418K weights, plus the losses |
| Gives | 3D surfaces, contours, 1D slices (landscape shape) | gradient sizes, cos(g_E, g_F), in-plane arrow fields (conflict) |
| Cost | ~2,040 structures/s per GPU | ~630 structures/s per GPU (3.2x; the force term needs a double backward) |

The gradient sweep also records L_E and L_F at every point (gradients need the loss), so the two folders' loss
numbers agree exactly, which cross-checks the runs. Conflict is about gradient directions, so only the gradient
sweep can show it.

## Main results across zoom levels

| range | L_E corner / centre | L_F corner / centre | energy local minima (force) | frac cos(g_E,g_F) < 0 | median cos | median abs(g_F)/abs(g_E) |
|---|---|---|---|---|---|---|
| +-1 | 3.9e18 | 5.9e17 | 8 (1) | 14.5% | 0.43 | 123 |
| +-0.1 | 1.6e5 | 99 | 9 (1) | 5.7% | 0.40 | 0.79 |
| +-0.03 | 1.4e4 | 8.8 | 9 (1) | 5.9% | 0.41 | 0.75 |
| +-0.01 | 1,552 | 1.85 | 9 (1) | 6.0% | 0.41 | 0.75 |
| +-0.003 | 144 | 1.08 | 9 (1) | 6.5% | 0.41 | 0.76 |
| +-0.001 | 17.8 | 1.009 | 9 (1) | 6.2% | 0.41 | 0.77 |

cos(g_E, g_F) at the trained model is +0.40. Losses at the centre: L_E = 1.23e-6 (1.1 meV/atom RMSE),
L_F = 3.42e-3 (0.0585 eV/A RMSE).

## Patterns: zoomed in versus zoomed out

1. **Zoomed in, both losses are smooth quadratic bowls.** From +-0.1 down to +-0.001 the corner loss falls like
   R^2 for both (energy 1.6e5 -> 17.8, force 99 -> 1.009), and the fitted curvatures are identical at +-0.01,
   +-0.003 and +-0.001. Zoomed out (+-1) the surface is not quadratic: a curved crescent-shaped energy valley, a
   raised ridge shared by both losses, 8 local minima and 27% concave points for the energy loss.
2. **The energy loss is a narrow straight canyon.** A quadratic fit to the 2D plane gives, in units of L/L0 per
   alpha^2: energy stiff and soft curvatures 2.0e7 and ~0.9-1.0e4 (ratio ~2,000-2,400); force 9.7e3 and 2.3e3 (ratio
   4.2, a mildly elongated ellipse). The energy canyon is very steep across (loss doubles after a step of ~3e-4) and
   flat along its floor (floor loss 0.97-0.995 x the centre value over the whole +-0.001 window).
3. **Energy and force share principal axes in this plane** (within 3.3 degrees at every zoom <= 0.01).
4. **In absolute loss units the stiff curvatures are comparable** (energy ~25, force ~33), while along the soft axis the
   energy loss is ~700x flatter (~0.01 against ~7.9). The much larger "relative" energy sensitivity (~930x at
   alpha = 0.067, ~1,800x at the corners) mostly reflects its very low minimum value (1.2e-6 vs 3.4e-3).
5. **The extra energy local minima are a sampling artifact, not roughness:** the straight canyon floor crosses 9
   grid columns at every range from +-0.1 to +-0.001 (one sampled dip per crossing), and the valley is smooth
   once resolved at +-0.001 (see the 3D plot).
6. **Conflict is a thin wedge along the canyon. The negative-cos fraction falls as you zoom in from +-1, then
   stays flat.** The fraction of points with negative cos(g_E, g_F) falls from 14.5% at +-1 to 5.7% at +-0.1, then
   stays at 5.7-6.5% at every range down to +-0.001. From +-0.1 down the cos map has the same shape at every zoom, so
   the conflict there is set by the local quadratic structure and not by higher-order ruggedness; the extra
   conflict at +-1 comes from the non-quadratic far field. The in-plane angle exceeds 90 degrees at 6.8% of points at
   +-1 and only 1.2-2.6% from +-0.1 down.
7. **The two gradients are comparable near the trained model but not far from it.** Median abs(g_F)/abs(g_E) is
   ~0.75-0.79 for R <= 0.1, against 123 at +-1 where the force gradient dominates. The median cos is ~+0.41
   throughout (aligned on average).

## Interpretation (hypotheses are marked)

- Near the trained model the energy and force landscapes are smooth; the conflict there does not come from a
  jagged surface. The energy loss is **ill-conditioned** (a ~2,000x stiffness ratio between canyon directions),
  not rugged. Residual connections target jagged landscapes, so they are unlikely to address what is seen here.
  This is limited to one random plane (see caveats), and says nothing about the early, far-from-minimum phase of
  training, where the +-1 view (irregular energy field, 14.5% conflict) may be more relevant.
- *Hypothesis, not tested:* across a canyon, g_E points toward the floor and flips sign depending on the side, so
  stepping across it produces a gradient that is almost one direction with a flipping sign. That matches the
  earlier training diagnostic (energy gradient nearly identical between batches, sign flipping in ~12% of steps).

## Requested commit-message bullets (from the earlier +-1 run) and what superseded them

- The energy surface is rougher than the force surface. On the 31×31 grid it has 8 local minima (force: 1), and 27% of grid points have a concave direction (force: 3%).
- The force descent field is a smooth bowl. All the arrows point toward the trained model. The energy field is lopsided and irregular.
- Conflict is patchy. cos(g_E, g_F) is negative at 13.8% of grid points (as low as −0.72). It is about 0 at the trained model itself.

Status against the matched 31x31 / 1000-structure runs: bullet 1 holds at +-1 but the zoom shows the extra minima
are a sampling artifact of the energy canyon (pattern 5). Bullet 2 holds at +-1; zoomed in the energy field is a
smooth convergent field toward the canyon floor. Bullet 3 changed: 14.5% (not 13.8%) of +-1 grid points have negative
cos, and cos at the trained model is **+0.40**, not about 0; the earlier near-zero value came from the
200-structure subset and was noise.

## Caveats and corrections

1. **One random plane.** A 2D slice with two random directions in a 418K-dimensional space; per the paper, its
   curvature is a weighted average of the full Hessian, so it likely understates the stiffest directions. Only seeds
   0 and 1 were used.
2. **+-1 is far-field.** alpha = 1 shifts every filter by its own norm; the basin is a few percent wide there.
3. **Sensitivity.** Earlier "about 100x" was a misread; the measured ratio is ~930x at alpha = 0.067 and is a *relative*
   statement (see pattern 4).
4. **Cos ~ 0 at the trained model** was noise from the small subset; at an exact stationary point of the weighted loss
   one would expect g_E = -g_F, but the EMA weights are not one, and the gradients there are tiny.
5. **Initialization** is not shown on these plots (random directions are ~orthogonal to the init direction). In the
   CPU diagnostic the weights had moved 28% of their norm after 600 steps; the distance for the final model was
   not measured.
6. Surface heights are log10, which exaggerates small differences near the minimum; grid-dependent statistics
   (local-minima counts, concave fractions) depend on the grid.

## Next steps (not yet done)

- More random direction pairs, to check the canyon and the conflict wedge are generic.
- Use true Hessian directions (Hessian-vector products) instead of random ones, to measure the real stiffness.
- Add the 1D line from the trained weights to the initial weights (same model seed); repeat at an early checkpoint.
- Test the canyon hypothesis against training (which side of the floor each step lands on), and try loss
  normalization or preconditioning, before considering residual connections (`convnet_resnet`, currently off).
