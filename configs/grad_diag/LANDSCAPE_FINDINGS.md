# Loss / gradient landscape of the tutorial NequIP model: findings so far

Status: **preliminary**. One GPU job (Polaris job 7841533, 2026-11-13, 1 node, ~4.5 min) at a single
range (alpha, beta in [-1, 1]). Method: Li et al. 2018 (arXiv:1712.09913), filter-wise normalized random
directions around the trained model (EMA weights of `outputs/2026-10-24/04-09-50/best.ckpt`).
Code: `loss_landscape.py`, `plot_loss_landscape.py`, `run_landscape.pbs`.
Data: `../landscape_loss_sweep/` (31x31, 1000 train structures, loss only) and `../landscape_gradient_sweep/`
(17x17, 200 train structures, with gradients). Plots in each `plots/` folder; raw numbers in
`landscape_metrics.txt`.

## How the results were produced

All three output sets came from one PBS job (7841533); a benchmark and a smoke test ran first and are not part of the results.

1. Loss sweep (`landscape_loss_sweep/`): 31×31 grid, 1,000 structures, no gradients. It ran as 4 processes, one per GPU, each doing a block of rows, which is why the folder has 4 .npz files.
2. Gradient sweep (`landscape_gradient_sweep/`): 17×17 grid, 200 structures, with gradients. Also 4 processes, so 4 .npz files.

### Why two sweeps

Both sweeps cover the same 2D plane around the trained model but compute different things at each grid point:

| | Loss sweep (`landscape_loss_sweep/`) | Gradient sweep (`landscape_gradient_sweep/`) |
|---|---|---|
| Computes | the two losses, L_E and L_F | the gradients of L_E and L_F w.r.t. all 418K weights, plus the losses |
| Gives | 3D surfaces, contours, 1D slices (landscape shape) | gradient sizes, cos(g_E, g_F), in-plane arrow fields (where the gradients conflict) |
| Cost per structure | cheap: ~2,000 structures/s per GPU | ~3.4-4.4x more (the force term needs a double backward) |
| Size chosen | 31×31 grid, 1,000 structures | 17×17 grid, 200 structures |

- **Why separate.** The loss sweep alone cannot show conflict: conflict is about gradient directions, not loss
  heights. The gradient sweep can, but costs several times more per point, so it used a smaller grid and subset
  (the whole job took ~4.5 min). The loss sweep could then use a finer grid and more data for the surface shape.
- **Overlap.** The gradient sweep also records L_E and L_F at each point, so `landscape_gradient_sweep/` has its own 3D and
  contour plots. They match the loss sweep's shapes (same needle pit, ridge, crescent and dips), which
  cross-checks the two runs and shows the features do not depend on the subset or grid. The loss sweep is the
  higher-resolution version of that part; running only the gradient sweep at the larger size would cost more.

## Results to put in the commit message (requested verbatim)

- The energy surface is rougher than the force surface. On the 31x31 grid it has 8 local minima (force: 1), and 27% of grid points have a concave direction (force: 3%).
- The force descent field is a smooth bowl. All the arrows point toward the trained model. The energy field is lopsided and irregular.
- Conflict is patchy. cos(g_E, g_F) is negative at 13.8% of grid points (as low as -0.72). It is about 0 at the trained model itself.

## Caveats and corrections (do not drop these when quoting the results)

1. **+-1 is mostly far-field.** alpha = 1 means every filter is perturbed by a random vector as large as the
   filter itself (100% relative). The loss there is 1e8-1e10 x its minimum along the axes and ~1e18 x at the
   corners. The basin is only a few percent wide, so the +-1 grid has ~3 points inside it. The roughness
   numbers above mostly describe the far field and may partly be **undersampling artifacts** (grid step 0.067
   is much larger than the energy basin, see 3). They do not yet say whether the landscape is rugged
   near the trained model.
2. **Sensitivity (corrects an earlier "~100x" remark).** Measured as L(alpha)/L(0) averaged over +-alpha along
   both random directions: at alpha = 0.067 the energy loss is x1.2e4 and the force loss x13, so the energy loss
   is ~930x more sensitive *relative to its own minimum* (the gap shrinks with alpha: 2x at alpha = 1).
   This is a relative statement. The absolute increases are dL_E = 0.015 (eV^2/atom^2) and dL_F = 0.041
   (eV^2/A^2); these have different units and cannot be compared directly. The energy minimum is simply very
   low (1.2e-6 vs 3.4e-3).
3. **Basin widths.** Loss growth from alpha = 0.067 to 0.133 is x3.85 for both losses (quadratic would be x4), so a
   quadratic bowl is a decent local model. Extrapolating, the loss reaches 10x its minimum at
   alpha ~ 0.0018 (energy) and ~ 0.058 (force): the energy basin is ~30x narrower. (Rough estimate.)
4. **Where is initialization?** Not shown on these plots: the init point is in a specific direction, random
   directions are ~orthogonal to it in 418K dimensions. In the CPU diagnostic run the weights had moved 28% of
   their norm from init after only 600 steps (~0.5 epoch), so init-to-trained distance is likely >= 0.3 in
   alpha-like units (not measured for the final model). The init losses (step 0, batch of 5 structures:
   L_E ~ 22.7, L_F ~ 28.5) are ~2e7 x and ~8e3 x the trained minima, comparable to random-direction
   perturbations of radius ~0.5-0.9, so +-1 is more than enough to reach init-level loss values.
5. **cos ~ 0 at the trained model is not evidence of orthogonality.** At an exact stationary point of the
   weighted-sum loss one expects g_E = -g_F (cos = -1). The EMA weights are not an exact stationary point and
   the gradients there are tiny (log10|g| ~ -2.4), so the value is noise-dominated.
6. 200-structure subset, 17x17 grid for the gradient maps; the gradient-map statistics are coarse.

## Steep versus rugged: what the 930x figure does and does not show

- **Definition.** alpha = step along a filter-normalized random direction (alpha 0.067 shifts each filter by 6.7%
  of its own norm). "930x" = (L_E/L_E,min) / (L_F/L_F,min) at that step = 12,100 / 13. Supersedes the earlier
  misread "~100x".
- **Shows: the energy loss is steeper** (narrower basin, ~30x by the quadratic estimate). Both losses grow x3.85
  when alpha doubles, so near the minimum both look like smooth, nearly quadratic bowls that differ in width, not shape.
- **Does not show: that energy is more erratic.** Steep is not rugged. The "rugged" evidence (8 vs 1 grid minima,
  27% vs 3% concave points, lopsided arrows) comes from the +-1 grid: mostly far field, sampled at a step (0.067)
  much larger than the energy basin (~0.002), so it may be a sampling artifact.
- **Test:** the zoom runs. A smooth bowl at alpha ~0.002-0.003 means artifact; several minima or irregular arrows
  there means the energy landscape really is more irregular.

## Planned next run (not yet submitted)

Zoom ladder alpha in +-0.1, 0.03, 0.01, 0.003, 0.001 (basin estimates above: force ~0.06, energy ~0.002),
finer grids (loss 41x41, gradients 21x21), plus the 1D interpolation theta* -> theta_init
(same model seed) to place initialization on the loss scale. Key test: is the energy surface still rougher
than the force surface at its own basin scale, or were the local minima an undersampling artifact?
