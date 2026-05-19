# BEADS — Baseline Estimation And Denoising with Sparsity

Ning & Selesnick 2014 BEADS, implemented against the
`pybaselines.Baseline().beads` full banded contract for the public c4a
parameter surface.

The exposed options are:

- `freq_cutoff = 0.005`
- `asymmetry = 6`
- `filter_type = 1`
- `cost_function = 2`
- `eps_0 = eps_1 = 1e-6`
- `fit_parabola = true`
- no derivative smoothing

For each row `y` of length `n`:

1. Subtract pybaselines' endpoint parabola.
2. Build the second-order high-pass filter matrices `A` and `B`.
3. Iterate the 9-diagonal BEADS system with L1-v2 derivative weights,
   asymmetric signal weights, and `lam_0 / lam_1 / lam_2`.
4. Reconstruct the low-pass baseline and output `out = y - baseline`.

## Parameters
- `lam_0: double` (default `1e2`) — signal sparsity/data-fidelity weight.
- `lam_1: double` (default `0.5`) — first-difference sparsity weight.
- `lam_2: double` (default `0.5`) — second-difference sparsity weight.
- `max_iter: int` (default `50`).
- `tol: double` (default `1e-3`) — relative objective-change convergence
  threshold.

## ABI
```c
c4a_status_t c4a_pp_beads_create(c4a_pp_beads_handle_t** out,
                                  double lam_0, double lam_1, double lam_2,
                                  int32_t max_iter, double tol);
void         c4a_pp_beads_destroy(c4a_pp_beads_handle_t* h);
c4a_status_t c4a_pp_beads_transform(const c4a_pp_beads_handle_t* h,
                                     c4a_matrix_view_t X, c4a_matrix_view_t out);
```

## Numerical Contract
- Full pybaselines-compatible banded formulation, specialized to
  `filter_type=1`.
- The iterative linear system has four lower and four upper diagonals.
- On `max_iter` reached without convergence, the last iterate is returned.
- External benchmark gate: `pybaselines.Baseline().beads` with the parameters
  above and `tol=1e-3`.

## Reference
- Ning, X., Selesnick, I. W. & Duval, L. (2014). "Chromatogram baseline
  estimation and denoising using sparsity (BEADS)." Chemometrics and
  Intelligent Laboratory Systems 139, 156-167.
- Pybaselines BEADS reference: `pybaselines.misc.beads`.
