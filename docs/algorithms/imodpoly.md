# IModPoly — Improved Modified Polynomial baseline

Gan, Ruan & Mo 2006 σ-stopping variant of ModPoly. For each row `y` of length `n`:

1. Fit polynomial baseline `z`; compute `devr = std(y - z, ddof = 0)`.
2. Repeat up to `max_iter`:
   - Clip peaks: `y[i] := y[i] if y[i] < z[i] + devr else (z[i] + devr)`.
   - Re-fit polynomial -> new `z`.
   - `devr_new = std(y - z, ddof = 0)`.
   - `conv = |devr_new - devr| / max(devr_new, DBL_MIN)`.
   - `devr = devr_new`.
   - If `conv < tol`, break.
3. Output: `out = y_original - z`.

## Parameters
- `polyorder: int` (default 2).
- `max_iter: int` (default 250).
- `tol: double` (default 1e-3) — relative residual-stdev change convergence.

## ABI
```c
c4a_status_t c4a_pp_imodpoly_create(c4a_pp_imodpoly_handle_t** out,
                                     int32_t polyorder,
                                     int32_t max_iter, double tol);
void         c4a_pp_imodpoly_destroy(c4a_pp_imodpoly_handle_t* h);
c4a_status_t c4a_pp_imodpoly_transform(const c4a_pp_imodpoly_handle_t* h,
                                        c4a_matrix_view_t X, c4a_matrix_view_t out);
```

## Numerical contract
- Same Householder-QR Vandermonde factorisation as ModPoly (shared from `core/common/linalg.h`).
- Residual stdev uses ddof = 0 (population); the convergence check protects the denominator with `max(devr_new, DBL_MIN)`.
- `mask_initial_peaks` (pybaselines' optional pre-threshold step) is disabled; this matches `num_std = 1` default behaviour on clean spectra and keeps the parity surface compact.
- On `max_iter` reached without convergence: silently returns the last iterate.
- Parity tolerance vs frozen NumPy reference: `1e-9 abs / 1e-10 rel`.

## Reference
- Gan, F., Ruan, G. & Mo, J. (2006). "Baseline correction by improved iterative polynomial fitting with automatic threshold." Chemometrics and Intelligent Laboratory Systems 82(1-2), 59-65.
- Frozen Python reference: `parity/python_generator/src/c4a_parity_pybaselines_ref/imodpoly.py`.
