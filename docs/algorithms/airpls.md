# AirPLS — Adaptive iteratively reweighted penalized least squares

Zhang 2010 baseline correction. Same skeleton as AsLS but with an exponential weight update that concentrates on negative residuals.

For each row `y`:

1. Initialize `w = 1`, `y_l1 = sum(|y|)`.
2. Repeat up to `max_iter` (iteration index `i`, starting at 1):
   - Solve `(diag(w) + λ D²ᵀD²) z = w · y` (pentadiagonal LDLT).
   - `d = y - z`; `d_neg = -min(d, 0)` (magnitude of negative residuals).
   - `sum_neg = sum(d_neg)`.
   - If `y_l1 > 0` and `sum_neg / y_l1 < tol`: break.
   - If `count(d_neg > 0) < 2`: break (no negative residuals to fit).
   - `t = min(i, 50)` (clip exponent multiplier).
   - `new_w[i] = exp(clip(t * d_neg / sum_neg, [0, log(DBL_MAX) - spacing]))` on negative-residual indices, `0` elsewhere.
   - `w ← new_w`.
3. Output: `out[i] = y[i] - z[i]`.

## Parameters
- `lam: double` (default 1e6) — smoothness regularisation.
- `max_iter: int` (default 50) — iteration cap.
- `tol: double` (default 1e-2) — convergence threshold on `sum(|d_neg|) / ||y||_1`.

## ABI
```c
c4a_status_t c4a_pp_airpls_create(c4a_pp_airpls_handle_t** out,
                                   double lam,
                                   int32_t max_iter, double tol);
void         c4a_pp_airpls_destroy(c4a_pp_airpls_handle_t* h);
c4a_status_t c4a_pp_airpls_transform(const c4a_pp_airpls_handle_t* h,
                                      c4a_matrix_view_t X, c4a_matrix_view_t out);
```

## Numerical contract
- Pentadiagonal symmetric LDLT solver.
- Exp argument clipped to `[0, log(DBL_MAX) - spacing(log(DBL_MAX))]` to avoid `+inf` from large iterates (matches pybaselines `_airpls_loop` overflow guard).
- Convergence metric `sum(d_neg) / ||y||_1` evaluated BEFORE weight update (matches pybaselines).
- Parity tolerance vs internal parity fixture: `1e-7 abs / 1e-8 rel` for typical regularisation; `1e-6 abs / 1e-7 rel` for stiff regularisation (`lam >= 1e7`) where LDLT-vs-spsolve ULP differences compound across iterations.

## Reference
- Zhang, Z.-M.; Chen, S.; Liang, Y.-Z. (2010). "Baseline Correction Using Adaptive Iteratively Reweighted Penalized Least Squares." *Analyst*, 135 (5), 1138–1146.
- Internal parity fixture: `parity/python_generator/src/c4a_parity_pybaselines_ref/airpls.py`.
