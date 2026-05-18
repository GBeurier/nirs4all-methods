# ArPLS — Asymmetrically reweighted penalized least squares

Baek et al. 2015. Same skeleton as AsLS / AirPLS but with a logistic weight update centred on the negative-residual statistics.

For each row `y`:

1. Initialize `w = 1`.
2. Repeat up to `max_iter`:
   - Solve `(diag(w) + λ D²ᵀD²) z = w · y` (pentadiagonal LDLT).
   - `d = y - z`; `d_neg = d[d < 0]` (negative residuals).
   - `μ⁻ = mean(d_neg)`, `σ⁻ = std(d_neg, ddof=1)` (sample std; clamped to `DBL_MIN`).
   - `new_w[i] = 1 / (1 + exp(2 * (d[i] - (2σ⁻ - μ⁻)) / σ⁻))`  (numerically stable two-branch logistic).
   - If `||new_w - w||_2 / max(||w||_2, ε) < tol`: break.
   - `w ← new_w`.
3. Output: `out[i] = y[i] - z[i]`.

## Parameters
- `lam: double` (default 1e5) — smoothness regularisation.
- `max_iter: int` (default 50) — iteration cap.
- `tol: double` (default 1e-3) — relative weight-change convergence.

## ABI
```c
c4a_status_t c4a_pp_arpls_create(c4a_pp_arpls_handle_t** out,
                                  double lam,
                                  int32_t max_iter, double tol);
void         c4a_pp_arpls_destroy(c4a_pp_arpls_handle_t* h);
c4a_status_t c4a_pp_arpls_transform(const c4a_pp_arpls_handle_t* h,
                                     c4a_matrix_view_t X, c4a_matrix_view_t out);
```

## Numerical contract
- Pentadiagonal symmetric LDLT solver.
- Sample std uses `ddof=1` (matches `numpy.std(d_neg, ddof=1)`).
- Std clamped to `DBL_MIN` to avoid divide-by-zero on degenerate cases (all-zero residuals).
- Logistic stabilised via two-branch computation: for `arg >= 0` use `exp(-arg) / (1 + exp(-arg))`; for `arg < 0` use `1 / (1 + exp(arg))`. Avoids overflow for `|arg| >> 700`.
- Parity tolerance vs frozen NumPy reference: `1e-7 abs / 1e-8 rel`.

## Reference
- Baek, S.-J.; Park, A.; Ahn, Y.-J.; Choo, J. (2015). "Baseline Correction Using Asymmetrically Reweighted Penalized Least Squares Smoothing." *Analyst*, 140 (1), 250–257.
- Frozen Python reference: `parity/python_generator/src/c4a_parity_pybaselines_ref/arpls.py`.
