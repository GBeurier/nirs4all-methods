# AsLS — Asymmetric Least Squares baseline

Eilers & Boelens 2005 asymmetric least squares baseline estimation. For each row `y` of length `n`:

1. Initialize weights `w = 1`.
2. Repeat up to `max_iter`:
   - Solve `(diag(w) + λ D²ᵀD²) z = w · y` (pentadiagonal LDLT).
   - Compute new weights: `new_w[i] = p` if `y[i] > z[i]` else `1 - p`.
   - If `||new_w - w||_2 / max(||w||_2, ε) < tol`, break.
   - `w ← new_w`.
3. Output: `out[i] = y[i] - z[i]` (baseline-subtracted spectrum).

## Parameters
- `lam: double` (default 1e6) — smoothness regularisation. Higher = smoother baseline.
- `p: double` (default 0.001) — asymmetry. `p ∈ (0, 1)`; values close to 0 pull the baseline below peaks.
- `max_iter: int` (default 50) — iteration cap.
- `tol: double` (default 1e-3) — relative weight-change convergence threshold.

## ABI
```c
n4m_status_t n4m_pp_asls_create(n4m_pp_asls_handle_t** out,
                                 double lam, double p,
                                 int32_t max_iter, double tol);
void         n4m_pp_asls_destroy(n4m_pp_asls_handle_t* h);
n4m_status_t n4m_pp_asls_transform(const n4m_pp_asls_handle_t* h,
                                    n4m_matrix_view_t X, n4m_matrix_view_t out);
```

## Numerical contract
- Pentadiagonal symmetric LDLT solver in `core/common/banded_solver.{c,h}` (O(n) per iteration, vs pls4all's dense O(n²)).
- Per-row scratch buffers reused across all iterations to minimize allocator churn.
- Convergence uses `relative_difference(w, new_w)` matching pybaselines's `_safe_relative_difference` (L2 norms with `max(||w||, DBL_MIN)` denominator).
- On `max_iter` reached without convergence: silently returns the last iterate (matches pybaselines behaviour).
- Parity tolerance vs internal parity fixture: `1e-7 abs / 1e-8 rel`.

## Reference
- Eilers, P. H. C. & Boelens, H. F. M. (2005). "Baseline Correction with Asymmetric Least Squares Smoothing." Leiden University Medical Centre Report.
- Internal parity fixture: `parity/python_generator/src/n4m_parity_pybaselines_ref/asls.py` (historically validated against `pybaselines==1.1.4`); current upstream `pybaselines` drift is tracked by the reference parity gate.
