# IAsLS — Improved Asymmetric Least Squares baseline

He et al. 2014 IAsLS baseline, aligned with
`pybaselines.whittaker.Baseline.iasls` for `diff_order = 2`. For each row `y`
of length `n`:

1. Fit polynomial of degree `polyorder` to `(positions, y)` -> initial baseline `z0`.
2. Initialise weights `w[i] = p` if `y[i] > z0[i]` else `1 - p`.
3. Build the fixed first-derivative residual term
   `d1_y = lam_1 * (D1^T D1) y`.
4. Repeat up to `max_iter`:
   - Solve `(diag(w^2) + lam D2^T D2 + lam_1 D1^T D1) z =
     w^2 * y + d1_y` (pentadiagonal LDLT).
   - `w_new[i] = p` if `y[i] > z[i]` else `1 - p`.
   - If `rel_l2_diff(w, w_new) < tol`, break.
   - `w = w_new`.
5. Output: `out = y - z`.

## Parameters
- `lam: double` (default 1e6) — smoothness regularisation.
- `p: double` (default 1e-2) — asymmetry weight, `p ∈ (0, 1)`.
- `lam_1: double` (default 1e-4) — first-derivative residual penalty.
- `polyorder: int` (default 2) — polynomial pre-fit degree.
- `diff_order: int` (default 2) — supported value in this ABI revision.
- `max_iter: int` (default 50).
- `tol: double` (default 1e-3) — relative weight-change convergence threshold.

## ABI
```c
n4m_status_t n4m_pp_iasls_create(n4m_pp_iasls_handle_t** out,
                                  double lam, double p,
                                  int32_t polyorder,
                                  int32_t max_iter, double tol);
n4m_status_t n4m_pp_iasls_create_ex(n4m_pp_iasls_handle_t** out,
                                     double lam, double p,
                                     double lam_1,
                                     int32_t polyorder,
                                     int32_t diff_order,
                                     int32_t max_iter, double tol);
void         n4m_pp_iasls_destroy(n4m_pp_iasls_handle_t* h);
n4m_status_t n4m_pp_iasls_transform(const n4m_pp_iasls_handle_t* h,
                                     n4m_matrix_view_t X, n4m_matrix_view_t out);
```

## Numerical contract
- Shared 1st/2nd-difference penalties + in-place pentadiagonal LDLT from `core/common/banded_solver.h`. The fixed penalty is built once per matrix; the LDLT factor reuses row-scope `L` / `D` scratch buffers across iterations.
- Polynomial pre-fit uses the shared Householder QR (`core/common/linalg.h`), factored once per matrix and replayed per row.
- Convergence uses `n4m_relative_l2_diff` (shared with AsLS / ArPLS).
- On `max_iter` reached without convergence: silently returns the last iterate.
- Parity tolerance vs pybaselines snapshot: `5e-6 abs / 1e-5 rel`.

## Reference
- He, S., Zhang, X. et al. (2014). "Baseline correction for Raman spectra using an improved asymmetric least squares method." Analytical Methods 6, 4402-4407.
- External reference snapshot: `pybaselines.iasls` output stored under
  `benchmarks/reference_snapshots/cross_binding/iasls/`.
