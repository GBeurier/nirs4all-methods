# IAsLS — Improved Asymmetric Least Squares baseline

He et al. 2014 IAsLS baseline. Identical to AsLS except for the initial weight vector, which is derived from a polynomial pre-fit rather than initialised to ones. For each row `y` of length `n`:

1. Fit polynomial of degree `polyorder` to `(positions, y)` -> initial baseline `z0`.
2. Initialise weights `w[i] = p` if `y[i] > z0[i]` else `1 - p`.
3. Repeat up to `max_iter`:
   - Solve `(diag(w) + λ D²ᵀD²) z = w · y` (pentadiagonal LDLT).
   - `w_new[i] = p` if `y[i] > z[i]` else `1 - p`.
   - If `rel_l2_diff(w, w_new) < tol`, break.
   - `w = w_new`.
4. Output: `out = y - z`.

## Parameters
- `lam: double` (default 1e6) — smoothness regularisation.
- `p: double` (default 1e-2) — asymmetry weight, `p ∈ (0, 1)`.
- `polyorder: int` (default 2) — polynomial pre-fit degree.
- `max_iter: int` (default 50).
- `tol: double` (default 1e-3) — relative weight-change convergence threshold.

## ABI
```c
c4a_status_t c4a_pp_iasls_create(c4a_pp_iasls_handle_t** out,
                                  double lam, double p,
                                  int32_t polyorder,
                                  int32_t max_iter, double tol);
void         c4a_pp_iasls_destroy(c4a_pp_iasls_handle_t* h);
c4a_status_t c4a_pp_iasls_transform(const c4a_pp_iasls_handle_t* h,
                                     c4a_matrix_view_t X, c4a_matrix_view_t out);
```

## Numerical contract
- Shared 2nd-difference penalty builder + in-place pentadiagonal LDLT from `core/common/banded_solver.h`. The penalty is built once per matrix; the LDLT factor reuses row-scope `L` / `D` scratch buffers across iterations.
- Polynomial pre-fit uses the shared Householder QR (`core/common/linalg.h`), factored once per matrix and replayed per row.
- Convergence uses `c4a_relative_l2_diff` (shared with AsLS / ArPLS).
- On `max_iter` reached without convergence: silently returns the last iterate.
- Parity tolerance vs frozen NumPy reference: `1e-7 abs / 1e-8 rel`.

## Reference
- He, S., Zhang, X. et al. (2014). "Baseline correction for Raman spectra using an improved asymmetric least squares method." Analytical Methods 6, 4402-4407.
- Frozen Python reference: `parity/python_generator/src/c4a_parity_pybaselines_ref/iasls.py`.
