# ModPoly — Modified Polynomial baseline

Lieber & Mahadevan-Jansen 2003 iterative polynomial baseline. For each row `y` of length `n`:

1. Fit polynomial of degree `polyorder` to `(positions = [0..n-1], y)`; evaluate at positions to get baseline `z`.
2. Repeat up to `max_iter`:
   - Clip peaks: `y[i] = min(y[i], z[i])`.
   - Re-fit polynomial -> new baseline `z_new`.
   - If `rel_l2_diff(z, z_new) < tol`, break.
   - `z = z_new`.
3. Output: `out[i] = y_original[i] - z[i]`.

## Parameters
- `polyorder: int` (default 2) — polynomial degree.
- `max_iter: int` (default 250) — iteration cap.
- `tol: double` (default 1e-3) — relative baseline-change convergence threshold.

## ABI
```c
n4m_status_t n4m_pp_modpoly_create(n4m_pp_modpoly_handle_t** out,
                                    int32_t polyorder,
                                    int32_t max_iter, double tol);
void         n4m_pp_modpoly_destroy(n4m_pp_modpoly_handle_t* h);
n4m_status_t n4m_pp_modpoly_transform(const n4m_pp_modpoly_handle_t* h,
                                       n4m_matrix_view_t X, n4m_matrix_view_t out);
```

## Numerical contract
- Per-row Householder QR factorisation of the `(n, polyorder + 1)` Vandermonde matrix in descending powers (shared with `Detrend` / `IModPoly` / `IAsLS` via `core/common/linalg.{c,h}`). Factor once per row; re-apply `Q^T` against the clipped y at each iteration.
- Convergence uses `relative_difference(z_old, z_new)` from `banded_solver.h` (L2 norms, `max(||z_old||, DBL_MIN)` denominator).
- On `max_iter` reached without convergence: silently returns the last iterate.
- Parity tolerance vs internal parity fixture: `1e-9 abs / 1e-10 rel`.

## Reference
- Lieber, C. A. & Mahadevan-Jansen, A. (2003). "Automated method for subtraction of fluorescence from biological raman spectra." Applied Spectroscopy 57(11), 1363-1367.
- Internal parity fixture: `parity/python_generator/src/n4m_parity_pybaselines_ref/modpoly.py`.
