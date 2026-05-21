# Detrend (polynomial baseline subtraction)

Per-row polynomial detrending. For each row of `X` of length `p`:

1. Build positions `t = [0, 1, …, p-1]`.
2. Fit a polynomial of degree `polyorder` to `(t, X[i, :])` via least-squares (Householder QR).
3. Evaluate the fitted polynomial at the same positions.
4. Subtract: `out[i, j] = X[i, j] - polyval(coefs, t)[j]`.

## Parameters
- `polyorder: int` (default 1) — polynomial order. Must satisfy `polyorder >= 0` and `polyorder + 1 <= cols`.

## ABI
```c
n4m_status_t n4m_pp_detrend_create(n4m_pp_detrend_handle_t** out, int32_t polyorder);
void         n4m_pp_detrend_destroy(n4m_pp_detrend_handle_t* h);
n4m_status_t n4m_pp_detrend_transform(const n4m_pp_detrend_handle_t* h,
                                       n4m_matrix_view_t X, n4m_matrix_view_t out);
```

## Numerical contract
- Closed-form least-squares via the shared Householder QR helper in `core/common/linalg.{c,h}`.
- True per-element division and subtraction; no reciprocal-multiplication shortcut.
- Parity tolerance vs internal parity fixture: `1e-11 abs / 1e-12 rel`.
- Internal parity fixture: `parity/python_generator/src/n4m_parity_pybaselines_ref/detrend.py` (uses `np.polyfit`/`np.polyval`).

## Reference
- Standard polynomial detrending; see `numpy.polyfit` / `numpy.polyval` documentation.
