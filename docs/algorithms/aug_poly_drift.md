# PolynomialBaselineDrift — `n4m_aug_poly_drift_*`

Adds a per-sample polynomial baseline drift of configurable degree to spectra.

Algorithm (RNG draws are **order-major** to match nirs4all):

1. Build `lambdas = linspace(-1, 1, n_features)` (NumPy 1.26.4 conventions; for `n_features == 1` returns `[-1.0]`).
2. For each order `k` in `[0, degree]`, draw `n_samples` uniforms and form
   `coeffs[k, i] = coeff_min[k] + (coeff_max[k] - coeff_min[k]) * u_ik`.
3. `out[i, j] = X[i, j] + sum_k coeffs[k, i] * lambdas[j]^k`.

The polynomial accumulation uses the `lj_pow *= lj` recurrence rather than `np.polyval` so the multiplication order is bit-identical between the C reference and the Python ref.

## Parameters
- `degree: int32_t` (`>= 0`) — polynomial degree.
- `coeff_min, coeff_max: const double* (length degree + 1)` — per-order uniform ranges. `coeff_min[k]` and `coeff_max[k]` bracket the order-k coefficient for every sample.

## ABI
```c
n4m_status_t n4m_aug_poly_drift_create(
    n4m_aug_poly_drift_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    int32_t degree,
    const double* coeff_min, const double* coeff_max);
void         n4m_aug_poly_drift_destroy(n4m_aug_poly_drift_handle_t* h);
n4m_status_t n4m_aug_poly_drift_apply(
    const n4m_aug_poly_drift_handle_t* h,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
```

The state copies `coeff_min` / `coeff_max` into internal storage; caller retains ownership of the original arrays.

## Reference
- Internal parity fixture: `parity/python_generator/src/n4m_parity_augmenters_ref/poly_drift.py`.
- Parity tolerance: 1e-15 abs.
