# GaussianAdditiveNoise — `n4m_aug_gaussian_noise_*`

Adds per-sample std-scaled Gaussian noise to spectra.

For each input row `X[i, :]`:

1. Compute the row's biased standard deviation `stds[i] = std(X[i, :])` (ddof=0).
2. Draw a single bulk `noise = standard_normal(rows * cols)` from the borrowed PCG64 handle.
3. `out[i, j] = X[i, j] + noise[i*cols + j] * stds[i] * sigma`.

The single-call bulk draw fixes the RNG stream order, so the C reference matches the internal parity fixture at 1e-15 abs.

## Parameters
- `sigma: double` (`>= 0.0`) — relative noise scale; the per-row std is multiplied by `sigma` to get the actual noise stddev.

## ABI
```c
n4m_status_t n4m_aug_gaussian_noise_create(
    n4m_aug_gaussian_noise_handle_t** out,
    n4m_rng_pcg64_state_t* rng,         /* borrowed, MUST be non-NULL */
    double sigma);
void         n4m_aug_gaussian_noise_destroy(n4m_aug_gaussian_noise_handle_t* h);
n4m_status_t n4m_aug_gaussian_noise_apply(
    const n4m_aug_gaussian_noise_handle_t* h,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
```

## Determinism
The augmenter does not own the RNG. Re-seed the RNG with `n4m_rng_pcg64_set_seed(rng, seed)` before each `_apply` to lock the output. Tested in `cpp/tests/test_augmenters_noise_drift.cpp::aug_gaussian_noise_*`.

## Reference
- Internal parity fixture: `parity/python_generator/src/n4m_parity_augmenters_ref/gaussian_noise.py` (validated against `nirs4all.operators.augmentation.spectral.GaussianAdditiveNoise` with `variation_scope="sample"`).
- Parity tolerance: 1e-15 abs.
