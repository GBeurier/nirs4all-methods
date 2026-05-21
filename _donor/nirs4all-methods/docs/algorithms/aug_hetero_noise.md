# HeteroscedasticNoiseAugmenter — `n4m_aug_hetero_noise_*`

Adds signal-dependent Gaussian noise to spectra.

For each input cell:

1. `sigma[i, j] = noise_base + noise_signal_dep * |X[i, j]|`
2. Draw `noise = standard_normal(rows * cols)` (one bulk call, row-major).
3. `out[i, j] = X[i, j] + noise[i*cols + j] * sigma[i, j]`.

Mirrors `nirs4all.operators.augmentation.synthesis.HeteroscedasticNoiseAugmenter` with `variation_scope="sample"` (default).

## Parameters
- `noise_base: double` (`>= 0.0`) — signal-independent noise stddev.
- `noise_signal_dep: double` (`>= 0.0`) — signal-dependent noise coefficient.

## ABI
```c
n4m_status_t n4m_aug_hetero_noise_create(
    n4m_aug_hetero_noise_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double noise_base, double noise_signal_dep);
void         n4m_aug_hetero_noise_destroy(n4m_aug_hetero_noise_handle_t* h);
n4m_status_t n4m_aug_hetero_noise_apply(
    const n4m_aug_hetero_noise_handle_t* h,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
```

## Reference
- Internal parity fixture: `parity/python_generator/src/n4m_parity_augmenters_ref/hetero_noise.py`.
- Parity tolerance: 1e-15 abs.
