# HeteroscedasticNoiseAugmenter — `c4a_aug_hetero_noise_*`

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
c4a_status_t c4a_aug_hetero_noise_create(
    c4a_aug_hetero_noise_handle_t** out,
    c4a_rng_pcg64_state_t* rng,
    double noise_base, double noise_signal_dep);
void         c4a_aug_hetero_noise_destroy(c4a_aug_hetero_noise_handle_t* h);
c4a_status_t c4a_aug_hetero_noise_apply(
    const c4a_aug_hetero_noise_handle_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t out);
```

## Reference
- Frozen Python: `parity/python_generator/src/c4a_parity_augmenters_ref/hetero_noise.py`.
- Parity tolerance: 1e-15 abs.
