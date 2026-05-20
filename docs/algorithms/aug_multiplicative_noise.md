# MultiplicativeNoise — `n4m_aug_multiplicative_noise_*`

Applies per-sample multiplicative gain to spectra.

For each input row `X[i, :]`:

1. Draw `eps[i] = sigma_gain * standard_normal()` from the borrowed PCG64 handle (one draw per row, bulk-filled).
2. `out[i, j] = X[i, j] * (1 + eps[i])`.

Mirrors `nirs4all.operators.augmentation.spectral.MultiplicativeNoise` with `per_wavelength=False, variation_scope="sample"` (default). Per-wavelength and batch-scope variants are deferred to a follow-up phase.

## Parameters
- `sigma_gain: double` (`>= 0.0`) — standard deviation of the per-sample gain factor.

## ABI
```c
n4m_status_t n4m_aug_multiplicative_noise_create(
    n4m_aug_multiplicative_noise_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double sigma_gain);
void         n4m_aug_multiplicative_noise_destroy(
    n4m_aug_multiplicative_noise_handle_t* h);
n4m_status_t n4m_aug_multiplicative_noise_apply(
    const n4m_aug_multiplicative_noise_handle_t* h,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
```

## Reference
- Internal parity fixture: `parity/python_generator/src/n4m_parity_augmenters_ref/multiplicative_noise.py`.
- Parity tolerance: 1e-15 abs.
