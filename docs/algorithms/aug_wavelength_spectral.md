# Phase 16 — Wavelength + spectral augmenters

Ten stochastic augmenters in the new `n4m_aug_*` ABI category. All ten
implement the locked 3-symbol ABI from
`roadmap/phase-15-18-augmenters-abi-contract.md` (`_create`, `_apply`,
`_destroy`); the RNG handle is the first constructor parameter and is
stored by reference (the augmenter does NOT own it).

| Operator                  | Symbol prefix                  | Algorithm |
| ------------------------- | ------------------------------ | --------- |
| WavelengthShift           | `n4m_aug_wavelength_shift_*`   | `out[i] = np.interp(lambdas - shift_i, lambdas, X[i])`, `shift_i ~ U(lo, hi)` |
| WavelengthStretch         | `n4m_aug_wavelength_stretch_*` | `query = center + (lambdas - center) / f_i`, `out[i] = np.interp(query, lambdas, X[i])`, `f_i ~ U(lo, hi)` |
| LocalWavelengthWarp       | `n4m_aug_local_warp_*`         | linearly-interp `n_control_points` random shifts `~ U(-mx, mx)` to wavelengths, then resample |
| BandPerturbation          | `n4m_aug_band_perturb_*`       | for each (sample, band): multiply by random gain, add random offset, within a random band |
| BandMasking               | `n4m_aug_band_mask_*`          | zero (or ramp-interpolate) `n_per_sample ~ U[n_lo, n_hi]` random bands per sample |
| ChannelDropout            | `n4m_aug_channel_dropout_*`    | mask `rng.random() < p` channels; zero or interpolate from kept neighbours |
| GaussianSmoothingJitter   | `n4m_aug_gauss_jitter_*`       | per-row reflect-padded Gaussian convolution with `sigma_i ~ U(sigma_lo, sigma_hi)` |
| UnsharpSpectralMask       | `n4m_aug_unsharp_mask_*`       | `out = X + amount_i * (X - convolve(X, gauss))`, `amount_i ~ U(amt_lo, amt_hi)` |
| SmoothMagnitudeWarp       | `n4m_aug_magnitude_warp_*`     | linearly-interp `n_control_points` random gains `~ U(g_lo, g_hi)`, multiply spectrum elementwise |
| LocalClipping             | `n4m_aug_local_clip_*`         | clip each of `n_regions` random bands to the 90th-percentile of the band |

## ABI contract

```c
typedef struct n4m_aug_<NAME>_handle_t n4m_aug_<NAME>_handle_t;

n4m_status_t n4m_aug_<NAME>_create(n4m_aug_<NAME>_handle_t** out,
                                    n4m_rng_pcg64_state_t*    rng,
                                    /* op-specific params */);
n4m_status_t n4m_aug_<NAME>_apply (const n4m_aug_<NAME>_handle_t* h,
                                    n4m_matrix_view_t X,
                                    n4m_matrix_view_t out);
void          n4m_aug_<NAME>_destroy(n4m_aug_<NAME>_handle_t* h);
```

The augmenters are stateless w.r.t. data (no `_fit`, no
`_inverse_transform`, no `_is_fitted`). Each `_apply` advances the RNG
state; tests reseed the RNG via `n4m_rng_pcg64_set_seed` immediately
before `_apply` to lock the stream.

## Determinism

NumPy 1.26.4 dispatches `rng.uniform` to PCG64's `next_double` callback
(`(uint64 >> 11) * 2^-53`) and `rng.integers(low, high)` to the buffered
32-bit Lemire path when `high - low <= 2^32`. The C engine replicates
both exactly:

- `n4m_aug_uniform`: `lo + (hi - lo) * next_double` — bit-equivalent to
  `rng.uniform(lo, hi)`.
- `n4m_aug_randint`: buffered 32-bit Lemire (or unbuffered 64-bit Lemire
  for wider ranges) — bit-equivalent to `rng.integers(lo, hi)`.

Random-stream order in each operator matches the Python reference's
allocation pattern (e.g. BandPerturbation draws centers, widths, gains,
offsets as four successive batches in row-major order).

## Linear-interp vs cubic-spline variants

Two operators differ from the nirs4all reference path:

- **LocalWavelengthWarp** and **SmoothMagnitudeWarp** use `np.interp`
  (linear interpolation) here, where the original
  `nirs4all.operators.augmentation.spectral` path uses
  `scipy.interpolate.splrep / splev` (cubic spline). The linear variant
  is chosen because cubic splines are non-trivial to reproduce
  bit-exactly across C/Python. The internal parity fixture under
  `parity/python_generator/src/n4m_parity_aug_spectral_ref/` mirrors
  the linear path, so the C engine matches at the 1e-15 PCG64 contract.

## Smoothing engine reuse

GaussianSmoothingJitter and UnsharpSpectralMask reuse the same
`n4m_aug_convolve_reflect` helper that lives in
`cpp/src/core/augmentations/spectral/spectral_common.c`. It applies a
reflect-padded 1-D correlation (which is the convolution for symmetric
kernels). The kernel is built via `n4m_aug_gauss_kernel_build` — a simple
normalized Gaussian of fixed odd width, byte-for-byte identical to
nirs4all's `_get_gaussian_kernel`. This is intentionally NOT the
`n4m_pp_gaussian_state_t` engine: that path is parameterised by
SciPy's `truncate * sigma` rule which computes a kernel length on the
fly, whereas the augmenters need a caller-fixed length.

## Parity floor

| Operator              | Reference                                                    | Tolerance         |
| --------------------- | ------------------------------------------------------------ | ----------------- |
| All ten Phase 16 ops  | `parity/python_generator/src/n4m_parity_aug_spectral_ref/*` | 1e-15 abs/rel     |

The reference modules are frozen NumPy + scipy.ndimage. Validated against
nirs4all 0.8.x once; subsequent nirs4all releases can drift without
breaking the nirs4all-methods parity gates.

## Files

- Core engines: `cpp/src/core/augmentations/wavelength/*.{c,h}` (3 ops)
  and `cpp/src/core/augmentations/spectral/*.{c,h}` (7 ops + shared
  helpers in `spectral_common.{c,h}`).
- C API wrapper: `cpp/src/c_api/c_api_augmenters_wavelength_spectral.cpp`
  + side-channel header `c_api_augmenters_wavelength_spectral.h`.
- Tests: `cpp/tests/test_augmenters_wavelength_spectral.cpp` —
  10 smoke tests + 10 parity tests in the
  `n4m_tests_aug` executable.
- Fixtures: `parity/fixtures/aug_<name>_v1.json` (10 files, 2 cases each).
- Internal parity fixture: `parity/python_generator/src/n4m_parity_aug_spectral_ref/`.
- Generator: `parity/python_generator/scripts/generate_phase16_fixtures.py`.

## ABI symbols added

30 new symbols total (3 per operator):

```
n4m_aug_wavelength_shift_create / _apply / _destroy
n4m_aug_wavelength_stretch_create / _apply / _destroy
n4m_aug_local_warp_create / _apply / _destroy
n4m_aug_band_perturb_create / _apply / _destroy
n4m_aug_band_mask_create / _apply / _destroy
n4m_aug_channel_dropout_create / _apply / _destroy
n4m_aug_gauss_jitter_create / _apply / _destroy
n4m_aug_unsharp_mask_create / _apply / _destroy
n4m_aug_magnitude_warp_create / _apply / _destroy
n4m_aug_local_clip_create / _apply / _destroy
```

All symbols are exported via the wildcard `n4m_*` Linux version script;
the canonical declarations land in `n4m.h` with the post-merge
integration commit that bundles Phases 15-18 (the public header is
frozen during parallel development to avoid edit conflicts).
