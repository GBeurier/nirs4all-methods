# Phase 3h - Haar Wavelet Preprocessing

Goal: activate `P4A_OP_WAVELET_DENOISE` as a deterministic Haar wavelet
denoising operator inside the fitted preprocessing pipeline.

## Scope

- Row-wise Haar forward and inverse transform.
- Non-power-of-two spectra are padded to the next power of two using the last
  observed value, then cropped back to the original wavelength count.
- Soft thresholding is applied to detail coefficients at each requested level.
- Default params: levels `1`, threshold `0.0`.
- Explicit params: `levels`, `threshold`.
- Validation:
  - levels integer in `[0, 20]`;
  - levels must not exceed the padded signal length;
  - threshold finite and non-negative.

## Acceptance

- ABI tests cover zero-threshold reconstruction and parameter validation.
- NumPy parity fixture covers a non-power-of-two spectrum with two denoising
  levels and a non-zero threshold.
- `pls4all_cli --selfcheck` exercises Haar wavelet denoising.
- CI, parity gate, ABI surface, coverage and sanitizers are green before tag.

## Deferred

- Additional wavelet families beyond Haar.
- Robust threshold estimators such as MAD/universal threshold.
- Supervised OSC / EPO preprocessing.
