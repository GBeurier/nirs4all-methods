# InstrumentalBroadeningAugmenter

## Algorithm

Models finite-resolution instrumental broadening via Gaussian convolution
with FWHM-derived sigma. The relationship between FWHM (in wavelength
units, typically nm) and Gaussian sigma in points is:

$$\sigma_{\mathrm{pts}} = \frac{\mathrm{FWHM}}{2\sqrt{2\ln 2} \cdot \mathrm{wl\_step}}$$

where `wl_step = median(diff(wavelengths))` when wavelengths is supplied,
else `1.0`.

Three behaviours:

- **Fixed FWHM** (`use_fwhm_range = 0`): one $\sigma_{\mathrm{pts}}$ for all rows; no RNG draws.
- **Per-sample range** (`use_fwhm_range = 1`, `variation_scope = 0`): one
  `FWHM ~ U(low, high)` per row.
- **Batch range** (`use_fwhm_range = 1`, `variation_scope = 1`): one
  `FWHM ~ U(low, high)` shared across the batch.

Each row is then row-wise convolved with a Gaussian of the corresponding
$\sigma_{\mathrm{pts}}$. The convolution is delegated to the existing
`n4m_pp_gaussian` engine (mode `reflect`, `truncate=4.0`), which has
$10^{-9}$ parity with `scipy.ndimage.gaussian_filter1d`.

## Determinism

`use_fwhm_range = 0` makes the augmenter fully deterministic — no RNG draws,
output is uniquely determined by `fwhm`, `wavelengths`, and `X`. The parity
fixture exercises this case (`instrument_broaden_fixed_fwhm5`).

## Reference

`nirs4all.operators.augmentation.synthesis.InstrumentalBroadeningAugmenter`.
