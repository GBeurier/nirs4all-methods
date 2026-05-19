# `aug_instrument_broaden` — InstrumentalBroadeningAugmenter

_Group_: **Augmentation** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/aug_instrument_broaden.md`](../algorithms/aug_instrument_broaden.md)

## Description

`aug_instrument_broaden` is a chemometrics4all augmentation method exposed through the C ABI and the generated language bindings.

### Parameters

No public constructor parameters are required for the documented default call.

## Explanations

### Bibliographic source

`nirs4all.operators.augmentation.synthesis.InstrumentalBroadeningAugmenter`.

### Mathematical principle

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
`c4a_pp_gaussian` engine (mode `reflect`, `truncate=4.0`), which has
$10^{-9}$ parity with `scipy.ndimage.gaussian_filter1d`.

### Implementation

`use_fwhm_range = 0` makes the augmenter fully deterministic — no RNG draws,
output is uniquely determined by `fwhm`, `wavelengths`, and `X`. The parity
fixture exercises this case (`instrument_broaden_fixed_fwhm5`).

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | — | C/C++ | Stable libc4a entry point family. |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
/* C ABI prefix */
c4a_*
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- ℹ No external parity reference row is registered for this public helper; the page is generated from the in-tree API and algorithm documentation.
:::

### Benchmarks

No cross-binding timing row is currently registered for this method. The implementation table above is still generated from the public API surface.

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
