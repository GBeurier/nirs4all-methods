# `aug_instrument_broaden` — Gaussian instrumental broadening from FWHM

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_instrument_broaden_*`

## Description

Instrumental spectral broadening via Gaussian convolution.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `fwhm` | `float` | `5.0` |
| `use_fwhm_range` | `bool` | `False` |
| `fwhm_low` | `float` | `3.0` |
| `fwhm_high` | `float` | `8.0` |
| `variation_scope` | `int` | `0` |
| `wavelengths` | `—` | `None` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_instrument_broaden_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/scattering.h#L90) · [`n4m_augmentation_instrument_broaden_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/scattering.h#L83) · [`n4m_augmentation_instrument_broaden_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/scattering.h#L93). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.scattering import InstrumentalBroadeningAugmenter
```

Source signature: [`InstrumentalBroadeningAugmenter(fwhm: float = 5.0, use_fwhm_range: bool = False, fwhm_low: float = 3.0, fwhm_high: float = 8.0, variation_scope: int = 0, wavelengths = None, rng: Optional[PCG64] = None, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L859).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No single paper defines this implementation; Gaussian line-spread convolution is a standard instrumental-resolution model, with exact kernel construction specified in source.

### Mathematical principle

Convert full width at half maximum to channel standard deviation as $\sigma_p=\mathrm{FWHM}/[2\sqrt{2\ln2}\,\Delta\lambda]$, where $\Delta\lambda$ is the median wavelength step (or 1 without wavelengths), and Gaussian-convolve each row. A fixed FWHM or uniform range may be used per row or batch.

### Appropriate uses

Testing transfer across instruments with different spectral resolution.

### Limits and validation

The line-spread function is assumed Gaussian and stationary. Irregular wavelength grids are summarized by one median step, and broadening cannot sharpen a low-resolution spectrum.

### Implementation

Python role API `n4m.augmentation.scattering.InstrumentalBroadeningAugmenter`; ABI 2 family `n4m_augmentation_instrument_broaden_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_augmentation_instrument_broaden_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/scattering/instrument_broaden.h


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)