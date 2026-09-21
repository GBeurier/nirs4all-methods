# `aug_band_mask` — Random contiguous-band masking

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_band_mask_*`

## Description

Mask random spectral bands with zero-fill or interpolation.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `n_bands_lo` | `int` | `1` |
| `n_bands_hi` | `int` | `3` |
| `bw_lo` | `int` | `5` |
| `bw_hi` | `int` | `15` |
| `mode` | `str \| int` | `'zero'` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_band_mask_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/spectral.h#L37) · [`n4m_augmentation_band_mask_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/spectral.h#L31) · [`n4m_augmentation_band_mask_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/spectral.h#L40). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.spectral import BandMasking
```

Source signature: [`BandMasking(n_bands_lo: int = 1, n_bands_hi: int = 3, bw_lo: int = 5, bw_hi: int = 15, mode: str | int = 'zero', rng: Optional[PCG64] = None, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L591).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No single spectroscopy paper defines this operator. It is analogous to structured feature dropout, with exact band sampling defined by the native source.

### Mathematical principle

For each spectrum, draw a band count in `n_bands_range`; for each band draw a center and integer width in `band_width_range`. In `zero` mode set the half-open slice to zero; in `interp` mode replace it by the line joining the nearest edge values. Overlapping bands are applied sequentially.

### Appropriate uses

Testing whether a model survives missing or corrupted contiguous wavelength regions.

### Limits and validation

Zero is not a neutral absorbance for every representation, while interpolation can create unrealistically straight segments. Edge clipping and band overlap mean realized masked width may be below the requested sum.

### Implementation

Python role API `n4m.augmentation.spectral.BandMasking`; ABI 2 family `n4m_augmentation_band_mask_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_augmentation_band_mask_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/spectral/band_mask.h


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)