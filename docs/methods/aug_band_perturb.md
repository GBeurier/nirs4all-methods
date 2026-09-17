# `aug_band_perturb` — Local band gain-and-offset perturbation

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_band_perturb_*`

## Description

Random band-local gain and offset perturbations.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `n_bands` | `int` | `3` |
| `bw_lo` | `int` | `5` |
| `bw_hi` | `int` | `15` |
| `gain_lo` | `float` | `0.9` |
| `gain_hi` | `float` | `1.1` |
| `offset_lo` | `float` | `-0.01` |
| `offset_hi` | `float` | `0.01` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_band_perturb_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/spectral.h#L26) · [`n4m_augmentation_band_perturb_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/spectral.h#L19) · [`n4m_augmentation_band_perturb_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/spectral.h#L29). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.spectral import BandPerturbationAugmenter
```

Source signature: [`BandPerturbationAugmenter(n_bands: int = 3, bw_lo: int = 5, bw_hi: int = 15, gain_lo: float = 0.9, gain_hi: float = 1.1, offset_lo: float = -0.01, offset_hi: float = 0.01, rng: Optional[PCG64] = None, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L564).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No canonical paper specifies this bandwise affine perturbation. It is a native spectral corruption heuristic related to offset/gain augmentation.

### Mathematical principle

For each row and each of `n_bands`, draw a center, width $w$, gain $g$ and offset $a$. Within the clipped band, update values sequentially as $X'_{ij}=gX_{ij}+a$. Parameter ranges `bw_*`, `gain_*`, and `offset_*` control extent and amplitude.

### Appropriate uses

Robustness to wavelength-local sensitivity and baseline changes.

### Limits and validation

Band boundaries are abrupt and overlapping bands compound in draw order. The operator exposes per-sample variation only, not one shared perturbation per batch.

### Implementation

Python role API `n4m.augmentation.spectral.BandPerturbationAugmenter`; ABI 2 family `n4m_augmentation_band_perturb_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_augmentation_band_perturb_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/spectral/band_perturb.h


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)