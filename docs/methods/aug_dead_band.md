# `aug_dead_band` — Random noisy detector dead bands

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_dead_band_*`

## Description

Simulate dead spectral detector bands.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `n_bands` | `int` | `1` |
| `width_low` | `int` | `5` |
| `width_high` | `int` | `10` |
| `noise_std` | `float` | `0.05` |
| `probability` | `float` | `0.0` |
| `variation_scope` | `int` | `0` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_dead_band_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/scattering.h#L105) · [`n4m_augmentation_dead_band_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/scattering.h#L98) · [`n4m_augmentation_dead_band_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/scattering.h#L108). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.scattering import DeadBandAugmenter
```

Source signature: [`DeadBandAugmenter(n_bands: int = 1, width_low: int = 5, width_high: int = 10, noise_std: float = 0.05, probability: float = 0.0, variation_scope: int = 0, rng: Optional[PCG64] = None, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L886).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No canonical paper defines this simulator. The number, width, probability, and replacement-noise rules are implementation-specific.

### Mathematical principle

With `probability`, select `n_bands`; sample each integer width and start, and replace the slice by independent $N(0,\text{noise_std}^2)$ values. Per-sample scope draws locations for each row; batch scope shares locations but still draws independent row noise.

### Appropriate uses

Stress-testing models against failed or unusable contiguous detector regions.

### Limits and validation

Replacement is centered at zero rather than the local baseline. At the default `probability=0`, the operator is a no-op. Overlapping bands may reduce affected coverage.

### Implementation

Python role API `n4m.augmentation.scattering.DeadBandAugmenter`; ABI 2 family `n4m_augmentation_dead_band_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_augmentation_dead_band_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/scattering/dead_band.h


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)