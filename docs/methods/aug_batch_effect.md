# `aug_batch_effect` — Offset, slope, and gain batch effects

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_batch_effect_*`

## Description

Random offset, slope and gain batch effects.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `offset_std` | `float` | `0.0` |
| `slope_std` | `float` | `0.0` |
| `gain_std` | `float` | `0.0` |
| `variation_scope` | `int` | `0` |
| `wavelengths` | `—` | `None` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_batch_effect_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/scattering.h#L72) · [`n4m_augmentation_batch_effect_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/scattering.h#L66) · [`n4m_augmentation_batch_effect_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/scattering.h#L75). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.scattering import BatchEffectAugmenter
```

Source signature: [`BatchEffectAugmenter(offset_std: float = 0.0, slope_std: float = 0.0, gain_std: float = 0.0, variation_scope: int = 0, wavelengths = None, rng: Optional[PCG64] = None, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L836).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No canonical publication defines this three-component simulator. It is an internal instrument/session perturbation model related to affine spectral augmentation.

### Mathematical principle

On centered normalized wavelength $t_j$, draw $a\sim N(0,\sigma_a^2)$, $b\sim N(0,\sigma_b^2)$ and $g\sim N(1,\sigma_g^2)$, then $X'_{ij}=gX_{ij}+a+bt_j$. `variation_scope=0` draws a tuple per row; scope 1 shares one tuple across the batch.

### Appropriate uses

Simulating acquisition-session or instrument-to-instrument affine shifts.

### Limits and validation

Zero defaults make this a no-op until standard deviations are set. A shared batch draw models one batch only and does not attach batch labels.

### Implementation

Python role API `n4m.augmentation.scattering.BatchEffectAugmenter`; ABI 2 family `n4m_augmentation_batch_effect_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_augmentation_batch_effect_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/scattering/batch_effect.h


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)