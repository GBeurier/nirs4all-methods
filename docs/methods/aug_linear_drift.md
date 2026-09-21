# `aug_linear_drift` — Random affine baseline drift

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_linear_drift_*`

## Description

Add random offset and linear slope drift.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `offset_min` | `float` | `-0.05` |
| `offset_max` | `float` | `0.05` |
| `slope_min` | `float` | `-0.01` |
| `slope_max` | `float` | `0.01` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_linear_drift_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/drift.h#L21) · [`n4m_augmentation_linear_drift_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/drift.h#L15) · [`n4m_augmentation_linear_drift_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/drift.h#L20). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.drift import LinearBaselineDrift
```

Source signature: [`LinearBaselineDrift(offset_min: float = -0.05, offset_max: float = 0.05, slope_min: float = -0.01, slope_max: float = 0.01, rng: Optional[PCG64] = None, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L449).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No unique paper defines this augmenter. Additive offset and slope perturbation is a spectroscopy augmentation pattern used by Bjerrum et al. (2017), arXiv:1710.01927 (https://arxiv.org/abs/1710.01927).

### Mathematical principle

For spectrum $i$, draw $a_i\sim U(a_{min},a_{max})$ and $b_i\sim U(b_{min},b_{max})$, then set $X'_{ij}=X_{ij}+a_i+b_i(j-\bar j)$. Centering the implicit channel index makes `offset_*` the drift at the spectral midpoint and `slope_*` the per-index gradient.

### Appropriate uses

Robustness to baseline displacement and linear tilt between acquisitions.

### Limits and validation

The ABI uses channel index, not physical wavelength. Consequently the same slope parameter has a different physical meaning after resampling or cropping.

### Implementation

Python role API `n4m.augmentation.drift.LinearBaselineDrift`; ABI 2 family `n4m_augmentation_linear_drift_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_augmentation_linear_drift_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/drift/linear_drift.h; https://arxiv.org/abs/1710.01927


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)