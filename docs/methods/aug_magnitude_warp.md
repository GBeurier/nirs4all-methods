# `aug_magnitude_warp` — Smooth multiplicative magnitude warp

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_magnitude_warp_*`

## Description

Smooth multiplicative magnitude warp.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `n_control_points` | `int` | `5` |
| `gain_lo` | `float` | `0.9` |
| `gain_hi` | `float` | `1.1` |
| `wavelengths` | `—` | `None` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_magnitude_warp_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/spectral.h#L78) · [`n4m_augmentation_magnitude_warp_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/spectral.h#L72) · [`n4m_augmentation_magnitude_warp_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/spectral.h#L81). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.spectral import MagnitudeWarp
```

Source signature: [`MagnitudeWarp(n_control_points: int = 5, gain_lo: float = 0.9, gain_hi: float = 1.1, wavelengths = None, rng: Optional[PCG64] = None, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L674).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No paper is canonical for this exact spectral warp. The current native linear interpolation path is the normative algorithm.

### Mathematical principle

At equally spaced control wavelengths draw gains from $U(\text{gain_lo},\text{gain_hi})$, linearly interpolate a gain field $g_i(\lambda)$, and return $X'_i(\lambda)=g_i(\lambda)X_i(\lambda)$. `n_control_points` controls the field's spatial frequency.

### Appropriate uses

Simulating smooth wavelength-dependent sensitivity or multiplicative scatter drift.

### Limits and validation

The C path is piecewise linear and intentionally differs from the historical cubic-spline oracle. Gains are unconstrained beyond user bounds and may cross zero.

### Implementation

Python role API `n4m.augmentation.spectral.MagnitudeWarp`; ABI 2 family `n4m_augmentation_magnitude_warp_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_augmentation_magnitude_warp_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/spectral/magnitude_warp.h


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)