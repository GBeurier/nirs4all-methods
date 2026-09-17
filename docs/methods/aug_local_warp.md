# `aug_local_warp` — Piecewise-linear local wavelength warp

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_local_warp_*`

## Description

Random local wavelength warping.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `n_control_points` | `int` | `5` |
| `max_shift` | `float` | `1.0` |
| `wavelengths` | `—` | `None` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_local_warp_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/wavelength.h#L41) · [`n4m_augmentation_local_warp_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/wavelength.h#L35) · [`n4m_augmentation_local_warp_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/wavelength.h#L44). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.wavelength import LocalWarpAugmenter
```

Source signature: [`LocalWarpAugmenter(n_control_points: int = 5, max_shift: float = 1.0, wavelengths = None, rng: Optional[PCG64] = None, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L547).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No paper is canonical for this exact local-warp heuristic. The native source, rather than the historical cubic Python variant, defines the shipped behavior.

### Mathematical principle

Choose `n_control_points` equally spaced wavelengths, draw their shifts from $U(-m,m)$ where $m$ is `max_shift`, linearly interpolate the shift field $s_i(\lambda)$, and compute $X'_i(\lambda)=X_i(\lambda-s_i(\lambda))$ by linear interpolation.

### Appropriate uses

Robustness to smooth, non-uniform wavelength calibration distortions.

### Limits and validation

The current C implementation uses piecewise-linear control interpolation; it deliberately differs from the older cubic `splrep/splev` oracle. Large shifts can fold the query grid or create endpoint plateaus.

### Implementation

Python role API `n4m.augmentation.wavelength.LocalWarpAugmenter`; ABI 2 family `n4m_augmentation_local_warp_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_augmentation_local_warp_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/wavelength/local_warp.h


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)