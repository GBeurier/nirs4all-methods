# `aug_local_clip` — Random local 90th-percentile clipping

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_local_clip_*`

## Description

Clip random local spectral regions.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `n_regions` | `int` | `1` |
| `width_lo` | `int` | `5` |
| `width_hi` | `int` | `15` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_local_clip_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/spectral.h#L88) · [`n4m_augmentation_local_clip_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/spectral.h#L83) · [`n4m_augmentation_local_clip_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/spectral.h#L91). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.spectral import LocalClip
```

Source signature: [`LocalClip(n_regions: int = 1, width_lo: int = 5, width_hi: int = 15, rng: Optional[PCG64] = None, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L694).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No canonical paper specifies this saturation heuristic. Its fixed 90th-percentile rule is defined by the native implementation.

### Mathematical principle

For every row, sample `n_regions` centers and widths. In each resulting slice, compute its linearly interpolated 90th percentile $q_{0.9}$ and replace each value by $\min(X_{ij},q_{0.9})$. Overlapping regions operate on the already modified row.

### Appropriate uses

Simulating local high-end detector saturation or peak truncation.

### Limits and validation

Only positive excursions are clipped, the 90% level is fixed, and the threshold depends on the randomly selected region. It is not a detector transfer function.

### Implementation

Python role API `n4m.augmentation.spectral.LocalClip`; ABI 2 family `n4m_augmentation_local_clip_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_augmentation_local_clip_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/spectral/local_clip.h


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)