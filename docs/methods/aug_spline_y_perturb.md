# `aug_spline_y_perturb` — Smooth additive spline perturbation

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_spline_y_perturbations_*`

## Description

Spline y-axis perturbation augmenter.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `spline_points` | `int` | `-1` |
| `perturbation_intensity` | `float` | `0.005` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_spline_y_perturbations_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/splines.h#L48) · [`n4m_augmentation_spline_y_perturbations_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/splines.h#L43) · [`n4m_augmentation_spline_y_perturbations_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/splines.h#L51). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.splines import SplineYPerturbationAugmenter
```

Source signature: [`SplineYPerturbationAugmenter(spline_points: int = -1, perturbation_intensity: float = 0.005, rng: Optional[PCG64] = None, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L1149).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No publication canonically defines this control-point noise augmenter. Its random-control construction is specified by the native source.

### Mathematical principle

Let $v=\max(X)$ times `perturbation_intensity` and draw one batch-shared baseline from $U(-v,v)$. For every row, draw y offsets at `spline_points` controls (default $p/2$, minimum 4) from the baseline-shifted interval, fit a not-a-knot cubic curve on [0,p], evaluate it on channels 0 through p-1, and add that curve to the row.

### Appropriate uses

Generating smooth additive baseline fluctuations instead of channelwise white noise.

### Limits and validation

Intensity is tied to the global maximum, not absolute magnitude; all-negative data therefore invert the intended interval. Many controls can approach high-frequency noise, while few reduce the effect to broad drift. The last anchor lies at p, one step beyond the final queried channel p-1.

### Implementation

Python role API `n4m.augmentation.splines.SplineYPerturbationAugmenter`; ABI 2 family `n4m_augmentation_spline_y_perturbations_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_augmentation_spline_y_perturbations_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/splines/spline_y_perturbations.h


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)