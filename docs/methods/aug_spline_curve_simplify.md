# `aug_spline_curve_simplify` — Curve-control cubic-spline simplification

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_spline_curve_simplification_*`

## Description

Simplify each spectrum via a cubic B-spline through a random control
subset along the curve (nirs4all ``Spline_Curve_Simplification``). Same
behaviour as the x-axis variant, differing only in the uniform path's
np.unique handling.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `spline_points` | `int` | `-1` |
| `uniform` | `bool` | `False` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_spline_curve_simplification_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/splines.h#L78) · [`n4m_augmentation_spline_curve_simplification_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/splines.h#L73) · [`n4m_augmentation_spline_curve_simplification_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/splines.h#L81). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.splines import SplineCurveSimplificationAugmenter
```

Source signature: [`SplineCurveSimplificationAugmenter()`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L1184).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No canonical paper defines this augmenter. It shares the internal interpolating B-spline engine with x simplification and exists for historical parity.

### Mathematical principle

Reduce each spectrum to `spline_points` controls (default $p/4$), fit a not-a-knot cubic interpolant, and reconstruct the dense curve. Random mode is identical to x simplification; in uniform mode this variant additionally deduplicates rounded linspace indices.

### Appropriate uses

Parity with the historical curve-simplification operator and sparse-curve stress tests.

### Limits and validation

The distinction from x simplification is only the uniform-index deduplication rule. It is not a geometry-aware arc-length simplifier despite the name.

### Implementation

Python role API `n4m.augmentation.splines.SplineCurveSimplificationAugmenter`; ABI 2 family `n4m_augmentation_spline_curve_simplification_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_augmentation_spline_curve_simplification_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/splines/spline_curve_simplification.h


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)