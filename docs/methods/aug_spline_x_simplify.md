# `aug_spline_x_simplify` — Cubic-spline reconstruction from sparse x controls

_Group_: **Augmentation** · _C ABI_: `n4m_augmentation_spline_x_simplification_*`

## Description

Simplify each spectrum via a cubic B-spline through a random control
subset on the x-axis (nirs4all ``Spline_X_Simplification``). ``spline_points
<= 0`` selects the reference default of n_features // 4. Mirrors the
reference defaults (``uniform=False``).

## Parameters

| Name | Type | Default |
|------|------|---------|
| `spline_points` | `int` | `-1` |
| `uniform` | `bool` | `False` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_augmentation_spline_x_simplification_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/splines.h#L63) · [`n4m_augmentation_spline_x_simplification_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/splines.h#L58) · [`n4m_augmentation_spline_x_simplification_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/augmentation/splines.h#L66). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.augmentation.splines import SplineXSimplificationAugmenter
```

Source signature: [`SplineXSimplificationAugmenter(spline_points: int = -1, uniform: bool = False, rng: Optional[PCG64] = None, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/augmentation.py#L1165).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No canonical paper defines this augmentation heuristic. It uses a classical interpolating B-spline but the control subset and parity rules are implementation-specific.

### Mathematical principle

Keep `spline_points` channel/value controls (default $p/4$), selected uniformly or by PCG64 sampling without replacement. Fit an interpolating not-a-knot cubic B-spline through those controls and evaluate it at all original channels.

### Appropriate uses

Testing robustness to reduced effective spectral resolution and sparse sampling.

### Limits and validation

Fine features between controls are irrecoverable and cubic interpolation may overshoot. This x variant does not apply the curve variant's final `unique` step in uniform mode.

### Implementation

Python role API `n4m.augmentation.splines.SplineXSimplificationAugmenter`; ABI 2 family `n4m_augmentation_spline_x_simplification_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_augmentation_spline_x_simplification_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/augmentation/splines/spline_simplify_common.h


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)