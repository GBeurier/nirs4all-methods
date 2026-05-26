# `aug_spline_x_simplify` — Spline X Simplification Augmenter

_Group_: **Augmentation** · _Binding_: `n4m.sklearn.SplineXSimplificationAugmenter` · _C ABI_: `n4m_aug_spline_x_simplify_*`

## Description

Spline x simplification stub; current C ABI returns NOT_IMPLEMENTED.

### Parameters

| Name | Type | Default |
|------|------|---------|
| `spline_points` | `int` | `-1` |
| `uniform` | `bool` | `True` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

Spline x simplification stub; current C ABI returns NOT_IMPLEMENTED.

### Implementation

C ABI `n4m_aug_spline_x_simplify_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.SplineXSimplificationAugmenter`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import SplineXSimplificationAugmenter
op = SplineXSimplificationAugmenter()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)