# `aug_spline_x_perturb` — Spline X Perturbation Augmenter

_Group_: **Augmentation** · _Binding_: `n4m.sklearn.SplineXPerturbationAugmenter` · _C ABI_: `n4m_aug_spline_x_perturb_*`

## Description

Spline x-axis perturbation augmenter.

### Parameters

| Name | Type | Default |
|------|------|---------|
| `spline_degree` | `int` | `3` |
| `perturbation_density` | `float` | `0.05` |
| `perturbation_range_min` | `float` | `-0.1` |
| `perturbation_range_max` | `float` | `0.1` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

Spline x-axis perturbation augmenter.

### Implementation

C ABI `n4m_aug_spline_x_perturb_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.SplineXPerturbationAugmenter`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import SplineXPerturbationAugmenter
op = SplineXPerturbationAugmenter()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)