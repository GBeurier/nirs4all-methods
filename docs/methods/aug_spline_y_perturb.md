# `aug_spline_y_perturb` — Spline Y Perturbation Augmenter

_Group_: **Augmentation** · _Binding_: `n4m.sklearn.SplineYPerturbationAugmenter` · _C ABI_: `n4m_aug_spline_y_perturb_*`

## Description

Spline y-axis perturbation augmenter.

### Parameters

| Name | Type | Default |
|------|------|---------|
| `spline_points` | `int` | `-1` |
| `perturbation_intensity` | `float` | `0.005` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

Spline y-axis perturbation augmenter.

### Implementation

C ABI `n4m_aug_spline_y_perturb_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.SplineYPerturbationAugmenter`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import SplineYPerturbationAugmenter
op = SplineYPerturbationAugmenter()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)