# `aug_edge_curve` — Edge Curvature Augmenter

_Group_: **Augmentation** · _Binding_: `n4m.sklearn.EdgeCurvatureAugmenter` · _C ABI_: `n4m_aug_edge_curve_*`

## Description

Curved edge response artifact.

### Parameters

| Name | Type | Default |
|------|------|---------|
| `curvature_strength` | `float` | `0.02` |
| `curvature_type` | `int` | `0` |
| `asymmetry` | `float` | `0.0` |
| `edge_focus` | `float` | `0.7` |
| `wavelengths` | `—` | `None` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

Curved edge response artifact.

### Implementation

C ABI `n4m_aug_edge_curve_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.EdgeCurvatureAugmenter`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import EdgeCurvatureAugmenter
op = EdgeCurvatureAugmenter()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)