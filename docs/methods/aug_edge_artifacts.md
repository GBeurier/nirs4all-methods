# `aug_edge_artifacts` — Edge Artifacts Augmenter

_Group_: **Augmentation** · _Binding_: `n4m.sklearn.EdgeArtifactsAugmenter` · _C ABI_: `n4m_aug_edge_artifacts_*`

## Description

Combined edge artifact augmenter.

### Parameters

| Name | Type | Default |
|------|------|---------|
| `enabled_flags` | `int` | `15` |
| `overall_strength` | `float` | `1.0` |
| `detector_model` | `int` | `4` |
| `wavelengths` | `—` | `None` |
| `rng` | `Optional[PCG64]` | `None` |
| `seed` | `int` | `0` |

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

Combined edge artifact augmenter.

### Implementation

C ABI `n4m_aug_edge_artifacts_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.EdgeArtifactsAugmenter`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import EdgeArtifactsAugmenter
op = EdgeArtifactsAugmenter()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)