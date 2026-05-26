# `pp_saps` — Score Augmented Projection Standardization

_Group_: **Signal transforms** · _Binding_: `n4m.sklearn.ScoreAugmentedProjectionStandardization` · _C ABI_: `n4m_pp_saps_*`

## Description

Score-augmented projection standardization inspired by SA-PBS.

### Parameters

| Name | Type | Default |
|------|------|---------|
| `n_components` | `int` | `5` |
| `score_weight` | `float` | `1.0` |
| `fit_intercept` | `bool` | `True` |
| `ridge` | `float` | `0.0` |

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

Score-augmented projection standardization inspired by SA-PBS.

### Implementation

C ABI `n4m_pp_saps_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.ScoreAugmentedProjectionStandardization`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import ScoreAugmentedProjectionStandardization
op = ScoreAugmentedProjectionStandardization()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)