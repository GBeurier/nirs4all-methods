# `pp_direct_standardization` — Direct Standardization

_Group_: **Signal transforms** · _Binding_: `n4m.sklearn.DirectStandardization` · _C ABI_: `n4m_pp_direct_standardization_*`

## Description

Direct standardization transfer map between paired instruments.

### Parameters

| Name | Type | Default |
|------|------|---------|
| `fit_intercept` | `bool` | `True` |
| `ridge` | `float` | `0.0` |

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

Direct standardization transfer map between paired instruments.

### Implementation

C ABI `n4m_pp_direct_standardization_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.DirectStandardization`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import DirectStandardization
op = DirectStandardization()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)