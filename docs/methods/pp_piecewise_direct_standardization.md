# `pp_piecewise_direct_standardization` — Piecewise Direct Standardization

_Group_: **Signal transforms** · _Binding_: `n4m.sklearn.PiecewiseDirectStandardization` · _C ABI_: `n4m_pp_piecewise_direct_standardization_*`

## Description

PDS: local regressions mapping source windows to target wavelengths.

### Parameters

| Name | Type | Default |
|------|------|---------|
| `window_size` | `int` | `5` |
| `fit_intercept` | `bool` | `True` |
| `ridge` | `float` | `0.0` |

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

PDS: local regressions mapping source windows to target wavelengths.

### Implementation

C ABI `n4m_pp_piecewise_direct_standardization_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.PiecewiseDirectStandardization`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import PiecewiseDirectStandardization
op = PiecewiseDirectStandardization()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)