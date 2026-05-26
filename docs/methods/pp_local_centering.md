# `pp_local_centering` — Local Centering

_Group_: **Signal transforms** · _Binding_: `n4m.sklearn.LocalCentering` · _C ABI_: `n4m_pp_local_centering_*`

## Description

Transfer by subtracting source mean and adding target mean.

### Parameters

_No constructor parameters._

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

Transfer by subtracting source mean and adding target mean.

### Implementation

C ABI `n4m_pp_local_centering_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.LocalCentering`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import LocalCentering
op = LocalCentering()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)