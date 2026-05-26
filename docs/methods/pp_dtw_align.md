# `pp_dtw_align` — Dynamic Time Warping Alignment

_Group_: **Signal transforms** · _Binding_: `n4m.sklearn.DynamicTimeWarpingAlignment` · _C ABI_: `n4m_pp_dtw_align_*`

## Description

Dynamic-time-warping alignment to a fixed-length reference.

### Parameters

| Name | Type | Default |
|------|------|---------|
| `reference` | `—` | `None` |

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

Dynamic-time-warping alignment to a fixed-length reference.

### Implementation

C ABI `n4m_pp_dtw_align_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.DynamicTimeWarpingAlignment`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import DynamicTimeWarpingAlignment
op = DynamicTimeWarpingAlignment()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)