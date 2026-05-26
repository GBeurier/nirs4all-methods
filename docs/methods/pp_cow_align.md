# `pp_cow_align` — Correlation Optimized Warping

_Group_: **Signal transforms** · _Binding_: `n4m.sklearn.CorrelationOptimizedWarping` · _C ABI_: `n4m_pp_cow_align_*`

## Description

Segment-wise correlation optimized warping approximation.

### Parameters

| Name | Type | Default |
|------|------|---------|
| `reference` | `—` | `None` |
| `interval_size` | `int` | `32` |
| `max_shift` | `int` | `5` |

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

Segment-wise correlation optimized warping approximation.

### Implementation

C ABI `n4m_pp_cow_align_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.CorrelationOptimizedWarping`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import CorrelationOptimizedWarping
op = CorrelationOptimizedWarping()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)