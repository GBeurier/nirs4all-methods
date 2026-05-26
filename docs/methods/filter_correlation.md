# `filter_correlation` — Correlation Filter

_Group_: **Signal transforms** · _Binding_: `n4m.sklearn.CorrelationFilter` · _C ABI_: `n4m_filter_correlation_*`

## Description

Model-agnostic feature filter by absolute correlation to ``y``.

### Parameters

| Name | Type | Default |
|------|------|---------|
| `threshold` | `float` | `0.0` |
| `top_k` | `int | None` | `None` |

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

Model-agnostic feature filter by absolute correlation to ``y``.

### Implementation

C ABI `n4m_filter_correlation_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.CorrelationFilter`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import CorrelationFilter
op = CorrelationFilter()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)