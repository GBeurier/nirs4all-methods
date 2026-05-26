# `filter_x_outlier` — X Outlier Filter

_Group_: **Sample / feature filters** · _Binding_: `n4m.sklearn.XOutlierFilter` · _C ABI_: `n4m_filter_x_outlier_*`

## Description

Multivariate outlier filter on the design matrix ``X``.

<details>
<summary>Full binding docstring</summary>

```text
Multivariate outlier filter on the design matrix ``X``.

``method`` selects one of six scoring strategies; see
:c:type:`n4m_filter_x_outlier_method_t`.
```
</details>

### Parameters

| Name | Type | Default |
|------|------|---------|
| `method` | `str` | `'mahalanobis'` |
| `use_threshold` | `bool` | `False` |
| `threshold` | `float` | `0.0` |
| `n_components` | `int` | `0` |
| `contamination` | `float` | `0.1` |
| `seed` | `int` | `0` |
| `n_estimators` | `int` | `100` |
| `max_samples` | `int` | `256` |

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

Multivariate outlier filter on the design matrix ``X``.

### Implementation

C ABI `n4m_filter_x_outlier_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.XOutlierFilter`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import XOutlierFilter
op = XOutlierFilter()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)