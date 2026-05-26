# `filter_y_outlier` — Y Outlier Filter

_Group_: **Sample / feature filters** · _Binding_: `n4m.sklearn.YOutlierFilter` · _C ABI_: `n4m_filter_y_outlier_*`

## Description

Univariate outlier filter on the target vector ``y``.

<details>
<summary>Full binding docstring</summary>

```text
Univariate outlier filter on the target vector ``y``.

``method`` is one of ``"iqr"``, ``"zscore"``, ``"percentile"``, ``"mad"``.
Threshold semantics follow nirs4all's :class:`YOutlierFilter`:

* For ``"iqr"`` / ``"zscore"`` / ``"mad"``: ``threshold`` is the
  multiplier on the IQR / σ / MAD bounds.
* For ``"percentile"``: ``lower_percentile`` and ``upper_percentile``
  define the keep band.
```
</details>

### Parameters

| Name | Type | Default |
|------|------|---------|
| `method` | `str` | `'iqr'` |
| `threshold` | `float` | `1.5` |
| `lower_percentile` | `float` | `1.0` |
| `upper_percentile` | `float` | `99.0` |

## Explanations

### Bibliographic source

_Standard spectroscopic operator — see the nirs4all preprocessing / augmentation handbook and the cited literature within the binding docstring._

### Mathematical principle

Univariate outlier filter on the target vector ``y``.

### Implementation

C ABI `n4m_filter_y_outlier_*` in libn4m (create / apply / destroy lifecycle), wrapped by `n4m.sklearn.YOutlierFilter`. The same numerical kernel backs every language binding.

### Usage

```python
from n4m.sklearn import YOutlierFilter
op = YOutlierFilter()
X_transformed = op.fit_transform(X)
```

---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)