# `filter_y_outlier` — Univariate target outlier filter

_Group_: **Sample / feature filters** · _C ABI_: `n4m_outlier_detection_y_outlier_*`

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

## Parameters

| Name | Type | Default |
|------|------|---------|
| `method` | `str` | `'iqr'` |
| `threshold` | `float` | `1.5` |
| `lower_percentile` | `float` | `1.0` |
| `upper_percentile` | `float` | `99.0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_outlier_detection_y_outlier_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/outlier_detection.h#L50) · [`n4m_outlier_detection_y_outlier_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/outlier_detection.h#L41) · [`n4m_outlier_detection_y_outlier_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/outlier_detection.h#L45) · [`n4m_outlier_detection_y_outlier_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/outlier_detection.h#L47) · [`n4m_outlier_detection_y_outlier_is_fitted`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/outlier_detection.h#L54). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.outlier_detection import YOutlierFilter
```

Source signature: [`YOutlierFilter(method: str = 'iqr', threshold: float = 1.5, lower_percentile: float = 1.0, upper_percentile: float = 99.0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/filters.py#L40).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No single paper defines this four-rule facade. IQR fences, z scores, percentiles, and scaled MAD are classical robust/descriptive rules; the exact NumPy-compatible percentile interpolation is source-defined.

### Mathematical principle

Fit one interval on y: IQR uses $[Q_1-tIQR,Q_3+tIQR]$; z score uses $[\mu-t\sigma,\mu+t\sigma]$; percentile uses configured quantiles; MAD uses $[m-t(1.4826\,MAD),m+t(1.4826\,MAD)]$. Apply keeps values inside the learned bounds and returns mask plus counts. Quantiles use linear interpolation.

### Appropriate uses

Auditable removal or flagging of extreme reference values before fitting a calibration model.

### Limits and validation

Filtering y narrows the calibration domain and can bias validation. Z score assumes a meaningful mean/scale; IQR and MAD can degenerate on tied values. Fit bounds only on the training partition.

### Implementation

Python role API `n4m.outlier_detection.YOutlierFilter`; ABI 2 family `n4m_outlier_detection_y_outlier_{create,fit,apply,is_fitted,destroy}`.

The ABI-2 implementation is the `n4m_outlier_detection_y_outlier_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/filters/y_outlier.h


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)