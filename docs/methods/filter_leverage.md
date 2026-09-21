# `filter_leverage` — Hat-matrix or PCA-score leverage filter

_Group_: **Sample / feature filters** · _C ABI_: `n4m_outlier_detection_high_leverage_*`

## Description

Hat-matrix or PCA score-space leverage filter.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `method` | `str \| int` | `'hat'` |
| `threshold_multiplier` | `float` | `2.0` |
| `absolute_threshold` | `float \| None` | `None` |
| `n_components` | `int` | `0` |
| `center` | `bool` | `True` |

## API and bindings

**C ABI (ABI 2):** [`n4m_outlier_detection_high_leverage_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/outlier_detection.h#L73) · [`n4m_outlier_detection_high_leverage_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/outlier_detection.h#L59) · [`n4m_outlier_detection_high_leverage_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/outlier_detection.h#L67) · [`n4m_outlier_detection_high_leverage_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/outlier_detection.h#L69) · [`n4m_outlier_detection_high_leverage_is_fitted`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/outlier_detection.h#L71) · [`n4m_outlier_detection_high_leverage_threshold`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/outlier_detection.h#L77). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.outlier_detection import HighLeverageFilter
```

Source signature: [`HighLeverageFilter(method: str | int = 'hat', threshold_multiplier: float = 2.0, absolute_threshold: float | None = None, n_components: int = 0, center: bool = True)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/filters.py#L226).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Rousseeuw & van Zomeren (1990), *Unmasking Multivariate Outliers and Leverage Points*, JASA 85, 633–639, DOI 10.1080/01621459.1990.10474920 (https://doi.org/10.1080/01621459.1990.10474920), provides the leverage/outlier context; the fallback and thresholds here are implementation-specific.

### Mathematical principle

Hat mode computes $h_i=x_i^T(X^TX+10^{-10}I)^{-1}x_i$ after optional centering. PCA mode computes leverage in retained score space; `n_components<=0` selects $\min(n-1,p,50)$. Unless `absolute_threshold` in (0,1) is supplied, the cutoff is `threshold_multiplier` times mean training leverage. Rows at or below the cutoff are kept.

### Appropriate uses

Detecting calibration samples geometrically remote from the fitted X design space.

### Limits and validation

When rows are not greater than columns, requested hat mode silently uses PCA. Leverage measures influence geometry, not spectral quality or response error, and thresholds are training-distribution dependent.

### Implementation

Python role API `n4m.outlier_detection.HighLeverageFilter`; ABI 2 family `n4m_outlier_detection_high_leverage_{create,fit,apply,threshold,destroy}`.

The ABI-2 implementation is the `n4m_outlier_detection_high_leverage_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/filters/high_leverage.h; https://doi.org/10.1080/01621459.1990.10474920


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)