# `filter_quality` — Rule-based row-level spectral quality filter

_Group_: **Sample / feature filters** · _C ABI_: `n4m_outlier_detection_spectral_quality_*`

## Description

Stateless row-level spectrum quality filter.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `max_nan_ratio` | `float` | `0.1` |
| `max_zero_ratio` | `float` | `0.5` |
| `min_variance` | `float` | `1e-08` |
| `max_value` | `float \| None` | `None` |
| `min_value` | `float \| None` | `None` |
| `check_inf` | `bool` | `True` |

## API and bindings

**C ABI (ABI 2):** [`n4m_outlier_detection_spectral_quality_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/outlier_detection.h#L91) · [`n4m_outlier_detection_spectral_quality_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/outlier_detection.h#L82) · [`n4m_outlier_detection_spectral_quality_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/outlier_detection.h#L89). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.outlier_detection import SpectralQualityFilter
```

Source signature: [`SpectralQualityFilter(max_nan_ratio: float = 0.1, max_zero_ratio: float = 0.5, min_variance: float = 1e-08, max_value: float | None = None, min_value: float | None = None, check_inf: bool = True)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/filters.py#L315).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No canonical paper defines this conjunction of data-quality checks. It is an explicit internal quality-control rule set whose thresholds are user policy.

### Mathematical principle

A row is kept only if every enabled rule passes: NaN fraction $\le$ `max_nan_ratio`; no infinity when `check_inf`; exact-zero fraction $\le$ `max_zero_ratio`; NaN-aware population variance $\ge$ `min_variance`; and optional finite extrema within `min_value`/`max_value`. Fit is an idempotent no-op.

### Appropriate uses

Rejecting empty, flat, non-finite, saturated, or out-of-range acquisitions before modeling.

### Limits and validation

Thresholds are scale- and representation-specific. NaNs count as nonzero for the zero-ratio rule, and a row can pass these syntactic checks while remaining chemically implausible.

### Implementation

Python role API `n4m.outlier_detection.SpectralQualityFilter`; ABI 2 family `n4m_outlier_detection_spectral_quality_{create,apply,destroy}`.

The ABI-2 implementation is the `n4m_outlier_detection_spectral_quality_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/filters/spectral_quality.h


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)