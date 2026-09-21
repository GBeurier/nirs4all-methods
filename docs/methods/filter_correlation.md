# `filter_correlation` — Absolute Pearson-correlation feature filter

_Group_: **Signal transforms** · _C ABI_: `n4m_feature_selection_correlation_*`

## Description

Model-agnostic feature filter by absolute correlation to ``y``.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `threshold` | `float` | `0.0` |
| `top_k` | `int \| None` | `None` |

## API and bindings

**C ABI (ABI 2):** [`n4m_feature_selection_correlation_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/feature_selection.h#L26) · [`n4m_feature_selection_correlation_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/feature_selection.h#L28) · [`n4m_feature_selection_correlation_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/feature_selection.h#L30) · [`n4m_feature_selection_correlation_is_fitted`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/feature_selection.h#L38) · [`n4m_feature_selection_correlation_output_cols`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/feature_selection.h#L36) · [`n4m_feature_selection_correlation_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/feature_selection.h#L33). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.feature_selection.filter import CorrelationFilter
```

Source signature: [`CorrelationFilter()`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/advanced.py#L500).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

There is no single canonical feature-selection paper for this use of Pearson correlation. The absolute-score threshold and top-k policy are implementation-specific.

### Mathematical principle

For each channel, compute the centered Pearson coefficient $r_j=\sum_i(x_{ij}-\bar x_j)(y_i-\bar y)/[\sqrt{\sum_i(x_{ij}-\bar x_j)^2}\sqrt{\sum_i(y_i-\bar y)^2}]$ and score it by $|r_j|$. Zero-denominator features score zero. Retain scores strictly above `threshold` or the `top_k` largest.

### Appropriate uses

Fast supervised screening of wavelengths with a strong marginal linear relation to one target.

### Limits and validation

It is univariate, linear, and ignores redundancy between selected wavelengths. Absolute value removes sign information; constant y makes every score zero; using it before cross-validation would leak targets.

### Implementation

Python role API `n4m.feature_selection.filter.CorrelationFilter`; ABI 2 family `n4m_feature_selection_correlation_{create,fit,transform,output_cols,destroy}`.

The ABI-2 implementation is the `n4m_feature_selection_correlation_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_advanced.cpp#L1138


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)