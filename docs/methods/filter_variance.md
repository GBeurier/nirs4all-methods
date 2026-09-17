# `filter_variance` — Population-variance feature filter

_Group_: **Signal transforms** · _C ABI_: `n4m_feature_selection_variance_*`

## Description

Model-agnostic feature filter by variance.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `threshold` | `float` | `0.0` |
| `top_k` | `int \| None` | `None` |

## API and bindings

**C ABI (ABI 2):** [`n4m_feature_selection_variance_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/feature_selection.h#L12) · [`n4m_feature_selection_variance_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/feature_selection.h#L14) · [`n4m_feature_selection_variance_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/feature_selection.h#L15) · [`n4m_feature_selection_variance_is_fitted`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/feature_selection.h#L22) · [`n4m_feature_selection_variance_output_cols`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/feature_selection.h#L20) · [`n4m_feature_selection_variance_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/feature_selection.h#L17). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.feature_selection.filter import VarianceFilter
```

Source signature: [`VarianceFilter(threshold: float = 0.0, top_k: int | None = None)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/advanced.py#L451).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Variance thresholding is a basic descriptive-statistics filter with no single canonical method paper. The shipped scoring and tie behavior are defined by the ABI implementation.

### Mathematical principle

At fit time, score channel $j$ by the population variance $v_j=n^{-1}\sum_i(X_{ij}-\bar X_j)^2$. With no `top_k`, retain channels satisfying $v_j>\text{threshold}$ (strict inequality). With `top_k`, retain the k largest scores; stable ascending ranking makes original channel order the tie-break before the retained indices are emitted.

### Appropriate uses

Removing constant or nearly constant wavelengths before modeling.

### Limits and validation

Variance depends on units and preprocessing and says nothing about association with y. The selector does not expose a Boolean support mask through the Python API, only transformed columns.

### Implementation

Python role API `n4m.feature_selection.filter.VarianceFilter`; ABI 2 family `n4m_feature_selection_variance_{create,fit,transform,output_cols,destroy}`.

The ABI-2 implementation is the `n4m_feature_selection_variance_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_advanced.cpp#L1138


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)