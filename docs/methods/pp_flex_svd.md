# `pp_flex_svd` — Flexible truncated singular-value decomposition

_Group_: **Feature extraction** · _C ABI_: `n4m_decomposition_flexible_svd_*`

## Description

SVD with integer component selection.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `n_components` | `float` | `5.0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_decomposition_flexible_svd_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/decomposition.h#L28) · [`n4m_decomposition_flexible_svd_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/decomposition.h#L30) · [`n4m_decomposition_flexible_svd_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/decomposition.h#L31) · [`n4m_decomposition_flexible_svd_is_fitted`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/decomposition.h#L36) · [`n4m_decomposition_flexible_svd_output_cols`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/decomposition.h#L38) · [`n4m_decomposition_flexible_svd_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/decomposition.h#L33). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.decomposition import FlexibleSVD
```

Source signature: [`FlexibleSVD()`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/feature_extraction.py#L164).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Eckart & Young (1936), *The approximation of one matrix by another of lower rank*, Psychometrika 1, 211–218, https://doi.org/10.1007/BF02288367.

### Mathematical principle

Compact SVD factorizes the uncentered input as $X=U\Sigma V^T$ and emits the first $k$ coordinates $XV_k$. A parameter at least one is truncated to an integer rank; a parameter in $(0,1)$ selects the smallest rank reaching that fraction of total column variance using the projected-component variance ratios.

### Appropriate uses

Low-rank compression when retaining the mean direction is intentional and rank should be chosen directly or from a variance target.

### Limits and validation

Uncentered SVD can devote its first direction to mean offset and is scale-sensitive. The fractional criterion is based on projected variance rather than simply squared singular-value energy, and fitting must remain inside validation folds.

### Implementation

`n4m.decomposition.FlexibleSVD` calls `n4m_decomposition_flexible_svd_*`; the compact factorization and transform are in `flexible_svd.c`.

The ABI-2 implementation is the `n4m_decomposition_flexible_svd_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1007/BF02288367; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/feature_selection/flexible_svd.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)