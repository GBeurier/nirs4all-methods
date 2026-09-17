# `pp_flex_pca` — Flexible principal component analysis

_Group_: **Feature extraction** · _C ABI_: `n4m_decomposition_flexible_pca_*`

## Description

PCA with integer or explained-variance component selection.

## Parameters

| Name | Type | Default |
|------|------|---------|
| `n_components` | `float` | `5.0` |

## API and bindings

**C ABI (ABI 2):** [`n4m_decomposition_flexible_pca_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/decomposition.h#L13) · [`n4m_decomposition_flexible_pca_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/decomposition.h#L15) · [`n4m_decomposition_flexible_pca_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/decomposition.h#L16) · [`n4m_decomposition_flexible_pca_is_fitted`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/decomposition.h#L21) · [`n4m_decomposition_flexible_pca_output_cols`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/decomposition.h#L23) · [`n4m_decomposition_flexible_pca_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/decomposition.h#L18). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.decomposition import FlexiblePCA
```

Source signature: [`FlexiblePCA()`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/feature_extraction.py#L158).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Pearson (1901), *On Lines and Planes of Closest Fit to Systems of Points in Space*, Philosophical Magazine 2, 559–572, https://doi.org/10.1080/14786440109462720.

### Mathematical principle

After column centering, compact SVD gives $X_c=U\Sigma V^T$ and scores $T=U_k\Sigma_k$. `n_components` is interpreted either as an integer count or as a target cumulative explained-variance fraction.

### Appropriate uses

Unsupervised compression, visualization, noise reduction, or a fixed latent feature stage before regression.

### Limits and validation

PCA maximizes variance rather than relevance to a response, is scale-sensitive, and requires fold-local fitting. A variance threshold may choose different dimensions across folds.

### Implementation

`n4m.decomposition.FlexiblePCA` wraps `n4m_decomposition_flexible_pca_*`; centering, SVD, variance selection, and projection are in `flexible_pca.c`.

The ABI-2 implementation is the `n4m_decomposition_flexible_pca_*` lifecycle in libn4m.

### Sources and provenance

https://doi.org/10.1080/14786440109462720; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/feature_selection/flexible_pca.c


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)