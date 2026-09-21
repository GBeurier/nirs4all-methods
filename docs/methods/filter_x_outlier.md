# `filter_x_outlier` — Multistrategy X-space outlier filter

_Group_: **Sample / feature filters** · _C ABI_: `n4m_outlier_detection_x_outlier_*`

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

## Parameters

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

## API and bindings

**C ABI (ABI 2):** [`n4m_outlier_detection_x_outlier_apply`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/outlier_detection.h#L157) · [`n4m_outlier_detection_x_outlier_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/outlier_detection.h#L142) · [`n4m_outlier_detection_x_outlier_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/outlier_detection.h#L152) · [`n4m_outlier_detection_x_outlier_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/outlier_detection.h#L153) · [`n4m_outlier_detection_x_outlier_is_fitted`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/outlier_detection.h#L155). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):**

```python
from n4m.outlier_detection import XOutlierFilter
```

Source signature: [`XOutlierFilter(method: str = 'mahalanobis', use_threshold: bool = False, threshold: float = 0.0, n_components: int = 0, contamination: float = 0.1, seed: int = 0, n_estimators: int = 100, max_samples: int = 256)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/filters.py#L131).

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

The facade combines distinct methods: Liu, Ting & Zhou (2008), Isolation Forest, DOI 10.1109/ICDM.2008.17 (https://doi.org/10.1109/ICDM.2008.17); Breunig et al. (2000), LOF, DOI 10.1145/342009.335388 (https://doi.org/10.1145/342009.335388); and Rousseeuw & Van Driessen (1999), FAST-MCD, DOI 10.1080/00401706.1999.10485670 (https://doi.org/10.1080/00401706.1999.10485670). Mahalanobis and PCA Q/T² options are classical.

### Mathematical principle

`method` selects empirical Mahalanobis distance, simplified FAST-MCD robust Mahalanobis, PCA residual $Q_i=\|x_i-\hat x_i\|^2$, PCA-score Hotelling-like $T_i^2$, isolation-forest mean path score, or LOF with $k=\min(20,n-1)$. Defaults use a chi-square-derived distance cutoff for Mahalanobis, training 95th percentiles for PCA scores, and the `contamination` quantile for forest/LOF.

### Appropriate uses

Choosing a global, robust, subspace, isolation, or local-density detector under one mask API.

### Limits and validation

The native robust covariance, isolation forest, and LOF are independent vendored implementations, not calls to scikit-learn and not guaranteed numerically identical. Distance methods are sensitive to scaling; contamination fixes an expected fraction rather than an absolute scientific criterion.

### Implementation

Python role API `n4m.outlier_detection.XOutlierFilter`; ABI 2 family `n4m_outlier_detection_x_outlier_{create,fit,apply,destroy}`; vendored kernels live under `cpp/src/core/filters/_vendored/`.

The ABI-2 implementation is the `n4m_outlier_detection_x_outlier_*` lifecycle in libn4m.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/filters/x_outlier.h; https://doi.org/10.1109/ICDM.2008.17; https://doi.org/10.1145/342009.335388; https://doi.org/10.1080/00401706.1999.10485670


---

_See also_: [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)