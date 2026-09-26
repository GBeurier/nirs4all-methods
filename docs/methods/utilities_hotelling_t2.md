# `hotelling_t2` — n4m.outlier_detection.hotelling_t2

_Namespace_: **`n4m.outlier_detection`** · _Fully-qualified_: `n4m.outlier_detection.hotelling_t2` · _Catalog id_: `utilities.hotelling_t2`

## API surface

**C ABI (ABI 2):** [`n4m_outlier_detection_hotelling_t2`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/outlier_detection.h#L119). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):** `from n4m.outlier_detection import hotelling_t2`

**Signature:** [`hotelling_t2(X, n_components: int = 5, alpha: float = 0.05)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py#L10661)

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `X` | `—` | `required` |
| `n_components` | `int` | `5` |
| `alpha` | `float` | `0.05` |

## Explanations

### Bibliographic source

Hotelling (1931), *The Generalization of Student's Ratio*, Annals of Mathematical Statistics 2, 360–378, https://doi.org/10.1214/aoms/1177732979.

### Mathematical principle

The input matrix is column-centered and decomposed by compact SVD. For the first $k$ PCA scores, $T_i^2=\sum_{j=1}^k t_{ij}^2/\lambda_j$. The upper control limit is $k(n-1)(n+1)/[n(n-k)]\,F_{1-\alpha}(k,n-k)$, with the F quantile obtained through regularized incomplete-beta inversion.

### Appropriate uses

Flagging multivariate score-space extremes directly from a calibration matrix when no PLS model object is involved.

### Limits and validation

The implementation requires `rows >= cols`, $k< n$, finite dense F64 data, and treats the same input as both PCA fit and scored set. The finite-sample F limit assumes the classical approximately normal model.

### Implementation

`n4m.outlier_detection.hotelling_t2` wraps the stateless ABI-2 symbol `n4m_outlier_detection_hotelling_t2`; SVD, statistic, and UCL are in `cpp/src/core/utilities/hotelling_t2.c`.

### Sources and provenance

https://doi.org/10.1214/aoms/1177732979; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/utilities/hotelling_t2.c


_See also_: [methods index](index.md).