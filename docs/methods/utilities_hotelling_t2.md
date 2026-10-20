# `hotelling_t2` — n4m.outlier_detection.hotelling_t2

_Namespace_: **`n4m.outlier_detection`** · _Fully-qualified_: `n4m.outlier_detection.hotelling_t2` · _Catalog id_: `utilities.hotelling_t2`

## API surface

**C ABI (ABI 2):** [`n4m_outlier_detection_hotelling_t2`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/outlier_detection.h#L135) · [`n4m_outlier_detection_hotelling_t2_resolved`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/outlier_detection.h#L143). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):** `from n4m.outlier_detection import hotelling_t2`

**Signature:** [`hotelling_t2(X, n_components: int = 5, alpha: float = 0.05)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py#L10699)

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

The input matrix is column-centered with a corrected two-pass mean (constant columns centre to exact zeros) and decomposed by compact SVD. Only resolved directions enter, $\sigma_j > 16\max(n,p)\,\varepsilon\,\sigma_1$; with $r\le k$ of them, $T_i^2=\sum_{j=1}^r t_{ij}^2/\lambda_j=(n-1)\sum_{j=1}^r u_{ij}^2$, evaluated in the second, scale-invariant form. The upper control limit is $r(n-1)(n+1)/[n(n-r)]\,F_{1-\alpha}(r,n-r)$, solved directly in the UCL $G$: with $\kappa=(n^2-1)/n$, $G/(\kappa+G)$ follows a Beta$(r/2,(n-r)/2)$ law whose small tail (upper for $\alpha\le 0.5$, lower otherwise) is evaluated in logarithms from $G/\kappa$ and inverted by bisection on the normal doubles; $r=0$ gives $T^2=0$ and UCL $0$.

### Appropriate uses

Flagging multivariate score-space extremes directly from a calibration matrix when no PLS model object is involved.

### Limits and validation

The implementation requires `rows >= cols`, $k< n$, finite dense F64 data, and treats the same input as both PCA fit and scored set. The finite-sample F limit assumes the classical approximately normal model. The measured relative error of the UCL is below $10^{-12}$ against mpmath references ($n$ up to $10^6$, $\alpha$ from $10^{-300}$ to $1-2^{-53}$), not a proven bound. A UCL that is not a finite normal double (only for extreme $\alpha$, e.g. $n=3$, $r=1$, $\alpha<7.4\times10^{-309}$) returns `N4M_ERR_NUMERICAL_FAILURE`.

### Implementation

`n4m.outlier_detection.hotelling_t2` wraps the stateless ABI-2 symbol `n4m_outlier_detection_hotelling_t2`; SVD, statistic, and UCL are in `cpp/src/core/utilities/hotelling_t2.c`.

### Sources and provenance

https://doi.org/10.1214/aoms/1177732979; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/utilities/hotelling_t2.c


_See also_: [methods index](index.md).