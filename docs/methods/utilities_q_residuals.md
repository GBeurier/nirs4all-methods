# `q_residuals` — n4m.outlier_detection.q_residuals

_Namespace_: **`n4m.outlier_detection`** · _Fully-qualified_: `n4m.outlier_detection.q_residuals` · _Catalog id_: `utilities.q_residuals`

## API surface

**C ABI (ABI 2):** [`n4m_outlier_detection_q_residuals`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/outlier_detection.h#L163). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):** `from n4m.outlier_detection import q_residuals`

**Signature:** [`q_residuals(X, n_components: int = 5, alpha: float = 0.05)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py#L10717)

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

Jackson & Mudholkar (1979), *Control Procedures for Residuals Associated With Principal Component Analysis*, Technometrics 21, 341–349, https://doi.org/10.1080/00401706.1979.10489779.

### Mathematical principle

After column centering (the corrected two-pass mean shared with Hotelling T²) and compact SVD, the first $k$ components reconstruct each row. $Q_i=\|x_i-\hat x_i\|_2^2$. The upper limit uses the Jackson–Mudholkar moment approximation from residual eigenvalue sums $\theta_1,\theta_2,\theta_3$ and the upper standard-normal quantile at $\alpha$. Everything runs at one power-of-two scale and is rescaled at the end, so the moments cannot overflow or underflow, and scaling X by $2^s$ scales Q and the UCL by exactly $4^s$.

### Appropriate uses

Detecting observations whose variation lies outside a retained PCA subspace, complementing score-space Hotelling T².

### Limits and validation

The implementation requires `rows >= cols` and fits/scores the same matrix. When no residual direction is resolved ($k=p$, constant X, or numerical rank at most $k$), Q and the UCL are exactly 0, a documented state rather than a measurement; Q values at or below $(16\max(n,p)\,\varepsilon\,\sigma_1)^2$ are rounding noise. Values that are not representable, or a non-positive Jackson–Mudholkar base (possible only for $\alpha>0.5$), return `N4M_ERR_NUMERICAL_FAILURE`. The Jackson–Mudholkar approximation can be poor for small samples, nonnormal residuals, or near-degenerate residual eigenvalues.

### Implementation

`n4m.outlier_detection.q_residuals` wraps `n4m_outlier_detection_q_residuals`; reconstruction and the moment-based UCL are in `cpp/src/core/utilities/q_residuals.c`.

### Sources and provenance

https://doi.org/10.1080/00401706.1979.10489779; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/utilities/q_residuals.c


_See also_: [methods index](index.md).