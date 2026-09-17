# `pls_diagnostics` — n4m.metrics.diagnostics.pls_diagnostics

_Namespace_: **`n4m.metrics.diagnostics`** · _Fully-qualified_: `n4m.metrics.diagnostics.pls_diagnostics` · _Catalog id_: `diagnostics.pls_diagnostics`

## API surface

**C ABI (ABI 2):** [`n4m_metrics_pls_diagnostics_compute`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/metrics.h#L103). Use the linked public header for the exact signature, configuration, and result handles.

**R (source-verified):** [`pls_diagnostics(model, X)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/diagnostics.R).

The source signature has additional required inputs, so no example call is fabricated.

**MATLAB / Octave (source-verified):** [`pls_diagnostics(X, Y, n_components, X_reference)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/matlab/+n4m/pls_diagnostics.m).

The source signature has additional required inputs, so no example call is fabricated.

## Explanations

### Bibliographic source

Hotelling (1931), *The Generalization of Student's Ratio*, https://doi.org/10.1214/aoms/1177732979; Jackson & Mudholkar (1979), *Control Procedures for Residuals Associated With Principal Component Analysis*, https://doi.org/10.1080/00401706.1979.10489779; Wold et al. (2001), *PLS-regression: a basic tool of chemometrics*, https://doi.org/10.1016/S0169-7439(01)00155-1.

### Mathematical principle

Rows are centered by the fitted PLS $X$ mean and projected with the model rotations. $T^2_i=\sum_a t_{ia}^2/s_a^2$ measures score-space distance; $Q_i=\|x_i-\hat{x}_i\|^2$ measures reconstruction residual; DModX scales $\sqrt{Q_i}$ by residual degrees of freedom and, when supplied, a reference residual scale. One call returns all three vectors plus model dimensions.

### Appropriate uses

Diagnosing extrapolation, leverage, and spectral lack of fit for observations evaluated against one fitted PLS model.

### Limits and validation

Statistics are model-relative and component-count dependent. T² needs stored training scores or an explicit reference; DModX scaling is unreliable with too few residual degrees of freedom. This is distinct from the standalone PCA utilities below.

### Implementation

The C ABI entry is `n4m_metrics_pls_diagnostics_compute`, producing an `n4m_method_result_t`. The current public Python `n4m.metrics.diagnostics` module does not yet expose a wrapper; formulas are in `cpp/src/core/pls_diagnostics.cpp`.

### Sources and provenance

https://doi.org/10.1214/aoms/1177732979; https://doi.org/10.1080/00401706.1979.10489779; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/pls_diagnostics.cpp


_See also_: [methods index](index.md).