# `transfer_metrics` — n4m.domain_adaptation.metrics.transfer_metrics

_Namespace_: **`n4m.domain_adaptation.metrics`** · _Fully-qualified_: `n4m.domain_adaptation.metrics.transfer_metrics` · _Catalog id_: `utilities.transfer_metrics`

## API surface

**C ABI (ABI 2):** [`n4m_domain_adaptation_transfer_metrics_compute`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/domain_adaptation.h#L36). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):** `from n4m.domain_adaptation.metrics import transfer_metrics`

**Signature:** [`transfer_metrics(X_source, X_target, n_components: int = 10, k_neighbors: int = 10, seed: int = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py#L10783)

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `X_source` | `—` | `required` |
| `X_target` | `—` | `required` |
| `n_components` | `int` | `10` |
| `k_neighbors` | `int` | `10` |
| `seed` | `int` | `0` |

## Explanations

### Bibliographic source

This bundle has no single canonical paper. Its components include linear CKA (Kornblith et al., 2019, https://proceedings.mlr.press/v97/kornblith19a.html), Procrustes analysis (Gower, 1975, https://doi.org/10.1007/BF02291478), and trustworthiness (Venna & Kaski, 2001, https://doi.org/10.1016/S0893-6080(01)00150-6).

### Mathematical principle

Source and target are separately centered and reduced by PCA to a common effective rank. The result contains centroid distance, linear CKA, Grassmann principal-angle distance, RV coefficient, two-dimensional Procrustes disparity, neighborhood trustworthiness, a covariance-plus-nearest-cloud spread distance, and source/target explained-variance ratios.

### Appropriate uses

Quantifying several complementary aspects of dataset shift before or after calibration transfer.

### Limits and validation

The nine values have different scales and directions and should not be collapsed without a declared policy. Grassmann is NaN when feature counts differ; Procrustes and trustworthiness need enough rows, while spread subsampling depends deterministically on `seed`.

### Implementation

`n4m.domain_adaptation.metrics.transfer_metrics` wraps `n4m_domain_adaptation_transfer_metrics_compute`; the PCA/Jacobi and all nine exact definitions are in `cpp/src/core/utilities/transfer_metrics.c`.

### Sources and provenance

https://proceedings.mlr.press/v97/kornblith19a.html; https://doi.org/10.1007/BF02291478; https://doi.org/10.1016/S0893-6080(01)00150-6; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/utilities/transfer_metrics.c


_See also_: [methods index](index.md).