# `pca_reference_qc` — n4m.diagnostics.pca_reference_qc

_Namespace_: **`n4m.diagnostics`** · _Fully-qualified_: `n4m.diagnostics.pca_reference_qc` · _Catalog id_: `diagnostics.pca_reference_qc`

## API surface

**C ABI (ABI 2):** [`n4m_procedure_run`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/estimator.h#L8). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):** `from n4m.roles import PCAReferenceQc`

**Signature:** [`PCAReferenceQc(*, valid_mask = (), n_components = 5)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/roles/_generated.py#L2032)

**R (source-verified):** [`n4m_pca_reference_qc(valid_mask = c(), n_components = 5L)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/estimator_roles_generated.R).

```r
library(n4m)
result <- n4m_pca_reference_qc()
```

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `valid_mask` | `—` | `()` |
| `n_components` | `—` | `5` |

## Explanations

### Bibliographic source

Principal-component projection under sample-covariance normalization (ddof1), with a declared numerical resolution. The independent full-SVD oracle is documented at https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html.

### Mathematical principle

Fit means and retained loading vectors on X_target only using the existing native exact Gram/Jacobi solver. Project calibration and query rows through that same fitted state. Retained eigenvalues use calibration score sums of squares divided by n_train-1. Q is squared resolved-subspace residual; T2 sums score squares divided by resolved eigenvalues.

### Appropriate uses

Train-only spectral-domain diagnostics on ordinary wide NIRS matrices, with means, loadings, eigenvalues and frozen query projection available independently of exact full moments.

### Limits and validation

No implicit scaling, imputation, confidence limit or causal calibration is provided. Unresolved components do not contribute to Q/T2; zero resolved rank makes T2 unavailable. Exact full-spectrum moments remain explicitly not requested, never replaced by truncated moments. Whole-computation and peak-buffer budgets may refuse a request before allocation.

### Implementation

The new generic procedure reuses FlexiblePCA state and native exact Gram/Jacobi kernels. Its fast randomized estimator default and serialized state remain unchanged. Independent SVD plus four real 390–541 by2151 matrices validate retained geometry and frozen state.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/estimator/intrinsic_diagnostics.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/tests/test_intrinsic_diagnostics.py; https://github.com/GBeurier/nirs4all-methods/blob/main/docs/intrinsic_diagnostics_contract.v1.json

## Catalog note

Versioned descriptive protocol with native aggregates and explicit per-row availability; see docs/intrinsic_diagnostics_contract.v1.json.


_See also_: [methods index](index.md).