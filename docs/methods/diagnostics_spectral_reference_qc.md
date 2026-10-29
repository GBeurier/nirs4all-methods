# `spectral_reference_qc` — n4m.diagnostics.spectral_reference_qc

_Namespace_: **`n4m.diagnostics`** · _Fully-qualified_: `n4m.diagnostics.spectral_reference_qc` · _Catalog id_: `diagnostics.spectral_reference_qc`

## API surface

**C ABI (ABI 2):** [`n4m_procedure_run`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/estimator.h#L8). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):** `from n4m.roles import SpectralReferenceQc`

**Signature:** [`SpectralReferenceQc(*, valid_mask = (), reference_mode = 'dataset_mean', reference_valid_mask = ())`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/roles/_generated.py#L2182)

**R (source-verified):** [`n4m_spectral_reference_qc(valid_mask = c(), reference_mode = "dataset_mean", reference_valid_mask = c())`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/estimator_roles_generated.R).

```r
library(n4m)
result <- n4m_spectral_reference_qc()
```

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `valid_mask` | `—` | `()` |
| `reference_mode` | `—` | `'dataset_mean'` |
| `reference_valid_mask` | `—` | `()` |

## Explanations

### Bibliographic source

This descriptive protocol fixes the reference, valid coordinates, OLS normalization and finite-output availability explicitly. The least-squares reference definition is documented by NumPy: https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html.

### Mathematical principle

Compute RMS difference and arccos of the normalized inner product on common valid bands. Regress each spectrum against the reference with an intercept, then report offset, gain and residual RMS with denominator equal to the eligible coordinate count. Median and type-7 p95 summaries, including absolute offset and gain-minus-one, remain native.

### Appropriate uses

Comparing spectra against a dataset mean or a supplied calibration-only frozen reference without silently changing the reference when an evaluation batch changes.

### Limits and validation

Zero norms make SAM unavailable; a constant reference makes affine coefficients unidentifiable. Pairwise-valid masks differ from column imputation. These distances do not identify a physical cause or calibrated anomaly probability.

### Implementation

One generic native procedure over finite X plus explicit masks; supplied reference uses X_target. Bindings only marshal the existing typed parameters and named result arrays. Independent OLS, RMS/SAM and layout/mask witnesses qualify the calculation.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/estimator/intrinsic_diagnostics.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/tests/test_intrinsic_diagnostics.py; https://github.com/GBeurier/nirs4all-methods/blob/main/docs/intrinsic_diagnostics_contract.v1.json

## Catalog note

Versioned descriptive protocol with native aggregates and explicit per-row availability; see docs/intrinsic_diagnostics_contract.v1.json.


_See also_: [methods index](index.md).