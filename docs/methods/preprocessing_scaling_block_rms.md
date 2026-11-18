# `block_rms` — n4m.transform.scaling.block_rms

_Namespace_: **`n4m.transform.scaling`** · _Fully-qualified_: `n4m.transform.scaling.block_rms` · _Catalog id_: `preprocessing.scaling.block_rms`

## API surface

**C ABI:** no standalone exported symbol is declared for this method.

**Python (verified public re-export):** `from n4m.roles import BlockRms`

**Signature:** [`BlockRms()`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/roles/_generated.py#L4009)

**R (source-verified):** [`n4m_block_rms()`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/estimator_roles_generated.R).

```r
library(n4m)
result <- n4m_block_rms()
```

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No single canonical paper is asserted for this elementary block scaling rule. The exact normalization, finite-input boundary and retained-scalar state are specified by the native source and catalog declaration linked below; no external-reference parity is claimed.

### Mathematical principle

For a source block X with n training rows, fit one scalar s=sqrt(sum_ij X_ij^2/n), without implicit centering, and transform new rows as X/s. An all-zero training block retains s=1. The native implementation uses a scaled sum of squares to avoid forming overflowing X_ij^2. With population-standardized columns, s=sqrt(p), so the block multiplier is 1/sqrt(p), where p is the block width.

### Appropriate uses

Balancing source magnitudes before early feature fusion or a multiblock model when a single training-fitted scale per source is intended. Apply it after explicitly chosen source preprocessing and inside each training scope; holdout rows never contribute to the fitted scalar.

### Limits and validation

No centering, featurewise scaling or imputation occurs. Input values must be finite, and an unrepresentable fitted scale or transformed value is refused. This is not per-sample L2 normalization. A downstream model's per-column autoscaling or internal branch normalization can absorb the source scalar, so it must not be presented as an effective intervention when that downstream normalization cancels it. The shared role fixture predates this addition; registration is not a cross-binding parity claim.

### Implementation

The native preprocessing.scaling.block_rms transformer uses cpp/src/core/estimator/block_rms_adapter.cpp through the generic n4m_estimator_* lifecycle. Python and JS expose BlockRms and R exposes n4m_block_rms through generated thin role bindings. The BRS1 state inside N4ME stores only input width and fitted scalar; prediction-time transforms reuse that state without fitting. Strided F32/F64 input and output views are accepted; optional y is ignored. No public C symbol, ABI change or host numerical implementation is added.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/estimator/block_rms_adapter.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/catalog/methods/preprocessing.scaling.block_rms.yaml; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/estimator.h

## Catalog note

Train-fitted scalar block RMS normalization: s=sqrt(sum(X_train^2)/n_train), without implicit centering. Transform divides a source block by its fitted scalar; an all-zero block keeps scale one. Width and scalar are persisted as BRS1 state; optional y is ignored. No new public ABI symbol or host numerical implementation.


_See also_: [methods index](index.md).