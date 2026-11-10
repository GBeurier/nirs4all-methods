# `finite_column_quantiles` — n4m.utilities.finite_column_quantiles

_Namespace_: **`n4m.utilities`** · _Fully-qualified_: `n4m.utilities.finite_column_quantiles` · _Catalog id_: `utilities.finite_column_quantiles`

## API surface

**C ABI:** no standalone exported symbol is declared for this method.

**Python (verified public re-export):** `from n4m.roles import FiniteColumnQuantiles`

**Signature:** [`FiniteColumnQuantiles(*, probabilities = None)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/roles/_generated.py#L5629)

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `probabilities` | `—` | `None` |

## Explanations

### Bibliographic source

No single canonical paper is asserted for this implementation-specific utility. Its finite-value filtering, type-7 interpolation and named output contract are specified by the native implementation and public-C contract test linked below. These sources document behavior, not external-reference parity.

### Mathematical principle

For each column of X, collect and sort only finite observations. With n finite values sorted as v[0], ..., v[n-1], compute h=(n-1)*p, i=floor(h) and f=h-i for each requested probability p. Return v[i]+f*(v[i+1]-v[i]) when interpolation is needed; exact nodes use v[i] directly. Probabilities zero and one return the minimum and maximum, respectively, and a singleton returns its sole finite value for every p. Preserve caller probability order and duplicates. A column with no finite values returns NaN for every p.

### Appropriate uses

Computing descriptive per-channel summaries of incomplete numerical matrices while reporting the finite sample count for each channel. If summaries inform a learned threshold or preprocessing choice, callers must restrict their inputs to the appropriate training scope.

### Limits and validation

X must be an F64 matrix with positive row and column counts. The required probabilities parameter is a non-empty double array of finite values in [0,1]; the Python constructor's None default does not supply probabilities. NaN and both infinities in X are excluded rather than imputed, so columns may summarize different populations. No weighting, fitted state, uncertainty interval or statistical coverage guarantee is provided. The existing C test exercises this contract; no external-reference parity is claimed.

### Implementation

The registered utilities.finite_column_quantiles procedure calls run_finite_column_quantiles in cpp/src/core/estimator/procedures.cpp through the shared n4m_procedure_run ABI in cpp/include/n4m/estimator.h. It reads X using explicit row and column strides and returns quantiles as an F64 matrix of shape (number of probabilities, number of columns), plus finite_counts as an int64 vector with one entry per column. Python exposes n4m.roles.FiniteColumnQuantiles(probabilities=...).run(X) through the generic native procedure transport without recomputing quantiles.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/estimator.h; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/estimator/procedures.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/catalog/methods/utilities.finite_column_quantiles.yaml; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/tests/test_finite_column_quantiles.c

## Catalog note

Per-column type-7 quantiles of finite X entries only. Required non-empty probabilities in [0,1], in caller order. Outputs: quantiles (probabilities x columns, F64), finite_counts (one int64 per column). Empty finite columns return NaN and count zero. No fitted state or imputation.


_See also_: [methods index](index.md).