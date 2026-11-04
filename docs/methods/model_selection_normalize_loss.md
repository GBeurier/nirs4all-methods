# `normalize_loss` — n4m.component_analysis.normalize_loss

_Namespace_: **`n4m.component_analysis`** · _Fully-qualified_: `n4m.component_analysis.normalize_loss` · _Catalog id_: `model_selection.normalize_loss`

## API surface

**C ABI (ABI 2):** [`n4m_component_normalize_loss_v1`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/component_analysis.h#L11). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):** `from n4m.component_analysis import normalize_loss`

**Signature:** [`normalize_loss(values: ArrayLike, baselines: ArrayLike)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/component_analysis.py#L92)

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `values` | `ArrayLike` | `required` |
| `baselines` | `ArrayLike` | `required` |

## Explanations

### Bibliographic source

The relative-baseline scale is defined by the native protocol at https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/component_analysis.h; this arithmetic utility does not claim a new statistical estimator.

### Mathematical principle

For finite nonnegative loss L and finite strictly positive baseline B, return L/B. Zero loss is valid; zero baseline and overflowing ratios are refused.

### Appropriate uses

Place losses with dataset-dependent units on a dimensionless relative-baseline scale before a separately specified component analysis.

### Limits and validation

Matching, baseline selection and dependence assumptions are host responsibilities. Ratios do not remove baseline uncertainty or establish comparable difficulty, causal effects or independent repeated folds.

### Implementation

ABI 2.22 computes ratios natively into a temporary buffer and commits all outputs only after validation. Python and JS/WASM translate buffers without calculating ratios. Input/output aliases are supported; errors leave outputs unchanged.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_component_analysis.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/component_analysis.h; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/tests/test_component_analysis.py

## Catalog note

Finite nonnegative loss divided by a finite strictly-positive matched baseline. Native calculation, atomic output, no inferred matching or uncertainty.


_See also_: [methods index](index.md).