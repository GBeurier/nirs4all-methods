# `linear_stack_compress` — n4m.ensemble.linear_stack_compress

_Namespace_: **`n4m.ensemble`** · _Fully-qualified_: `n4m.ensemble.linear_stack_compress` · _Catalog id_: `aom_pop.linear_stack_compress`

## API surface

**C ABI (ABI 2):** [`n4m_ensemble_linear_stack_compress`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/ensemble.h#L161). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):** `from n4m.ensemble import compress_linear_stack`

**Signature:** [`compress_linear_stack(base_coefficients, base_intercepts, meta_weights, meta_intercept)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/linear_ridge_stack.py#L39)

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `base_coefficients` | `—` | `required` |
| `base_intercepts` | `—` | `required` |
| `meta_weights` | `—` | `required` |
| `meta_intercept` | `—` | `required` |

## Explanations

### Bibliographic source

No separate publication defines this ABI-2 deployment operation; it is the exact affine composition used by the versioned AOM stack contract. Source contract: https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/ensemble.h.

### Mathematical principle

Given base coefficient columns B, base intercepts b, meta weights w, and a meta intercept c, compute the deployment predictor with coefficients B w and intercept b w + c, preserving every output column without refitting.

### Appropriate uses

Collapsing an already fitted fully affine stack into one portable coefficient matrix and intercept vector for lower-cost replay.

### Limits and validation

Every preprocessing and base predictor must already be affine in the supplied input coordinates. The operation performs no fitting, validation, or leakage check and cannot compress nonlinear or sample-adaptive transformations.

### Implementation

`n4m.ensemble.compress_linear_stack`; C ABI `n4m_ensemble_linear_stack_compress`.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/linear_stack.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_linear_stack.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/linear_ridge_stack.py

## Catalog note

Native affine coefficient/intercept composition, including multiple outputs. No fitting or prediction changes; fully affine inputs required.

_Timing benchmark_: `benchmarks/cross_binding/bench_aom_calibration_contract.py`


_See also_: [methods index](index.md).