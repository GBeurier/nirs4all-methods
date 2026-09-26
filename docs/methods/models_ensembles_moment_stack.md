# `moment_stack` — n4m.ensemble.moment_stack

_Namespace_: **`n4m.ensemble`** · _Fully-qualified_: `n4m.ensemble.moment_stack` · _Catalog id_: `models.ensembles.moment_stack`

## API surface

**C ABI:** no standalone exported symbol is declared for this method.

**Python (verified public re-export):** `from n4m.ensemble import moment_stack`

**Signature:** [`moment_stack(X, y, **kwargs)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py#L8796)

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `X` | `—` | `required` |
| `y` | `—` | `required` |
| `kwargs` | `—` | `required` |

## Explanations

### Bibliographic source

No canonical paper defines this n4m moment-stack product route; its implementation is a documented linear stack over moment-compatible candidate predictions.

### Mathematical principle

Build a candidate prediction matrix from the declared moment-compatible routes and fit a linear meta-level combination using out-of-fold predictions, so the stack learns weights without reusing a candidate's own fit rows.

### Appropriate uses

Combining a fixed, auditable set of linear/moment-compatible spectral predictors when complementary errors are expected.

### Limits and validation

The stack requires a valid OOF plan and can overfit when many highly correlated candidates are included. It is a product composition rather than a claim of a new statistical estimator; assess it in an outer validation loop.

### Implementation

`n4m.ensemble.moment_stack` and `MomentStack`; no standalone C ABI symbol.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/ensemble.py; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/moment_facade.py

## Catalog note

Python-backed train-only OOF linear stack over native moment-compatible Ridge, PLS sweep, PCR, continuum, ECR and CPPLS heads. Base predictions are fit on training folds only; the final reusable estimator fits the same base heads on all rows and applies a small Ridge meta-model over base predictions. This is a moment-model composition, not a nonlinear feature lift and not a fused CUDA grinder.

_Timing benchmark_: `benchmarks/cross_binding/bench_moment_stack_timing.py`


_See also_: [methods index](index.md).