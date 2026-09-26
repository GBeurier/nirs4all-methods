# `aom_ridge_blender` — n4m.ensemble.aom_ridge_blender

_Namespace_: **`n4m.ensemble`** · _Fully-qualified_: `n4m.ensemble.aom_ridge_blender` · _Catalog id_: `aom_pop.ridge_blender`

## API surface

**C ABI (ABI 2):** [`n4m_ensemble_aom_ridge_blender_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/ensemble.h#L39). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):** `from n4m.ensemble import aom_ridge_blender`

**Signature:** [`aom_ridge_blender(X, y, *, profile: str | int = 'compact', cv: int = 5, fold_ids = None, ridge_lambdas = (0.0001, 0.01, 1.0, 100.0), regularizer: float = 0.01, center_x: bool | None = None, scale_x: bool | None = None, center_y: bool | None = None, scale_y: bool | None = None)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py#L6812)

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `X` | `—` | `required` |
| `y` | `—` | `required` |
| `profile` | `str \| int` | `'compact'` |
| `cv` | `int` | `5` |
| `fold_ids` | `—` | `None` |
| `ridge_lambdas` | `—` | `(0.0001, 0.01, 1.0, 100.0)` |
| `regularizer` | `float` | `0.01` |
| `center_x` | `bool \| None` | `None` |
| `scale_x` | `bool \| None` | `None` |
| `center_y` | `bool \| None` | `None` |
| `scale_y` | `bool \| None` | `None` |

## Explanations

### Bibliographic source

No canonical paper defines this exact n4m candidate bank; it uses non-negative stacking. Beurier, G. et al. (2026). *AOM-PLS / POP-PLS* paper companion, arXiv:2605.13587, https://arxiv.org/abs/2605.13587. The product variants below are implementation-specific extensions; the paper does not by itself specify their ABI-2 orchestration.

### Mathematical principle

Generate OOF predictions for every strict-linear AOM Ridge candidate, solve non-negative weights constrained to sum to one, refit each candidate on all rows, and blend their raw-space linear coefficients.

### Appropriate uses

Stable combination of several plausible strict AOM Ridge candidates while retaining a replayable linear predictor.

### Limits and validation

Weights must be learned from OOF rather than in-sample predictions. Highly correlated candidates can yield unstable weights; the native bank is finite and is not a generic arbitrary-estimator blender.

### Implementation

`n4m.ensemble.aom_ridge_blender` and `AOMRidgeBlenderRegressor`; C ABI `n4m_ensemble_aom_ridge_blender_fit`.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/aom_ridge_blender.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/ensemble.py

## Catalog note

Native strict-linear AOM Ridge OOF simplex blender over compact/wide chain banks and positive Ridge-lambda grids. Compact has 12 chains; wide has 31 strict-linear chains, including Gaussian, FCK and Whittaker variants. It returns candidate predictions, OOF candidate predictions, non-negative blend weights, and final input_coefficients plus intercept for replay through NativeAOMRidgeBlenderRegressor; native v1 builds in CUDA-enabled configurations but is not yet a fused batched GPU blender.

_Timing benchmark_: `benchmarks/cross_binding/bench_aom_ridge_blender_timing.py`


_See also_: [methods index](index.md).