# `ridge_global` — n4m.model_selection.aom_search.ridge_global

_Namespace_: **`n4m.model_selection.aom_search`** · _Fully-qualified_: `n4m.model_selection.aom_search.ridge_global` · _Catalog id_: `aom_pop.ridge_global`

## API surface

**C ABI:** no standalone exported symbol is declared for this method.

**Python (verified public re-export):** `from n4m.model_selection.aom_search import aom_ridge_global`

**Signature:** [`aom_ridge_global(X, y, *, operators = None, cv: int = 5, fold_ids = None, ridge_lambdas = (0.0001, 0.01, 1.0, 100.0), center_x: bool | None = None, scale_x: bool | None = None, center_y: bool | None = None, scale_y: bool | None = None, moment_policy: str | int = 'auto')`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py#L6721)

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `X` | `—` | `required` |
| `y` | `—` | `required` |
| `operators` | `—` | `None` |
| `cv` | `int` | `5` |
| `fold_ids` | `—` | `None` |
| `ridge_lambdas` | `—` | `(0.0001, 0.01, 1.0, 100.0)` |
| `center_x` | `bool \| None` | `None` |
| `scale_x` | `bool \| None` | `None` |
| `center_y` | `bool \| None` | `None` |
| `scale_y` | `bool \| None` | `None` |
| `moment_policy` | `str \| int` | `'auto'` |

## Explanations

### Bibliographic source

No canonical paper defines this strict-linear Ridge route. Beurier, G. et al. (2026). *AOM-PLS / POP-PLS* paper companion, arXiv:2605.13587, https://arxiv.org/abs/2605.13587. The product variants below are implementation-specific extensions; the paper does not by itself specify their ABI-2 orchestration.

### Mathematical principle

Treat each declared strict AOM operator as a one-step chain, choose the operator and positive Ridge penalty by CV RMSE, then export the selected linear predictor in original input coordinates.

### Appropriate uses

A concise, auditable global preprocessing choice for Ridge calibration models.

### Limits and validation

It cannot model interactions among sequential operators. A low CV score after selection is optimistic unless evaluated in an outer split; only strict linear transformations can be folded back.

### Implementation

`n4m.model_selection.aom_search.aom_ridge_global` and `AOMRidgeGlobalRegressor`; no standalone C ABI symbol.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/model_selection/aom_search.py

## Catalog note

Python-backed donor-style AOM Ridge global selector constrained to strict-linear single-operator AOM views. It delegates scoring and final fit to the native aom_chain_sweep_run Ridge-only path, selects one operator plus one positive Ridge alpha by train CV, and returns folded input_coefficients plus intercept for replay through NativeAOMRidgeGlobalRegressor. It intentionally excludes donor branch_global, MKL/kernel, row-reference-dependent preprocessing and nonlinear AOM Ridge modes; native v1 builds in CUDA-enabled configurations but this is not yet a fused GPU grinder.

_Timing benchmark_: `benchmarks/cross_binding/bench_aom_ridge_global_timing.py`


_See also_: [methods index](index.md).