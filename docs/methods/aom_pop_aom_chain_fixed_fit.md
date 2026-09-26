# `aom_chain_fixed_fit` — n4m.model_selection.aom_search.aom_chain_fixed_fit

_Namespace_: **`n4m.model_selection.aom_search`** · _Fully-qualified_: `n4m.model_selection.aom_search.aom_chain_fixed_fit` · _Catalog id_: `aom_pop.aom_chain_fixed_fit`

## API surface

**C ABI (ABI 2):** [`n4m_model_selection_aom_chain_fixed_fit_run`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L387). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):** `from n4m.model_selection.aom_search import aom_chain_fixed_fit_run`

**Signature:** [`aom_chain_fixed_fit_run(X, y, chain, *, head: str | int = 'ridge', param: float = 0.1, center_x: bool | None = None, scale_x: bool | None = None, center_y: bool | None = None, scale_y: bool | None = None, moment_policy: str | int = 'auto', cuda_pls_parallel_folds: bool | None = None, cuda_pls_min_device_features: int | None = None, cuda_pls_many_batched: bool | None = None)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py#L6662)

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `X` | `—` | `required` |
| `y` | `—` | `required` |
| `chain` | `—` | `required` |
| `head` | `str \| int` | `'ridge'` |
| `param` | `float` | `0.1` |
| `center_x` | `bool \| None` | `None` |
| `scale_x` | `bool \| None` | `None` |
| `center_y` | `bool \| None` | `None` |
| `scale_y` | `bool \| None` | `None` |
| `moment_policy` | `str \| int` | `'auto'` |
| `cuda_pls_parallel_folds` | `bool \| None` | `None` |
| `cuda_pls_min_device_features` | `int \| None` | `None` |
| `cuda_pls_many_batched` | `bool \| None` | `None` |

## Explanations

### Bibliographic source

No single canonical paper defines this fixed-candidate ABI wrapper. Beurier, G. et al. (2026). *AOM-PLS / POP-PLS* paper companion, arXiv:2605.13587, https://arxiv.org/abs/2605.13587. The product variants below are implementation-specific extensions; the paper does not by itself specify their ABI-2 orchestration.

### Mathematical principle

Fit one already-selected strict AOM chain and head on the supplied calibration data. The operation separates candidate choice from the final refit, preserving the selected chain's configuration and prediction surface.

### Appropriate uses

Materializing a model selected by a prior, recorded search or by an external validation protocol.

### Limits and validation

This operation performs no independent model selection and cannot repair leakage in the upstream choice. Its result is valid only for the recorded chain/head parameters and compatible input wavelength layout.

### Implementation

`n4m.model_selection.aom_search.aom_chain_fixed_fit_run`; C ABI `n4m_model_selection_aom_chain_fixed_fit_run`.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_method_result.cpp

## Catalog note

Native final-only reusable fit for one already-selected caller-provided strict-linear AOM chain/head/parameter. This is the individual winner reuse surface behind `NativeAOMFixedCandidateRegressor(fit_mode="final_only")` and the final model-building step of `NativeAOMScreenRefitRegressor`: it fits the selected Ridge lambda or PLS component count on all training rows, folds transformed coefficients back into original input space, and returns final predictions, input_coefficients and intercept without running CV. It is not a ranking endpoint: OOF/fold outputs are empty and CV score fields are NaN unless a higher-level wrapper injects an exact-CV score that was already verified by `aom_refit_candidates`.

_Timing benchmark_: `benchmarks/cross_binding/bench_aom_sweep_timing.py`


_See also_: [methods index](index.md).