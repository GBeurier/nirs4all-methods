# `robust_hpo` — n4m.model_selection.aom_search.robust_hpo

_Namespace_: **`n4m.model_selection.aom_search`** · _Fully-qualified_: `n4m.model_selection.aom_search.robust_hpo` · _Catalog id_: `aom_pop.robust_hpo`

## API surface

**C ABI (ABI 2):** [`n4m_model_selection_robust_hpo_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L429). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):** `from n4m.model_selection.aom_search import aom_robust_hpo`

**Signature:** [`aom_robust_hpo(X, y, *, profile: str | int = 'compact', cv: int = 5, heads = ('ridge', 'pls'))`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py#L8897)

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `X` | `—` | `required` |
| `y` | `—` | `required` |
| `profile` | `str \| int` | `'compact'` |
| `cv` | `int` | `5` |
| `heads` | `—` | `('ridge', 'pls')` |

## Explanations

### Bibliographic source

No single canonical paper defines this ABI-2 compact/wide screen. Beurier, G. et al. (2026). *AOM-PLS / POP-PLS* paper companion, arXiv:2605.13587, https://arxiv.org/abs/2605.13587. The product variants below are implementation-specific extensions; the paper does not by itself specify their ABI-2 orchestration.

### Mathematical principle

Evaluate a finite bank of strict-linear spectral chains with Ridge and/or PLS heads by contiguous-fold CV RMSE, select the minimum-score tuple, refit on all calibration rows, and fold its linear coefficients back to the original feature space.

### Appropriate uses

Fast, reproducible comparison of a declared preprocessing bank when the final model must remain a replayable linear predictor.

### Limits and validation

Candidate selection must be nested inside external validation. Native v1 excludes stateful, sample-fitted, nonlinear, and source-routed transformations; a CUDA build does not make the complete candidate bank a fused GPU search.

### Implementation

`n4m.model_selection.aom_search.aom_robust_hpo` and `AOMRobustHPOSweepRegressor`; C ABI `n4m_model_selection_robust_hpo_fit`.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/aom_robust_hpo.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_method_result.cpp

## Catalog note

Native strict-linear AOM robust-HPO screen over compact/wide preprocessing banks and Ridge/PLS heads. Compact has 12 chains; wide has 31 strict-linear chains, including Gaussian, FCK and Whittaker variants. Exposes input_coefficients plus intercept for replay in original feature space and backs NativeAOMRobustHPORegressor. Builds in CPU and CUDA-enabled libn4m configurations; native v1 is not the fused batched GPU grinder.

_Timing benchmark_: `benchmarks/cross_binding/bench_aom_robust_hpo_timing.py`


_See also_: [methods index](index.md).