# `calibration` — n4m.model_selection.aom_calibration.calibration

_Namespace_: **`n4m.model_selection.aom_calibration`** · _Fully-qualified_: `n4m.model_selection.aom_calibration.calibration` · _Catalog id_: `aom_pop.calibration`

## API surface

**C ABI (ABI 2):** [`n4m_model_selection_aom_calibration_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L514) · [`n4m_model_selection_aom_calibration_predict`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L521). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):** `from n4m.model_selection.aom_calibration import AOMPLSRegressor`

**Signature:** [`AOMPLSRegressor()`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/model_selection/aom_calibration.py#L266)

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

No canonical publication defines this complete ABI-2 calibration protocol. Beurier, G. et al. (2026). *AOM-PLS / POP-PLS* paper companion, arXiv:2605.13587, https://arxiv.org/abs/2605.13587. The product variants below are implementation-specific extensions; the paper does not by itself specify their ABI-2 orchestration.

### Mathematical principle

Evaluate the declared raw, SNV, and MSC branches fold-locally with the strict10 operator bank, select the branch, chain, and PLS component or Ridge penalty by training-fold RMSE, then refit the selected configuration on every calibration row.

### Appropriate uses

Reproducible global or screened AOM calibration when branch state, candidate identity, and the final refit must share one versioned native contract.

### Limits and validation

The fast protocol screens candidates and does not prove that discarded candidates are inferior. Selection scores require an outer assessment, and sample-adaptive SNV or MSC branches cannot be reduced to one raw-input affine map.

### Implementation

`n4m.model_selection.aom_calibration` exposes the global and Fast AOM PLS/Ridge estimators; C ABI `n4m_model_selection_aom_calibration_fit` and `n4m_model_selection_aom_calibration_predict`.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/aom_calibration.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_aom_calibration.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/model_selection/aom_calibration.py

## Catalog note

Versioned strict10 native global/Fast PLS and Ridge branch-CV protocols. Also exposes AOMRidgeRegressor, FastAOMPLSRegressor and experimental FastAOMRidgeRegressor. See docs/methods/aom_calibration.md.

_Timing benchmark_: `benchmarks/cross_binding/bench_aom_calibration_contract.py`


_See also_: [methods index](index.md).