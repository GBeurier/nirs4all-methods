# `aom_operator_pls_stack` — n4m.ensemble.aom_operator_pls_stack

_Namespace_: **`n4m.ensemble`** · _Fully-qualified_: `n4m.ensemble.aom_operator_pls_stack` · _Catalog id_: `aom_pop.operator_pls_stack`

## API surface

**C ABI (ABI 2):** [`n4m_ensemble_aom_operator_pls_stack_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/ensemble.h#L91). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):** `from n4m.ensemble import aom_operator_pls_stack`

**Signature:** [`aom_operator_pls_stack(X, y, *, profile: str | int = 'compact', cv: int = 5, fold_ids = None, components = (2, 4, 8), alphas = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0), std_penalty: float = 0.0, gap_penalty: float = 0.0, center_x: bool | None = None, scale_x: bool | None = None, center_y: bool | None = None, scale_y: bool | None = None)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py#L8410)

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
| `components` | `—` | `(2, 4, 8)` |
| `alphas` | `—` | `(0.001, 0.01, 0.1, 1.0, 10.0, 100.0)` |
| `std_penalty` | `float` | `0.0` |
| `gap_penalty` | `float` | `0.0` |
| `center_x` | `bool \| None` | `None` |
| `scale_x` | `bool \| None` | `None` |
| `center_y` | `bool \| None` | `None` |
| `scale_y` | `bool \| None` | `None` |

## Explanations

### Bibliographic source

No single paper defines this ABI-2 composition. Beurier, G. et al. (2026). *AOM-PLS / POP-PLS* paper companion, arXiv:2605.13587, https://arxiv.org/abs/2605.13587. The product variants below are implementation-specific extensions; the paper does not by itself specify their ABI-2 orchestration.

### Mathematical principle

For each declared strict AOM view, standardize its $n\times p$ matrix, fit a PLS1 projector with $k_{eff}$ retained components, and concatenate the resulting latent scores into an $n\times\sum_b k_{eff,b}$ design. A Ridge head is fitted on that design. Cross-validation scores and selects the component count and Ridge penalty; it does not use base-model prediction stacking.

### Appropriate uses

Combining complementary strict AOM views through a compact supervised latent design for a univariate response.

### Limits and validation

The implementation supports one response and relies on each view's PLS1 score projection. OOF predictions are used to score candidate component/penalty settings, not as the columns of the final Ridge design. Correlated views can still make the Ridge weights unstable and selection needs an outer assessment.

### Implementation

`n4m.ensemble.aom_operator_pls_stack` and `AOMOperatorPLSStackRegressor`; C ABI `n4m_ensemble_aom_operator_pls_stack_fit`.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/aom_operator_pls_stack.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/ensemble.py

## Catalog note

Native strict-linear AOM operator PLS1 score stack over compact/wide operator banks with a Ridge head. Compact has 12 operators; wide has 31 strict-linear operators, including Gaussian, FCK and Whittaker variants. Single-target only. It returns fold scores, final stack features, Ridge-head coefficients for audit, and folded input_coefficients plus input_intercept for replay through NativeAOMOperatorPLSStackRegressor; native v1 builds in CUDA-enabled configurations but is not yet a fused batched GPU stack.

_Timing benchmark_: `benchmarks/cross_binding/bench_aom_operator_pls_stack_timing.py`


_See also_: [methods index](index.md).