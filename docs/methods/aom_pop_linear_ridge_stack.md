# `linear_ridge_stack` — n4m.ensemble.linear_ridge_stack

_Namespace_: **`n4m.ensemble`** · _Fully-qualified_: `n4m.ensemble.linear_ridge_stack` · _Catalog id_: `aom_pop.linear_ridge_stack`

## API surface

**C ABI:** no standalone exported symbol is declared for this method.

**Python (verified public re-export):** `from n4m.ensemble import LinearRidgeStackRegressor`

**Signature:** [`LinearRidgeStackRegressor(*, operators = None, alphas = None, meta_alphas = None, cv = 3, fold_ids = None, inner_cv = 3)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/linear_ridge_stack.py#L66)

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `operators` | `—` | `None` |
| `alphas` | `—` | `None` |
| `meta_alphas` | `—` | `None` |
| `cv` | `—` | `3` |
| `fold_ids` | `—` | `None` |
| `inner_cv` | `—` | `3` |

## Explanations

### Bibliographic source

No canonical publication defines this exact AOM candidate bank and affine-export surface; it is a product-specific stacked generalization protocol. Beurier, G. et al. (2026). *AOM-PLS / POP-PLS* paper companion, arXiv:2605.13587, https://arxiv.org/abs/2605.13587. The product variants below are implementation-specific extensions; the paper does not by itself specify their ABI-2 orchestration.

### Mathematical principle

Tune every affine base route inside each outer training fold, assemble only its out-of-fold predictions, fit a Ridge meta-head on that OOF design, refit the bases on all calibration rows, and compose the fitted stack into one affine predictor.

### Appropriate uses

Combining complementary strict-linear AOM views while retaining an auditable OOF training path and a compact deployment representation.

### Limits and validation

The inner and outer fold identities are part of the estimator contract. Meta-model selection is not an unbiased test estimate, correlated bases can destabilize weights, and nonlinear or sample-adaptive bases cannot be exported as one affine predictor.

### Implementation

`n4m.ensemble.LinearRidgeStackRegressor` orchestrates native Ridge sweeps and `n4m_ensemble_linear_stack_compress`; it has no standalone fit C ABI symbol.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/linear_ridge_stack.py; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/linear_stack.cpp

## Catalog note

Python orchestration over native Ridge sweep and compression: nested-CV base tuning, OOF Ridge meta-head, deployment-only affine export. Distinct from the simplex blender.

_Timing benchmark_: `benchmarks/cross_binding/bench_aom_calibration_contract.py`


_See also_: [methods index](index.md).