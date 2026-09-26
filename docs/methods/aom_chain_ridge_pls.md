# `aom_chain_ridge_pls` — n4m.model_selection.aom_search.aom_chain_ridge_pls

_Namespace_: **`n4m.model_selection.aom_search`** · _Fully-qualified_: `n4m.model_selection.aom_search.aom_chain_ridge_pls` · _Catalog id_: `aom_pop.aom_chain_ridge_pls`

## API surface

**C ABI:** no standalone exported symbol is declared for this method.

**Python (verified public re-export):** `from n4m.model_selection.aom_search import aom_chain_ridge_pls`

**Signature:** [`aom_chain_ridge_pls(X, y, chains = None, *, profile: str = 'compact', families: dict | None = None, templates: Sequence[Sequence[str]] | None = None, max_chains: int | None = None, n_components: int = 2, pls_components: Sequence[int] | None = None, ridge_lambda: float | None = None, ridge_lambdas: Sequence[float] = (0.0, 0.1, 1.0, 10.0), cv: int = 5, fold_ids = None, center_x: bool = True, center_y: bool = True)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py#L8183)

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `X` | `—` | `required` |
| `y` | `—` | `required` |
| `chains` | `—` | `None` |
| `profile` | `str` | `'compact'` |
| `families` | `dict \| None` | `None` |
| `templates` | `Sequence[Sequence[str]] \| None` | `None` |
| `max_chains` | `int \| None` | `None` |
| `n_components` | `int` | `2` |
| `pls_components` | `Sequence[int] \| None` | `None` |
| `ridge_lambda` | `float \| None` | `None` |
| `ridge_lambdas` | `Sequence[float]` | `(0.0, 0.1, 1.0, 10.0)` |
| `cv` | `int` | `5` |
| `fold_ids` | `—` | `None` |
| `center_x` | `bool` | `True` |
| `center_y` | `bool` | `True` |

## Explanations

### Bibliographic source

No canonical publication defines this strict-chain Ridge-PLS product surface. Beurier, G. et al. (2026). *AOM-PLS / POP-PLS* paper companion, arXiv:2605.13587, https://arxiv.org/abs/2605.13587. The product variants below are implementation-specific extensions; the paper does not by itself specify their ABI-2 orchestration.

### Mathematical principle

For every declared strict-linear chain, PLS component count, and Ridge penalty, fit within each CV fold and select the lowest validation RMSE. Composition of linear chain maps permits the final Ridge-PLS coefficients to be expressed in raw input space.

### Appropriate uses

Selecting among a small, auditable set of linear preprocessing chains while retaining a single coefficient vector for deployment.

### Limits and validation

Only chains that are genuinely linear and sample-independent can be folded back exactly. Cross-validation chooses among many candidates and therefore needs an outer assessment; it does not cover MSC/SNV/EMSC or nonlinear candidate families.

### Implementation

`n4m.model_selection.aom_search.aom_chain_ridge_pls` and `AOMChainRidgePLSRegressor`; there is no standalone C ABI entry point.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/aom_chain_ridge_pls.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/model_selection/aom_search.py

## Catalog note

Native L2-augmented-design Ridge-PLS on strict/raw-base chains, distinct from donor Ridge-on-PLS-scores SingleChainPLSRidge; not a parity port. It applies strict-linear AOM chains sequentially, selects one chain plus PLS component count and Ridge-PLS lambda by train CV, fits through the native ridge_pls binding, and folds final coefficients back to original-input input_coefficients plus intercept. It intentionally excludes SNV, MSC, EMSC, OSC, row-reference-dependent preprocessing, nonlinear lifts, kernels and dataset/source routing; native v1 builds in CUDA-enabled configurations but this is not a fused many-chain GPU Ridge-PLS grinder.

_Timing benchmark_: `benchmarks/cross_binding/bench_aom_chain_ridge_pls_timing.py`


_See also_: [methods index](index.md).