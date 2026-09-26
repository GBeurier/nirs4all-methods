# `aom_sweep` — n4m.model_selection.aom_search.aom_sweep

_Namespace_: **`n4m.model_selection.aom_search`** · _Fully-qualified_: `n4m.model_selection.aom_search.aom_sweep` · _Catalog id_: `aom_pop.aom_sweep`

## API surface

**C ABI (ABI 2):** [`n4m_model_selection_aom_sweep_run`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L364). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):** `from n4m.model_selection.aom_search import aom_sweep_run`

**Signature:** [`aom_sweep_run(X, y, *, profile: str | int = 'compact', cv: int = 5, fold_ids = None, ridge_lambdas = (0.01, 0.1, 1.0, 10.0), pls_components = None, heads = ('ridge',), center_x: bool | None = None, scale_x: bool | None = None, center_y: bool | None = None, scale_y: bool | None = None, moment_policy: str | int = 'auto', pls_score_mode: str | int = 'cv', score_only: bool = False, cuda_pls_parallel_folds: bool | None = None, cuda_pls_min_device_features: int | None = None, cuda_pls_many_batched: bool | None = None)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py#L6399)

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
| `ridge_lambdas` | `—` | `(0.01, 0.1, 1.0, 10.0)` |
| `pls_components` | `—` | `None` |
| `heads` | `—` | `('ridge',)` |
| `center_x` | `bool \| None` | `None` |
| `scale_x` | `bool \| None` | `None` |
| `center_y` | `bool \| None` | `None` |
| `scale_y` | `bool \| None` | `None` |
| `moment_policy` | `str \| int` | `'auto'` |
| `pls_score_mode` | `str \| int` | `'cv'` |
| `score_only` | `bool` | `False` |
| `cuda_pls_parallel_folds` | `bool \| None` | `None` |
| `cuda_pls_min_device_features` | `int \| None` | `None` |
| `cuda_pls_many_batched` | `bool \| None` | `None` |

## Explanations

### Bibliographic source

The global AOM-PLS paper motivates operator selection; this ABI-2 sweep has no separate canonical paper. Beurier, G. et al. (2026). *AOM-PLS / POP-PLS* paper companion, arXiv:2605.13587, https://arxiv.org/abs/2605.13587. The product variants below are implementation-specific extensions; the paper does not by itself specify their ABI-2 orchestration.

### Mathematical principle

Build the internal compact or wide bank of supported strict-linear preprocessing chains, including selected two-operator chains, score each chain/head/parameter configuration by fold-local CV, and refit the selected configuration on all training rows while returning its descriptor and diagnostics.

### Appropriate uses

A bounded, reproducible comparison of the built-in strict-linear AOM chain bank with Ridge and compatible PLS heads.

### Limits and validation

The bank is finite and profile-defined rather than an arbitrary chain search. Candidate selection requires an outer evaluation, and results are comparable only under the same profile, folds, head mask, and preprocessing scope.

### Implementation

`n4m.model_selection.aom_search.aom_sweep_run`; C ABI `n4m_model_selection_aom_sweep_run`.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/aom_sweep.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/model_selection/aom_search.py

## Catalog note

Native configurable AOM strict-linear preprocessing sweep. Compact profile has 12 chains; wide profile has 31 chains, including Gaussian, FCK and Whittaker variants. Ridge and compatible single-target NIPALS PLS1 candidates use exact operator-moment scoring when dense guards allow it; identity/SavGol/Norris/finite/FCK chains additionally use a banded moment route and detrend_poly chains use a structured low-rank moment route, with materialized fallbacks for unsupported chain/head regimes in auto mode. Repeated strict-linear prefixes are cached in bounded medium-width operator-moment grids and exposed via n_moment_prefix_cache_hits/misses. The moment_policy knob can force the legacy materialized-chain route for timing and production guarding, or force_moments/moments_only can fail fast if a candidate screen would leave operator moments. Route counters are split by head for Ridge/PLS operator-moment and materialized candidates; candidate_routes gives per-candidate route provenance without changing the stable candidate_scores shape, and Python decoded rows expose score_route_id/score_route. PLS fit-cost counters are split by route and phase via n_pls_moment_cv_fits, n_pls_materialized_cv_fits, n_pls_moment_final_fits and n_pls_materialized_final_fits. score_only=True returns ranking outputs without selected-model matrices for broad first-pass campaigns, and the result exports the flat chain descriptor (`chain_offsets`, `op_kinds`, `param_offsets`, `chain_params`) so each chain_id can be decoded through `n4m.decode_aom_chains` or `n4m.aom_candidate_table`. Python also exposes NativeAOMSweepRegressor, which uses input_coefficients folded into the original feature space for sklearn-style prediction on new X.

_Timing benchmark_: `benchmarks/cross_binding/bench_aom_sweep_timing.py`


_See also_: [methods index](index.md).