# `aom_chain_sweep` — n4m.model_selection.aom_search.aom_chain_sweep

_Namespace_: **`n4m.model_selection.aom_search`** · _Fully-qualified_: `n4m.model_selection.aom_search.aom_chain_sweep` · _Catalog id_: `aom_pop.aom_chain_sweep`

## API surface

**C ABI (ABI 2):** [`n4m_model_selection_aom_chain_sweep_run`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/model_selection.h#L356). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):** `from n4m.model_selection.aom_search import aom_chain_sweep_run`

**Signature:** [`aom_chain_sweep_run(X, y, chains, *, cv: int = 5, fold_ids = None, ridge_lambdas = (0.01, 0.1, 1.0, 10.0), pls_components = None, heads = ('ridge',), center_x: bool | None = None, scale_x: bool | None = None, center_y: bool | None = None, scale_y: bool | None = None, moment_policy: str | int = 'auto', pls_score_mode: str | int = 'cv', score_only: bool = False, cuda_pls_parallel_folds: bool | None = None, cuda_pls_min_device_features: int | None = None, cuda_pls_many_batched: bool | None = None)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py#L6484)

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `X` | `—` | `required` |
| `y` | `—` | `required` |
| `chains` | `—` | `required` |
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

No single canonical paper defines this strict-chain ABI sweep. Beurier, G. et al. (2026). *AOM-PLS / POP-PLS* paper companion, arXiv:2605.13587, https://arxiv.org/abs/2605.13587. The product variants below are implementation-specific extensions; the paper does not by itself specify their ABI-2 orchestration.

### Mathematical principle

Enumerate a declared bank of strict-linear preprocessing chains and head hyperparameters, score each by fold-local CV, and return a ranked result table with a refitted winner.

### Appropriate uses

Reproducible comparison of transparent, finite AOM candidate sets where ranking diagnostics are as important as the selected model.

### Limits and validation

The sweep is a model-selection procedure, not a test-set estimate. It excludes stateful and nonlinear transformations from the fold-back guarantee and can be expensive as chains grow.

### Implementation

`n4m.model_selection.aom_search.aom_chain_sweep_run`; C ABI `n4m_model_selection_aom_chain_sweep_run`.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/aom_chain_sweep.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/model_selection/aom_search.py

## Catalog note

Native caller-provided strict-linear AOM chain sweep. Accepts flat descriptors for identity, detrend polynomial, Savitzky-Golay, Norris-Williams, finite difference, Whittaker and FCK chains; Ridge and compatible single-target NIPALS PLS1 candidates use dense, banded or structured exact operator-moment scoring where guarded, including structured low-rank detrend and pentadiagonal Whittaker transforms, with materialized fallbacks otherwise in auto mode. Repeated strict-linear prefixes are cached in bounded medium-width operator-moment grids and exposed via n_moment_prefix_cache_hits/misses. CPU auto routes Ridge p>n_train and PLS min_train<4p rows through exact materialized scorers; CUDA auto keeps the moment route. The moment_policy knob can force the legacy materialized-chain route for timing and production guarding, or force_moments/moments_only can fail fast if a candidate screen would leave operator moments. Route counters are split by head for Ridge/PLS operator-moment and materialized candidates; candidate_routes gives per-candidate route provenance without changing the stable candidate_scores shape, and Python decoded rows expose score_route_id/score_route. PLS fit-cost counters are split by route and phase via n_pls_moment_cv_fits, n_pls_materialized_cv_fits, n_pls_moment_final_fits and n_pls_materialized_final_fits. score_only=True returns ranking outputs without selected-model matrices for broad first-pass campaigns, and the result echoes the validated flat chain descriptor (`chain_offsets`, `op_kinds`, `param_offsets`, `chain_params`) so each chain_id can be decoded through `n4m.decode_aom_chains` or `n4m.aom_candidate_table`. Python also exposes `build_aom_strict_chain_grid`, `aom_chain_score_campaign`, `aom_evaluate_candidates`, `aom_candidate_rank_diagnostics`, `aom_candidate_report_records`, `aom_save_candidate_report`, `aom_load_candidate_report` and `aom_candidate_operator_summary` for deterministic strict-linear chain grids, chunked score-only execution, checkpoint/resume guarded by data/config fingerprints, bounded incremental max_chunks_per_run execution, route-counter aggregation, normalized throughput/route metrics including moment_prefix_cache_hit_fraction, PLS fit-cost metrics including pls_cv_fits_per_chain and pls_cv_fits_per_candidate, global top-k inspection, score-route summaries, operator/head candidate summaries, explicit CV-vs-holdout candidate reports, screen-recall rank diagnostics, and JSON/JSONL/CSV candidate report export/reload. NativeAOMChainSweepRegressor and NativeAOMFixedCandidateRegressor both use input_coefficients folded into the original feature space for sklearn-style prediction on new X, with the fixed-candidate wrapper intended for refitting one decoded top-k campaign row, including winners reloaded from candidate report files.

_Timing benchmark_: `benchmarks/cross_binding/bench_aom_sweep_timing.py`


_See also_: [methods index](index.md).