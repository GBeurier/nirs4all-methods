# `aom_chain_screen_refit` — n4m.model_selection.aom_campaign.aom_chain_screen_refit

_Namespace_: **`n4m.model_selection.aom_campaign`** · _Fully-qualified_: `n4m.model_selection.aom_campaign.aom_chain_screen_refit` · _Catalog id_: `aom_pop.aom_chain_screen_refit`

## API surface

**C ABI:** no standalone exported symbol is declared for this method.

**Python (verified public re-export):** `from n4m.model_selection.aom_campaign import aom_chain_screen_refit_campaign`

**Signature:** [`aom_chain_screen_refit_campaign(X, y, chains = None, *, profile: str = 'lab', families: dict | None = None, templates: Sequence[Sequence[str]] | None = None, max_chains: int | None = None, chain_chunk_size: int = 4096, top_k: int = 50, refit_top_k: int | None = None, refit_per_head_top_k: int | None = None, cv: int = 5, fold_ids = None, ridge_lambdas = (0.01, 0.1, 1.0, 10.0), pls_components = (1, 2, 4), heads = ('ridge', 'pls'), center_x: bool | None = None, scale_x: bool | None = None, center_y: bool | None = None, scale_y: bool | None = None, moment_policy: str | int = 'auto', refit_moment_policy: str | int | None = None, pls_score_mode: str | int = 'cv', chain_ordering: str = 'input', split_head_scoring: str = 'off', cuda_pls_parallel_folds: bool | None = None, cuda_pls_min_device_features: int | None = None, cuda_pls_many_batched: bool | None = None, backend_cuda_available: bool | None = None, backend_min_cuda_product: int | None = None, checkpoint_path: str | Path | None = None, resume: bool = True, max_chunks_per_run: int | None = None, refit_sort_by: str | None = 'refit_cv_rmse', refit_execution: str = 'auto', refit_auto_max_extra_fraction: float = 1.0, return_predictions: bool = False)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py#L3572)

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `X` | `—` | `required` |
| `y` | `—` | `required` |
| `chains` | `—` | `None` |
| `profile` | `str` | `'lab'` |
| `families` | `dict \| None` | `None` |
| `templates` | `Sequence[Sequence[str]] \| None` | `None` |
| `max_chains` | `int \| None` | `None` |
| `chain_chunk_size` | `int` | `4096` |
| `top_k` | `int` | `50` |
| `refit_top_k` | `int \| None` | `None` |
| `refit_per_head_top_k` | `int \| None` | `None` |
| `cv` | `int` | `5` |
| `fold_ids` | `—` | `None` |
| `ridge_lambdas` | `—` | `(0.01, 0.1, 1.0, 10.0)` |
| `pls_components` | `—` | `(1, 2, 4)` |
| `heads` | `—` | `('ridge', 'pls')` |
| `center_x` | `bool \| None` | `None` |
| `scale_x` | `bool \| None` | `None` |
| `center_y` | `bool \| None` | `None` |
| `scale_y` | `bool \| None` | `None` |
| `moment_policy` | `str \| int` | `'auto'` |
| `refit_moment_policy` | `str \| int \| None` | `None` |
| `pls_score_mode` | `str \| int` | `'cv'` |
| `chain_ordering` | `str` | `'input'` |
| `split_head_scoring` | `str` | `'off'` |
| `cuda_pls_parallel_folds` | `bool \| None` | `None` |
| `cuda_pls_min_device_features` | `int \| None` | `None` |
| `cuda_pls_many_batched` | `bool \| None` | `None` |
| `backend_cuda_available` | `bool \| None` | `None` |
| `backend_min_cuda_product` | `int \| None` | `None` |
| `checkpoint_path` | `str \| Path \| None` | `None` |
| `resume` | `bool` | `True` |
| `max_chunks_per_run` | `int \| None` | `None` |
| `refit_sort_by` | `str \| None` | `'refit_cv_rmse'` |
| `refit_execution` | `str` | `'auto'` |
| `refit_auto_max_extra_fraction` | `float` | `1.0` |
| `return_predictions` | `bool` | `False` |

## Explanations

### Bibliographic source

No single canonical paper defines this screen/refit orchestration surface. Beurier, G. et al. (2026). *AOM-PLS / POP-PLS* paper companion, arXiv:2605.13587, https://arxiv.org/abs/2605.13587. The product variants below are implementation-specific extensions; the paper does not by itself specify their ABI-2 orchestration.

### Mathematical principle

Screen a declared strict-linear candidate pool using the configured inexpensive criterion, retain the candidates prescribed by the policy, then refit those candidates with exact CV before choosing the final model.

### Appropriate uses

Large but finite strict-chain searches where an auditable two-stage reduction is needed before costly exact validation.

### Limits and validation

A screening score is not the final selection metric. The retained-pool policy changes the chance of losing the optimum and must be reported; outer validation remains necessary.

### Implementation

`n4m.model_selection.aom_campaign.aom_chain_screen_refit_campaign`; no standalone C ABI symbol.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/model_selection/aom_campaign.py

## Catalog note

Python-backed two-pass AOM preprocessing campaign over the native n4m_model_selection_aom_chain_sweep_run ABI, with final model reuse delegated to the catalogued final-only `aom_pop.aom_chain_fixed_fit` surface. The first pass runs a chunked score-only strict-linear chain screen, including optional pls_score_mode=gcv_proxy for explicit moment-only PLS proxy ranking, checkpoint/resume and global/per-head/per-route top-k aggregation from full chunk score tables. The second pass refits retained candidates by exact native CV with pls_score_mode=cv, preserving screen and refit reports for audit; `refit_per_head_top_k` can add each head's best rows to the exact-refit pool before deduplication, while score-only refit can group by chain/head, batch retained chains sharing a head/parameter signature, or explicitly union-batch retained parameters per head while keeping the same returned exact-CV scores. Union batching exposes scored and extra-scored candidate counters so its surplus cost is auditable, and aom_refit_execution_plan previews those counters for each mode without touching X/y. NativeAOMScreenRefitRegressor exposes the same workflow as a sklearn-style reusable regressor: fit runs the two-pass campaign, then fits the selected verified row through NativeAOMFixedCandidateRegressor in final_only mode so predict uses folded input-space coefficients on new X without replaying CV after exact refit. NativeAOMMomentScreenRefitRegressor, NativeAOMMomentPLSScreenRefitRegressor, NativeAOMMomentPLSExactScreenRefitRegressor and NativeAOMMomentRidgeScreenRefitRegressor are end-user presets over the same workflow for mixed Ridge/PLS GCV-proxy->exact-refit, PLS GCV-proxy->exact-refit, PLS exact-screen->exact-refit and Ridge exact-screen->exact-refit paths. This is intended for ultra-configurable preprocessing selection experiments and reusable selected models without dataset-name routing; it is not the fused CUDA/IKPLS grinder.

_Timing benchmark_: `benchmarks/cross_binding/bench_aom_screen_refit_scaling.py`


_See also_: [methods index](index.md).