# `aom_staged_chain_campaign` — n4m.model_selection.aom_campaign.aom_staged_chain_campaign

_Namespace_: **`n4m.model_selection.aom_campaign`** · _Fully-qualified_: `n4m.model_selection.aom_campaign.aom_staged_chain_campaign` · _Catalog id_: `aom_pop.aom_staged_chain_campaign`

## API surface

**C ABI:** no standalone exported symbol is declared for this method.

**Python (verified public re-export):** `from n4m.model_selection.aom_campaign import aom_staged_chain_campaign`

**Signature:** [`aom_staged_chain_campaign(X, y, stages = None, *, plan: str = 'compact_wide_lab', cv: int = 5, fold_ids = None, ridge_lambdas = (0.01, 0.1, 1.0, 10.0), pls_components = (1, 2, 4), heads = ('ridge', 'pls'), top_k: int = 50, refit_top_k: int | None = None, refit_per_head_top_k: int | None = 10, families: dict | None = None, templates: Sequence[Sequence[str]] | None = None, max_chains: int | None = None, chain_chunk_size: int = 4096, checkpoint_dir: str | Path | None = None, resume: bool = True, max_chunks_per_run: int | None = None, center_x: bool | None = None, scale_x: bool | None = None, scale_x_values: Sequence[bool | None] | None = None, center_y: bool | None = None, scale_y: bool | None = None, moment_policy: str | int = 'auto', refit_moment_policy: str | int | None = None, pls_score_mode: str | int = 'cv', chain_ordering: str = 'input', split_head_scoring: str = 'off', backend_cuda_available: bool | None = None, backend_min_cuda_product: int | None = None, cuda_pls_parallel_folds: bool | None = None, cuda_pls_min_device_features: int | None = None, cuda_pls_many_batched: bool | None = None, refit_sort_by: str | None = 'refit_cv_rmse', refit_execution: str = 'auto', refit_auto_max_extra_fraction: float = 1.0, return_predictions: bool = False, impact: bool = True, rank_diagnostics: bool = True, impact_top_k: int | None = None, X_audit = None, y_audit = None, audit_top_k: int | None = None, return_stage_screens: bool = False)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py#L4147)

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `X` | `—` | `required` |
| `y` | `—` | `required` |
| `stages` | `—` | `None` |
| `plan` | `str` | `'compact_wide_lab'` |
| `cv` | `int` | `5` |
| `fold_ids` | `—` | `None` |
| `ridge_lambdas` | `—` | `(0.01, 0.1, 1.0, 10.0)` |
| `pls_components` | `—` | `(1, 2, 4)` |
| `heads` | `—` | `('ridge', 'pls')` |
| `top_k` | `int` | `50` |
| `refit_top_k` | `int \| None` | `None` |
| `refit_per_head_top_k` | `int \| None` | `10` |
| `families` | `dict \| None` | `None` |
| `templates` | `Sequence[Sequence[str]] \| None` | `None` |
| `max_chains` | `int \| None` | `None` |
| `chain_chunk_size` | `int` | `4096` |
| `checkpoint_dir` | `str \| Path \| None` | `None` |
| `resume` | `bool` | `True` |
| `max_chunks_per_run` | `int \| None` | `None` |
| `center_x` | `bool \| None` | `None` |
| `scale_x` | `bool \| None` | `None` |
| `scale_x_values` | `Sequence[bool \| None] \| None` | `None` |
| `center_y` | `bool \| None` | `None` |
| `scale_y` | `bool \| None` | `None` |
| `moment_policy` | `str \| int` | `'auto'` |
| `refit_moment_policy` | `str \| int \| None` | `None` |
| `pls_score_mode` | `str \| int` | `'cv'` |
| `chain_ordering` | `str` | `'input'` |
| `split_head_scoring` | `str` | `'off'` |
| `backend_cuda_available` | `bool \| None` | `None` |
| `backend_min_cuda_product` | `int \| None` | `None` |
| `cuda_pls_parallel_folds` | `bool \| None` | `None` |
| `cuda_pls_min_device_features` | `int \| None` | `None` |
| `cuda_pls_many_batched` | `bool \| None` | `None` |
| `refit_sort_by` | `str \| None` | `'refit_cv_rmse'` |
| `refit_execution` | `str` | `'auto'` |
| `refit_auto_max_extra_fraction` | `float` | `1.0` |
| `return_predictions` | `bool` | `False` |
| `impact` | `bool` | `True` |
| `rank_diagnostics` | `bool` | `True` |
| `impact_top_k` | `int \| None` | `None` |
| `X_audit` | `—` | `None` |
| `y_audit` | `—` | `None` |
| `audit_top_k` | `int \| None` | `None` |
| `return_stage_screens` | `bool` | `False` |

## Explanations

### Bibliographic source

No single canonical paper defines this staged orchestration layer. Beurier, G. et al. (2026). *AOM-PLS / POP-PLS* paper companion, arXiv:2605.13587, https://arxiv.org/abs/2605.13587. The product variants below are implementation-specific extensions; the paper does not by itself specify their ABI-2 orchestration.

### Mathematical principle

Run several declared score-only strict-chain screens, merge and deduplicate their retained candidates, then exact-CV-refit the retained union. Optional impact and rank diagnostics compare screen and refit evidence.

### Appropriate uses

Campaigns that need resumable, auditable search across compact, wide, or focused strict-linear banks.

### Limits and validation

Stage labels and external audits must not choose the winner; only train-side exact-CV refit may do so. A partial screen is not evidence that unvisited candidates are inferior.

### Implementation

`n4m.model_selection.aom_campaign.aom_staged_chain_campaign`; Python orchestration over libn4m, no new C ABI.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/model_selection/aom_campaign.py

## Catalog note

Python-backed staged strict-chain AOM preprocessing campaign over the same native chain-screen/refit helpers as `aom_pop.aom_chain_screen_refit`. It runs compact, wide, lab, focused strict-family plans such as `savgol_focus` / `strict_family_focus`, or caller-provided strict-linear score-only stages over Ridge, PLS or mixed heads, merges cross-stage global and per-head retained candidates, exact-CV refits the retained union once, and attaches preprocessing-impact plus screen-vs-refit rank diagnostics. Optional `X_audit`/`y_audit` scoring is audit-only; production selection remains train exact-CV (`selection_uses_test_set=False`) and the sklearn wrapper omits audit inputs before fitting the selected row through final-only `aom_pop.aom_chain_fixed_fit`. `NativeAOMSavgolFocusRegressor` is the fast SavGol-focused preset wrapper over this method, using the locally validated `savgol_focus` / `scale_x_values=[False, True]` recipe while preserving selected-candidate reuse. `NativeAOMStrictFamilyLiteRegressor` is the cost-safe strict-family audit preset over the same method, sampling SavGol, Norris-Williams, finite-difference, Gaussian, FCK and Whittaker stages with small screen/refit defaults and no scale grid by default. The workflow supports checkpoint/resume and `max_chunks_per_run` for incremental campaigns, one-GPU CUDA PLS route knobs through the underlying screen/refit helpers, no dataset-name routing, and no hors-moment nonlinear lifts. This is the catalogued global staged preprocessing-selection method for benchmark campaigns, not the future fused CUDA/IKPLS grinder.

_Timing benchmark_: `benchmarks/cross_binding/bench_aom_staged_chain_campaign_timing.py`


_See also_: [methods index](index.md).