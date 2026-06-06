# `aom_chain_sweep_run` - user-defined native AOM chain sweep

_Group_: **Diagnostic / AOM** · _ABI_: `n4m_aom_chain_sweep_run`

## Description

`aom_chain_sweep_run` is the configurable native preprocessing-campaign
surface. Instead of selecting the built-in `compact` or `wide` AOM bank, the
caller supplies the chain list directly.

Current ABI v1 is intentionally restricted to strict-linear, shape-preserving
operators:

- `identity` / `raw`
- `detrend` / `detrend_poly`
- `savgol_smooth`
- `savgol_derivative`
- `norris_williams` / `nw`
- `finite_difference`
- `whittaker`
- `fck`

Stateful or train-fitted preprocessings such as SNV, MSC, EMSC, OSC/EPO and
baseline families are rejected in this path. They need fold-local fitting and
remain in the Python reference estimator layer.

## Python Usage

The dedicated AOM facade is available as `n4m.aom`; it aliases the same native
runtime as the top-level functions and `n4m.sklearn` classes:

```python
import n4m.aom as aom

res = aom.aom_chain_sweep_run(X, y, chains, heads=("ridge", "pls"))
inventory = aom.available_methods()
```

```python
import n4m

chains = [
    ["identity"],
    [("detrend", [1])],
    [("savgol_smooth", [5, 2])],
    [("detrend", [1]), ("savgol_derivative", [7, 2, 1])],
    [("savgol_smooth", [5, 2]), ("finite_difference", [1])],
]

res = n4m.aom_chain_sweep_run(
    X,
    y,
    chains,
    fold_ids=fold_ids,
    ridge_lambdas=[0.01, 0.1, 1.0],
    pls_components=[1, 2, 4],
    heads=("ridge", "pls"),
    scale_x=False,
    moment_policy="auto",
)
```

Sklearn-style native estimator over the same descriptor format:

```python
from n4m.sklearn import NativeAOMChainSweepRegressor

model = NativeAOMChainSweepRegressor(
    chains,
    fold_ids=fold_ids,
    ridge_lambdas=[0.01, 0.1, 1.0],
    pls_components=[1, 2, 4],
    heads=("ridge", "pls"),
    scale_x=False,
    moment_policy="auto",
).fit(X_train, y_train)

y_pred = model.predict(X_test)
```

Operator specs can be strings, tuples, or dictionaries:

```python
chains = [
    "identity",
    ("detrend", [2]),
    [{"kind": "savgol_derivative", "params": [11, 2, 1]}],
]
```

Use `"identity"` explicitly for a raw chain; empty chains are rejected.

`aom.available_methods()` returns JSON-safe metadata for the public AOM
surfaces, including global screen/refit presets, the ultra-configurable
campaign helpers, fixed-candidate winner reuse and linear AOM diversity heads.
It is an inventory for tooling and documentation, not a selector and not a
dataset-dependent router.

## C ABI Descriptor

```c
n4m_aom_chain_sweep_run(
    ctx, cfg, X, Y,
    cv, fold_ids, n_fold_ids,
    chain_offsets, n_chain_offsets,
    op_kinds, n_op_kinds,
    param_offsets, n_param_offsets,
    params, n_params,
    ridge_lambdas, n_ridge_lambdas,
    pls_components, n_pls_components,
    heads_mask,
    out_result)
```

Flat descriptor rules:

- `chain_offsets`: length `n_chains + 1`, monotonic, first `0`, last `n_ops`
- `op_kinds`: length `n_ops`, values from `n4m_operator_kind_t`
- `param_offsets`: length `n_ops + 1`, monotonic, first `0`, last `n_params`
- `params`: flat double parameter payload

Example for three chains:

- chain 0: `identity`
- chain 1: `detrend(1)`
- chain 2: `savgol_smooth(5,2) -> finite_difference(1)`

```c
int32_t chain_offsets[] = {0, 1, 2, 4};
int32_t op_kinds[] = {
    N4M_OP_IDENTITY,
    N4M_OP_DETREND_POLY,
    N4M_OP_SAVGOL_SMOOTH,
    N4M_OP_FINITE_DIFFERENCE,
};
int32_t param_offsets[] = {0, 0, 1, 3, 4};
double params[] = {1.0, 5.0, 2.0, 1.0};
```

## Outputs

Outputs match `aom_sweep_run`:

- `candidate_scores` `(n_candidates, 5)`: `candidate_id`, `chain_id`,
  `head_id`, `param`, `cv_rmse`
- `chain_offsets`, `op_kinds`, `param_offsets`, `chain_params`: flat
  descriptor of the validated strict-linear chain bank. For
  `aom_chain_sweep_run`, this echoes the caller-provided descriptor after
  native validation; for `aom_sweep_run`, it serializes the selected built-in
  profile.
- `candidate_routes` `(n_candidates)`: per-candidate scoring route code,
  `0=materialized`, `1=dense_operator_moment`, `2=banded_operator_moment`,
  `3=structured_operator_moment`.
- selected `oof_predictions`, final `predictions`, `coefficients`,
  `input_coefficients`, `intercept`, `x_mean`, `x_scale`, `y_mean`
- `fold_ids`
- scalars including `selected_chain_id`, `selected_head_id`,
  `selected_param`, `selected_cv_rmse`, `n_chains`, `n_candidates`,
  `n_operator_moment_candidates`,
  `n_ridge_operator_moment_candidates`,
  `n_pls_operator_moment_candidates`,
  `n_banded_operator_moment_candidates`,
  `n_structured_operator_moment_candidates`,
  `n_dense_operator_moment_candidates`, `n_materialized_candidates`,
  `n_ridge_materialized_candidates`, `n_pls_materialized_candidates`,
  `n_moment_prefix_cache_hits`, `n_moment_prefix_cache_misses`,
  `n_pls_moment_cv_fits`, `n_pls_materialized_cv_fits`,
  `n_pls_moment_score_batch_calls`,
  `n_pls_moment_score_batch_jobs`, `n_pls_gcv_proxy_candidates`,
  `n_pls_gcv_proxy_fits`, `n_pls_gcv_proxy_batch_calls`,
  `n_pls_gcv_proxy_batch_jobs`,
  `n_pls_moment_final_fits`, `n_pls_materialized_final_fits`,
  `aom_pls_score_mode`, and `score_only`

The scalar `profile` is `-1` for caller-provided chains.

`coefficients` are in the selected transformed-chain feature space.
`input_coefficients` are folded back into the original feature space, so
`X_new @ input_coefficients + intercept` reproduces the selected native model
without replaying the chain in Python.

`moment_policy="auto"` is the default and enables guarded exact
operator-moment scoring. Use `moment_policy="materialized"` or `"legacy"` to
force the legacy materialized-chain route for every chain/head. This is useful
when comparing route timings or when a small-cell workload is faster without
moment transforms.

Use `moment_policy="force_moments"` when the candidate screen must be
moment-only. Any chain/head/regime that would need a materialized fallback
returns `UNSUPPORTED` instead of being silently screened outside the moment
route. Python also accepts `"moments_only"`, `"operator_moments_only"`, and
`"strict_moments"`. The selected chain can still be materialized once after
ranking to expose OOF/final predictions and `input_coefficients`.

When the operator-moment route is used, repeated strict-linear chain prefixes
are cached for bounded medium-width grids. This is an exact reuse of
transformed all-sample and held-out moment sets; it does not affect ranking.
The cache is visible through `n_moment_prefix_cache_hits`,
`n_moment_prefix_cache_misses`, and, in `aom_chain_score_campaign`,
`moment_prefix_cache_hit_fraction`.

Use `score_only=True` for broad chain-ranking campaigns when no selected model
artifact is needed yet. The result keeps `candidate_scores`, selected ids,
route counters, `fold_ids` and chain descriptors; model-output matrices are
empty `0 x 0` matrices and scalar `score_only` is `1`. This avoids
selected-model refits and OOF/model output buffers in both operator-moment and
materialized candidate-screen routes. Materialized routes still pay fold-local
scoring fits, so this is not yet a replacement for batched IKPLS or a fully
fused CUDA grinder.
The PLS fit counters expose that residual cost: `n_pls_moment_cv_fits` and
`n_pls_materialized_cv_fits` count CV fits in the screen, and
`n_pls_moment_final_fits` / `n_pls_materialized_final_fits` count selected
final refits only when model outputs are requested. For PLS-only exact-CV
operator-moment screens, the native scorer batches eligible chains through one
internal score-only dispatch, preserving exact fold-CV scores while avoiding a
separate native PLS scoring call per chain. This is the exact screen path; it
is distinct from the cheaper `gcv_proxy` first pass below. The
`n_pls_moment_score_batch_calls` and `n_pls_moment_score_batch_jobs` counters
report how many native many-chain exact dispatches were used and how many
chain-fold jobs they contained.

Use `pls_score_mode="gcv_proxy"` only for explicit first-pass PLS screens. It
requires `score_only=True` and stays inside operator moments; if a requested
chain/head cannot be scored through moments, the call fails instead of falling
back to materialized scoring. PLS candidate scores then use a deterministic
PLS1 GCV RMSE proxy from all-sample transformed moments, so PLS rows expose
`score_metric="pls_gcv_proxy_rmse"` and `n_pls_gcv_proxy_*` counters. This is
not exact fold CV; use it to cheaply retain/rank many chains, then refit or
evaluate selected rows with the default `pls_score_mode="cv"` path. For
PLS-only operator-moment screens, the native proxy path also batches eligible
chains in one internal score-only dispatch and skips held-out moment
transforms, because the proxy only uses all-sample moments. The
`n_pls_gcv_proxy_fits` counter reports one proxy fit per chain, while
`n_pls_gcv_proxy_batch_calls` and `n_pls_gcv_proxy_batch_jobs` report the
many-chain dispatch shape.

Python helpers:

- `n4m.decode_aom_chains(res)` decodes the flat descriptor into operator
  chains.
- `n4m.aom_candidate_table(res, sort=True)` attaches the decoded chain to each
  candidate score row for top-k campaign reports, including `score_route_id`
  and the readable `score_route` label. PLS proxy rows also expose
  `score_metric="pls_gcv_proxy_rmse"`; exact-CV rows keep
  `score_metric="cv_rmse"`.

## Campaign Helpers

For larger strict-linear preprocessing screens, Python exposes two convenience
helpers over the same native ABI:

```python
chains = n4m.build_aom_strict_chain_grid(
    "lab",
    max_chains=5000,
)

campaign = n4m.aom_chain_score_campaign(
    X,
    y,
    chains=chains,
    fold_ids=fold_ids,
    ridge_lambdas=[0.01, 0.1, 1.0],
    pls_components=[1, 2, 4],
    heads=("ridge", "pls"),
    chain_chunk_size=1024,
    top_k=50,
    moment_policy="auto",
    backend_cuda_available=True,
    backend_min_cuda_product=512 * 512,
    checkpoint_path="reports/aom_lab_campaign_checkpoint.json",
    max_chunks_per_run=10,
)

best = campaign["best"]
print(best["chain"], best["head"], best["param"], best["cv_rmse"])

verified = n4m.aom_refit_candidates(
    X_train,
    y_train,
    campaign,
    top_k=20,
    fold_ids=fold_ids,
    scale_x=False,
)
print(verified["best_cv"]["chain"], verified["best_cv"]["refit_cv_rmse"])

screen_refit = n4m.aom_chain_screen_refit_campaign(
    X_train,
    y_train,
    chains=chains,
    fold_ids=fold_ids,
    ridge_lambdas=[0.01, 0.1, 1.0],
    pls_components=[1, 2, 4],
    heads=("ridge", "pls"),
    chain_chunk_size=1024,
    top_k=50,
    refit_top_k=20,
    moment_policy="force_moments",
    pls_score_mode="gcv_proxy",
    backend_cuda_available=True,
    backend_min_cuda_product=512 * 512,
    checkpoint_path="reports/aom_lab_campaign_checkpoint.json",
)
print(screen_refit["best_refit"]["chain"], screen_refit["best_refit"]["refit_cv_rmse"])

from n4m.sklearn import (
    NativeAOMFixedCandidateRegressor,
    NativeAOMScreenRefitRegressor,
)

screen_refit_model = NativeAOMScreenRefitRegressor(
    chains=chains,
    fold_ids=fold_ids,
    ridge_lambdas=[0.01, 0.1, 1.0],
    pls_components=[1, 2, 4],
    heads=("ridge", "pls"),
    chain_chunk_size=1024,
    top_k=50,
    refit_top_k=20,
    scale_x=False,
    moment_policy="force_moments",
    pls_score_mode="gcv_proxy",
).fit(X_train, y_train)

y_pred = screen_refit_model.predict(X_test)

model = NativeAOMFixedCandidateRegressor.from_candidate(
    best,
    fold_ids=fold_ids,
    scale_x=False,
).fit(X_train, y_train)

y_pred = model.predict(X_test)

holdout = n4m.aom_evaluate_candidates(
    X_train,
    y_train,
    X_test,
    y_test,
    campaign,
    top_k=20,
    fold_ids=fold_ids,
    scale_x=False,
)

print(holdout["best_eval"]["chain"], holdout["best_eval"]["eval_rmse"])
rank_diag = n4m.aom_candidate_rank_diagnostics(holdout, cutoffs=(1, 5, 10, 20))

n4m.aom_save_candidate_report("reports/aom_topk_eval.json", holdout)
n4m.aom_save_candidate_report("reports/aom_topk_eval.csv", holdout)

rows = n4m.aom_load_candidate_report("reports/aom_topk_eval.csv")
summary = n4m.aom_candidate_operator_summary(rows)

model = NativeAOMFixedCandidateRegressor.from_candidate(
    rows[0],
    fold_ids=fold_ids,
    scale_x=False,
).fit(X_train, y_train)

best_pls_model = NativeAOMFixedCandidateRegressor.from_campaign(
    campaign,
    head="pls",
    fold_ids=fold_ids,
    scale_x=False,
).fit(X_train, y_train)
```

`build_aom_strict_chain_grid("compact")` and `"wide"` reproduce the native
built-in chain banks. `"lab"` / `"cartesian"` builds a deterministic broader
strict-linear grid with multiple Savitzky-Golay smooth/derivative variants,
Norris-Williams, finite differences, Gaussian/FCK kernels and Whittaker chains.
Custom `families` and `templates` can define larger cartesian screens without
routing by dataset identity. The AOM `gaussian` family is the strict fixed
zero-padding banded variant used by the moment screen; the full
`n4m.sklearn.Gaussian` / `pp_gaussian` transformer remains the SciPy-compatible
preprocessing surface.
Use `iter_aom_strict_chain_grid(...)` when the same deterministic grid should
be consumed incrementally instead of materialized as one list. It accepts the
same grid arguments plus `start`, `stop`, `chunk_size` and `with_ids`; ids are
stable after de-duplication and `include_identity` filtering, so checkpointed
campaign launchers can resume by chain-id ranges without changing scores.

`aom_chain_score_campaign` always calls
`aom_chain_sweep_run(..., score_only=True)` and aggregates a global top-k over
chunks. It also keeps `top_candidates_by_head` and `best_by_head`, so a broad
mixed Ridge/PLS campaign can inspect the best preprocessing chains per model
head even when the global top-k is dominated by one head. It also keeps
`top_candidates_by_score_route` and `best_by_score_route`, so CPU/GPU audits
can inspect the best candidates scored through materialized, dense, banded or
structured moment routes. These per-head and per-route lists are audit outputs
only; they do not alter the global `top_candidates` order or the native scores.
Reports also expose `moment_backend_recommendations`, keyed by requested head,
using the same launch-planning policy as `moment_screen_backend_recommendation`.
That diagnostic uses only `n_samples`, `n_features`, `head`, `cuda_available`,
`backend_min_cuda_product`, plus the explicit PLS CUDA threshold and
many-batched flag; pass `backend_cuda_available=True` from an external launcher
when a CUDA build is available but the current process has not loaded it yet.
Use `backend_min_cuda_product` to reproduce or override the source-free launch
threshold in campaign reports without changing candidate scores.
The backend recommendation is not part of checkpoint fingerprints and does not
change candidate scoring or ranking.
The report also sums the route counters, so a campaign can state how many rows
used operator moments versus materialized fallback. Passing
`pls_score_mode="gcv_proxy"` to the campaign applies the explicit PLS proxy
screen described above and fingerprints checkpoints separately from exact-CV
campaigns. This helper is for reproducible ranking and inspection; it is not a
fused batched IKPLS or custom CUDA grinder.

For very large cartesian screens, pass `chain_ordering="prefix"` to
`aom_chain_score_campaign` or `aom_chain_screen_refit_campaign` to sort the
chain list by operator-prefix key before chunking. This does not change native
candidate scores: top rows keep their original `chain_id` and also expose
`ordered_chain_id` for audit. It only improves the chance that chains sharing a
strict-linear prefix land in the same native call and hit the per-call
moment-prefix cache. The default `chain_ordering="input"` preserves caller order.

For mixed Ridge/PLS campaigns, pass `split_head_scoring="auto"` to score each
chunk as two native score-only calls, Ridge-only then PLS-only, and merge the
candidate rows before top-k aggregation. This preserves the `(chain_id, head,
param)` scores and ranking semantics, but lets *both* halves use their native
head-homogeneous batch path: a single mixed call uses none of the batched fast
paths, so splitting turns on the Ridge moment score batch
(`n_ridge_moment_score_batch_calls`/`_jobs`) and the PLS exact or GCV-proxy
batch (`n_pls_moment_score_batch_calls`/`_jobs` for `pls_score_mode="cv"`,
`n_pls_gcv_proxy_batch_calls`/`_jobs` for `pls_score_mode="gcv_proxy"`).
Reports expose `n_split_head_chunks` and `n_chunk_score_calls`.

The lower-level campaign helpers (`aom_chain_score_campaign` /
`aom_chain_screen_refit_campaign`) default to `split_head_scoring="off"` for a
backwards-compatible launch shape. The sklearn screen/refit estimators default
to `"auto"`: `NativeAOMScreenRefitRegressor` (whose default heads are the mixed
`("ridge", "pls")` pair) and its `NativeAOMMomentScreenRefitRegressor` preset.
For single-head screens `"auto"` is inert and `n_split_head_chunks` stays `0`.

Use `n4m.aom_moment_screen_refit_campaign` when you want the same fast moment
profile as a function instead of an estimator. It wraps
`aom_chain_screen_refit_campaign` with `moment_policy="force_moments"`,
`chain_ordering="prefix"`, `split_head_scoring="auto"`,
`pls_score_mode="gcv_proxy"`, `refit_per_head_top_k=10`, and
`refit_execution="auto"`, while still accepting explicit chains, folds, grids,
CUDA flags, checkpoints and refit budgets. The combined report keeps the normal
`n4m.aom_chain_screen_refit_campaign.v1` schema and adds
`campaign_preset="moment_fast_screen_refit"`.

On CUDA builds, pass `cuda_pls_parallel_folds=True` to `aom_chain_sweep_run`,
`aom_chain_score_campaign`, `aom_refit_candidates`,
`aom_chain_screen_refit_campaign`, or the native sklearn screen/refit
wrappers to run eligible exact PLS1 moment jobs in bounded stream-parallel
batches on the selected single GPU. This preserves exact CV scores and reports
`n_pls_moment_cuda_parallel_fold_batches` plus
`n_pls_moment_cuda_parallel_fold_jobs`. It is a scheduling option over the
current exact moment jobs, not fused IKPLS.

An experimental cuBLAS-only many-job scheduler is also available for profiling
with `cuda_pls_many_batched=True` or the `N4M_CUDA_PLS_MANY_BATCHED=1`
environment fallback. It tiles independent exact PLS1 moment jobs on one GPU
and batches the dominant `p^2` operations with `cublasDgemmStridedBatched`,
while preserving the same scores. If both CUDA PLS schedulers are requested,
`cuda_pls_many_batched=True` is tried before `cuda_pls_parallel_folds=True`.
It is not the default because current smoke timings did not beat the legacy
sequential-many workspace path. Use `N4M_CUDA_PLS_MANY_LEGACY=1` to force the
legacy non-batched path even when an explicit flag or env opt-in is set, and
`N4M_CUDA_PLS_BATCH_MAX_BYTES=<bytes>` to cap the experimental tile memory.

Pass `cuda_pls_min_device_features=<positive int>` to the same calls to change
the CUDA PLS1 moment device-route threshold from the default 1024 features.
This is useful for controlled CPU/CUDA crossover sweeps on medium-width NIRS
datasets. The value is included in campaign fingerprints, reports and sklearn
diagnostics, so checkpoint resume and benchmark CSVs do not mix different
GPU-route configurations.

Campaign and per-chunk reports include normalized timing and route metrics:
`chains_per_second`, `candidates_per_second`, `ms_per_chain`,
`ms_per_candidate`, `operator_moment_candidate_fraction`,
`materialized_candidate_fraction`, and route-specific Ridge/PLS plus
dense/banded/structured fractions. They also include `pls_cv_fits_per_chain`
and `pls_cv_fits_per_candidate`, derived from exact-CV PLS fit counters, plus
`pls_gcv_proxy_fits_per_chain` and `pls_gcv_proxy_fits_per_candidate` when the
proxy screen is enabled. These fields are derived from elapsed chunk times and
native route counters, and are intended for CPU/GPU campaign comparison and
for spotting chunks that leave the operator-moment route or pay excess
fold-local PLS fitting.
`benchmarks/cross_binding/bench_aom_screen_refit_scaling.py` gives the focused
timing for proxy screen plus exact-CV refit as `refit_top_k` increases; use it
to size retained-candidate budgets and to compare future batched IKPLS/CUDA
work against the current exact refit path. Pass `--head ridge` to the same
benchmark to measure grouped and batched exact-CV refit over Ridge lambda
grids. Pass `--head mixed --refit-per-head-top-k K` to measure the mixed
Ridge/PLS workflow that exact-refits the union of global top rows and per-head
top rows. Pass `--chain-ordering prefix` to measure prefix-aware chunk packing
and compare the emitted screen prefix-cache hit counters. Pass
`--split-head-scoring auto` on mixed screens to measure the PLS-only batched
subcall path separately from the historical single mixed call. On CUDA builds,
pass `--cuda-pls-parallel-folds` to time the bounded stream-parallel exact
PLS1 moment scheduling path and inspect the emitted CUDA-parallel
batches/jobs counters. Pass `--cuda-pls-min-device-features 256` or another
positive threshold to test medium-width PLS device routing explicitly.

When `checkpoint_path` is provided, the campaign writes a JSON checkpoint after
each completed chunk and resumes it by default on the next call. The
checkpoint contains the current global, per-head and per-route top-k rows,
per-chunk route counters and a fingerprint of the chain grid, folds,
hyperparameters and `X/y` contents. A mismatched checkpoint raises instead of
mixing scores from different screens. When a partial checkpoint is resumed,
top-k rows are filtered to the chunks actually present in the checkpoint before
new chunks are appended. This is intended for long 50k/200k-chain ranking runs
where process or GPU interruptions should not force a full restart.

Use `max_chunks_per_run` to advance a long campaign incrementally. For
example, a scheduler can run ten chunks, persist the checkpoint, then relaunch
the same call later. The returned report includes `complete`,
`n_remaining_chunks` and `processed_chunks_this_run`. The chunk budget itself
is not part of the checkpoint fingerprint, so it can be changed between
relaunches without invalidating the campaign.

`NativeAOMFixedCandidateRegressor` is the reuse surface for a selected row. It
fits exactly one decoded chain/head/parameter candidate through the same native
ABI and stores folded `input_coefficients`, so `predict(X_new)` does not replay
Python preprocessing objects. Use `from_candidate(row)` for an explicit row,
or `from_campaign(report, head="ridge"|"pls", rank=0)` to reuse the global
winner or a per-head campaign winner directly. Use
`from_refit_report(verified, rank=0)` after `aom_refit_candidates`, or
directly after `aom_chain_screen_refit_campaign`, to reuse the best exact-CV
row from a second-pass report. `rank` is zero-based inside the chosen global,
per-head, or refit-CV ordering.
By default the fixed-candidate estimator uses `fit_mode="cv"` and recomputes
the one-candidate exact CV score. When the row already has a verified exact-CV
score, pass `fit_mode="final_only"` and `precomputed_cv_rmse=...` to fit the
selected chain/head/parameter on all rows without CV replay. The underlying
native endpoint is `n4m.aom_chain_fixed_fit_run`; it returns final predictions,
folded input-space coefficients and intercept, but no OOF predictions or fold
ids because it is not a ranking/CV endpoint.
This endpoint is catalogued as `aom_pop.aom_chain_fixed_fit`.
The cross-binding timing benchmark reports this individual-winner reuse cost
as `native_aom_chain_fixed_fit_pls` and
`native_aom_chain_fixed_fit_ridge` rows in
`benchmarks/cross_binding/aom_sweep_timing.csv` and the matching CUDA smoke
CSV.

`n4m.aom_refit_candidates` is the train-only verification helper for broad
score-only screens. It refits each decoded row as a single exact native
candidate with `pls_score_mode="cv"` and reports `refit_cv_rmse`, `oof_rmse`,
`train_rmse`, screen score metadata and exact refit route/fitting counters.
This is the intended second pass after `pls_score_mode="gcv_proxy"` screens:
the proxy can retain many candidates cheaply, then this helper re-ranks the
retained rows by exact CV without using a holdout/test set.
Use `n4m.aom_refit_execution_plan(candidates, top_k=...,
auto_max_extra_fraction=...)` before the refit to audit the execution cost of
each exact score mode without touching `X` or `y`. It reports
`n_refit_groups`, `n_refit_scored_candidates`, and
`n_refit_extra_scored_candidates` for `individual`, `grouped_score`,
`batched_score`, and `union_batched_score`, plus the `recommended_mode` used by
`execution_mode="auto"`.
Use `execution_mode="grouped_score"` when only exact CV scores are needed:
rows sharing the same decoded chain/head are scored together, so multiple PLS
components or Ridge lambdas avoid redundant fold-local fits. The ranking is
still exact CV; grouped rows do not include per-candidate prediction arrays.
Use `execution_mode="batched_score"` to keep the same exact-CV scores while
batching multiple retained chains that share the same head and retained
parameter set into one native `aom_chain_sweep_run` call. This can reduce
Python/native call overhead and lets native strict-linear prefix caches span
retained chains. It still reports scores only; use `individual` when
per-candidate train/OOF prediction arrays are required.
Use `execution_mode="union_batched_score"` to batch all retained chains for a
head with the union of retained parameters for that head. This may score extra
chain/parameter pairs that are not returned as refit rows; the report exposes
`n_refit_scored_candidates` and `n_refit_extra_scored_candidates` so that
surplus is explicit. It can help when the parameter grid is small relative to
Python/native call overhead.
Use `execution_mode="auto"` when no prediction arrays are needed. It uses the
same plan as `aom_refit_execution_plan`: it selects `union_batched_score` only
when that reduces native refit groups and the extra scored candidates are no
more than `auto_max_extra_fraction * n_retained_candidates`; otherwise it uses
`batched_score`, which never scores unretained parameters.

`n4m.aom_chain_screen_refit_campaign` is the one-call version of that workflow:
it runs the chunked score-only campaign, then exact-CV refits the retained
`refit_top_k` rows. The combined report exposes `screen`, `refit`,
`best_screen`, `best_refit`, `screen_complete`, top-level `rows` and
`best_cv`, so it can be passed directly to
`NativeAOMFixedCandidateRegressor.from_refit_report`. If `max_chunks_per_run`
or an incomplete checkpoint leaves the screen partial, the helper still refits
the current top rows and marks `screen_complete=False`.
Set `refit_per_head_top_k` to include each head's best screen rows in the
exact-CV refit pool in addition to the global `refit_top_k` rows. This is useful
for mixed Ridge/PLS campaigns where PLS may be screened by a GCV proxy while
Ridge rows use exact CV. The helper deduplicates candidates by decoded
chain/head/parameter and reports `n_refit_global_candidates`,
`n_refit_per_head_candidates`, `n_refit_per_head_extra_candidates` and
`n_refit_union_candidates`.
By default it uses `refit_execution="auto"` and
`refit_auto_max_extra_fraction=1.0`, so the second pass can choose
`union_batched_score` when the plan says the reduced native calls justify the
bounded extra exact scores. If `return_predictions=True`, auto mode falls back
to individual replay because score-only batched modes do not return per-row
prediction arrays.

`NativeAOMScreenRefitRegressor` is the sklearn-style estimator form of the
same workflow. Its `fit` runs the two-pass campaign, stores
`campaign_report_`, `screen_report_` and `refit_report_`, then fits the chosen
verified row as a reusable fixed candidate through final-only native fit.
`predict(X_new)` uses the final folded input-space coefficients and does not
replay Python preprocessing objects. `get_diagnostics()` separates
screen/refit/final counters; after exact-CV refit, the `final_*` fields should
show zero final CV fits and only the selected all-row fit needed to build the
reusable model.

Reusable sklearn presets wrap the same estimator for the common end-user
workflows:

```python
from n4m.sklearn import (
    NativeAOMMomentScreenRefitRegressor,
    NativeAOMMomentPLSScreenRefitRegressor,
    NativeAOMMomentPLSExactScreenRefitRegressor,
    NativeAOMMomentRidgeScreenRefitRegressor,
)

mixed_model = NativeAOMMomentScreenRefitRegressor(
    profile="lab",
    max_chains=5000,
    ridge_lambdas=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
    pls_components=(1, 2, 3, 4, 6, 8),
    top_k=100,
    refit_top_k=50,
    refit_per_head_top_k=25,
    fold_ids=fold_ids,
).fit(X_train, y_train)

pls_model = NativeAOMMomentPLSScreenRefitRegressor(
    profile="lab",
    max_chains=5000,
    pls_components=(1, 2, 3, 4, 6, 8),
    top_k=100,
    refit_top_k=25,
    fold_ids=fold_ids,
).fit(X_train, y_train)

pls_exact_model = NativeAOMMomentPLSExactScreenRefitRegressor(
    profile="lab",
    max_chains=5000,
    pls_components=(1, 2, 3, 4, 6, 8),
    top_k=100,
    refit_top_k=25,
    fold_ids=fold_ids,
).fit(X_train, y_train)

ridge_model = NativeAOMMomentRidgeScreenRefitRegressor(
    profile="lab",
    max_chains=5000,
    ridge_lambdas=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
    top_k=100,
    refit_top_k=25,
    fold_ids=fold_ids,
).fit(X_train, y_train)
```

`NativeAOMMomentScreenRefitRegressor` is the mixed global preset. It fixes
`heads=("ridge", "pls")`, uses exact Ridge CV and
`pls_score_mode="gcv_proxy"` for the first pass, then exact-CV refits the
retained union of the global screen top rows and the per-head screen top rows.
The per-head inclusion is controlled by `refit_per_head_top_k`; it is a
train-only retention budget for exact verification, not a new score.

`NativeAOMMomentPLSScreenRefitRegressor` fixes `heads=("pls",)`,
`ridge_lambdas=()`, `pls_score_mode="gcv_proxy"`,
`moment_policy="force_moments"` and `chain_ordering="prefix"`, then exact-CV
refits retained rows with `pls_score_mode="cv"`.
`NativeAOMMomentPLSExactScreenRefitRegressor` fixes the same PLS-only moment
surface but uses `pls_score_mode="cv"` for the first-pass screen too; it is the
auditable exact-screen preset when proxy recall is the question.
`NativeAOMMomentRidgeScreenRefitRegressor` fixes `heads=("ridge",)`,
`pls_components=()`, `moment_policy="force_moments"` and the same prefix-aware
chunk ordering. All presets keep `profile`, custom
`chains`/`families`/`templates`, checkpointing, incremental
`max_chunks_per_run`, top-k budgets and exact-refit execution parameters
configurable. Because these presets are strict moment presets, they raise
`UNSUPPORTED` when the current fold geometry or chain/head regime would leave
the operator-moment route; use the generic
`NativeAOMScreenRefitRegressor(moment_policy="auto", ...)` when a production
run should allow guarded materialized fallbacks.

`n4m.aom_evaluate_candidates` is an explicit analysis helper for comparing
screen or refit rank against a caller-provided holdout/test split. It refits
each decoded candidate on `X_train, y_train`, predicts `X_eval`, and reports
`screen_cv_rmse`, `refit_cv_rmse`, `eval_rmse`, `eval_r2`, `cv_rank`,
`eval_rank`, and `rank_delta`. The eval set is not used to alter the fit,
choose a route, or select by dataset identity.

`n4m.aom_candidate_rank_diagnostics(report_or_rows)` turns a holdout report
into screen-recall metrics. It compares the screen score, `screen_cv_rmse` by
default, against `eval_rmse`, and reports Spearman rank correlation,
mean/median/max absolute rank drift, the eval rank of the screen winner, the
screen rank of the eval winner, and top-k overlap/recall for caller-provided
cutoffs. It can also consume rows reloaded by
`n4m.aom_load_candidate_report`.

`n4m.aom_candidate_report_records(report)` flattens campaign or holdout
candidate rows into JSON-safe dictionaries. `n4m.aom_save_candidate_report`
writes those rows as `.json`, `.jsonl` / `.ndjson`, or `.csv` without requiring
pandas. Prediction arrays produced by `return_predictions=True` are omitted by
default; pass `include_predictions=True` only for small reports. CSV exports
include `chain_json`, a compact JSON encoding of the decoded strict-linear
preprocessing chain, so a saved top-k row can be refit later with
`NativeAOMFixedCandidateRegressor.from_candidate(row)`.

`n4m.aom_load_candidate_report(path)` reads `.json`, `.jsonl` / `.ndjson`, or
`.csv` candidate reports and restores rows as refittable dictionaries. In
particular, CSV rows recover `chain` from `chain_json` and convert the standard
rank/id/score fields back to numeric types.

`n4m.aom_candidate_operator_summary(report_or_rows)` groups already-scored
candidate rows by model head, preprocessing operator, operator/head pair,
chain length, and scoring route when route labels are present. It reports
count, best score, mean/median score and rank stats using `eval_rmse` when
present, otherwise `cv_rmse`, `refit_cv_rmse` or `screen_cv_rmse`. This is an
analysis surface for pruning or expanding future preprocessing grids; it does
not alter candidate scores or select by dataset identity.

`n4m.aom_candidate_preprocessing_impact(report_or_rows)` is the more detailed
post-hoc impact view. It groups scored rows by inferred preprocessing stage,
operator, concrete option such as `savgol_smooth(7,2)`, position in the chain
and head/stage combinations. When an identity-chain baseline is present, it
also reports best-score improvement versus identity. This is for understanding
which preprocessing options deserve more cartesian budget; it does not rerank
or select candidates.

`n4m.aom_candidate_route_summary(report_or_rows)` is the route-coverage audit.
It consumes campaign, refit, holdout or reloaded candidate rows and reports the
materialized vs dense/banded/structured operator-moment counts and fractions
for the rows it received, globally, by head and by chain. When the input is a
campaign/refit report with aggregate counters, it also adds `reported_total`
for the full scored/refit candidate set, so a `top_k` report can distinguish
retained-row coverage from full-screen coverage. Use `all_operator_moment`,
`reported_total["all_operator_moment"]` and `materialized_or_unknown_chains` to
verify whether a broad preprocessing screen actually stayed in the moment
routes before reusing or expanding that grid. It is an audit surface only; it
does not rerank candidates or change routing.

## CUDA Facade Smoke

The AOM and moment Python facades can be checked against the CUDA build with:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/cross_binding/aom_moment_cuda_facade_smoke.py
```

The smoke loads `build/cuda-on`, runs `n4m.moment.sweep_run` and
`n4m.aom.aom_chain_sweep_run` on a wide PLS1 moment case, and fails if the
reported PLS CV route is host or materialized instead of CUDA-device moments.

## Backend Status

The method builds and tests in CPU and CUDA-enabled libn4m configurations. It
uses exact operator-moment scoring when a chain can be represented cheaply in
moment space. Dense transforms represent a chain by its feature-space operator
matrix and apply `x_sum A`, `A' X'X A`, and `A' X'Y`; they are guarded by
`p <= n_train` or the medium dense cap `p <= 48` with strictly positive Ridge
lambdas. Local linear operators (`identity`, Savitzky-Golay
smooth/derivative, Norris-Williams, finite difference, Gaussian and FCK) also
use a banded descriptor, avoiding dense chain matrices. The banded route is
enabled up to `p <= 256` for Ridge scoring and `p <= 1024` for compatible
single-target NIPALS PLS1 scoring. Chains containing `detrend_poly` use an
exact structured low-rank projection transform in moment space and can compose
with those banded local operators under the same wide guards. Chains containing
`whittaker` use an exact structured pentadiagonal solve for
`(I + lambda D2'D2)^-1` and can also compose with the banded local operators.
On CPU builds, `auto` routes Ridge rows with `p > n_train` through the exact
materialized dual-Ridge scorer because that is cheaper than feature-space
moment Ridge in this geometry. CPU `auto` also routes compatible PLS1 rows
through the exact materialized prefix scorer when `min_train < 4p`. CUDA builds
keep the operator-moment route in those cells.

Unsupported moment routes fall back per chain/head to the materialized native
sweep in `auto`, or return `UNSUPPORTED` in `force_moments`. Selected chains
are always materialized once to populate public OOF/final predictions. Batched
IKPLS, fully fused operator-moment updates for all regimes and custom CUDA
kernels are future acceleration layers.
