# ABI — Changes Log

## 2026-07-11 — ABI 2.2.0: HPO terminal lifecycle and owning rich trace

Additive MINOR change: `n4m_trial_status_t` adds the public terminal value
`N4M_TRIAL_CANCELLED=4`. No function symbol or public struct layout changed.
The existing `n4m_optimizer_tell_result` now persists a structured
`{code, message, retryable}` error for `FAILED` and `CANCELLED`; explicit fields
use the fail-closed versioned text wire
`n4m.error.v1|CODE|0-or-1|message`.
Invalid UTF-8 is rejected before lifecycle mutation; search-space names and
string categorical labels are checked at their builder boundaries as well.

The existing `n4m_optimizer_get_trials` method-result payload retains its five
compatibility matrices and adds trace-v1 keys for exact int64 ids, ordered
parameters and activation/category metadata, optimizer-global event order,
intermediate values, and structured errors. Its result is an owning snapshot
whose buffers survive optimizer destruction. Python exposes the payload as
owning `TrialRecord`, `TrialParameter`, `IntermediateValue`, and `TrialError`
objects through `Optimizer.get_trials(since_id=...)`.

The already-exported `n4m_optimizer_save` and `n4m_optimizer_load` reservations
are now activated without a symbol, signature, enum, struct-layout, ABI-version,
or SONAME change. They read/write the portable little-endian N4MOPT v1 study
checkpoint. Load is transactional and fail-closed for corruption, future
versions, excessive sizes/counts, and inconsistent option/search-space/sampler
state. Exported-symbol snapshots remain 733/733, and SONAME remains
`libn4m.so.2`.

## 2026-07-10 — ABI 2.1.0: native HPO optimizer surface (`optimization` role)

Additive change (MINOR). New role header `cpp/include/n4m/optimization.h` — a
handle-based ask/tell hyperparameter optimizer + typed search space. All
sampler/pruner algorithms sit behind reserved enum values
(`n4m_sampler_kind_t`, `n4m_pruner_kind_t`); F0 implements only `random` +
`none`, and reserved values return `N4M_ERR_NOT_IMPLEMENTED`, so Phase-1
samplers/pruners (F1–F4) add no further public symbols. See
`docs/FINETUNING_F0_PR.md`.

31 new `N4M_API` symbols:

- search space (9): `n4m_search_space_create`, `_destroy`, `_add_int`,
  `_add_float`, `_add_categorical`, `_add_ordinal`, `_add_sorted_tuple`,
  `_add_constraint`, `_num_params`.
- optimizer (13): `n4m_optimizer_options_init`, `n4m_optimizer_create`,
  `_destroy`, `_enqueue`, `_ask`, `_ask_batch`, `_tell`, `_tell_result`,
  `_tell_intermediate`, `_best`, `_get_trials`, `_save`, `_load`
  (`_save`/`_load` were reserved in 2.1 and activated by N4MOPT v1 in 2.2 without
  changing their signatures).
- trial accessors (8): `n4m_trial_get_id`, `_get_int`, `_get_float`,
  `_get_category`, `_is_active`, `_get_rung`, `_get_status`, `_get_duration`.
- pure-native driver (1): `n4m_finetune_estimator`.

New 4-byte enums (guard-railed in `optimization.h`): `n4m_param_kind_t`,
`n4m_cat_type_t`, `n4m_constraint_kind_t`, `n4m_sampler_kind_t`,
`n4m_pruner_kind_t`, `n4m_opt_direction_t`, `n4m_eval_mode_t`, `n4m_metric_t`,
`n4m_liar_kind_t`, `n4m_trial_status_t`. New value struct
`n4m_optimizer_options_t` (forward-compatible via a `struct_size` prologue).

**Later within 2.1.0 (still additive, no symbol/size change):** the reserved tail
of `n4m_optimizer_options_t` was narrowed from `reserved[64]` to two named
`int32_t` fields — `max_resource`, `reduction_factor` (for `hyperband`/`asha`) —
plus `reserved[56]`. `sizeof(n4m_optimizer_options_t)` is unchanged (verified
C == Python == 120 bytes) and no exported symbol changed, so this is
ABI-compatible: callers that zero-initialise via `n4m_optimizer_options_init`
get `0` (= auto/default) for both. All nine `n4m_sampler_kind_t` values and all
five `n4m_pruner_kind_t` values are now implemented (reserved-value dispatch now
only fires for out-of-range enums).

## 2026-06-14 — ABI 2.0.0: namespace clean break (all method symbols renamed)

Breaking change. Every public **method** symbol was renamed to the canonical
role-based convention `n4m_<role>_<leaf><tail>`, where `role` is the top-level
ML namespace (`augmentation`, `compose`, `decomposition`, `domain_adaptation`,
`ensemble`, `estimators`, `feature_selection`, `lowlevel`, `metrics`,
`model_selection`, `outlier_detection`, `transform`), `leaf` is the catalog
leaf, and `<tail>` is the existing operation suffix preserved byte-for-byte
(`_fit`, `_create`, `_destroy`, `_transform`, `_select`, `_split`,
`_result_get_*`, …). No runtime aliases were kept — the terse legacy exports
(`n4m_pp_*`, `n4m_aug_*`, `n4m_split_*`, `n4m_filter_*`, `n4m_aom_*`,
`n4m_metric_*`, `n4m_util_*`, `n4m_ridge_fit`, `n4m_pls_fit_simple`, …) are gone.

- **566 method symbols renamed** (deterministically, from `catalog/methods.yaml`
  via `proposals/namespace/_build_rename_map.py` → `_rename_map.tsv`; 0
  collisions). Examples: `n4m_ridge_fit → n4m_estimators_ridge_fit`,
  `n4m_pp_snv_create → n4m_transform_snv_create`,
  `n4m_pls_fit_simple → n4m_estimators_pls_fit` (drop "simple"),
  `n4m_aom_global_result_get_best_score → n4m_model_selection_aom_pls_result_get_best_score`,
  `n4m_util_hotelling_t2 → n4m_outlier_detection_hotelling_t2`,
  `n4m_metric_rmse → n4m_metrics_regression_metrics_rmse`.
- **136 infra symbols unchanged** (context, config, matrix view, RNG, model,
  pipeline, validation plan, method-result, serialization, array, backends).
- The 10 Python-only methods (`c_surface: "none"`) export no C symbol.
- Exported surface total: 702 (566 method + 136 infra). Linux version node
  bumped `N4M_1 → N4M_2`; SONAME is now `libn4m.so.2`.
- Public headers were split into role headers + 2nd-level subheaders
  (`transform/*.h`, `estimators/*.h`, `augmentation/*.h`, `lowlevel.h`, …);
  the umbrella `n4m.h` keeps the shared infra and includes the role headers.
  The flat `pls.h` and the empty category stubs (`aom_pop.h`, `preprocessing.h`,
  `models.h`, `selection.h`, `splitters.h`, `filters.h`, `diagnostics.h`,
  `transfer.h`, `utilities.h`, `context.h`) were deleted (no compat include).
- ABI snapshots regenerated for all three platforms; `N4M_ABI_VERSION_*` set to
  `2.0.0`. Numerics are unchanged — this is a surface rename only.

## 2026-06-06 — ABI 1.22.0: PLS CV reference surface

One additive public symbol:

- `n4m_pls_cross_validate`

This is a C/Python ABI entry point for exact PLS-only cross-validation over one
input matrix. The current implementation delegates to the PLS branch of
`n4m_sweep_run`, so candidate scores and CPU/CUDA route counters match the
existing sweep path. It is intentionally catalogued as ABI infrastructure, not
as a production method. The future fused/batched IKPLS-style multi-chain
executor can replace the internals without changing this signature.

## 2026-06-05 — ABI 1.21.0: CUDA PLS many-design batching config

Two additive public config helpers:

- `n4m_config_set_cuda_pls_many_batched`
- `n4m_config_get_cuda_pls_many_batched`

The default remains off. When enabled on a CUDA build, eligible PLS1 moment
many-design jobs may use the experimental tiled/strided-batched route that
also remains reachable through the `N4M_CUDA_PLS_MANY_BATCHED` environment
fallback. This changes only GPU scheduling and timings; candidate scores remain
fold-level exact for the selected scoring path.

## 2026-06-05 — ABI 1.20.0: CUDA PLS device threshold config

Two additive public config helpers:

- `n4m_config_set_cuda_pls_min_device_features`
- `n4m_config_get_cuda_pls_min_device_features`

The default threshold remains 1024 features, matching the conservative
historical CUDA PLS1 moment guard. Lower positive values let CPU/CUDA
crossover campaigns explicitly test medium-width PLS moment screens on the
selected single GPU without recompiling. This changes only route eligibility
and timing; candidate scores are unchanged for a given exact scoring path.

## 2026-06-05 — ABI 1.19.0: CUDA PLS fold scheduling config

Two additive public config helpers:

- `n4m_config_set_cuda_pls_parallel_folds`
- `n4m_config_get_cuda_pls_parallel_folds`

When enabled on a CUDA build, eligible exact PLS1 moment CV jobs may run in
bounded stream-parallel batches on the single selected GPU. This changes only
scheduling and timings; candidate scores are unchanged. Sweep and AOM
MethodResults also expose additive scalar counters
`n_pls_moment_cuda_parallel_fold_batches` and
`n_pls_moment_cuda_parallel_fold_jobs` for fit-cost auditing.

## 2026-06-05 — ABI 1.18.x: strict AOM Gaussian operator kind

No public symbol or result layout change. The public operator enum gains one
additive value:

- `N4M_OP_GAUSSIAN = 18`

This value is accepted by the strict AOM chain sweep and represents a fixed,
shape-preserving zero-padding Gaussian convolution with a banded
operator-moment descriptor. It is distinct from the full `pp_gaussian`
preprocessing transformer surface.

## 2026-06-05 — ABI 1.18.x: AOM chain fixed final fit

One additive public symbol:

- `n4m_aom_chain_fixed_fit_run(n4m_context_t*, const n4m_config_t*,
  const n4m_matrix_view_t* X, const n4m_matrix_view_t* Y,
  const int32_t* chain_offsets, int64_t n_chain_offsets,
  const int32_t* op_kinds, int64_t n_op_kinds,
  const int32_t* param_offsets, int64_t n_param_offsets,
  const double* params, int64_t n_params, int32_t head_id, double param,
  n4m_method_result_t** out_result)`

This fits one already-selected caller-provided strict-linear AOM
chain/head/parameter on all rows without running CV. It is a model-building
endpoint, not a ranking endpoint: CV score fields are NaN unless a higher-level
wrapper injects an externally verified exact-CV score. Python uses this in
`NativeAOMScreenRefitRegressor` after exact-CV refit so reusable model
construction no longer repays one-candidate CV.

## 2026-06-05 — ABI 1.18.x: AOM score-only screen output mode

Two additive public config helpers:

- `n4m_config_set_aom_score_only`
- `n4m_config_get_aom_score_only`

When enabled for `n4m_aom_sweep_run` or `n4m_aom_chain_sweep_run`, the result
keeps the candidate-score table, selected identifiers, route counters and fold
ids, but omits selected-model matrices by returning them as `0 x 0`. This is
an additive output/cost-control knob for large preprocessing ranking passes.

## 2026-06-04 — ABI 1.18.0: native AOM operator PLS score stack

One additive public symbol (ABI MINOR bump 1.17.0 -> 1.18.0),
backward-compatible (no signature/layout change, nothing removed):

- `n4m_aom_operator_pls_stack_fit(n4m_context_t*, const n4m_config_t*,
  const n4m_matrix_view_t* X, const n4m_matrix_view_t* Y, int32_t profile,
  int32_t cv, const int32_t* fold_ids, int64_t n_fold_ids,
  const int32_t* components, int64_t n_components, const double* alphas,
  int64_t n_alphas, double std_penalty, double gap_penalty,
  n4m_method_result_t** out_result)`

This exposes a native strict-linear AOM operator PLS1 score stack. The method
builds compact or wide AOM operator banks, fits fold-local PLS1 score
projectors per operator, concatenates the scores, selects `(n_components,
alpha)` by train-only CV criterion, and refits the selected stack on all rows
with a Ridge head.

The returned `n4m_method_result_t` carries `candidate_scores`, `fold_scores`,
`oof_predictions`, `predictions`, `stack_features`, `coefficients`,
`intercept`, `fold_ids` and `operator_feature_offsets`. `candidate_scores`
columns are `spec_id`, `n_components`, `alpha`, `mean_oof_rmse`,
`std_oof_rmse`, `mean_train_rmse`, `criterion`.

Native v1 is single-target (`Y.cols == 1`) and not yet a fused batched GPU
stack. Custom Python operator matrices, shuffled/both CV and baseline admission
gating remain in the Python `AOMOperatorPLSStack` estimator.

The implementation lives in `cpp/src/core/aom_operator_pls_stack.cpp` and is
dispatched from `cpp/src/c_api/c_api_method_result.cpp`. The symbol is declared
in `cpp/include/n4m/pls.h`, exported in all ABI snapshots, catalogued as
`aom_pop.operator_pls_stack`, wrapped in Python as
`n4m.aom_operator_pls_stack`, and documented in
`docs/methods/aom_operator_pls_stack.md`.

## 2026-06-04 — ABI 1.17.0: native AOM Ridge OOF simplex blender

One additive public symbol (ABI MINOR bump 1.16.0 -> 1.17.0),
backward-compatible (no signature/layout change, nothing removed):

- `n4m_aom_ridge_blender_fit(n4m_context_t*, const n4m_config_t*,
  const n4m_matrix_view_t* X, const n4m_matrix_view_t* Y, int32_t profile,
  int32_t cv, const int32_t* fold_ids, int64_t n_fold_ids,
  const double* ridge_lambdas, int64_t n_ridge_lambdas, double regularizer,
  n4m_method_result_t** out_result)`

This exposes a native strict-linear AOM Ridge candidate blender. The method
builds compact or wide AOM chain banks, scores each chain/lambda candidate by
fold-local OOF Ridge predictions, solves a regularized non-negative simplex
blend, and refits all candidates on the full training data for final blended
predictions.

The returned `n4m_method_result_t` carries `candidate_scores`, `weights`,
`oof_predictions`, `predictions`, `oof_candidate_predictions`,
`candidate_predictions` and `fold_ids`. `candidate_scores` columns are
`candidate_id`, `chain_id`, `lambda`, `cv_rmse`, `weight`.

Native v1 requires strictly positive Ridge lambdas and is not yet a fused
batched GPU blender. It builds in CUDA-enabled configurations, but the
candidate loop still uses the existing native Ridge path per fold/candidate.

The implementation lives in `cpp/src/core/aom_ridge_blender.cpp` and is
dispatched from `cpp/src/c_api/c_api_method_result.cpp`. The symbol is declared
in `cpp/include/n4m/pls.h`, exported in all ABI snapshots, catalogued as
`aom_pop.ridge_blender`, wrapped in Python as `n4m.aom_ridge_blender`, and
documented in `docs/methods/aom_ridge_blender.md`.

## 2026-06-04 — ABI 1.16.0: user-defined AOM chain sweep

One additive public symbol (ABI MINOR bump 1.15.0 -> 1.16.0),
backward-compatible (no signature/layout change, nothing removed):

- `n4m_aom_chain_sweep_run(n4m_context_t*, const n4m_config_t*,
  const n4m_matrix_view_t* X, const n4m_matrix_view_t* Y, int32_t cv,
  const int32_t* fold_ids, int64_t n_fold_ids,
  const int32_t* chain_offsets, int64_t n_chain_offsets,
  const int32_t* op_kinds, int64_t n_op_kinds,
  const int32_t* param_offsets, int64_t n_param_offsets,
  const double* params, int64_t n_params,
  const double* ridge_lambdas, int64_t n_ridge_lambdas,
  const int32_t* pls_components, int64_t n_pls_components,
  int32_t heads_mask, n4m_method_result_t** out_result)`

This exposes a flat descriptor for caller-provided strict-linear preprocessing
chains. `chain_offsets` partitions `op_kinds`; `param_offsets` partitions the
flat `params` payload. Empty chains are rejected; callers use an explicit
identity operator for raw spectra. Supported operators are identity, polynomial
detrend, Savitzky-Golay smooth/derivative, Norris-Williams, finite difference,
Whittaker, FCK and Gaussian.

The result shape matches `n4m_aom_sweep_run`; `candidate_scores` columns are
`candidate_id`, `chain_id`, `head_id`, `param`, `cv_rmse`, and scalar
`profile` is `-1` for caller-provided chains.

This is the first ABI-stable arbitrary strict-linear preprocessing-chain
surface. It still materializes transformed matrices per chain and uses
materialized PLS CV; fused operator-moment updates, batched IKPLS and CUDA
kernels remain later acceleration work.

The implementation lives in `cpp/src/core/aom_sweep.cpp` and is dispatched from
`cpp/src/c_api/c_api_method_result.cpp`. The symbol is declared in
`cpp/include/n4m/pls.h`, catalogued as `aom_pop.aom_chain_sweep`, wrapped in
Python as `n4m.aom_chain_sweep_run`, and documented in
`docs/methods/aom_chain_sweep_run.md`.

## 2026-06-04 — ABI 1.15.0: configurable native AOM preprocessing sweep

One additive public symbol (ABI MINOR bump 1.14.0 -> 1.15.0),
backward-compatible (no signature/layout change, nothing removed):

- `n4m_aom_sweep_run(n4m_context_t*, const n4m_config_t*,
  const n4m_matrix_view_t* X, const n4m_matrix_view_t* Y, int32_t profile,
  int32_t cv, const int32_t* fold_ids, int64_t n_fold_ids,
  const double* ridge_lambdas, int64_t n_ridge_lambdas,
  const int32_t* pls_components, int64_t n_pls_components,
  int32_t heads_mask, n4m_method_result_t** out_result)`

The symbol applies the native strict-linear AOM compact/wide preprocessing
chain bank, then delegates candidate scoring to `n4m_sweep_run` over Ridge
lambdas and/or PLS component counts. It returns `candidate_scores`,
`oof_predictions`, final `predictions`, coefficients/intercept and fold ids.
`candidate_scores` has columns `candidate_id`, `chain_id`, `head_id`, `param`,
`cv_rmse`; `head_id` is `0` for Ridge and `1` for PLS.

This is a configurable product sweep over the fixed AOM strict-linear banks. It
is not yet the arbitrary operator-descriptor layer or fused batched IKPLS/CUDA
grinder.

The implementation lives in `cpp/src/core/aom_sweep.cpp` and is dispatched from
`cpp/src/c_api/c_api_method_result.cpp`. The symbol is declared in
`cpp/include/n4m/pls.h`, exported in all ABI snapshots, catalogued as
`aom_pop.aom_sweep`, wrapped in Python as `n4m.aom_sweep_run`, and documented
in `docs/methods/aom_sweep_run.md`.

## 2026-06-04 — ABI 1.14.0: native Ridge/PLS sweep

One additive public symbol (ABI MINOR bump 1.13.0 -> 1.14.0),
backward-compatible (no signature/layout change, nothing removed):

- `n4m_sweep_run(n4m_context_t*, const n4m_config_t*,
  const n4m_matrix_view_t* X, const n4m_matrix_view_t* Y, int32_t cv,
  const int32_t* fold_ids, int64_t n_fold_ids, const double* ridge_lambdas,
  int64_t n_ridge_lambdas, const int32_t* pls_components,
  int64_t n_pls_components, int32_t heads_mask,
  n4m_method_result_t** out_result)`

ABI v1 supports exact Ridge CV over row-additive moments where efficient, with
a precomputed dual Ridge path when `p > n_train`. It also supports fold-local
materialized PLS component screening through the existing native PLS model path.
The returned `n4m_method_result_t` carries `candidate_scores`,
`oof_predictions`, final `predictions`, coefficients/intercept and fold ids.
`candidate_scores[:,1]` is `0` for Ridge and `1` for PLS; `param` is lambda for
Ridge and `n_components` for PLS.

The fused batched IKPLS/operator-descriptor grinder is not part of ABI v1.

The implementation lives in `cpp/src/core/sweep.cpp` and is dispatched from
`cpp/src/c_api/c_api_method_result.cpp`. The symbol is declared in
`cpp/include/n4m/pls.h`, exported in all ABI snapshots, catalogued as
`utilities.sweep`, wrapped in Python as `n4m.sweep_run`, and documented in
`docs/methods/sweep_run.md`.

## 2026-06-04 — ABI 1.13.0: native row-additive moment substrate

Three additive public symbols (ABI MINOR bump 1.12.0 -> 1.13.0),
backward-compatible (no signature/layout change, nothing removed):

- `n4m_moments_compute(n4m_context_t*, const n4m_matrix_view_t* X,
  const n4m_matrix_view_t* Y, n4m_method_result_t** out_result)`
- `n4m_moments_subset_compute(n4m_context_t*, const n4m_matrix_view_t* X,
  const n4m_matrix_view_t* Y, const int64_t* row_indices, int64_t n_indices,
  n4m_method_result_t** out_result)`
- `n4m_moments_subtract(n4m_context_t*, const n4m_method_result_t* lhs,
  const n4m_method_result_t* rhs, n4m_method_result_t** out_result)`

The result is a `n4m_method_result_t` carrying raw additive moments
(`x_sum`, `y_sum`, `xtx`, `xty`, `yty`) and centered moments recomputed from
the raw sums (`x_mean`, `y_mean`, `cxx`, `cxy`, `cyy`). This gives an exact
fold-subtraction primitive for PLS/Ridge screens: compute all rows, compute the
held-out rows, subtract raw moments, then recenter on the remaining train rows.

The implementation lives in `cpp/src/core/moments.cpp` and is dispatched from
`cpp/src/c_api/c_api_method_result.cpp`. The symbols are declared in
`cpp/include/n4m/pls.h`, exported in all ABI snapshots, catalogued as
`utilities.moments`, wrapped in Python as `n4m.moments` /
`n4m.moments_train_from_heldout`, and documented in `docs/methods/moments.md`.

## 2026-06-04 — ABI 1.12.0: native AOM robust-HPO screen

One additive public symbol (ABI MINOR bump 1.11.0 -> 1.12.0),
backward-compatible (no signature/layout change, nothing removed):

- `n4m_aom_robust_hpo_fit(n4m_context_t*, const n4m_config_t*,
  const n4m_matrix_view_t* X, const n4m_matrix_view_t* Y, int32_t profile,
  int32_t cv, int32_t heads_mask, n4m_method_result_t** out_result)`

This exposes the product AOM robust-HPO preprocessing screen through the public
C ABI. Native v1 screens compact/wide banks of strict-linear, shape-preserving
AOM preprocessing chains and Ridge/PLS heads by contiguous K-fold CV RMSE. It
returns a `n4m_method_result_t` carrying in-sample predictions after refitting
the selected candidate, transformed-space coefficients, intercept, scalar
selection diagnostics and the full `candidate_scores` matrix
(`chain_id`, `head_id`, `param`, `mean_cv_rmse`).

The implementation lives in `cpp/src/core/aom_robust_hpo.cpp` and is dispatched
from `cpp/src/c_api/c_api_method_result.cpp`. The symbol is declared in
`cpp/include/n4m/pls.h`, exported in all ABI snapshots, catalogued as
`aom_pop.robust_hpo`, wrapped in Python as `n4m.aom_robust_hpo`, and documented
in `docs/methods/aom_robust_hpo.md`.

## 2026-06-03 — ABI 1.11.0: direct (closed-form) Ridge regression

One additive public symbol (ABI MINOR bump 1.10.0 → 1.11.0), backward-compatible
(no signature/layout change, nothing removed):

- `n4m_ridge_fit(n4m_context_t*, const n4m_config_t*, const n4m_matrix_view_t* X,
  const n4m_matrix_view_t* Y, const double* lambdas, int64_t n_lambdas,
  n4m_method_result_t** out_result)`

This is a **genuine closed-form** multi-output Ridge — `beta = (Xc'Xc + lambda I)^-1
Xc'Yc` on column-centered X/Y with `intercept = y_mean - x_mean.beta` (the penalty is
not applied to the intercept, for `sklearn.linear_model.Ridge` parity). It is distinct
from the pre-existing `n4m_ridge_pls_fit` (ridge-augmented SIMPLS, rank-truncated by
`n_components`). The solver is chosen automatically by shape (PRIMAL augmented-QR for
p ≤ n, DUAL Gram-on-samples for p > n; identical coefficients up to round-off).

Declared with `N4M_API` in `cpp/include/n4m/pls.h` (after `n4m_continuum_regression_fit`),
implemented in `cpp/src/c_api/c_api_method_result.cpp` over the new core kernel
`cpp/src/core/ridge.cpp`. Result keys: `coefficients` (p×q), `intercept` (1×q),
`x_mean`, `x_scale` (1×p), `y_mean` (1×q), `predictions` (n×q), scalar `rmse`,
scalar `lambda`.

Snapshots regenerated for all three platforms via
`scripts/regen_abi_snapshots.sh --derive` (linux from the lib; macos/windows derived
= linux minus the `N4M_1` version node). `n4m_ridge_fit` is present in
`cpp/abi/expected_symbols_{linux,macos,windows}.txt`. Header
`N4M_ABI_VERSION_MINOR` and `bindings/python/src/n4m/_ffi.py:ABI_VERSION_MINOR`
both bumped 10 → 11; `bump_version.sh --check` is green (project version unchanged
at 0.98.0).

## 2026-06-03 — macOS/Windows snapshot correction + cross-platform gate enforced

No ABI surface change (still ABI 1.10.0). This is an **audit-trail and CI
correction**: the 2026-05-30 entry below claimed "Snapshots regenerated for all
three platforms", but `expected_symbols_{macos,windows}.txt` were in fact a stale,
truncated copy of an old Linux `nm -D` dump — 500 lines, carrying the Linux-only
`@@N4M_1` version tag (which macOS `nm -gU` / Windows `dumpbin` never emit), and
missing ~171 symbols (the whole selection / method-result / aom / config family).
They were also **not diffed by CI** on macOS/Windows (only Linux was fail-closed).

Corrected here:

- `expected_symbols_{macos,windows}.txt` regenerated to the real 671-symbol set
  — identical to the Linux `n4m_*` names minus the Linux-only `N4M_1` version
  node (the only legitimate cross-platform difference).
- `.github/workflows/abi-check.yml` now diffs the committed snapshot **fail-closed
  on all three platforms** (macOS `diff`, Windows `Compare-Object` set comparison),
  with `LC_ALL=C`-pinned sorts so ordering is reproducible.
- Added a SONAME / RPATH-RUNPATH linkage gate to the Linux job (asserts
  `SONAME == libn4m.so.1` and no baked-in absolute search path).
- Added `scripts/regen_abi_snapshots.sh` — the single canonical regenerator
  (`--check` for CI/pre-commit, `--derive` to produce the macOS/Windows files
  from the Linux snapshot when only a Linux box is available).

## 2026-05-18 — Linux export baseline for ABI 1.16.0

`build/dev-release/cpp/src/libn4m.so.1.16.0` exports 27 additional
`n4m_*` symbols compared with the previous Linux baseline. Each added symbol is
declared with `N4M_API` in the public header `cpp/include/pls4all/p4a.h`, so the
Linux ABI gate now treats them as intentional public additions:

- `n4m_method_result_get_int64_vector`
- `n4m_mb_pls_fit`, `n4m_lw_pls_fit`, `n4m_pls_lda_fit`,
  `n4m_pls_logistic_fit`, `n4m_aom_preprocess_fit`
- `n4m_variable_select_rank`, `n4m_interval_select`,
  `n4m_stability_select`, `n4m_uve_select`, `n4m_spa_select`,
  `n4m_cars_select`, `n4m_random_frog_select`, `n4m_scars_select`,
  `n4m_ga_select`, `n4m_shaving_select`, `n4m_bve_select`,
  `n4m_t2_select`, `n4m_wvc_select`, `n4m_wvc_threshold_select`,
  `n4m_emcuve_select`, `n4m_randomization_select`, `n4m_bipls_select`,
  `n4m_sipls_select`, `n4m_rep_select`, `n4m_ipw_select`, `n4m_st_select`

## 2026-05-30 — ABI 1.10.0: additive RNG-kind config selector

Two additive public symbols (ABI MINOR bump 1.9.0 → 1.10.0), backward-compatible
(no signature/layout change, nothing removed):

- `n4m_config_set_rng_kind(n4m_config_t*, n4m_rng_kind_t)`
- `n4m_config_get_rng_kind(const n4m_config_t*, n4m_rng_kind_t*)`

New enum `n4m_rng_kind_t` { `N4M_RNG_SPLITMIX64`=0 (default), `N4M_RNG_PCG64`=1,
`N4M_RNG_MT_R`=2, `N4M_RNG_NUMPY_MT`=3 } selects the RNG engine a stochastic
method draws from, so its output can match an external reference library's exact
RNG (numpy default_rng / base R / numpy RandomState) for parity. Default
SPLITMIX64 reproduces n4m's historical streams bit-for-bit — leaving it unset
changes nothing. Snapshots regenerated for all three platforms
(expected_symbols_{linux,macos,windows}.txt). Engines verified bit-exact:
docs/dev/RNG_TIER0_INVENTORY.md, cpp/tests/test_rng_engine.cpp.
