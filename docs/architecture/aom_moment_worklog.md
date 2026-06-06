# AOM / Moment Integration Worklog

## 2026-06-06 - Force-moments bypasses CPU wide PLS materialization heuristic

Purpose:

- Make `moment_policy="force_moments"` mean "try the admissible
  operator-moment route" even when the CPU wide-PLS heuristic would normally
  materialize for speed.

Changes:

- In `run_aom_chain_sweep`, CPU wide Ridge/PLS materialization heuristics are
  disabled under the explicit `force_moments` policy before route selection.
- This unblocks exact-CV PLS AOM moment screens for wider strict-moment chains
  on CPU. A previously rejected identity PLS screen at `n=120, p=32`,
  `cv=4`, `pls_components=[1,2,3]` now stays on the PLS moment score-batch
  route with zero materialized candidates.
- Added
  `test_aom_pls_force_moments_bypasses_cpu_wide_materialization_heuristic`.

Validation:

- Rebuilt `build/dev-release`, `build/cuda-on`, `build/omp-on` and
  `build/blas-on` `n4m_c`.
- Targeted pytest:
  `3 passed, 78 deselected` for the new force-moments guard plus existing
  BLAS/OMP score-parity guards.
- Manual dev-release width sweep over `p=16,32,48,64,80,96` now returns
  exact PLS moment score-batch rows with `n_materialized_candidates=0`.
- Manual one-GPU CUDA smoke at `p=96` with
  `CUDA_VISIBLE_DEVICES=0`, `cuda_pls_min_device_features=1` and
  `cuda_pls_parallel_folds=True` reports four CUDA PLS fold jobs and zero
  materialized candidates.
- Manual dev/OMP/BLAS smoke at `p=96` reports matching selected scores and
  zero materialized candidates.

Follow-up:

- This is a semantics/coverage fix for forced moment screens, not the full
  fused many-chain IKPLS executor.

## 2026-06-06 - PLS cross-validate reference ABI artifacts

Purpose:

- Pin the reserved `n4m.pls_cross_validate` /
  `n4m_pls_cross_validate` reference surface with small CPU/CUDA smoke timing
  artifacts before the future grouped/fused PLS grinder exists.

Changes:

- Added `bench_pls_cross_validate_timing.py`, which times full and
  score-only `n4m.pls_cross_validate` calls across three synthetic shapes.
- Each row compares the public PLS CV hook against
  `n4m.sweep_run(heads=("pls",))` on the same folds/component grid and records
  max absolute candidate-score, OOF-prediction and prediction deltas.
- Regenerated CPU and one-GPU CUDA smoke CSVs:
  `pls_cross_validate_timing.csv` and
  `pls_cross_validate_timing_cuda_smoke.csv`.
- Added artifact guards proving CPU rows use host exact-CV routing while CUDA
  rows use the device PLS CV/final-fit route with parallel folds enabled,
  many-batched off and numerical-zero equivalence deltas.
- Documented the benchmark command in `benchmarks/cross_binding/README.md`.

Validation:

- `test_aom_moment_cuda_smoke_artifacts.py`: `25 passed`.
- Python `py_compile` on the new benchmark and artifact test passed.
- `git diff --check` passed.

Follow-up:

- This is a reference ABI/timing hook only. The real open item remains the
  fused/batched host/device IKPLS-style executor for many preprocessing chains,
  folds and candidates.

## 2026-06-06 - Staged benchmark PLS score-mode switch

Purpose:

- Make staged benchmark campaigns directly compare exact-CV PLS screens against
  explicit GCV-proxy first-pass screens without code edits, so screen-recall
  studies can be run incrementally from the standard runners.

Changes:

- Added `--pls-score-mode {cv,gcv_proxy}` to
  `run_aom_staged_real_cohort.py` and
  `bench_aom_staged_chain_campaign_timing.py`.
- Forwarded the mode to `n4m.aom_staged_chain_campaign` and persisted it in
  real-cohort CSV rows, diagnostics JSON runner metadata and staged timing CSV
  rows.
- Updated benchmark README guidance: `cv` remains default exact-CV screen;
  `gcv_proxy` is explicit for proxy-vs-exact recall/timing campaigns, while
  retained-candidate refit remains exact-CV.

Validation:

- `test_aom_benchmark_tools.py`: `21 passed`.
- Synthetic staged timing smoke with `--pls-score-mode gcv_proxy` wrote a CSV
  row containing `pls_score_mode=gcv_proxy`.
- py_compile and `git diff --check` passed.

## 2026-06-06 - CUDA facade smoke covers PLS exact preset

Purpose:

- Make the new public PLS exact-CV screen/refit preset prove the same
  CPU/GPU facade and CUDA-route readiness as the existing AOM/moment reusable
  presets.

Changes:

- Extended `aom_moment_cuda_facade_smoke.py` to assert
  `NativeAOMMomentPLSExactScreenRefitRegressor` aliases through both
  `n4m.moment` and `n4m.aom`.
- Added a tiny exact-CV PLS screen/refit fit on one visible GPU, checking
  `pls_score_mode="cv"`, `refit_pls_score_mode="cv"`, zero GCV-proxy fits,
  CUDA device CV counters for both screen and refit, and zero host PLS CV fits.
- Regenerated `aom_moment_cuda_facade_smoke.json` with ABI `1.22.0` and added
  artifact assertions for the new `pls_exact_screen_refit_estimator` section.

Validation:

- Regenerated the CUDA facade smoke artifact through `build/cuda-on` with
  `CUDA_VISIBLE_DEVICES=0`.
- `test_aom_moment_cuda_smoke_artifacts.py`: `22 passed`.
- py_compile and `git diff --check` passed.

## 2026-06-06 - PLS exact-CV screen/refit reusable preset

Purpose:

- Expose a simple PLS-only end-user preset for exact-CV preprocessing screens,
  separate from the existing PLS GCV-proxy screen preset, so screen-recall
  audits and reuse experiments can opt into exact train-CV ranking directly.

Changes:

- Added `NativeAOMMomentPLSExactScreenRefitRegressor`, a PLS-only
  `NativeAOMScreenRefitRegressor` preset with `pls_score_mode="cv"`,
  `ridge_lambdas=()`, `heads=("pls",)`, `moment_policy="force_moments"` and
  prefix-aware chain ordering.
- Exported the preset through `n4m`, `n4m.sklearn`, `n4m.aom` and
  `n4m.moment`; added `moment_pls_exact_screen_refit` inventory entries beside
  the existing PLS GCV and Ridge exact presets.
- Updated method docs, coverage matrix and catalog notes to distinguish PLS
  GCV-proxy -> exact-refit from PLS exact-screen -> exact-refit.

Validation:

- Targeted wrapper/facade pytest, py_compile and diff checks run in the
  corresponding patch validation.

## 2026-06-06 - PLS exact batch fallback prefix reuse

Purpose:

- Reduce the cost of the robust PLS exact score-only fallback after a global
  batched prefix fit fails on a rank-deficient late component.

Changes:

- `score_pls1_moment_sweeps_score_only` now builds a unique descending list of
  requested component prefixes and, in the fallback path, tries only the largest
  still-needed prefix for each chain/fold job before descending.
- When a lower prefix is recovered, all requested components at or below that
  prefix reuse the same `RidgeMomentFit` prefix vector. Components above the
  recovered prefix are marked failed/`inf`, matching the previous exact-CV
  fallback semantics.
- Already-failed component candidates are not retried on later folds.
- Counters now report actual fallback prefix-fit attempts. In mixed batches, a
  healthy job whose max prefix succeeds after the global batch failed pays one
  fit per fold rather than one fit per component per fold.
- No ABI or public Python surface change.

Validation:

- Rebuilt dev-release `n4m_c` and `n4m_internal_tests`.
- Rebuilt CUDA `n4m_c` and `n4m_internal_tests` with `CUDA_VISIBLE_DEVICES=0`.
- Dev and CUDA `n4m_internal_tests` passed.
- Targeted dev pytest over PLS fallback/reference degeneracy tests: `3 passed`.
- Targeted one-GPU CUDA pytest over the same plus many-batched precedence:
  `4 passed`.
- Full dev-release `test_moment_model_wrappers.py`: `80 passed`.

## 2026-06-06 - Fixed-candidate CUDA option surface alignment

Purpose:

- Keep winner-reuse APIs aligned with the AOM/moment facade inventories so a
  selected PLS candidate can be reused with the same public CPU/CUDA option
  names exposed by screen/refit methods.

Changes:

- Added `cuda_pls_parallel_folds` and `cuda_pls_many_batched` to
  `n4m.aom_chain_fixed_fit_run`.
- Added `cuda_pls_parallel_folds` and `cuda_pls_many_batched` to
  `NativeAOMFixedCandidateRegressor`, storing them and forwarding them both to
  final-only fixed fits and to `fit_mode="cv"` replay.
- The already-present `cuda_pls_min_device_features` remains forwarded.
- No C ABI change. Final-only fits currently consume the threshold knob on the
  native PLS component path; fold/many-batch knobs are API-symmetric and matter
  when the wrapper replays the exact-CV path.

Validation:

- Targeted pytest:
  `test_native_aom_chain_fixed_fit_run_matches_single_candidate_final_fit` and
  the AOM/moment facade inventory tests.
- Full dev-release `test_moment_model_wrappers.py`: `80 passed`.
- Python `py_compile` on touched modules/tests and `git diff --check` passed.

## 2026-06-06 - AOM exact PLS batch partial-failure guard

Purpose:

- Keep broad strict-moment AOM PLS screens usable when a degenerate late
  component fails but smaller component prefixes are still scoreable.

Changes:

- `score_pls1_moment_sweeps_score_only` still uses the fast batched prefix path
  when it succeeds.
- If that global batched prefix fit fails, it now clears the transient error and
  retries via per-chain/fold/component moment fits. Scoreable component
  candidates keep finite exact-CV scores; failed component candidates are marked
  with `inf`.
- A chain whose all component candidates fail now returns all-`inf` candidates
  instead of aborting the whole batch. The higher-level AOM screen still returns
  `N4M_ERR_NUMERICAL_FAILURE` if no finite candidate exists anywhere.
- The fallback remains moment-only and does not materialize transformed `X`;
  successful fast-path runs keep the existing `n_pls_moment_score_batch_*`
  counters.
- Added
  `test_aom_pls_moment_batch_degenerate_components_do_not_abort_screen`.

Validation:

- Rebuilt `build/dev-release` and `build/cuda-on` `n4m_c`.
- Manual CPU/CUDA reproduction on a rank-deficient identity-chain AOM PLS screen
  now returns finite component-1 scores, `inf` component-2 scores,
  `n_materialized_candidates=0`, `n_pls_materialized_cv_fits=0` and moment CV
  fit counters.
- Targeted dev pytest:
  `test_aom_pls_moment_batch_degenerate_components_do_not_abort_screen`,
  `test_pls_moment_fallback_builds_fold_designs_on_demand` and
  `test_pls_cross_validate_reference_matches_pls_sweep`.
- Targeted CUDA pytest on one GPU: same tests plus `many_batched_precedes`.
- Full dev-release `test_moment_model_wrappers.py`: `80 passed`.
- Python `py_compile` on touched tests/modules and `git diff --check` passed.

## 2026-06-06 - Rank-deficient PLS moment fallback guard

Purpose:

- Make the PLS moment sweep robust when a numerically degenerate fold cannot
  use the exact moment-prefix route and must fall back to materialized PLS.

Changes:

- Added a lazy `ensure_fold_designs()` helper in `run_moment_sweep`.
- The PLS fallback path now builds fold-local materialized designs on demand
  before reading `fold_designs[fold]`. Previously, compatible PLS1 moment
  screens could skip early fold-design construction, then fail the moment route
  on a rank-deficient fold and dereference an empty vector during fallback.
- Added `test_pls_moment_fallback_builds_fold_designs_on_demand`, covering the
  previously crashing tiny fixture with `score_only=True`, `score_only=False`
  and the public `n4m.pls_cross_validate(..., score_only=True)` wrapper.

Validation:

- Rebuilt `build/dev-release` and `build/cuda-on` `n4m_c`.
- Manual reproduction on the previous segfault fixture now returns candidate
  scores, with component 1 finite, component 2 `inf`, `n_pls_moment_cv_fits=0`
  and materialized fallback counters.
- Targeted dev pytest:
  `test_pls_moment_fallback_builds_fold_designs_on_demand` and
  `test_pls_cross_validate_reference_matches_pls_sweep`.
- Targeted CUDA pytest on one GPU:
  `test_pls_moment_fallback_builds_fold_designs_on_demand`,
  `test_pls_cross_validate_reference_matches_pls_sweep` and
  `many_batched_precedes`.
- Full dev-release `test_moment_model_wrappers.py`: `79 passed`.
- Python `py_compile` on touched tests/modules and `git diff --check` passed.

## 2026-06-06 - PLS CV ABI reference surface

Purpose:

- Provide the public C/Python entry point needed by the future fused/batched
  IKPLS-style PLS grinder with an exact single-matrix reference implementation.

Changes:

- Added ABI 1.22.0 symbol `n4m_pls_cross_validate(ctx, cfg, X, Y, fold_ids,
  n_fold_ids, n_folds, component_grid, n_component_grid, out_result)`.
- Implemented it by validating obvious pointer/length errors, then delegating to
  `n4m_sweep_run` with `heads_mask=PLS`; candidate scores and CPU/CUDA route
  counters therefore match the existing exact PLS sweep path.
- Exposed the symbol through Python ctypes as `n4m.pls_cross_validate` and
  `n4m.moment.pls_cross_validate`, added it to the moment facade inventory,
  classified it as catalog ABI infra rather than a production method, updated
  ABI snapshots and documented that the grouped/fused executor remains open.

Validation:

- Rebuilt `build/dev-release` and `build/cuda-on` `n4m_c`; both produced
  `libn4m.so.1.22.0`.
- Targeted dev/CUDA Python equivalence tests:
  `test_pls_cross_validate_reference_matches_pls_sweep` and the moment facade
  alias/inventory guard.
- Full dev-release `test_moment_model_wrappers.py`: `78 passed`.
- Python `py_compile` on touched modules/tests passed.
- Catalog checks: `validate.py --strict-abi`, `validate.py --check-references`,
  `split_legacy_methods.py --check`, `reconcile_abi.py --check`,
  `git diff --check` and `scripts/bump_version.sh --check`.

Follow-up:

- The tiny rank-deficient PLS fallback crash found during this validation pass
  is fixed by the later "Rank-deficient PLS moment fallback guard" entry above.

## 2026-06-06 - Compact-wide audit10 benchmark follow-up

Purpose:

- Use the new audit-rank diagnostics on a controlled 10-row real-cohort
  benchmark instead of adding more features.

Run:

- Ran `run_aom_staged_real_cohort.py` on one GPU with
  `plan=compact_wide`, `heads=ridge,pls`, `max_chains=12`,
  `chain_chunk_size=6`, `top_k=12`, `refit_top_k=6`,
  `refit_per_head_top_k=2`, `scale_x_grid=false,true`,
  `split_head_scoring=auto`, `cuda_pls_min_device_features=1`,
  `cuda_pls_parallel_folds`, `backend_min_cuda_product=1` and
  `max_features=1200`.
- The first 10 cohort rows were run in two chunks and merged into
  `benchmarks/cross_binding/aom_staged_real_cohort_compact_wide_audit10_20260606.csv`.

Artifacts:

- `benchmarks/cross_binding/aom_staged_real_cohort_compact_wide_audit10_20260606.csv`
- `benchmarks/cross_binding/aom_staged_compact_wide_audit10_20260606_rank_audit.csv`
- `benchmarks/cross_binding/aom_staged_compact_wide_audit10_20260606_rank_audit.md`
- `benchmarks/cross_binding/aom_staged_real_cohort_compact_wide_audit10_20260606_oracle_compare.csv`
- `benchmarks/cross_binding/aom_staged_compact_wide_audit10_20260606_oracle_summary.md`
- `benchmarks/cross_binding/aom_staged_compact_wide_audit10_vs_compact_20260606.csv`
- `benchmarks/cross_binding/aom_staged_compact_wide_audit10_vs_compact_20260606.md`
- `benchmarks/cross_binding/aom_staged_compact_wide_audit10_20260606_impact_summary.csv`
- `benchmarks/cross_binding/aom_staged_compact_wide_audit10_20260606_impact_summary.md`

Results:

- 10 rows total: 8 OK, 2 property-skipped (`BERRY` and `FUSARIUM`,
  both `n_features>1200`), all with `selection_uses_test_set=False`.
- All 8 OK rows selected `ridge` and `selected_campaign_stage=compact`; the
  extra wide stage did not win under this retained/refit budget.
- PLS routing stayed on CUDA: screen PLS moment CV fits `1920/0/1920`
  total/host/device; refit PLS moment CV fits `220/0/220`. Split-head counters
  were `64` split chunks and `128` score calls; Ridge screen counters were
  `2160/24/2160`.
- Timing: total OK fit time `1562.03s`, median OK fit time `155.04s`.
- Versus the compact split-head-auto baseline:
  8 paired, 0 wins, 0 losses, 8 ties, median ratio `1`.
- Oracle comparison:
  AOM-PLS oracle median ratio `1.03079` (1/7 target wins), AOM-Ridge oracle
  median ratio `1.08068` (0/8 target wins), TabPFN oracle median ratio
  `0.978527` (4/8 target wins).
- Rank audit:
  median test-rank delta `2.5`, max `5`; median oracle-gap ratio `0.02354`,
  max `0.57732` on ECOSIS; median CV/test Spearman `0.64286`, min `-0.73810`
  on ECOSIS.
- `summarize_aom_rank_audit.py` now emits selected/oracle head, parameter,
  preprocessing-chain labels and audit top-1/top-3/top-5 recall columns. The
  ECOSIS failure is visible without opening raw JSON: selected
  `ridge:0.1 savgol_smooth(7,2)` versus offline oracle
  `pls:1 detrend_poly(2)`, oracle CV-rank `7`, selected test-rank `5`, and
  top-1/top-3 recall `0/0`.
- Impact summary:
  `detrend_poly` is the top operator by dataset wins (`5/7`), identity follows
  (`2/5`), SavGol smooth has one win (`1/8`), and derivatives did not win under
  this profile.

Interpretation:

- This does not improve over compact; `compact_wide` currently adds cost
  without selected-score gain on the 8 OK datasets.
- The useful new signal is the rank-audit failure mode: ECOSIS has poor
  CV/test rank alignment and a large offline oracle gap even though the
  selected production model remains train-CV-only. That points to improving
  screen/refit recall or validation stability rather than expanding wide-stage
  chains blindly.

ECOSIS retention stress test:

- Reran only ECOSIS with the same `compact_wide` screen but wider retained
  refit budget: `top_k=60`, `refit_top_k=40`, `refit_per_head_top_k=10`.
- Artifact:
  `benchmarks/cross_binding/aom_staged_real_cohort_ecosis_compact_wide_refit40_audit_20260606.csv`
  plus diagnostics dir
  `benchmarks/cross_binding/aom_staged_ecosis_compact_wide_refit40_audit_20260606/`.
- It retained/refit `44` candidates. Production selection stayed the same as
  audit10: `ridge:0.1 savgol_smooth(7,2)`, `RMSEP=41.3984`.
- The offline audit oracle changed to an even better retained candidate:
  `ridge:10 finite_difference(1)`, `eval_rmse=10.5817`, `cv_rank=11`;
  the selected production candidate was test-rank `41/44`, with
  top-1/top-3/top-5 recall `0/0/0` and CV/test Spearman `0.0667`.
- Offline oracle comparison for this one row:
  AOM-Ridge oracle paired ratio `2.94852`; TabPFN paired ratio `0.691148`
  (target wins vs TabPFN on this row).
- Interpretation: increasing retained/refit budget discovers much better
  held-out candidates but does not change production selection, so this is a
  train-CV ranking/stability problem rather than a pure screen-recall problem.

## 2026-06-06 - Audit/test-rank diagnostics persistence and rank-audit summarizer

Changes:

- `run_aom_staged_real_cohort.py`: `diagnostics_payload()` now includes a
  compact `audit` section in the per-dataset diagnostics JSON when
  `report['audit']` is present. The compact payload includes `audit_only=True`,
  `n_candidates`, `selected_cv` (train-CV selected candidate's test-set
  performance, predictions stripped), `oracle` (test-rank best candidate,
  predictions stripped), and `audit_rank_diagnostics` (CV-vs-test Spearman
  correlation and related stats). Existing fields are preserved unchanged.
  `selection_uses_test_set` remains `False`; audit data is offline only.
  New helpers: `_compact_candidate_row()` (strips prediction arrays) and
  `_compact_audit_payload()` (builds the compact audit dict).
- Added `benchmarks/cross_binding/summarize_aom_rank_audit.py`: reads a
  directory (or glob) of `*.diagnostics.json` files with audit payloads and
  writes a CSV comparing the production (train-CV) selected candidate's
  CV rank/score vs its test rank/score vs the oracle (best-by-test-rank)
  candidate. Also writes an optional Markdown table. Files without an `audit`
  section (pre-feature diagnostics) are counted and reported but not written.
  The script is explicitly marked as offline audit only; it never changes
  production selection and does not route by dataset name.
- Tests: `test_aom_benchmark_tools.py` now asserts `audit` in diagnostics JSON
  (existing `test_real_cohort_runner_writes_route_counters_and_diagnostics`),
  adds `test_real_cohort_runner_diagnostics_compact_audit_payload` verifying
  audit compaction, prediction stripping and `selection_uses_test_set=False`,
  and adds `test_rank_audit_summarizer_reads_diagnostics_and_writes_summary`
  exercising the new summarizer with two-row audit vs one no-audit file.

Limitation: existing compact10 diagnostics artifacts (generated before this
patch) do not contain an `audit` section. The summarizer will count them as
pre-feature files and skip them in the output CSV. Rerun
`run_aom_staged_real_cohort.py --diagnostics-dir` on those datasets to
regenerate diagnostics with the audit payload.

## 2026-06-06 - Real-cohort split-head scoring audit path

Decision:

- Expose the existing score-preserving mixed-head split route in the
  real-cohort staged runner. Broad Ridge+PLS preprocessing screens should be
  able to use the head-homogeneous Ridge and PLS fast paths without changing
  candidate scores or selection policy.

Changes:

- Added `--split-head-scoring {auto,off,force}` to
  `benchmarks/cross_binding/run_aom_staged_real_cohort.py`.
- The runner default is now `auto`, while `off` keeps the legacy single native
  call per mixed chunk for timing comparisons.
- The output CSV records `split_head_scoring`,
  `n_screen_split_head_chunks` and `n_screen_chunk_score_calls`.
- `n4m.aom_staged_chain_campaign` now aggregates those two screen counters
  across stages, and each stage summary records its local values.
- The staged report and stage summaries also expose Ridge screen counters
  (`n_ridge_moment_cv_fits`, `n_ridge_moment_score_batch_calls`,
  `n_ridge_moment_score_batch_jobs`) so the real-cohort diagnostics payload no
  longer lists keys that are absent from staged reports.
- Model-config grids (`scale_x_values`) now sum split-head and Ridge screen
  counters across all evaluated configs. Before this fix, those counters came
  from the selected config only while PLS counters were already aggregated.
- `NativeAOMStagedChainCampaignRegressor.get_diagnostics()` exposes the same
  top-level screen counters for sklearn users.
- `NativeAOMStagedChainCampaignRegressor`,
  `NativeAOMSavgolFocusRegressor` and
  `NativeAOMStrictFamilyLiteRegressor` default to
  `split_head_scoring="auto"` for reusable mixed-head sklearn use; the
  lower-level staged helper keeps `off` as the explicit timing-compatible
  default.
- Regenerated `benchmarks/cross_binding/aom_moment_cuda_facade_smoke.json` with
  a new `staged_mixed_default_estimator` section. It proves the reusable sklearn
  default on one GPU: `split_head_scoring="auto"`,
  `n_screen_split_head_chunks=1`, `n_screen_chunk_score_calls=2`, PLS screen
  CUDA/host CV fits `8/0`, Ridge screen counters `8/1/8`, and
  `selection_uses_test_set=False`.
- `compare_aom_staged_variants.py` includes `split_head_scoring` in its
  configuration key so `auto` and `off` runs are not grouped together.

Validation:

- `py_compile` on touched Python modules/tests: pass.
- `pytest bindings/python/tests/test_aom_benchmark_tools.py bindings/python/tests/test_aom_staged_campaign.py -q`:
  `33 passed`.
- Full targeted benchmark-tool/catalog/CUDA-artifact/facade/wrapper/staged
  pytest suite: `154 passed`.
- Catalog split, reference coverage and strict-ABI validation: pass.
- CUDA facade smoke artifact test passed after the JSON regeneration.
- Synthetic score-preserving smoke: `aom_chain_score_campaign` with
  `split_head_scoring="off"` vs `"auto"` produced identical top candidates on a
  fixed mixed Ridge+PLS campaign; `auto` reported 2 split chunks and 4 native
  screen calls versus 2 calls for `off`.
- Synthetic staged smoke: `aom_staged_chain_campaign(...,
  split_head_scoring="auto")` reported `n_screen_split_head_chunks=2`,
  `n_screen_chunk_score_calls=4` and `selection_uses_test_set=False`.
- One-row real-cohort CUDA CLI smoke:
  `/tmp/n4m_aom_staged_split_head_auto_smoke.csv` and
  `/tmp/n4m_aom_staged_split_head_auto_diag/`; BEEFMARBLING completed with
  `split_head_scoring=auto`, `n_screen_split_head_chunks=2`,
  `n_screen_chunk_score_calls=4`, PLS screen CUDA/host `12/0`, and
  `selection_uses_test_set=False`.
- Compact10 one-GPU follow-up:
  `benchmarks/cross_binding/aom_staged_real_cohort_compact10_split_head_auto_20260606.csv`
  plus diagnostics dir
  `benchmarks/cross_binding/aom_staged_compact10_split_head_auto_20260606/`.
  It produced 8 OK rows and 2 property-skipped rows; all OK rows had
  `selection_uses_test_set=False`, `split_head_scoring=auto`, and selected the
  Ridge head. Against
  `aom_staged_real_cohort_compact10_mixed_diag_20260606.csv`, the paired score
  comparison was 0 wins, 0 losses, 8 ties with median ratio `1`, confirming the
  split route is score-preserving on the real compact profile. Route counters
  stayed clean: screen PLS CUDA/host `960/0`, refit PLS CUDA/host `220/0`;
  screen split counters were 32 split chunks and 64 score calls across 8 OK
  rows after summing both `scale_x` configs. Oracle ratios matched the compact
  mixed profile: AOM-PLS median
  `1.03079` (1/7 target wins), AOM-Ridge median `1.08068` (0/8), TabPFN median
  `0.978527` (4/8). The CSV now records Ridge screen-counter columns; total OK
  Ridge moment/batch counters were `1080/12/1080`. The diagnostics were
  regenerated after adding the Ridge and config-grid screen counters;
  BEEFMARBLING records Ridge moment/batch counters `360/4/360`.

## 2026-06-06 - Focused preprocessing diagnostic campaign

Decision:

- Run a short follow-up campaign from the compact10 diagnostics instead of
  widening the whole cartesian again. The tested stages focus on identity,
  `detrend_poly(1/2)`, the observed `detrend_poly -> norris_williams` branch,
  and a SavGol-variety branch with six smooth windows plus three derivatives.

Setup:

- One GPU (`CUDA_VISIBLE_DEVICES=0`) through `build/cuda-on`, `--limit 10`,
  `--max-features 1200`, Ridge+PLS heads, `--scale-x-grid false,true`,
  `--cuda-pls-parallel-folds` and `--cuda-pls-min-device-features 1`.
- Diagnostics output:
  `benchmarks/cross_binding/aom_staged_compact10_diag_focused_20260606/`.
- Result CSV:
  `benchmarks/cross_binding/aom_staged_real_cohort_compact10_diag_focused_20260606.csv`.
- Oracle comparison:
  `benchmarks/cross_binding/aom_staged_real_cohort_compact10_diag_focused_20260606_oracle_compare.csv`.
- Focused-vs-compact comparison:
  `benchmarks/cross_binding/aom_staged_compact10_diag_focused_vs_compact_20260606.md`.
- Impact summary:
  `benchmarks/cross_binding/aom_staged_compact10_diag_focused_20260606_impact_summary.md`.

Results:

- 8 OK rows and 2 property-skipped rows (`n_features>1200`), all with
  `selection_uses_test_set=False`.
- The selected production head was still Ridge on all 8 OK rows.
- Selected stages: `identity_detrend_norris` on 5/8 and `savgol_variety` on
  3/8; `scale_x=True` on 7/8.
- PLS routing stayed on GPU for the campaign: screen CUDA/host `1280/0`, refit
  CUDA/host `215/0`.
- Versus compact mixed diagnostics on the same 8 rows: 2 wins, 4 losses, 2 ties
  and median ratio `1.00125`; median fit time was `8.55s`.
- Against local oracles: AOM-PLS oracle median ratio `1.05178` (1/7 target
  wins), AOM-Ridge oracle median ratio `1.09528` (1/8 target wins), TabPFN
  oracle median ratio `0.992611` (4/8 target wins).

Interpretation:

- This focused branch confirms that SavGol diversity can be selected on real
  rows without test-set routing, but it did not improve the compact baseline in
  aggregate. The practical ceiling remains the linear Ridge-dominated selector,
  not the lack of a specific SavGol variant in the small profile.

## 2026-06-06 - Real-cohort staged diagnostics output

Decision:

- Keep `run_aom_staged_real_cohort.py` as the controlled real-dataset campaign
  runner, but make it able to persist the post-hoc preprocessing impact data
  needed for incremental family/option selection studies.

Changes:

- Added `--diagnostics-dir DIR` to
  `benchmarks/cross_binding/run_aom_staged_real_cohort.py`.
- Default behavior is unchanged. When the flag is omitted, the runner only
  writes the existing result CSV.
- For each `ok` row with diagnostics enabled, the runner writes
  `<safe_dataset_key>.diagnostics.json` containing dataset identity for audit,
  `selection_uses_test_set`, plan/head/scale metadata, `best`, `impact`,
  `rank_diagnostics`, selected/model-config summaries and route/counter fields.
- The runner also appends impact group rows to `impact_groups.csv`, with
  `group_kind` covering `by_operator`, `by_stage_family`, `by_stage_option` and
  `by_head_stage_option`. This gives a compact cross-dataset table for ranking
  preprocessing families/options without selecting by dataset name.
- Added `benchmarks/cross_binding/summarize_aom_impact_groups.py`, an offline
  summarizer for `impact_groups.csv` that emits aggregate CSV/Markdown ranked by
  dataset wins, rank-1 occurrences and train-CV rank.

Validation:

- `py_compile` on `run_aom_staged_real_cohort.py` and
  `test_aom_benchmark_tools.py`: pass.
- `pytest bindings/python/tests/test_aom_benchmark_tools.py -q -k real_cohort_runner`:
  `5 passed, 11 deselected`.

Follow-up diagnostic campaign:

- Reran the compact mixed Ridge+PLS 10-row real-cohort profile with
  `--diagnostics-dir benchmarks/cross_binding/aom_staged_compact10_mixed_diag_20260606`.
- Result CSV:
  `benchmarks/cross_binding/aom_staged_real_cohort_compact10_mixed_diag_20260606.csv`.
- Diagnostics: 8 JSON files and `impact_groups.csv` with 169 impact-group rows.
- All OK rows kept `selection_uses_test_set=False`; PLS route counters remained
  screen CUDA/host `960/0` and refit CUDA/host `220/0`.
- Summary artifact:
  `benchmarks/cross_binding/aom_staged_compact10_mixed_diag_20260606_summary.md`.
- Automated impact summary artifacts:
  `benchmarks/cross_binding/aom_staged_compact10_mixed_diag_20260606_impact_summary.csv`
  and
  `benchmarks/cross_binding/aom_staged_compact10_mixed_diag_20260606_impact_summary.md`.
- Main signal: best train-CV impact groups were `detrend_poly` on 5/8 datasets,
  `identity` on 2/8 and `savgol_smooth` on 1/8. The selected production head
  was still Ridge for all 8 OK rows.

## 2026-06-06 - Compact10 staged CUDA benchmark against local oracles

Decision:

- Stop adding wrappers for this pass and run the handoff-recommended controlled
  campaign on a small real cohort to separate Ridge, PLS and mixed-head behavior.

Setup:

- Ran the real-cohort staged campaign on one GPU with
  `CUDA_VISIBLE_DEVICES=0`,
  `N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so`, default diverse11 cohort,
  `--limit 10`, `--max-features 1200`, `--plan compact`, `--max-chains 12`,
  `--top-k 12`, `--refit-top-k 6`, `--refit-per-head-top-k 2`,
  `--scale-x-grid false,true`, `--cuda-pls-parallel-folds` and
  `--cuda-pls-min-device-features 1`.
- Ran three variants: mixed Ridge+PLS, Ridge-only and PLS-only.
- Compared each result file to the local AOM-PLS oracle, AOM-Ridge oracle and
  TabPFN oracle with `compare_aom_staged_to_oracles.py`.

Results:

- Each run produced 8 OK rows and 2 property-skipped rows
  (`n_features>1200`), with `selection_uses_test_set=False` on every OK row.
- Mixed selected `ridge` on all OK rows and matched Ridge-only RMSEP exactly.
- Mixed/Ridge oracle ratios: AOM-PLS paired median `1.03079` with 1 target win
  out of 7, AOM-Ridge paired median `1.08068` with 0 wins out of 8, TabPFN
  paired median `0.978527` with 4 wins out of 8.
- PLS-only oracle ratios: AOM-PLS paired median `1.22592` with 0 wins out of
  7, AOM-Ridge paired median `1.21773` with 0 wins out of 8, TabPFN paired
  median `1.27884` with 1 win out of 8.
- PLS CUDA route counters were clean: mixed screen/refit PLS CUDA/host
  `960/0` and `220/0`; PLS-only `960/0` and `450/0`.

Artifacts:

- `benchmarks/cross_binding/aom_staged_real_cohort_compact10_cuda_20260606.csv`
- `benchmarks/cross_binding/aom_staged_real_cohort_compact10_ridge_cuda_20260606.csv`
- `benchmarks/cross_binding/aom_staged_real_cohort_compact10_pls_cuda_20260606.csv`
- `benchmarks/cross_binding/aom_staged_real_cohort_compact10_mixed_cuda_20260606_oracle_compare.csv`
- `benchmarks/cross_binding/aom_staged_real_cohort_compact10_ridge_cuda_20260606_oracle_compare.csv`
- `benchmarks/cross_binding/aom_staged_real_cohort_compact10_pls_cuda_20260606_oracle_compare.csv`
- `benchmarks/cross_binding/aom_staged_compact10_cuda_20260606_summary.md`

Interpretation:

- For this compact budget, the production mixed selector is not adding
  diversity beyond Ridge; it chooses Ridge everywhere.
- PLS-only is weaker overall, but the ECOSIS Chla+b split is a concrete case
  where PLS-only beats Ridge materially, so per-family/per-dataset-property
  audit remains useful.
- This benchmark supports the current handoff interpretation: method wiring is
  no longer the bottleneck; the open gap is either stronger budget/selection
  policy within train-CV constraints or true fused/batched engines.

## 2026-06-06 - Robust-HPO native sklearn CUDA smoke artifact

Decision:

- Extend the committed `aom_robust_hpo_timing_cuda_smoke.csv` artifact to cover
  both the native ABI path (`native_abi`) and the `NativeAOMRobustHPORegressor`
  sklearn wrapper replay path (`native_sklearn`), replacing the old native-only
  3-row artifact.

Changes:

- Updated `bench_aom_robust_hpo_timing.py` so its sklearn backend uses the
  reusable native `NativeAOMRobustHPORegressor` wrapper instead of the older
  Python portfolio wrapper path. The row builder now records
  `prediction_replay_max_abs_error` by comparing `model.predict(X)` against
  native fitted predictions.
- Regenerated `benchmarks/cross_binding/aom_robust_hpo_timing_cuda_smoke.csv`
  without `--native-only` on one GPU (`CUDA_VISIBLE_DEVICES=0`,
  `N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so`). The new CSV has 6 rows:
  3 shapes x 2 backends (`native_abi` + `native_sklearn`), `profile=compact`,
  all pointing to `build/cuda-on/cpp/src/libn4m.so`.
- `prediction_replay_max_abs_error` for `native_sklearn` rows: <= 4.5e-15
  (machine epsilon level), well within the `<= 1e-10` gate.
- Updated `benchmarks/cross_binding/README.md`: new "AOM Robust-HPO Timing
  Smoke" section with the exact CUDA smoke command (no `--native-only`) and
  expected 6-row layout.
- Updated `docs/architecture/aom_moment_coverage_matrix.md` robust-HPO row:
  added explicit mention of the `native_abi` + `native_sklearn` CUDA artifact
  and the `prediction_replay_max_abs_error <= 1e-10` gate.

Validation:

- `test_aom_moment_cuda_smoke_artifacts.py -k robust_hpo`: `1 passed`.
- Targeted benchmark-tool/catalog/CUDA-artifact/facade/wrapper/staged pytest
  set: `152 passed`.
- Catalog split, reference validation and strict ABI validation pass.
- `git diff --check`: no whitespace issues.

## 2026-06-06 — Strict-family lite reusable audit preset

Decision:

- Keep `strict_family_focus` as the broad family-audit staged recipe, but do
  not present it as a fast default. Add a small-budget preset for reusable
  end-user audits across SavGol, Norris-Williams, finite-difference, Gaussian,
  FCK and Whittaker stages.

Changes:

- Added `NativeAOMStrictFamilyLiteRegressor`, a subclass preset over
  `NativeAOMStagedChainCampaignRegressor`.
- The preset fixes `plan="strict_family_focus"` and defaults to
  `max_chains=2`, `top_k=6`, `refit_top_k=4`,
  `refit_per_head_top_k=1`, `scale_x=False` and no `scale_x_values` grid.
- Exported the preset from `n4m`, `n4m.sklearn`, `n4m.aom` and `n4m.moment`.
- Added facade inventory rows as `preset_sklearn_wrapper` over
  `aom_pop.aom_staged_chain_campaign`, with the same no-plan/no-stages option
  surface as the SavGol-focused preset.

Validation:

- Added a synthetic fit/replay test that verifies train-CV-only selection,
  stage family coverage, no dataset/source/id stage metadata, and selected
  final-model replay.
- One-GPU CUDA facade smoke now covers the preset through `build/cuda-on`.
  The regenerated JSON reports `strict_family_lite_estimator` with screen PLS
  CUDA CV fits `36`, host `0`, refit PLS CUDA CV fits `4`, host `0`, and
  `selection_uses_test_set=False`.
- `bindings/python/tests/test_aom_staged_campaign.py`: `15 passed`.
- `bindings/python/tests/test_aom_moment_facade.py`: `13 passed`.
- `bindings/python/tests/test_aom_moment_facade.py` now also guards every
  `catalog_role="preset_sklearn_wrapper"` row: the advertised object must be
  the sklearn class, must wrap a catalog binding, must expose a `preset_plan`,
  and must not expose `plan`, `stages`, `families` or `templates`.
- The same facade test now guards that all AOM/moment facade entries are shared
  with the top-level `n4m` package, and that every relevant catalogued
  `n4m.python` binding under `aom_pop`, `utilities`, direct
  PLS/regularized/specialized heads and `moment_stack` is exposed by at least
  one facade.
- `bindings/python/tests/test_aom_moment_cuda_smoke_artifacts.py -q -k facade`:
  `1 passed, 20 deselected`.
- Targeted benchmark-tool/catalog/CUDA-artifact/facade/wrapper/staged pytest
  set: `152 passed`.
- Catalog split, reference validation and strict ABI validation pass.

## 2026-06-06 — SavGol-focus reusable sklearn preset

Decision:

- Promote the locally useful `savgol_focus` staged campaign from "plan value"
  to an end-user reusable method surface. This keeps the global staged campaign
  ultra-configurable while giving users a preconfigured winner-like estimator
  that can be fit/predicted directly.

Changes:

- Added `NativeAOMSavgolFocusRegressor`, a subclass preset over
  `NativeAOMStagedChainCampaignRegressor`.
- Defaults match the validated fast campaign recipe:
  `plan="savgol_focus"`, `max_chains=6`, `top_k=10`, `refit_top_k=8`,
  `refit_per_head_top_k=2`, `ridge_lambdas=(0.1, 1.0, 10.0)`,
  `pls_components=(1, 2)`, `scale_x_values=(False, True)`, and the one-GPU PLS
  route knobs used in the benchmark.
- Exported the preset from `n4m`, `n4m.sklearn`, `n4m.aom` and `n4m.moment`.
- Added facade inventory rows as `preset_sklearn_wrapper` over
  `aom_pop.aom_staged_chain_campaign`; the preset advertises only the options
  it actually accepts and does not expose `plan`, `stages`, `families` or
  `templates`.
- Updated staged campaign docs, coverage matrix and catalog notes.

Validation:

- `py_compile` passed for the touched Python modules.
- `bindings/python/tests/test_aom_staged_campaign.py`: `14 passed`; the new
  test fits the preset on a tiny synthetic dataset, verifies train-CV-only
  selection, selected `scale_x`, stage names and final-model replay.
- `benchmarks/cross_binding/aom_moment_cuda_facade_smoke.py` now covers
  `NativeAOMSavgolFocusRegressor` through the `build/cuda-on` shared library.
  The regenerated JSON reports `savgol_focus_estimator` with screen PLS CUDA CV
  fits `64`, host `0`, refit PLS CUDA CV fits `8`, host `0`, and
  `selection_uses_test_set=False`.
- `test_aom_moment_cuda_smoke_artifacts.py -k facade`: passed.

## 2026-06-06 — Focused strict-family staged campaign plans

Decision:

- Stop treating wider `compact_wide` / `lab` profiles as the only way to test
  preprocessing diversity. With a small `max_chains`, profile order can screen
  many SavGol variants before ever reaching Gaussian, FCK or Whittaker. Add
  fixed source-free stage recipes that put target strict-linear families first.

Changes:

- Added `plan="savgol_focus"` to `n4m.aom_staged_chain_campaign`.
  It runs compact, SavGol smooth, SavGol derivative and SavGol-combination
  stages over the existing strict-linear lab families.
- Added `plan="strict_family_focus"`. It runs compact plus separate
  SavGol, Norris-Williams, finite-difference, Gaussian, FCK, Whittaker and
  strict-combination stages so a small per-stage `max_chains` reaches each
  family.
- Existing plan names and explicit `stages=[...]` behavior are unchanged. The
  focused plans are deterministic stage labels only; they do not read dataset,
  source, id or name metadata and production selection remains train exact-CV.
- Updated staged campaign docs, benchmark README and catalog notes.

Validation:

- `py_compile` passed for `bindings/python/src/n4m/python.py`,
  `bindings/python/tests/test_aom_staged_campaign.py` and
  `benchmarks/cross_binding/run_aom_staged_real_cohort.py`.
- `test_aom_staged_campaign.py`: `13 passed`.
- One-GPU CLI smoke on one real cohort row with `plan=strict_family_focus`
  passed. It wrote 9 focused stage checkpoints, selected
  `strict_combinations`, kept `selection_uses_test_set=False`, and routed PLS
  screen/refit CV through CUDA with host counters at `0`.
- `catalog/scripts/validate.py --check-references`: PASS, 208/208.
- `catalog/scripts/validate.py --strict-abi`: PASS, 701/701.
- `git diff --check`: exit 0, with only known CRLF warnings on existing CSV
  artifacts.

Benchmark evidence:

- `savgol_focus --max-chains 6 --scale-x-grid false,true` on the local
  diverse-10 cohort with `--max-features 1200`:
  - 8 OK rows, 2 property-skipped rows;
  - selected `scale_x=True` on 7/8 OK rows;
  - selected stages: compact, SavGol smooth, SavGol derivative and
    SavGol-combinations;
  - versus compact scale-grid on the 8 paired rows: 5 wins, 2 losses, 1 tie,
    median paired ratio `0.995259`, mean ratio `0.996638`;
  - versus AOM-PLS oracle: paired median ratio `1.03051`, target wins `1`;
  - versus AOM-Ridge oracle: paired median ratio `1.07519`, target wins `1`;
  - versus TabPFN oracle: paired median ratio `0.971371`, target wins `4`;
  - median fit time `16.06s`, screen PLS CUDA CV fits `1920`, host `0`, refit
    PLS CUDA CV fits `245`, host `0`.
- `strict_family_focus --max-chains 4 --scale-x-grid false,true` was stopped as
  a partial run after 5 OK rows and 2 skipped rows because MANURE stalled in the
  refit/finalization after all focused stage checkpoints were written. On the 5
  paired rows it had 2 wins, 2 losses and 1 tie vs compact, but median fit time
  was already `26.63s`; treat this as a heavier family audit, not the fast
  default campaign.

## 2026-06-06 — Staged campaign model-config `scale_x` grid

Decision:

- Treat feature scaling as a train-CV-selected model/preprocessing config in the
  staged campaign. The compact 10-dataset calibration showed that forcing
  `scale_x=False` globally was too weak, while forcing `scale_x=True` globally
  hurt some datasets. The correct production rule is to select the config by
  exact train CV, not by held-out/test score.

Changes:

- Added `scale_x_values` to `n4m.aom_staged_chain_campaign`.
- When provided, the campaign runs one normal staged sub-campaign per value,
  scopes checkpoints under `scale_x_<value>`, selects the sub-campaign with the
  lowest `best.refit_cv_rmse`, and returns the selected rows/best/audit while
  aggregating route counters across every config evaluated.
- The report now records `model_config_grid`, `model_config_summaries`,
  `selected_model_config_id`, `selected_model_config`, `scale_x_values` and the
  selected `scale_x`.
- `NativeAOMStagedChainCampaignRegressor` accepts `scale_x_values` and fits the
  final reusable candidate with the selected `scale_x`.
- `run_aom_staged_real_cohort.py` accepts `--scale-x-grid false,true` and writes
  `scale_x`, `scale_x_values` and `selected_model_config_id` columns.

Benchmark evidence:

- Compact 10-dataset CUDA run without scaling:
  - AOM-PLS paired median ratio `1.17183`, target wins `0`.
  - AOM-Ridge paired median ratio `1.22661`, target wins `0`.
  - TabPFN paired median ratio `1.2894`, target wins `1`.
- Compact 10-dataset CUDA run with forced `scale_x=True`:
  - AOM-PLS paired median ratio `1.03079`, target wins `1`.
  - AOM-Ridge paired median ratio `1.0931`, target wins `0`.
  - TabPFN paired median ratio `1.05956`, target wins `4`.
- Compact 10-dataset CUDA run with `scale_x_values=[False, True]`, selected by
  train CV:
  - selected `scale_x=True` on 8/10 rows and `False` on 2/10 rows;
  - AOM-PLS paired median ratio `1.03079`, target wins `1`;
  - AOM-Ridge paired median ratio `1.05918`, target wins `0`;
  - TabPFN paired median ratio `1.05956`, target wins `4`;
  - PLS screen/refit stayed on one CUDA build route: screen CUDA CV fits `1200`,
    host `0`; refit CUDA CV fits `280`, host `0`.

Validation:

- CLI smoke with `--scale-x-grid false,true` on one dataset passed and wrote
  selected config / CUDA route columns.
- Targeted benchmark/catalog/CUDA-artifact/facade/wrapper/staged pytest suite:
  `143 passed`.
- `catalog/scripts/validate.py --check-references`: PASS, 208/208 production
  methods covered.
- `catalog/scripts/validate.py --strict-abi`: PASS, ABI coverage `701/701`.
- `catalog/scripts/split_legacy_methods.py --check`: PASS.
- `git diff --check`: exit 0, with only known CRLF warnings on existing CSV
  artifacts.

Remaining interpretation:

- This closes a real selection gap without using dataset identity or test-set
  selection. It still does not include SNV, MSC, EMSC, OSC or ASLS-style
  row/reference-dependent preprocessing, which explains several remaining gaps
  versus historical AOM-Ridge oracle rows.
- A follow-up `compact_wide` scale-grid run with a source-free
  `--max-features 1200` property filter produced 8 OK rows and 2 skipped rows.
  On the 8 rows paired with the compact scale-grid baseline it had 2 wins, 1
  loss and 5 ties, with median paired ratio `1.0` and mean paired ratio
  `0.994179`; the oracle summary was AOM-PLS median ratio `1.03051`,
  AOM-Ridge `1.08068`, TabPFN `0.968453`. It paid more screen work
  (`2880` CUDA PLS CV fits, host `0`) for marginal score movement, so the next
  campaign should target preprocessing families/options rather than blindly
  increasing the stage width.

## 2026-06-06 — Strict single-chain AOM Ridge-PLS selector

Decision:

- Add the strict/raw-base donor SingleChainRidgePLS-style surface without
  widening the port to non-moment preprocessing or nonlinear routes. This gives
  users a reusable single-chain Ridge-PLS model alongside the existing
  superblock and campaign methods.

Changes:

- Added `n4m.aom_chain_ridge_pls`, exposed through top-level `n4m`, `n4m.aom`
  and `n4m.moment`.
- Added `NativeAOMChainRidgePLSRegressor` in `n4m.sklearn`.
- Catalogued the method as `aom_pop.aom_chain_ridge_pls` with docs and timing
  smoke coverage.
- The function scores `(chain, n_components, ridge_lambda)` by train-only CV,
  applies strict-linear AOM chains sequentially, fits the selected final model
  through native `ridge_pls`, and folds the selected chain coefficients back to
  raw `input_coefficients` plus `intercept`.
- Candidate OOF predictions are tracked only for the current best candidate
  instead of storing an `n_candidates x n_samples x n_targets` buffer; the full
  score table remains available for ranking/audit.
- Added a one-GPU CUDA-build smoke artifact,
  `benchmarks/cross_binding/aom_chain_ridge_pls_timing_cuda_smoke.csv`, and
  extended the CUDA artifact guard to require function + sklearn replay rows.

Validation:

- CPU timing smoke wrote 6 rows to `/tmp/aom_chain_ridge_pls_timing.csv`.
- CUDA timing smoke wrote 6 rows to
  `benchmarks/cross_binding/aom_chain_ridge_pls_timing_cuda_smoke.csv` with
  `selection_mode=chain_ridge_pls`, `ridge_pls_backend=native`,
  `library_path=build/cuda-on/cpp/src/libn4m.so`, and replay error `0.0`.
- Targeted benchmark/catalog/CUDA-artifact/facade/wrapper/staged pytest suite:
  `141 passed`.
- `catalog/scripts/validate.py --check-references`: PASS, 208/208 production
  methods covered.
- `catalog/scripts/validate.py --strict-abi`: PASS, ABI coverage `701/701`.
- `catalog/scripts/split_legacy_methods.py --check`: PASS.
- `git diff --check`: exit 0, with only known CRLF warnings on existing CSV
  artifacts.

Scope note:

- This method intentionally excludes SNV, MSC, EMSC, OSC,
  row-reference-dependent preprocessing, nonlinear lifts, kernels, trees,
  TabPFN residuals and dataset/source routing. CUDA-build smoke proves build
  compatibility and replay, not a fused many-chain GPU Ridge-PLS grinder.

## 2026-06-05 — Direct AOM preprocessing strict-linear bank

Decision:

- Reuse the existing AOM strict-linear operator engine for standalone
  `n4m.aom_preprocess`, so the reusable primitive covers the same direct
  strict-linear families used by compact/wide AOM banks instead of only the
  subset accepted by the generic preprocessing pipeline.

Changes:

- Switched `cpp/src/core/aom_preprocessing.cpp` from a `Pipeline` fit/transform
  dispatch to `transform_aom_strict_operator`.
- Added a direct C ABI test,
  `aom_preprocess/direct_strict_linear_bank`, covering the 9-operator bank,
  hard/soft weights, operator kind ids, and exact replay of identity plus
  finite-difference outputs.
- Extended `bench_aom_preprocess_timing.py` and the CUDA smoke artifact to
  direct single-operator identity, degree-1 detrend, Savitzky-Golay
  smooth/derivative, Norris-Williams, finite difference, Gaussian, Whittaker
  and FCK rows.
- Updated the CUDA artifact guard to require 54 rows (9 operators x 2 gating
  modes x 3 shapes), expected operator kind ids and exact single-operator replay
  against the native `operator_outputs` buffer.
- Updated docs to state that strict chains and model-scoring diversity remain
  in AOM sweep/campaign helpers; the direct API now covers the reusable
  strict-linear single-operator bank.

Validation:

- Rebuilt both `dev-release` and `cuda-on` CMake presets after the C++ dispatch
  change.
- `py_compile` passed for the updated benchmark script and Python artifact
  tests.
- `test_aom_moment_cuda_smoke_artifacts.py`: `14 passed`.
- Targeted benchmark/catalog/CUDA-artifact/facade/wrapper/staged pytest suite:
  `108 passed`.
- `build/dev-release/cpp/tests/n4m_tests`: `351 passed, 0 failed`.
- `catalog/scripts/validate.py --strict-abi`: PASS, 201 methods and 701/701
  exported `n4m_*` symbols covered.
- `catalog/scripts/validate.py --check-references`: PASS, 201/201 production
  methods covered.
- `catalog/scripts/split_legacy_methods.py --check`: PASS.

## 2026-06-05 — Linear PLS variants sklearn replay wrappers

Decision:

- Promote `weighted_pls`, `robust_pls` and `ridge_pls` from ABI-close functions
  to reusable direct heads after proving `X @ coefficients + (y_mean - x_mean @
  coefficients)` replays native training predictions at numerical-noise level.
  `models.pls.kernel` remains intentionally excluded as non-linear kernel PLS.

Changes:

- Added `NativeWeightedPLSRegressor`, `NativeRobustPLSRegressor` and
  `NativeRidgePLSRegressor` in `n4m.sklearn`, top-level `n4m`, and the
  `n4m.moment` facade inventory.
- Updated `bench_direct_moment_heads_timing.py` to time those three heads as
  both `native_function` and `sklearn_fit_predict` rows.
- Regenerated `direct_moment_heads_timing_cuda_smoke.csv`; it now contains
  54 rows (9 methods x 3 shapes x 2 backends), all with
  `surface_status=function_and_sklearn_replay`.
- Updated direct-head docs, coverage matrix, benchmark README and handoff notes
  to remove stale function-only language.

Validation:

- Replay proof before wrapper addition: `weighted_pls`, `robust_pls` and
  `ridge_pls` all replayed native train predictions via reconstructed intercept
  at numerical-noise level across several CPU shapes/options.
- Focused wrapper/artifact pytest set:
  `bindings/python/tests/test_moment_model_wrappers.py` +
  `bindings/python/tests/test_aom_moment_cuda_smoke_artifacts.py`: `71 passed`.
- Targeted benchmark/catalog/CUDA-artifact/facade/wrapper/staged pytest suite:
  `107 passed`.
- `catalog/scripts/validate.py --strict-abi`: PASS, 201 methods and 701/701
  exported `n4m_*` symbols covered.
- `catalog/scripts/validate.py --check-references`: PASS, 201/201 production
  methods covered.
- `catalog/scripts/split_legacy_methods.py --check`: PASS.
- `git diff --check`: exit 0, only the known CRLF warnings on existing CSV
  artifacts.

## 2026-06-05 — Linear PLS variants doc/coverage cleanup

Decision:

- Document the three function-only linear PLS heads (`weighted_pls`,
  `robust_pls`, `ridge_pls`) that are already wired and tested but were not
  reflected in the coverage matrix, README or handoff notes. Fix two doc
  inaccuracies found during the pass. No new ABI was added; the only Python
  logic change is the timing-smoke `surface_status` cleanup.

Changes:

- `docs/architecture/aom_moment_coverage_matrix.md`: added three rows for
  weighted PLS, robust PLS and ridge-augmented PLS heads; updated Direct PLS
  head description to note the smoke CSV covers 9 methods / 45 rows; added
  `kernel_pls` exclusion invariant.
- `benchmarks/cross_binding/README.md`: updated "Direct Moment Heads CUDA
  Smoke" section to describe all 9 methods, 6 wrapper-backed vs 3
  function-only split, and 45-row CSV layout.
- `handoff.md`: added "Linear PLS variants" section recording what is wired,
  tested and why `kernel_pls` remains excluded.
- `benchmarks/cross_binding/bench_direct_moment_heads_timing.py`: made
  `surface_status` method-level — wrapper-backed `native_function` rows now
  report `function_and_sklearn_replay`; function-only rows keep `function_only`.
- `bindings/python/tests/test_aom_moment_cuda_smoke_artifacts.py`: tightened the
  direct-head artifact guard so wrapper-backed methods must report
  `surface_status=function_and_sklearn_replay` on every row.
- Regenerated `direct_moment_heads_timing_cuda_smoke.csv` (45 rows, CUDA ABI
  `1.21.0`) to reflect the `surface_status` change.
- `docs/methods/weighted_pls.md`: removed the misleading "no global coefficient
  export" claim; `n4m_weighted_pls_fit` does export `coefficients`; the note
  now describes the weight-dependent centering constraint that prevents a naive
  replay wrapper.
- `docs/methods/robust_pls.md`: corrected `max_irls_iter` default in parameter
  table from `20` to `5` (matches the Python helper signature).

Validation:

- Targeted benchmark/catalog/CUDA-artifact/facade/wrapper/staged pytest suite:
  `104 passed`.

## 2026-06-05 — Candidate preprocessing impact audit

Decision:

- Finish the current post-hoc cartesian analysis task without expanding the
  engine work: summarize which preprocessing stages/options are helping from
  already-scored candidate reports.

Changes:

- Added `n4m.aom_candidate_preprocessing_impact` and exposed it through
  `n4m.aom` and `n4m.moment`.
- The helper groups candidate rows by inferred stage, operator, concrete
  option, chain position and head/stage combinations.
- It reports best/mean/median score, ranks and improvement versus identity
  baselines when identity rows are present.

Validation:

- Covered by synthetic candidate-row tests and the facade inventory tests.

## 2026-06-05 — Moment-model OOF stack

Decision:

- Add the missing strict moment-model stacking surface from `PORTING.md`
  without adding nonlinear features or a new native ABI.

Changes:

- Added `NativeMomentStackRegressor`, a sklearn-style train-only OOF Ridge stack
  over native Ridge, PLS sweep, PCR, continuum, ECR and CPPLS base predictions.
- Exposed the stack through `n4m.sklearn`, top-level `n4m` and the
  `n4m.moment` inventory as `models.ensembles.moment_stack`.
- Added method docs, catalog metadata and a timing smoke benchmark.
- Added fitted `base_oof_diagnostics_` / `base_final_diagnostics_` and aggregate
  PLS route counters to make CPU/GPU routing auditable for the OOF base fits and
  final base refits.
- Extended `bench_moment_stack_timing.py` with custom shapes/base-models and
  CUDA PLS knobs, then generated
  `benchmarks/cross_binding/moment_stack_timing_cuda_smoke.csv`.

Validation:

- Smoke-tested Ridge/PCR/PLS and full six-base stacks against the rebuilt
  `libn4m`.
- CUDA smoke on one GPU with a PLS-only stack (`80x1024`, `cv=4`,
  `inner_cv=4`, `cuda_pls_min_device_features=1`,
  `cuda_pls_parallel_folds=True`) reported
  `n_base_oof_pls_moment_cuda_device_cv_fits=16`,
  `n_base_oof_pls_moment_host_cv_fits=0`,
  `n_base_final_pls_moment_cuda_device_cv_fits=4`, and
  `n_base_final_pls_moment_host_cv_fits=0`.

## 2026-06-05 — Direct PCR MethodResult head

Decision:

- Add PCR through the same direct-head pattern as Ridge, CPPLS, continuum and
  ECR, instead of exposing it through an untyped generic-model ctypes shortcut.

Changes:

- Added `n4m_pcr_fit` to the C ABI, forcing `Algorithm.PCR` + `Solver.SVD` on
  top of the caller's centering, scaling and component-count config.
- Added `n4m.pcr`, `n4m.moment.pcr` and `NativePCRRegressor` with replayable
  input-space coefficients and MethodResult predictions.
- Catalogued PCR as `models.pls.pcr` and registered the ABI symbol in the
  expected-symbol manifests.

Validation:

- Rebuilt `n4m_c`, ran the PCR direct/sklearn smoke, Python facade tests,
  catalog validation and strict ABI validation.

## 2026-06-05 — Deterministic streaming AOM chain grid

Decision:

- Add the missing streaming bank-generator surface from the porting backlog
  without changing the native scoring ABI or candidate scores.

Changes:

- Added `n4m.iter_aom_strict_chain_grid`, with the same grid semantics as
  `build_aom_strict_chain_grid` plus stable ids, `start`/`stop` slicing,
  optional `chunk_size`, and `with_ids`.
- Exposed the helper through `n4m.aom` and `n4m.moment` inventories as a
  non-catalog campaign/runtime helper.
- Documented the incremental launcher use case in
  `docs/methods/aom_chain_sweep_run.md` and the coverage matrix.

Validation:

- Added tests that compare iterator output against the existing builder, cover
  id slicing/chunking, and verify facade inventory exposure.

## 2026-06-05 — Campaign-level CPU/CUDA launch threshold override

Decision:

- Make the source-free CPU/CUDA launch recommendation fully configurable from
  AOM/moment campaigns and sklearn presets, not only from the standalone
  helper.

Changes:

- Added `backend_min_cuda_product` to `aom_chain_score_campaign`,
  `aom_chain_screen_refit_campaign`, and `aom_moment_screen_refit_campaign`.
- Propagated the option to `NativeAOMScreenRefitRegressor` and the mixed,
  PLS-only and Ridge-only moment screen-refit presets.
- Campaign reports now include `backend_min_cuda_product`, and
  `moment_backend_recommendation_policy_inputs` includes `min_cuda_product`.
- Updated `n4m.aom.available_methods()` and `n4m.moment.available_methods()`
  config inventories.
- Documented the option in `docs/methods/aom_chain_sweep_run.md`.
- Exposed `--backend-min-cuda-product` in
  `bench_aom_screen_refit_scaling.py` and the screen/refit rows of
  `bench_aom_sweep_timing.py`.

Validation:

- Added tests that force `backend_min_cuda_product=1` in campaign and sklearn
  wrapper paths and verify the recommendation/report changes without changing
  selected scores.
- Added benchmark script smokes for the new CLI flag.

## 2026-06-05 — Conservative CPU/CUDA backend recommendation v4

Decision:

- Align `moment_screen_backend_recommendation` with the latest one-GPU
  crossover evidence instead of recommending CUDA at the old `260x256`
  boundary.

Changes:

- Raised the default launch crossover threshold from `260 * 256` to
  `512 * 512`.
- Bumped the policy source string to `n4m.moment_gpu_crossover.v4`.
- Kept `min_cuda_product` as an explicit override for benchmark campaigns that
  want to test medium shapes.
- Updated tests and docs so `260x256` is CPU by default, while `512x512` and
  `256x1024` remain CUDA candidates.

Validation:

- Policy is based on `/tmp/moment_gpu_crossover_cuda_profiles.csv`
  (`repeats=1`, one GPU, ABI 1.21.0): CUDA default was slower than CPU for PLS
  at `260x256`, but faster for PLS at `512x512` and `256x1024`; Ridge was
  effectively flat/slightly slower at `260x256` and faster at the larger
  shapes.

## 2026-06-05 — Moment GPU crossover can compare CUDA PLS profiles

Decision:

- Measure the optional `cuda_pls_many_batched` route against the default CUDA
  PLS moment route in one controlled run before attempting larger fused CUDA
  work.

Changes:

- `bench_moment_gpu_crossover.py` now supports
  `--compare-cuda-pls-many-batched`.
- The script runs the CPU baseline once, then CUDA default and CUDA
  many-batched profiles on identical synthetic shapes.
- The CSV now includes `cuda_pls_profile` and `speedup_vs_cuda_default` in
  addition to `speedup_vs_cpu`, so the experimental route can be judged
  directly.

Validation:

- `py_compile` passed for `bench_moment_gpu_crossover.py`.
- One-GPU smoke:
  `CUDA_VISIBLE_DEVICES=0 bench_moment_gpu_crossover.py --shapes 64x16 --cv 3
  --repeats 1 --cuda-pls-min-device-features 1
  --compare-cuda-pls-many-batched` wrote
  `/tmp/moment_gpu_crossover_compare_smoke.csv` against ABI 1.21.0 with CPU,
  CUDA default and CUDA many-batched rows.
- One-GPU comparison, `repeats=1`, `cuda_pls_min_device_features=1`,
  shapes `260x256,512x512,256x1024`, wrote
  `/tmp/moment_gpu_crossover_cuda_profiles.csv`. On PLS rows, CUDA default was
  slower than CPU at `260x256` but faster at `512x512` and `256x1024`;
  `many_batched` did not beat CUDA default on those PLS shapes
  (`speedup_vs_cuda_default` about `0.99`, `0.88`, `0.88`). This makes the
  default CUDA PLS moment route the better current campaign baseline.

## 2026-06-05 — PLS moment route documented as IKPLS-style, not missing

Decision:

- Treat the current exact PLS1 moment route as an IKPLS-style
  sufficient-statistics implementation. The remaining open item is the fused
  many-chain/many-fold batching layer, not the existence of cross-product PLS1
  scoring itself.

Changes:

- Updated `cpp/src/core/sweep.hpp` comments to describe exact PLS1 moment
  scoring and the remaining materialized fallback.
- Renamed the coverage-matrix gap from generic "Batched IKPLS" to
  "Fused/batched IKPLS" and clarified the current single-target PLS1 coverage.

Validation:

- Documentation/comment-only change.

## 2026-06-05 — Backend recommendation reports many-batched PLS planning

Decision:

- Keep backend routing source-free, but make the diagnostic reflect every PLS
  CUDA knob that campaigns can pass.

Changes:

- `n4m.moment_screen_backend_recommendation(...)` now accepts
  `cuda_pls_many_batched`.
- AOM chain score campaign reports propagate the flag into
  `moment_backend_recommendations`.
- `n4m.moment.available_methods()` advertises the new backend-recommendation
  option.
- `docs/methods/sweep_run.md` and `docs/methods/aom_chain_sweep_run.md`
  document the extra diagnostic field.

Validation:

- `py_compile` passed for the touched Python modules and benchmark.
- `test_moment_model_wrappers.py`: 49 passed against ABI 1.21.0.

## 2026-06-05 — Moment GPU crossover benchmark aligned with ABI 1.21

Decision:

- Keep the crossover benchmark independent from a specific ABI patch version
  and make it measure the explicit CUDA PLS route that timing campaigns use.

Changes:

- `bench_moment_gpu_crossover.py` now defaults to the current build symlinks
  (`build/dev-release/.../libn4m.so` and `build/cuda-on/.../libn4m.so`)
  instead of hard-coding `libn4m.so.1.18.0`.
- Added `--cuda-pls-min-device-features` and `--cuda-pls-many-batched` to the
  crossover script.
- The CSV now records those knobs plus PLS score-batch and CUDA-parallel-fold
  counters, so a timing row shows whether the intended exact-CV PLS moment
  route actually ran on device.

Validation:

- `py_compile` passed for `bench_moment_gpu_crossover.py`.
- One-GPU smoke:
  `CUDA_VISIBLE_DEVICES=0 bench_moment_gpu_crossover.py --shapes 64x16 --cv 3
  --repeats 1 --cuda-pls-min-device-features 1 --cuda-pls-many-batched` wrote
  `/tmp/moment_gpu_crossover_smoke_many.csv` against ABI 1.21.0, with the CUDA
  PLS row reporting device CV fits.

## 2026-06-05 — CUDA PLS many-design batching config exposed

Decision:

- Keep the experimental CUDA tiled/strided-batched many-design PLS1 moment
  route opt-in, but expose it as a first-class config flag instead of requiring
  `N4M_CUDA_PLS_MANY_BATCHED`.
- Do not change default scoring or routing. The default remains the reusable
  sequential-many workspace; the new flag only selects the existing
  experimental one-GPU tiled path when the CUDA route is otherwise eligible.

Changes:

- Added public ABI helpers:
  `n4m_config_set_cuda_pls_many_batched` and
  `n4m_config_get_cuda_pls_many_batched`.
- Bumped ABI to 1.21.0 and updated Linux/macOS/Windows expected symbol lists
  plus `catalog/abi_infra.yaml`.
- Propagated `cuda_pls_many_batched` through `sweep_run`, `aom_sweep_run`,
  `aom_chain_sweep_run`, score/refit campaigns, sklearn sweep/screen-refit
  wrappers, AOM/moment facade inventories and the timing scripts.
- Added the flag to `bench_moment_sweep_timing.py`,
  `bench_aom_sweep_timing.py` and
  `bench_aom_screen_refit_scaling.py` so one-GPU timing campaigns can compare
  the path without env vars.

Validation:

- CPU build `build/dev-release`: rebuilt `libn4m.so.1.21.0`.
- CUDA build `build/cuda-on`: rebuilt `libn4m.so.1.21.0` with
  `CUDA_VISIBLE_DEVICES=0`.
- C++ tests: `n4m_tests` 350 passed on CPU and CUDA builds; internal tests
  passed on both builds.
- Python tests: `test_aom_moment_facade.py` +
  `test_moment_model_wrappers.py`: 58 passed against ABI 1.21.0.
- CUDA Python smokes: `aom_moment_cuda_facade_smoke.py` passed with CUDA
  device CV counters; explicit `sweep_run(...,
  cuda_pls_many_batched=True)` matched `False` scores with CUDA device fits.
- Catalog validation: 198 methods, PASS.
- ABI export set vs `expected_symbols_linux.txt`: 0 missing, 0 extra.
- `git diff --check`: PASS.

Remaining work:

- This is still not fused IKPLS or the final 200k-chain GPU grinder. The
  open performance gap remains a true batched/fused PLS CV path across
  preprocessing variants and broader operator-moment coverage.

## 2026-06-05 — AOM facade inventory parity

Decision:

- Keep `n4m.aom` as a logical facade over the existing top-level runtime
  functions, but make `n4m.aom.available_methods()` complete enough for
  scripts to discover every AOM campaign helper that the namespace already
  exports.

Changes:

- Added `n4m.aom.available_methods()` rows for the exported chain-grid,
  candidate decode/evaluate, retained-pool, exact-refit, fixed-fit,
  rank/route/operator audit and candidate-report IO helpers.
- Mirrored the relevant `config_options` metadata from `n4m.moment` so AOM
  users can inspect refit execution, CUDA PLS knobs, report serialization and
  audit controls from the AOM namespace without switching facades.
- Extended the AOM facade tests to assert top-level alias identity, inventory
  presence and representative `config_options` for those helpers.

Validation:

- `PYTHONPATH=bindings/python/src /home/delete/.venv/bin/python -m py_compile
  bindings/python/src/n4m/aom/__init__.py
  bindings/python/tests/test_moment_model_wrappers.py`: PASS.
- `PYTHONPATH=bindings/python/src
  N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so
  /home/delete/.venv/bin/python -m pytest
  bindings/python/tests/test_aom_moment_facade.py
  bindings/python/tests/test_moment_model_wrappers.py -q`: 58 passed.
- `PYTHONPATH=bindings/python/src
  /home/delete/.venv/bin/python catalog/scripts/validate.py`: PASS, 198
  methods.
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=bindings/python/src
  /home/delete/.venv/bin/python
  benchmarks/cross_binding/aom_moment_cuda_facade_smoke.py
  --cuda-visible-devices 0 --output
  /tmp/aom_moment_cuda_facade_smoke.json`: PASS.
- `git diff --check`: PASS.

## 2026-06-05 — Functional fast-profile AOM moment screen/refit wrapper

Decision:

- The native exact/proxy Ridge/PLS screen paths already batch head-homogeneous
  jobs, and the sklearn moment screen/refit presets already choose the fast
  defaults. The missing piece for scripts and campaigns was a low-friction
  function that exposes the same fast profile without changing the legacy
  defaults of `aom_chain_score_campaign` or `aom_chain_screen_refit_campaign`.

Changes:

- Added `n4m.aom_moment_screen_refit_campaign` and the `n4m.aom` /
  `n4m.moment` facade aliases.
  It wraps `aom_chain_screen_refit_campaign` with strict moment routes, prefix
  chain ordering, split-head mixed scoring, PLS GCV-proxy first-pass scoring,
  exact-CV refit and `refit_execution="auto"`.
- The combined report keeps schema
  `n4m.aom_chain_screen_refit_campaign.v1` and adds
  `campaign_preset="moment_fast_screen_refit"` for audit.
- Added facade inventory metadata under
  `moment_fast_screen_refit_campaign` for AOM and
  `aom_moment_screen_refit_campaign` for the moment namespace.
- Re-exported `NativeAOMMomentScreenRefitRegressor`,
  `NativeAOMMomentPLSScreenRefitRegressor` and
  `NativeAOMMomentRidgeScreenRefitRegressor` from `n4m.moment`, with matching
  inventory entries. The moment namespace now exposes both the functional
  campaign preset and the reusable estimator presets.
- Re-exported the retained-pool/refit/fixed-fit helpers from `n4m.moment`:
  `aom_screen_refit_candidate_pool`, `aom_refit_execution_plan`,
  `aom_refit_candidates`, `aom_chain_fixed_fit_run` and
  `NativeAOMFixedCandidateRegressor`. This lets a moment workflow go from
  screen report to exact-CV refit rows to a final-only reusable winner without
  switching namespaces.
- Re-exported the audit/report helpers from `n4m.moment`:
  `build_aom_strict_chain_grid`, `decode_aom_chains`,
  `aom_candidate_table`, `aom_evaluate_candidates`,
  `aom_candidate_operator_summary`, `aom_candidate_rank_diagnostics`,
  `aom_candidate_report_records`, `aom_save_candidate_report` and
  `aom_load_candidate_report`. The inventory records their explicit
  `config_options` as audit/report surfaces, not fit/predict catalog methods.
- Added `n4m.aom_candidate_route_summary` plus `n4m.aom` and `n4m.moment`
  aliases. It summarizes materialized, dense, banded and structured
  operator-moment coverage globally, by head and by chain from explicit
  candidate/report rows, so broad preprocessing screens can prove whether the
  retained rows actually stayed in moment routes before reuse.
- Extended the route summary with `reported_total` for campaign/refit reports
  that carry aggregate counters. This keeps the top-level summary focused on
  retained rows, while still exposing full-screen coverage when a campaign only
  stores top-k rows.

Validation:

- `PYTHONPATH=bindings/python/src /home/delete/.venv/bin/python -m py_compile
  bindings/python/src/n4m/python.py bindings/python/src/n4m/__init__.py
  bindings/python/src/n4m/aom/__init__.py
  bindings/python/tests/test_moment_model_wrappers.py`: PASS.
- `PYTHONPATH=bindings/python/src
  N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so
  /home/delete/.venv/bin/python -m pytest
  bindings/python/tests/test_moment_model_wrappers.py -q`: 49 passed.
- `PYTHONPATH=bindings/python/src
  N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so
  /home/delete/.venv/bin/python -m pytest
  bindings/python/tests/test_aom_moment_facade.py -q`: 9 passed.
- `PYTHONPATH=bindings/python/src
  N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so
  /home/delete/.venv/bin/python -m pytest
  bindings/python/tests/test_aom_moment_facade.py
  bindings/python/tests/test_moment_model_wrappers.py -q`: 58 passed after
  adding the functional campaign alias and the AOM/moment estimator aliases to
  `n4m.moment`, then 58 passed again after adding the retained-pool, exact-refit
  and fixed-candidate winner-reuse aliases, then 58 passed again after adding
  the audit/report helper aliases and JSON report save/load workflow coverage,
  then 58 passed again after adding route-coverage summary aliases and tests,
  then 58 passed again after adding `reported_total` coverage for top-k
  campaign reports.
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=bindings/python/src
  /home/delete/.venv/bin/python
  benchmarks/cross_binding/aom_moment_cuda_facade_smoke.py
  --cuda-visible-devices 0 --output /tmp/aom_moment_cuda_facade_smoke.json`:
  PASS, including the moment audit/report and route-summary alias assertions.
- `PYTHONPATH=bindings/python/src /home/delete/.venv/bin/python
  catalog/scripts/validate.py`: PASS.

## 2026-06-05 — Default mixed AOM screen/refit estimators to split-head scoring

Decision:

- A single mixed Ridge+PLS `aom_chain_sweep_run(score_only=True)` call uses
  *none* of the batched head-homogeneous fast paths. Probing a small mixed
  campaign shows every batch counter at `0` for `split_head_scoring="off"`,
  while `"auto"` (Ridge-only + PLS-only subcalls, merged) turns them on with
  bit-identical candidate scores. So mixed-by-default sklearn screen/refit
  workflows were silently leaving the batched grinder unused.
- Make the workflow/preset surfaces opt into `"auto"` by default while keeping
  the lower-level campaign helpers on the historical `"off"` launch shape (a
  pre-existing test pins `aom_chain_score_campaign(..., split_head_scoring="off")`
  behavior).

Changes:

- `NativeAOMScreenRefitRegressor` (base sklearn screen/refit estimator, default
  heads `("ridge", "pls")`) now defaults `split_head_scoring="auto"`; docstring
  documents the score-preserving merge and that single-head screens are inert.
  The `NativeAOMMomentScreenRefitRegressor` mixed preset already defaulted to
  `"auto"`; the PLS-only / Ridge-only presets inherit `"auto"` but it is a
  behavioral no-op (`can_split_head_scoring` is false for one head, so
  `n_split_head_chunks` stays `0`).
- `benchmarks/cross_binding/bench_aom_screen_refit_scaling.py`
  `--split-head-scoring` now defaults to `auto` (inert for the default
  `--head pls`; representative for `--head mixed`); the actual value is still
  written to each CSV row.
- `aom_chain_score_campaign` / `aom_chain_screen_refit_campaign` defaults are
  unchanged (`"off"`).
- `docs/methods/aom_chain_sweep_run.md` now spells out that splitting enables
  *both* the Ridge moment batch and the PLS exact/GCV-proxy batch (a single
  mixed call batches nothing), and which surfaces default to `off` vs `auto`.

Validation (CPU, `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so`, ABI 1.20):

- Empirical counters for a mixed `aom_chain_score_campaign` (C chains, L ridge
  lambdas, K folds): `off` -> all batch counters `0`; `auto` ->
  `n_ridge_moment_score_batch_jobs == C*L*K`, plus
  `n_pls_gcv_proxy_batch_jobs == C` (`pls_score_mode="gcv_proxy"`) or
  `n_pls_moment_score_batch_jobs == C*K == n_pls_moment_cv_fits`
  (`pls_score_mode="cv"`); `keyed (chain_id, head, param)` scores and `best`
  identical (max score diff `0.0`).
- `NativeAOMScreenRefitRegressor` default (`auto`) vs explicit `off`: identical
  `selected_chain_`, `selected_head_`, `selected_cv_rmse_`, `coef_` and
  `predict(X)` (max abs diff `0.0`).
- New/strengthened tests in
  `bindings/python/tests/test_moment_model_wrappers.py`
  (`test_aom_campaign_split_head_scoring_preserves_scores_and_enables_split`,
  `test_aom_campaign_split_head_exact_cv_enables_pls_moment_batch`,
  `test_native_aom_screen_refit_defaults_to_split_head_scoring`); full file
  `48 passed`.
- `python -m py_compile` of the bench, `native_sweeps.py` and
  `make_python_package.py` passed.

## 2026-06-05 — Add CUDA-device smoke controls to AOM sweep timing

Decision:

- Make `bench_aom_sweep_timing.py` able to exercise the CUDA PLS1 moment
  device route explicitly, instead of only timing CUDA builds through shapes
  that remain below the default device threshold.

Changes:

- Added `--cuda-pls-parallel-folds` and
  `--cuda-pls-min-device-features` to `bench_aom_sweep_timing.py`.
- Propagated those flags through AOM sweep, AOM chain sweep, exact PLS
  score-only, screen/refit, FCK/Gaussian/Whittaker and Ridge scenarios.
- Added `n_pls_moment_cuda_parallel_fold_batches`,
  `n_pls_moment_cuda_parallel_fold_jobs`,
  `n_refit_pls_moment_cuda_parallel_fold_batches` and
  `n_refit_pls_moment_cuda_parallel_fold_jobs` to the timing CSV rows.

Validation:

- CPU default smoke:
  `PYTHONPATH=bindings/python/src N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so /home/delete/.venv/bin/python benchmarks/cross_binding/bench_aom_sweep_timing.py --repeats 1 --cv 4 --profile compact --output /tmp/aom_sweep_cpu_full_smoke.csv`
  passed with 49 rows, Ridge/PLS exact score-only rows present and
  `pls_device_fits=0`.
- CUDA one-GPU smoke:
  `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=bindings/python/src N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so /home/delete/.venv/bin/python benchmarks/cross_binding/bench_aom_sweep_timing.py --repeats 1 --cv 4 --profile compact --cuda-pls-min-device-features 1 --cuda-pls-parallel-folds --output /tmp/aom_sweep_cuda_device_smoke.csv`
  passed with 49 rows, `pls_device_fits=824`, `parallel_batches=198`,
  `parallel_jobs=824` and `ridge_batch_calls=2`.
- `PYTHONPATH=bindings/python/src /home/delete/.venv/bin/python -m py_compile benchmarks/cross_binding/bench_aom_sweep_timing.py`
  passed.

## 2026-06-05 — Make AOM sweep timing smoke runnable under force_moments

Decision:

- Keep the FCK/Gaussian/exact PLS timing rows on strict moment routes, but make
  their synthetic shapes satisfy the exact PLS operator-moment invariant at the
  default `cv=4`.

Changes:

- Adjusted `bench_aom_sweep_timing.py` force-moment exact PLS shapes from
  `(160, 32)` / `(240, 48)` to `(192, 32)` / `(288, 48)`.
- Added a script comment documenting the `min_train >= 4p` constraint so the
  timing smoke does not drift back into unsupported materialized fallback
  territory.

Validation:

- `PYTHONPATH=bindings/python/src N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so /home/delete/.venv/bin/python benchmarks/cross_binding/bench_aom_sweep_timing.py --repeats 1 --cv 4 --profile compact --output /tmp/aom_sweep_full_smoke.csv`
  passed and produced 49 timing rows, including both
  `native_aom_chain_sweep_pls_exact_score_only` and
  `native_aom_chain_sweep_ridge_exact_score_only`.

## 2026-06-05 — Surface Ridge batch counters in campaigns and benches

Decision:

- Treat native Ridge batch score-only counters as campaign-level telemetry, not
  only low-level native diagnostics. Large preprocessing screens need the same
  visibility for Ridge as for PLS exact/proxy batch paths.

Changes:

- Added Ridge fit and batch counters to `_AOM_ROUTE_COUNTER_KEYS`, campaign
  chunk aggregation, checkpoint reports and split-head merged results.
- Added `ridge_cv_fits_per_chain` and `ridge_cv_fits_per_candidate` to AOM
  campaign throughput reports.
- Added Ridge batch counters to native sklearn sweep/screen-refit diagnostics.
- Added Ridge screen/refit/final-fit columns to
  `bench_aom_screen_refit_scaling.py`, plus Ridge batch columns and a
  `native_aom_chain_sweep_ridge_exact_score_only` row to
  `bench_aom_sweep_timing.py`.

Validation:

- `PYTHONPATH=bindings/python/src N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so /home/delete/.venv/bin/python -m pytest bindings/python/tests/test_moment_model_wrappers.py -k 'ridge_score_only_uses_batch_moment_path or moment_screen_refit_presets_are_reusable or pls_gcv_proxy_score_only' -q`
  passed.
- `PYTHONPATH=bindings/python/src /home/delete/.venv/bin/python -m py_compile bindings/python/src/n4m/python.py bindings/python/src/n4m/sklearn/native_sweeps.py benchmarks/cross_binding/bench_aom_sweep_timing.py benchmarks/cross_binding/bench_aom_screen_refit_scaling.py benchmarks/cross_binding/bench_moment_sweep_timing.py`
  passed.
- A reduced Ridge screen/refit benchmark smoke reported screen
  `n_ridge_moment_score_batch_calls=1`, `n_ridge_moment_score_batch_jobs=72`,
  refit `n_refit_ridge_moment_cv_fits=6`,
  `n_refit_ridge_moment_score_batch_calls=1` and
  `n_refit_ridge_moment_score_batch_jobs=6`.

## 2026-06-05 — Batch exact Ridge moment score-only screens

Decision:

- Apply the same many-chain native batch pattern used by exact PLS score-only
  screens to Ridge operator-moment AOM screens.
- Keep the scoring rule exact: every chain/lambda/fold job still fits Ridge
  from train moments and scores held-out SSE from held-out moments.

Changes:

- Added `score_ridge_moment_sweeps_score_only(...)`, which flattens
  `n_chains * n_lambdas * n_folds` Ridge moment jobs and uses the OpenMP
  `N4M_PARALLEL_FOR_STATIC` fallback when that build option is enabled.
- Added `n_ridge_moment_score_batch_calls` and
  `n_ridge_moment_score_batch_jobs` to sweep and AOM sweep results, and exposed
  the Ridge CV/final counters through AOM results.
- Routed Ridge-only AOM `score_only=True` operator-moment screens through one
  batch scorer when every chain is moment-eligible, with the existing
  materialized fallback kept for `moment_policy="auto"`.
- Added C++ internal coverage comparing batch Ridge scores against single-chain
  scoring, plus Python coverage for `n4m.aom_chain_sweep_run(...)` Ridge-only
  score-only batch counters.

Validation:

- `/home/delete/.venv/bin/cmake --build build/dev-release --target n4m_internal_tests -j2`
  passed.
- `/home/delete/.venv/bin/ctest --test-dir build/dev-release -R n4m_internal_tests --output-on-failure`
  passed.
- `/home/delete/.venv/bin/cmake --build build/dev-release --target n4m_tests -j2`
  passed.
- `/home/delete/.venv/bin/ctest --test-dir build/dev-release -R '^n4m_tests$' --output-on-failure`
  passed.
- `/home/delete/.venv/bin/cmake --build build/omp-on --target n4m_tests -j2`
  passed.
- `/home/delete/.venv/bin/ctest --test-dir build/omp-on -R n4m_internal_tests --output-on-failure`
  passed.
- `/home/delete/.venv/bin/ctest --test-dir build/omp-on -R '^n4m_tests$' --output-on-failure`
  passed.
- `/home/delete/.venv/bin/cmake --build build/cuda-on --target n4m_tests -j2`
  passed.
- `CUDA_VISIBLE_DEVICES=0 /home/delete/.venv/bin/ctest --test-dir build/cuda-on -R n4m_internal_tests --output-on-failure`
  passed.
- `CUDA_VISIBLE_DEVICES=0 /home/delete/.venv/bin/ctest --test-dir build/cuda-on -R '^n4m_tests$' --output-on-failure`
  passed.
- `PYTHONPATH=bindings/python/src N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so /home/delete/.venv/bin/python -m pytest bindings/python/tests/test_moment_model_wrappers.py -k 'ridge_score_only_uses_batch_moment_path' -q`
  passed.
- A manual Python smoke comparing Ridge `moment_policy="force_moments"` batch
  scoring against `moment_policy="materialized"` on 3 chains x 3 lambdas x 4
  folds reported `batch_calls=1`, `batch_jobs=36`, 9 candidates and max score
  difference `7.4e-16`.

## 2026-06-05 — Parallelize CPU fallback PLS1 moment fold fits

Decision:

- Treat the non-CUDA `fit_pls1_moment_prefixes_for_folds` loop as part of the
  fast preprocessing-screen path: batched exact-CV scoring already groups the
  fold work, but the host fallback still fitted each fold serially.
- Use the existing `N4M_PARALLEL_FOR_STATIC` abstraction so default builds remain
  serial and bit-for-bit scoped to the existing OpenMP option.

Changes:

- Added an OpenMP-annotated fallback loop over folds in `cpp/src/core/sweep.cpp`.
- Avoided sharing the ABI `Context` across OpenMP workers: each fold gets a
  local `Context`, and the first status/error is replayed to the caller after
  the parallel region.
- Preserved the existing many-fold CUDA path and the `used_cuda_*` flag
  semantics.
- Strengthened the internal sweep tests so exact PLS1 batch scoring and GCV
  proxy batch scoring must report the expected one-call/many-job counters.

Validation:

- `cmake --build build/dev-release --target n4m_tests -j2` passed.
- `ctest --test-dir build/dev-release -R n4m_internal_tests --output-on-failure`
  passed.
- `ctest --test-dir build/dev-release -R '^n4m_tests$' --output-on-failure`
  passed.
- `cmake --build build/omp-on --target n4m_tests -j2` passed, compiling the
  annotated `sweep.cpp` path with `N4M_WITH_OPENMP=ON`.
- `ctest --test-dir build/omp-on --output-on-failure` passed.
- After adding the counter assertions, targeted
  `cmake --build build/dev-release --target n4m_internal_tests -j2` and
  `ctest --test-dir build/dev-release -R n4m_internal_tests --output-on-failure`
  passed.
- The same targeted internal test passed in `build/omp-on`.
- The same targeted internal test passed in `build/cuda-on` with
  `CUDA_VISIBLE_DEVICES=0`.
- `cmake --build build/cuda-on --target n4m_tests -j2` passed.
- `CUDA_VISIBLE_DEVICES=0 ctest --test-dir build/cuda-on --output-on-failure`
  passed.
- Quick local timing smoke, exact PLS score-only with 8 folds and identical
  scores:
  - `512x384`, components `[1,2,4,8]`: dev-release serial median `44.8 ms`,
    `omp-on` with `OMP_NUM_THREADS=4` median `39.3 ms`.
  - `640x768`, components `[1,2,4,8]`: dev-release serial median `259.3 ms`,
    `omp-on` with `OMP_NUM_THREADS=4` median `246.2 ms`.

Remaining work:

- This parallelizes independent CPU fold fits. It is not fused IKPLS and does
  not reduce the algebraic cost per fold/component; large 100k-200k PP screens
  still need CUDA many-chain/fused kernels or a validated proxy screen to reach
  the intended latency.

## 2026-06-05 — Expose `aom_pop.aom_preprocessing` in `n4m`

Decision:

- Treat `aom_pop.aom_preprocessing` as part of the requested AOM product
  surface instead of leaving it reachable only through older `pls4all` internals
  or through selectors that use it indirectly.
- Add an ABI-close, NumPy-first helper rather than a low-level `ctx/bank/gate`
  wrapper: callers pass the same AOM operator specs used by the selector/sweep
  APIs and receive the native MethodResult fields.

Changes:

- Added FFI declarations for `n4m_gating_strategy_create`,
  `n4m_gating_strategy_destroy` and `n4m_aom_preprocess_fit`.
- Added `n4m.python.aom_preprocess(X, y=None, operators=..., gating_mode=...)`
  returning `transformed`, `operator_outputs`, `weights`, `operator_kinds`,
  `n_operators`, `n_samples`, `n_features` and `mode`.
- Re-exported it as `n4m.aom_preprocess` and `n4m.aom.aom_preprocess`; added it
  to `n4m.aom.available_methods()` with `catalog_id:
  aom_pop.aom_preprocessing`.
- Added the Python catalog binding for `aom_pop.aom_preprocessing` in both the
  split YAML and `catalog/methods.yaml`.
- Added smoke coverage for identity hard/soft gating and generated wheel
  smoke coverage including the optional-dependency child process.
- Updated the end-user `docs/methods/aom_preprocess.md` page and methods index
  so the documented product entry is `n4m.aom_preprocess` /
  `n4m.aom.aom_preprocess`, with the legacy lower-level bindings framed as
  compatibility surfaces.

Validation:

- Targeted AOM/moment facade, wrapper and optional-dependency tests passed, 55
  tests.
- Full Python binding tests passed, 296 tests with the 4 pre-existing UVE
  warnings.
- Generated package smokes passed: `nirs4all-methods` 2 tests, `pls4all` 1
  test.
- `catalog/scripts/validate.py --strict-abi` passed, 198 methods and 698/698
  exported `n4m_*` symbols covered.
- The six touched catalog entries are synchronized between `catalog/methods.yaml`
  and their split `catalog/methods/*.yaml` files.
- One-GPU CUDA facade smoke passed with `CUDA_VISIBLE_DEVICES=0`; the report now
  includes `aom_preprocess` identity coverage against the CUDA build in addition
  to moment/AOM PLS1 device-CV counters.
- `ruff check` on changed Python files and `git diff --check` passed.

Remaining work:

- The helper is an ABI-close operator-bank primitive. It proves availability in
  CPU and CUDA builds, but it is not a fused GPU preprocessing kernel.
- Superseded later on 2026-06-05: the catalog timing benchmark for
  `aom_pop.aom_preprocessing` now covers the direct strict-linear single-
  operator bank; strict chains and model scoring remain in AOM sweep/campaign
  helpers.
- Batched IKPLS, arbitrary-chain moment screening, fused CUDA/grouped kernels
  and optional WASM/WebGPU backends remain separate backlog items.

## 2026-06-05 — Fill direct AOM/moment Python bindings in the catalog

Decision:

- Close the remaining catalog discoverability gap for AOM/moment methods that
  were already callable from Python but still had `bindings: {}` in their YAML
  catalog entries.
- Keep the catalog binding pointed at the ABI-close `n4m.python` function. The
  sklearn estimators remain documented by the facade inventory as reusable
  wrappers through `wrapper_of`.

Changes:

- Added Python catalog bindings for `aom_pop.aom_pls`,
  `aom_pop.pop_pls`, `models.pls.cppls`,
  `models.regularized.continuum_regression` and `models.specialized.ecr` in
  both the split YAML files and `catalog/methods.yaml`.
- Preserved public aliases with `legacy_aliases` for `aom_pls` and `pop_pls`.
- Strengthened `test_aom_moment_facade.py`: every catalog-backed facade entry
  must now have a Python binding in the catalog, that binding must resolve in
  `n4m.python`, and alias-role facade rows must be declared as catalog legacy
  aliases.

Validation:

- Targeted facade/optional-dependency tests passed, 10 tests.
- Full Python binding tests passed, 295 tests with the 4 pre-existing UVE
  warnings.
- Generated package smokes passed: `nirs4all-methods` 2 tests, `pls4all` 1
  test.
- `catalog/scripts/validate.py --strict-abi` passed, 198 methods and 698/698
  exported `n4m_*` symbols covered.
- The five touched method entries are synchronized between `catalog/methods.yaml`
  and their split `catalog/methods/*.yaml` files. The global split check still
  reports two unrelated stale files (`augmentation.edge_artifacts.edge_artifacts`
  and `diagnostics.approximate_press`), so those were left untouched.
- One-GPU CUDA facade smoke passed again with `CUDA_VISIBLE_DEVICES=0`.
- `ruff check` on the changed Python facade test and `git diff --check` passed.

Remaining work:

- Batched IKPLS, arbitrary-chain moment screening, fused CUDA/grouped kernels
  and optional WASM/WebGPU backends remain separate backlog items.

## 2026-06-05 — Facade inventory catalog/doc traceability

Decision:

- Harden the product surface now exposed by `n4m.aom` and `n4m.moment`: an
  advertised method should not only resolve as Python, it should also point back
  to the catalog/doc artifact when it is a catalog-backed method.
- Keep this as metadata-only Python work. No ABI symbol, C++ implementation,
  numerical behavior, CUDA routing, or generated wheel dependency changed.

Changes:

- Added `catalog_id`, `catalog_role`, `wrapper_of` and `doc_path` metadata to
  the AOM/moment `available_methods()` inventories where applicable.
- Marked sklearn-style reusable wrappers as wrappers around the existing native
  catalog binding instead of inventing new catalog method IDs.
- Left `moment.backend_recommendation` as a non-catalog helper with an explicit
  `non_catalog_reason`.
- Extended `bindings/python/tests/test_aom_moment_facade.py` to assert facade
  entries still export/resolve, catalog files exist and contain the expected
  `method_id`, docs exist, and catalog Python bindings are coherent with direct
  entries or declared wrappers.

Validation:

- Targeted source tests: `test_aom_moment_facade.py` +
  `test_sklearn_optional.py` passed, 10 tests.
- Full Python binding tests passed, 295 tests with the 4 pre-existing UVE
  warnings.
- Generated package smokes passed: `nirs4all-methods` 2 tests, `pls4all` 1
  test.
- `catalog/scripts/validate.py --strict-abi` passed, 198 methods and 698/698
  exported `n4m_*` symbols covered.
- One-GPU CUDA facade smoke passed with `CUDA_VISIBLE_DEVICES=0`: moment PLS1
  reported 4 CUDA device-CV fits and AOM chain PLS1 reported 8 CUDA device-CV
  fits, with zero host/materialized PLS CV fits.
- `ruff check` on the changed Python files and `git diff --check` passed.

Remaining work:

- Batched IKPLS, arbitrary-chain moment screening, fused CUDA/grouped kernels
  and optional WASM/WebGPU backends remain separate backlog items.

## 2026-06-05 — Prove the optional-scikit-learn runtime contract for the core/AOM/moment surface

Decision:

- Product-hardening only, scoped to the Python packaging contract: the
  `nirs4all-methods` (`n4m`) and slim `pls4all` wheels declare **only `numpy`**
  as a runtime dependency and advertise scikit-learn/SciPy as optional. Verify
  the contract holds for `import n4m` + the `n4m.aom` / `n4m.moment` facades and
  prove it with a focused regression artifact.
- Do not add scikit-learn (or SciPy) as a runtime dependency, do not add ABI
  symbols, do not change numerical results, and do not touch the experimental
  CUDA many-chain path. Do not weaken the existing strong generated wheel smoke.

What I checked:

- Static audit of `bindings/python/src/n4m`: the only module-scope optional
  imports are `n4m/sklearn/_compat.py` (`from sklearn.base import ...` inside a
  `try/except` with a dependency-light `BaseEstimator`/`TransformerMixin`
  fallback) and a lazy, `try/except`-guarded `from scipy.optimize import minimize`
  inside `aom_portfolio._solve_simplex_qp` (projected-gradient fallback). Every
  other `sklearn`/`scipy` hit is a string literal, docstring, or the
  `__sklearn_is_fitted__` protocol method name — no hard import.
- Empirical verification with `sklearn` **and** `scipy` blocked at the meta-path
  level, loading the dev-release library via `N4M_LIB_PATH`
  (`build/dev-release/cpp/src/libn4m.so` — its SONAME filename is a stale
  `1.18.0` but the compiled content reports `0.98.0+abi.1.20.0`, which the
  `_ffi.py` ABI floor of 1.20 accepts). `import n4m`, `import n4m.aom`,
  `import n4m.moment` all succeed; `n4m.sklearn._compat.BaseEstimator` is the
  in-use base (`__module__ == "n4m.sklearn._compat"`); the direct function
  facades (`moment.moments/ridge/cppls/ecr`, `aom.aom_pls`,
  `aom/moment.available_methods`) run; and `NativeRidgeRegressor.fit/predict`
  reproduces the native ridge head exactly on the fallback base.

Finding:

- No import breakage — the runtime contract holds. But it was **unproven**:
  every test environment (the dev venv and the cibuildwheel test stage, which
  sets `CIBW_TEST_REQUIRES: pytest numpy scikit-learn`) installs scikit-learn,
  so a future hard `import sklearn`/`import scipy` at module scope would silently
  break the "works with NumPy alone" contract with nothing to catch it.

Changes (smallest regression artifact, no new dependencies):

- Added `bindings/python/tests/test_sklearn_optional.py`: spawns a child
  interpreter with `sklearn` and `scipy` blocked, then asserts the core import,
  the fallback `BaseEstimator`, the AOM/moment direct-function facades, a
  scikit-learn-style wrapper `fit/predict`, and `AOMRidgeBlender` import all work.
- Added `test_core_import_without_sklearn()` to the generated `nirs4all-methods`
  wheel smoke (`bindings/python/scripts/make_python_package.py`, `n4m` branch
  only) so the **shipped wheel** proves the same contract in a child interpreter
  with both optional deps blocked. The existing `test_import_and_load()` (which
  intentionally exercises the sklearn-wrapper facade) is unchanged; the slim
  `pls4all` smoke stays import/ABI-only. No runtime dependency was added —
  pyproject stays `numpy`-only.

Validation:

- `pytest bindings/python/tests`: 291 passed (was 290; +1), 4 pre-existing UVE
  warnings.
- Generated `nirs4all-methods` package smoke (`pytest
  bindings/python_nirs4all_methods/tests`): 2 passed (existing strong test +
  new optional-dep test).
- Slim `pls4all` package smoke: 1 passed, still import/ABI-only (unaffected).
- `ruff check` on `test_sklearn_optional.py` and `make_python_package.py`: all
  checks passed.
- Negative control: a throwaway package with an **unguarded** module-scope
  `import sklearn` produces a non-zero child exit under the block, confirming the
  guard bites (the test is not vacuous).

Note:

- scikit-learn remains optional (only the guarded `_compat.py` import) and SciPy
  remains optional (only the lazy, guarded `aom_portfolio` SLSQP path). The
  generated `bindings/python_nirs4all_methods/` tree is gitignored and rebuilt by
  `release-wheels.yml`; only the generator template was edited.

## 2026-06-05 — n4m.aom facade pop_pls wrapper gap + facade-consistency guards

Decision:

- The `n4m.aom` and `n4m.moment` facades are thin re-export layers over the
  single `libn4m` runtime. Their advertised `available_methods()` inventory must
  stay in lockstep with what the facade actually re-exports — an inventory entry
  that names a wrapper the facade does not export is an integration defect.
- Pure integration-layer fix. Do not add ABI symbols, change numerical results,
  or touch the experimental CUDA many-chain batched path. CPU and CUDA behavior
  are unchanged.

Finding and fix:

- `n4m.aom` advertised the `pop_pls` method with `entry="NativePOPPLSRegressor"`
  (`kind="sklearn_estimator"`), but the facade did not re-export that class:
  `n4m.aom.NativePOPPLSRegressor` raised `AttributeError`, while its sibling
  `n4m.aom.NativeAOMPLSRegressor` and the top-level `n4m.NativePOPPLSRegressor`
  both resolved. It was the only one of the 12 `n4m.aom` + 13 `n4m.moment`
  inventory entries that did not resolve as a facade attribute.
- Added `NativePOPPLSRegressor` to the `n4m.aom` import block and `__all__`
  (`bindings/python/src/n4m/aom/__init__.py`).

Regression guards (smallest missing integration artifact):

- Strengthened the generated `nirs4all-methods` wheel smoke test
  (`bindings/python/scripts/make_python_package.py`) to assert that every
  `n4m.aom` / `n4m.moment` inventory entry resolves as a facade attribute, so the
  generated wheel can never ship a facade that names a wrapper it does not export.
- Updated `release-wheels.yml` so that the strengthened generated-wheel smoke
  installs `scikit-learn` as a test dependency; the n4m package keeps sklearn
  optional at runtime, but this smoke intentionally exercises the sklearn wrapper
  facade.
- Added `bindings/python/tests/test_aom_moment_facade.py`: every inventory entry
  resolves and is exported in `__all__`; each facade entry is the *same object*
  as the underlying `n4m.python` / `n4m.sklearn` surface (thin-facade contract);
  the POP-PLS wrapper is shared across the `n4m.aom` facade, top-level `n4m`, and
  the sklearn layer.

Validation:

- `make_python_package.py --all` regenerates cleanly; generated
  `python_nirs4all_methods/src/n4m` is byte-identical to `bindings/python/src/n4m`.
- `pytest bindings/python/tests`: 290 passed (was 285; +5 facade tests).
- Generated `nirs4all-methods` package smoke test: 1 passed with the strengthened
  all-entries-resolve assertion.
- `catalog/scripts/validate.py --strict-abi`: PASS, 198 methods, 698/698 exported
  `n4m_*` symbols.
- `ruff check` on the three changed files: all checks passed.
- Functional smoke through the facades on libn4m ABI 1.20.0:
  `n4m.aom.NativePOPPLSRegressor` and `n4m.aom.NativeAOMPLSRegressor` fit/predict
  and `n4m.moment.ridge` runs.

Note:

- The bundled `bindings/python/src/n4m/lib/libn4m.so.1.18.0` is stale, untracked
  dev cruft. The ABI floor in `_ffi.py` (1.20) correctly rejects it and the
  loader falls back to the fresh `build/dev-release` 1.20.0 library; release
  wheels rebuild `libn4m`, so the stale dev copy never ships. Left as-is
  (out of scope, untracked).

## 2026-06-05 - Experimental CUDA PLS many strided-batched path

Decision:

- The exact PLS score-only many-chain dispatch now proves batch shape via
  counters, but the default CUDA implementation still runs each fold/chain job
  sequentially with a reused device workspace.
- A true production fix probably needs device-resident scalar reductions and
  small glue kernels, because the current host-compiled CUDA dispatch cannot
  remove all per-job cuBLAS scalar synchronizations.
- Add a safe cuBLAS-only prototype behind an opt-in environment flag, but do
  not make it the default until it is measurably faster.

Implementation:

- Added a tiled `pls1_moment_components_many` prototype that packs multiple
  independent moment jobs on one GPU and batches the dominant `C*w` and
  covariance-deflation `p^2` operations with `cublasDgemmStridedBatched`.
- Preserved the legacy sequential-many path as the default and as fallback.
  Enable the prototype with `N4M_CUDA_PLS_MANY_BATCHED=1`; keep
  `N4M_CUDA_PLS_MANY_LEGACY=1` as an explicit fallback override.
- Added `N4M_CUDA_PLS_BATCH_MAX_BYTES` to cap tile memory. Public ABI and
  Python signatures are unchanged.
- Added a public-surface C++ regression test that compares `n4m_sweep_run`
  default CUDA PLS many-job scoring against the opt-in batched mode when a CUDA
  backend is available.

Validation:

- CUDA build passed: `build/cuda-on/cpp/tests/n4m_tests` reported
  `349 passed, 0 failed` on `CUDA_VISIBLE_DEVICES=0`.
- CPU and CUDA `ctest --output-on-failure` both passed.
- Python CUDA smoke with forced device PLS (`cuda_pls_min_device_features=1`)
  showed default legacy vs opt-in batched max score diff
  `1.1102230246251565e-16`, identical selected candidate and `20/20` device
  CV fits.
- Timing smoke on 16 chains x 4 folds, `p=768`, PLS components 1..4, showed
  the cuBLAS-only batched prototype slightly slower than default legacy
  (`2327 ms` vs `2220 ms` median). Keep opt-in only.

## 2026-06-05 - Many-chain PLS batch dispatch counters

Decision:

- The exact-CV and GCV-proxy PLS-only AOM score-only paths already submit many
  transformed-chain moment jobs through one internal native dispatch. The
  existing counters showed total fold fits/proxy fits, but not whether those
  fits came from one many-chain batch dispatch or from per-chain calls.
- Add audit counters only. Do not change scores, route selection, candidate
  rows, or CUDA scheduling.

Implementation:

- Added `n_pls_moment_score_batch_calls` and
  `n_pls_moment_score_batch_jobs` to the exact-CV PLS moment score-only path.
- Added `n_pls_gcv_proxy_batch_calls` and
  `n_pls_gcv_proxy_batch_jobs` to the PLS GCV proxy score-only path.
- Propagated those scalars through `SweepResult`, `AomSweepResult`,
  MethodResult packing, Python result dictionaries, campaign aggregation,
  sklearn diagnostics, and timing CSV scripts.

Validation:

- Native C++ tests passed: `349 passed, 0 failed`.
- CTest passed: 2/2 tests.
- Targeted Python wrapper/tests passed: `44 passed`.
- Python smoke confirmed a PLS-only exact screen over 3 chains and 4 folds
  reports `batch_calls=1`, `batch_jobs=12`, and the proxy screen reports
  `batch_calls=1`, `batch_jobs=3`.

## 2026-06-05 - Direct native moment sklearn wrappers

Decision:

- Direct native Ridge, CPPLS, continuum regression and ECR were available as
  ABI-close Python functions, but not as reusable sklearn-style estimators.
  That left individual winning heads harder to reuse outside the global AOM
  sweep/refit wrappers.
- Add thin wrappers over the native MethodResults. Do not add new selection
  logic, new preprocessing routes or any out-of-moment method.

Implementation:

- Added `NativeRidgeRegressor`, `NativeCPPLSRegressor`,
  `NativeContinuumRegressionRegressor` and `NativeECRRegressor`.
- The wrappers replay predictions from input-space coefficients. Ridge uses
  the native intercept, while CPPLS, continuum and ECR reconstruct the
  intercept as `y_mean - x_mean @ coefficients`.
- Re-exported the classes through `n4m.sklearn`, top-level `n4m` and
  `n4m.moment`.
- Added `ridge_regressor`, `cppls_regressor`,
  `continuum_regression_regressor` and `ecr_regressor` rows to the moment
  method inventory.
- Added `benchmarks/cross_binding/bench_direct_moment_heads_timing.py` to time
  direct native functions and sklearn wrapper `fit+predict` replay overhead.

Validation:

- Added targeted tests for exact replay against native train predictions,
  sklearn coefficient/intercept shapes, diagnostics and multi-output support.
- Added generated-package smoke coverage for `NativeRidgeRegressor`.

## 2026-06-05 - Reusable AOM wrapper diagnostics

Decision:

- The native reusable AOM wrappers already replay predictions from folded
  input-space coefficients, but their diagnostics were still mostly numeric.
  For end-user reuse and incremental preprocessing campaigns, reports should
  expose source-free labels for the selected head and preconfigured bank.
- Add audit-only diagnostics. Do not change scoring, routing, candidate grids,
  or fitted coefficients.

Implementation:

- Added `profile_name` and `expected_bank_size` to diagnostics for
  `NativeAOMSweepRegressor`, `NativeAOMRobustHPORegressor`,
  `NativeAOMRidgeBlenderRegressor` and
  `NativeAOMOperatorPLSStackRegressor`.
- Added `selected_head` (`"ridge"` or `"pls"`) to
  `NativeAOMRobustHPORegressor` diagnostics.
- The expected preconfigured bank sizes are `12` for compact and `31` for
  wide; custom chain wrappers keep their existing explicit chain diagnostics.

Validation:

- Python syntax check passed for `native_sweeps.py`.
- Targeted Python wrapper tests passed, covering compact diagnostics, wide
  diagnostics and prediction replay.

## 2026-06-05 - FCK parity across native wide AOM portfolio banks

Decision:

- The configurable AOM sweep/campaign wide banks already carried Gaussian,
  FCK and Whittaker strict-linear variants. The native robust-HPO, Ridge
  blender and operator-PLS stack wide profiles still stopped at Gaussian plus
  Whittaker, so the preconfigured reusable methods did not reproduce the full
  strict-moment diversity available to the global screen.
- Add only FCK strict-linear variants. Stateful or nonlinear preprocessing
  families such as SNV, MSC, EMSC, OSC, EPO or baseline-centering remain out
  of these exact-moment native banks.

Implementation:

- Added `N4M_OP_FCK` alpha `0.0` and `1.0` to the wide native banks for:
  `aom_robust_hpo`, `aom_ridge_blender` and `aom_operator_pls_stack`.
- Wide native banks now align at 31 strict-linear chains/operators with the
  public `build_aom_strict_chain_grid("wide")` / AOM sweep profile.
- Updated catalog notes and method docs to state compact `12` vs wide `31`
  and to name Gaussian/FCK/Whittaker variants explicitly.

Validation:

- Rebuilt `build/dev-release --target n4m_tests`.
- Native test binary passed: `349 passed, 0 failed`.
- Added C++ tests proving wide robust-HPO, Ridge blender and operator-PLS
  stack expose 31 chains/operators after the FCK additions.

## 2026-06-05 - Public inventory config options for AOM/moment methods

Decision:

- The AOM and moment facades exposed public method inventories, but users still
  had to inspect long docs or signatures to know which entries accept global
  chain grids, checkpointing, exact/proxy refit controls, and CUDA PLS route
  knobs.
- Add audit-only `config_options` metadata to each inventory row. This is
  discoverability for end users and scripts, not a selector, router or scoring
  policy.

Implementation:

- `n4m.aom.available_methods()` now includes `config_options` for screen/refit
  presets, chain/profile sweeps, fixed-candidate reuse, Ridge blending,
  operator-PLS stacking, robust-HPO and legacy AOM/POP selectors.
- `n4m.moment.available_methods()` now includes `config_options` for raw
  moments, train-from-heldout moments, `sweep_run`, `NativeMomentSweepRegressor`,
  direct Ridge/CPPLS/continuum/ECR heads and backend recommendation.
- CUDA PLS controls (`cuda_pls_parallel_folds`,
  `cuda_pls_min_device_features`) are listed only on the moment/AOM screen
  surfaces that accept them.
- Updated the generated `nirs4all-methods` package smoke-test template so
  package regeneration preserves the inventory contract.

Validation:

- Python syntax check passed for the AOM facade, moment facade and package
  generator.
- Targeted facade tests passed and assert JSON-serializable inventories plus
  the expected CUDA/screen/refit options.
- Regenerated `bindings/python_nirs4all_methods`; generated package import
  smoke passed.

## 2026-06-05 - Configurable CUDA PLS moment device threshold

Decision:

- The PLS1 moment CUDA component loop was guarded by a hard-coded
  `p >= 1024` threshold. That is conservative, but it makes medium-width NIRS
  screens hard to test on GPU without recompiling.
- Expose the threshold as an explicit positive config option while keeping the
  default at 1024. This is a benchmark/control knob, not a new scoring rule.

Implementation:

- Added `n4m_config_set_cuda_pls_min_device_features` /
  `n4m_config_get_cuda_pls_min_device_features` and bumped the C ABI to 1.20.0.
- `sweep_run`, AOM sweep calls, AOM campaigns/refits/fixed fits, sklearn
  wrappers and screen/refit presets accept `cuda_pls_min_device_features`.
- Campaign fingerprints, reports and CPU/CUDA backend recommendations now
  include the threshold, so checkpoint resume cannot mix different GPU route
  configurations.
- `bench_moment_sweep_timing.py` and
  `bench_aom_screen_refit_scaling.py` accept
  `--cuda-pls-min-device-features`.

Validation:

- `build/dev-release --target n4m_c` rebuilt successfully after the ABI/config
  change.
- `build/cuda-on --target n4m_c` rebuilt successfully after the same change.
- Python source files and benchmark scripts pass `py_compile`.
- Targeted Python tests passed for backend recommendation, CPU score
  preservation and sklearn preset diagnostics.
- Full Python binding suite passed: `279 passed, 4 warnings`.
- Native CTest passed for `n4m_tests` and `n4m_internal_tests`.
- Catalog checks passed: legacy split check, reference validation and strict
  ABI coverage (`698/698` exported `n4m_*` symbols).
- Regenerated `bindings/python_nirs4all_methods` and the generated package
  import smoke passed.
- Smoke benchmark CLIs accepted the new flag and emitted
  `cuda_pls_min_device_features=256` in
  `moment_sweep_timing_min_device_smoke.csv` and
  `aom_screen_refit_min_device_smoke.csv`.
- CUDA smoke with `CUDA_VISIBLE_DEVICES=0`, `p=256` and
  `cuda_pls_min_device_features=256` switched PLS moment CV counters from
  host to device (`4 -> 0` host, `0 -> 4` device) while preserving candidate
  scores against the default threshold.

## 2026-06-05 - Configurable bounded CUDA PLS fold scheduling

Decision:

- The CUDA PLS1 moment helper already had an environment-driven profiling
  path for independent fold jobs, but it was not a reusable method option and
  could launch one thread/stream per job.
- Expose it as an explicit config/API option while keeping the old environment
  override for backwards-compatible profiling.
- Bound the stream-parallel work to small batches on the single selected GPU.
  This is still exact fold-CV over moment designs, not fused batched IKPLS.

Implementation:

- Added `n4m_config_set_cuda_pls_parallel_folds` /
  `n4m_config_get_cuda_pls_parallel_folds` and bumped the C ABI to 1.19.0.
- Added `cuda_pls_parallel_folds` to `sweep_run`, `aom_sweep_run`,
  `aom_chain_sweep_run`, `aom_chain_score_campaign`,
  `aom_refit_candidates`, `aom_chain_screen_refit_campaign`, the native
  sklearn sweep wrappers and the moment screen/refit presets.
- CUDA `pls1_moment_components_many(...)` now accepts the explicit flag,
  preserves `N4M_CUDA_PLS_PARALLEL_FOLDS=1` as an override, and runs bounded
  stream/cuBLAS batches instead of unbounded one-stream-per-job launches.
- Added `n_pls_moment_cuda_parallel_fold_batches` and
  `n_pls_moment_cuda_parallel_fold_jobs` counters to sweep, AOM sweep,
  campaign, refit and benchmark reports.
- Extended `bench_moment_sweep_timing.py` and
  `bench_aom_screen_refit_scaling.py` with `--cuda-pls-parallel-folds`.

Validation:

- Rebuilt `build/dev-release` and `build/cuda-on` targets `n4m_c`.
- ABI smoke from Python reports `(1, 19, 0)` and both new config symbols are
  exported from `build/dev-release/cpp/src/libn4m.so`.
- Targeted tests passed: the option is accepted, CPU-path candidate scores are
  unchanged, parallel CUDA counters remain zero off the CUDA-parallel path, and
  the mixed moment preset exposes the requested setting in diagnostics.

## 2026-06-05 - Split-head mixed campaign scoring to unlock PLS batching

Decision:

- The native exact/proxy PLS batch hook only activates when a score-only AOM
  call is PLS-only. Mixed Ridge+PLS chunk calls can therefore miss that faster
  PLS branch even when every chain is moment-eligible.
- Add an optional campaign-level split that scores Ridge and PLS in separate
  native calls per chunk, then merges candidate rows back into the same
  global/per-head/per-route top-k aggregation. This is not a new score or
  selector; it only changes chunk launch shape.
- Keep historical behavior as `split_head_scoring="off"` for generic calls.
  The mixed end-user moment preset uses `split_head_scoring="auto"`.

Implementation:

- Added `split_head_scoring` to `aom_chain_score_campaign`,
  `aom_chain_screen_refit_campaign` and `NativeAOMScreenRefitRegressor`.
- Added `split_head_scoring="auto"` to
  `NativeAOMMomentScreenRefitRegressor`.
- Split chunks expose `split_head_scoring`, `n_split_head_chunks` and
  `n_chunk_score_calls` in screen/campaign diagnostics.
- Extended `bench_aom_screen_refit_scaling.py` with `--split-head-scoring` and
  matching CSV columns.

Validation:

- Targeted source tests passed: split vs non-split mixed campaign scores match
  by `(chain_id, head, param)`, and the mixed preset reports the split
  diagnostics.
- Smoke benchmark generated
  `benchmarks/cross_binding/aom_mixed_screen_refit_split_smoke.csv`; the row
  reports `n_split_head_chunks=2`, `n_chunk_score_calls=4`, and preserved exact
  refit behavior on the tiny mixed case.

## 2026-06-05 - Public method inventories for AOM and moments

Decision:

- Add a source-free inventory surface to the logical facades so scripts and
  users can discover which entries are intended for global preprocessing
  campaigns, selected-winner reuse, direct moment heads and CPU/CUDA launch
  planning.
- Keep this as metadata only. It must not introduce a new selector, router or
  dataset-dependent policy.

Implementation:

- Added `n4m.aom.available_methods()`.
- Added `n4m.moment.available_methods()`.
- Each function returns JSON-safe dictionaries by copy, with fields for
  `name`, `entry`, `kind`, `role`, `heads`, `reuse`, `cpu`, `cuda` and `notes`.
- Updated the generated `nirs4all-methods` package smoke test to assert the
  key AOM and moment inventory entries are present.

Validation:

- Python syntax check passed for both facades, the generator and updated tests.
- Targeted source tests passed for the moment facade and AOM moment
  screen/refit presets, including inventory copy semantics.
- Regenerated `bindings/python_nirs4all_methods`, staged `libn4m.so`, and the
  generated package smoke test passed without `N4M_LIB_PATH`.

## 2026-06-05 - CUDA facade smoke for AOM and moments

Decision:

- Add a targeted GPU smoke for the new logical facades rather than relying only
  on import tests or broad timing CSVs. The smoke must prove that `n4m.aom` and
  `n4m.moment` still alias the shared runtime and that their wide PLS1 screens
  actually use the CUDA device-CV path.

Implementation:

- Added `benchmarks/cross_binding/aom_moment_cuda_facade_smoke.py`.
- The script spawns a child Python process with `N4M_LIB_PATH` pointed at
  `build/cuda-on/cpp/src/libn4m.so` and `CUDA_VISIBLE_DEVICES=0`.
- It runs `moment.sweep_run(...)` and `aom.aom_chain_sweep_run(...)` on an
  `80x1024`, 4-fold, PLS1 score-only case with `moment_policy="force_moments"`
  for the AOM route.
- It writes `benchmarks/cross_binding/aom_moment_cuda_facade_smoke.json`.

Validation:

- The smoke passed on the current CUDA build, ABI `1.18.0`.
- `moment.sweep_run` reported 4 CUDA-device PLS moment CV fits, 0 host PLS
  moment CV fits and 0 materialized PLS CV fits.
- `aom.aom_chain_sweep_run` over two chains reported 8 CUDA-device PLS moment
  CV fits, 0 host PLS moment CV fits and 0 materialized PLS CV fits.

## 2026-06-05 - Logical moment facade inside nirs4all-methods

Decision:

- Keep the historical `n4m.moments(...)` function unchanged.
- Add a singular `n4m.moment` namespace for discoverability, because a
  `n4m.moments` module would shadow the existing function after submodule
  import.
- Keep the facade as a thin alias layer over `n4m.python` and `n4m.sklearn`;
  there is still one C++/CUDA runtime and one binding stack.

Implementation:

- Added `bindings/python/src/n4m/moment/__init__.py` exposing
  `moments`, `moments_train_from_heldout`, `moment_screen_backend_recommendation`,
  `sweep_run`, `ridge`, `cppls`, `continuum_regression`, `ecr`, and
  `NativeMomentSweepRegressor`.
- Exported the facade as `n4m.moment` while keeping top-level function imports
  backwards-compatible.
- Updated source tests and generated package smoke tests to prove the facade
  aliases existing functions and does not shadow `n4m.moments`.

Validation:

- Python syntax check passed for the facade, top-level export, generator and
  updated tests.
- Source import smoke passed: `n4m.moment` aliases the existing native
  functions while `n4m.moments` remains callable.
- Targeted facade test passed with a real `moment.sweep_run(...)`.
- Regenerated `bindings/python_nirs4all_methods`, staged `libn4m.so`, and the
  generated package smoke test passed without `N4M_LIB_PATH`.

## 2026-06-05 - Logical AOM facade inside nirs4all-methods

Decision:

- Keep the C++/CUDA runtime and bindings in the single `nirs4all-methods`
  package. AOM currently shares operators, moments, Ridge/PLS kernels, catalog
  gates, ABI tracking and wheel staging with the rest of `libn4m`; extracting
  it into a second native package would duplicate release and ABI work.
- Add a logical Python boundary instead: `n4m.aom` is the dedicated public
  import surface for AOM campaigns, refit helpers, historical AOM portfolio
  classes and native sklearn AOM presets.

Implementation:

- Added `bindings/python/src/n4m/aom/__init__.py` as a thin facade over
  `n4m.python` and `n4m.sklearn`. It does not own a second implementation.
- Exported the facade as `n4m.aom` while keeping all existing top-level and
  `n4m.sklearn` imports backwards-compatible.
- Updated the generated `nirs4all-methods` package smoke test so wheels prove
  the facade is present and aliases the embedded runtime functions/classes.

Validation:

- Python syntax check passed for the new facade, top-level export, package
  generator and updated test.
- Source import smoke passed: `n4m.aom` aliases the top-level campaign helper
  and native sklearn preset class.
- Targeted mixed moment screen/refit preset test passed.
- Regenerated `bindings/python_nirs4all_methods`, staged `libn4m.so`, and the
  generated package smoke test passed without `N4M_LIB_PATH`.

## 2026-06-05 - End-user moment screen/refit sklearn presets

Decision:

- Keep the generic `NativeAOMScreenRefitRegressor` as the ultra-configurable
  experiment surface, but add named sklearn presets for the two common reusable
  workflows: mixed Ridge/PLS screen -> exact-CV refit, PLS proxy screen ->
  exact-CV refit, and Ridge exact screen -> exact-CV refit.
- Do not add a new scoring rule. Both presets delegate to the existing
  campaign/refit/final-only fixed-candidate path.
- For mixed campaigns, keep exact refit train-only but avoid relying only on a
  global screen top-k when Ridge exact-CV rows and PLS proxy rows share the
  same list. Include an optional per-head retention budget before exact refit.

Implementation:

- Changed `aom_chain_score_campaign` so `top_candidates_by_head` and
  `top_candidates_by_score_route` are collected from each chunk's full score
  table, not only from that chunk's global top rows.
- Added `refit_per_head_top_k` to `aom_chain_screen_refit_campaign` and
  `NativeAOMScreenRefitRegressor`. It exact-refits the deduplicated union of
  global top rows plus each head's top rows and reports the global/per-head
  union counters.
- Added `NativeAOMMomentScreenRefitRegressor`, the mixed Ridge/PLS preset. It
  uses exact Ridge CV plus PLS GCV proxy in the screen, then exact-CV refits
  the global/per-head retained union.
- Added `NativeAOMMomentPLSScreenRefitRegressor`, fixing
  `heads=("pls",)`, `ridge_lambdas=()`, `pls_score_mode="gcv_proxy"`,
  `moment_policy="force_moments"` and prefix-aware chain ordering.
- Added `NativeAOMMomentRidgeScreenRefitRegressor`, fixing
  `heads=("ridge",)`, `pls_components=()`, `moment_policy="force_moments"`
  and prefix-aware chain ordering.
- Re-exported the preset classes from `n4m.sklearn` and top-level `n4m`.
- Updated method docs, method index discoverability, and the catalog entry for
  `aom_pop.aom_chain_screen_refit`; its benchmark pointer now uses the focused
  screen/refit scaling benchmark.

Validation:

- Python syntax check passed for the source and regenerated
  `nirs4all-methods` package files.
- Targeted preset test passed, and the full moment wrapper test file passed:
  35 tests.
- Catalog gates passed: 198/198 split files up to date, 198/198 reference
  coverage and 694/694 ABI symbol coverage.
- Regenerated `bindings/python_nirs4all_methods`, staged the CPU
  `libn4m.so`, verified the staged hash matches the build lib, and smoke-fit
  both new presets from the package without `N4M_LIB_PATH`.
- Strengthened the generated `nirs4all-methods` package smoke test so wheel
  tests import the mixed AOM moment preset, run a tiny mixed Ridge/PLS
  screen-refit fit through the embedded library, and audit the retained
  candidate pool. The slim `pls4all` package keeps its import/ABI-only smoke.
- Added source-free `n4m.aom_screen_refit_candidate_pool(...)` and validated
  that the mixed preset refits the retained union of global and per-head rows.
- Extended `bench_aom_screen_refit_scaling.py` with `--head mixed` and
  `--refit-per-head-top-k`, then regenerated CPU and CUDA-smoke CSVs for PLS,
  Ridge and mixed runs. On the current CPU mixed run with 24 chains,
  `refit_per_head_top_k=4` retains 8 exact-refit candidates even when
  `refit_top_k=1`, and projects the 200k-chain mixed screen at about 169.4 s
  on the measured 260x48 shape. The CUDA smoke completes on one GPU and reports
  the same retained-pool counters.

Limit:

- These are ergonomic presets over the existing CPU/CUDA-capable substrate.
  They are not the fused CUDA/IKPLS grinder.
- Because the presets default to `moment_policy="force_moments"`, they reject
  small fold geometries or regimes that would need a materialized fallback.
  The generic `NativeAOMScreenRefitRegressor` remains the escape hatch for
  `moment_policy="auto"` production runs.

## 2026-06-05 - Screen/refit scaling refresh after PLS proxy batching

Decision:

- Refresh the proxy-screen plus exact-refit timing artefacts after batching the
  AOM PLS GCV proxy path. These CSVs are the direct benchmark for the intended
  large preprocessing campaign workflow.

Validation:

- Regenerated `aom_screen_refit_scaling.csv` and
  `aom_screen_refit_scaling_cuda_smoke.csv` for PLS proxy screen -> exact refit.
- Regenerated the matching Ridge artefacts,
  `aom_ridge_refit_scaling.csv` and `aom_ridge_refit_scaling_cuda_smoke.csv`,
  so all screen/refit CSVs share the current schema and auto-refit fields.
- PLS CPU screen/refit scaling now reports 24 chains, 48 proxy candidates,
  24 proxy fits, 2.22 ms screen median, and a 200k-chain screen projection of
  about 13.9 s before exact refit of retained rows.
- PLS auto refit keeps the plan-driven behavior: `top_k=1/2` selects
  `batched_score`, while `top_k=4/8/16` selects `union_batched_score` with
  explicit extra-score counters.
- Ridge rows remain on the exact Ridge screen path and keep the same auto-refit
  semantics; CPU screen projection is about 184.9 s for 200k chains on the
  measured 260x48 shape.

## 2026-06-05 - Batched AOM PLS GCV proxy score-only path

Decision:

- Apply the same many-chain native batching pattern to explicit
  `pls_score_mode="gcv_proxy"` AOM PLS screens, because this is the cheapest
  first-pass route for very large preprocessing campaigns.
- For proxy screens, transform only all-sample moments. Held-out moments are
  not needed by the GCV proxy and are no longer transformed in the PLS-only
  batch branch.
- Keep the existing exact-CV and refit semantics unchanged: GCV remains an
  explicit proxy score and must still be followed by exact-CV refit/evaluation
  for retained candidates.

Implementation:

- Added internal `score_pls1_gcv_moment_screens_score_only(...)`, the many-chain
  form of `score_pls1_gcv_moment_screen(...)`.
- Reused the existing PLS1 prefix fitter over all transformed all-sample moment
  sets, then mapped the nominal GCV RMSE back to one `SweepResult` per chain.
- Routed AOM PLS-only score-only proxy campaigns through the batch branch when
  all chains have an operator-moment route. Fallback behavior is unchanged for
  non-forced calls.

Validation:

- Rebuilt CPU and CUDA native targets.
- CTest CPU: 2/2 tests passed.
- CTest CUDA with `CUDA_VISIBLE_DEVICES=0`: 2/2 tests passed.
- Internal C++ test compares the GCV batch hook against single-chain scoring
  and passed in both builds.
- Targeted Python AOM PLS exact/proxy test passed on CPU and CUDA.
- Full Python CPU suite passed: 275 tests, 4 existing UVE warnings.
- Catalog checks passed: 198/198 references, 694/694 ABI symbols, and
  198/198 per-method files up to date.
- Regenerated `aom_sweep_timing.csv` and
  `aom_sweep_timing_cuda_smoke.csv`. CPU proxy rows now report 5 chains,
  10 candidates, 0 materialized candidates, and 5 proxy fits at 0.22 ms
  for 60x32, 0.30 ms for 80x48, and 0.44 ms for 96x64, projecting about
  8.8 s, 12.0 s, and 17.7 s for 200k chains on those shapes.
- CUDA large proxy smoke at 260x1024 with 2 chains completed with 0
  materialized candidates, 2 proxy fits, and 0 exact CV fits.
- Regenerated the `nirs4all-methods` Python package, staged the rebuilt CPU
  `libn4m.so`, and smoke-tested the packaged proxy PLS score-only path without
  `N4M_LIB_PATH`.

Limit:

- This is still a GCV proxy screen, not exact CV. It improves the cheap recall
  screen path but does not remove the need for exact refit/evaluation of the
  retained candidates.

## 2026-06-05 - Batched exact-CV AOM PLS score-only path

Decision:

- Add a native internal batch hook for exact-CV PLS1 moment screens over many
  transformed strict-linear chains, without changing the public C ABI.
- Use it only for AOM `heads=("pls",)`, exact-CV, `score_only=True` campaigns
  when every chain has an operator-moment route. If a chain needs fallback and
  `force_moments` is not set, keep the previous per-chain path.
- Keep the selected-model reuse path unchanged: score-only campaigns still
  select candidates, and final reusable fits are handled by the existing fixed
  candidate/final-only APIs.

Implementation:

- Added `score_pls1_moment_sweeps_score_only(...)` as a C++ internal helper.
  It builds all train-fold moment stats for `n_chains * n_folds` jobs, calls
  the existing prefix PLS1 moment fitter once, then maps exact held-out SSE
  back to one `SweepResult` per chain.
- Added an AOM PLS-only score-only branch that transforms all chain moments,
  calls the batch helper once, preserves candidate rows, route counters,
  selected ids, fold ids, and moment-prefix cache counters.
- Added a dedicated benchmark row
  `native_aom_chain_sweep_pls_exact_score_only` to separate exact-CV
  score-only PLS from the GCV proxy row.

Validation:

- Rebuilt CPU and CUDA native targets.
- CTest CPU: 2/2 tests passed.
- CTest CUDA with `CUDA_VISIBLE_DEVICES=0`: 2/2 tests passed.
- Internal C++ test compares the batch PLS1 moment hook against single-chain
  scoring and passed in both builds.
- Targeted Python exact/proxy AOM PLS test passed on CPU and CUDA.
- Full Python CPU suite passed: 275 tests, 4 existing UVE warnings.
- Catalog checks passed: 198/198 references, 694/694 ABI symbols, and
  198/198 per-method files up to date.
- Regenerated `aom_sweep_timing.csv` and
  `aom_sweep_timing_cuda_smoke.csv`. The new exact score-only rows report
  5 chains, 10 candidates, 0 materialized candidates, and 25 exact PLS moment
  CV fits. CPU medians were 0.58 ms at 160x32 and 1.25 ms at 240x48, projecting
  about 23.4 s and 49.8 s for 200k chains on those small shapes.
- CUDA large smoke at 260x1024 with 2 chains used the device path:
  `n_pls_moment_cuda_device_cv_fits=10`, host CV fits 0.
- Regenerated the `nirs4all-methods` Python package, staged the rebuilt CPU
  `libn4m.so`, and smoke-tested the packaged exact PLS score-only path without
  `N4M_LIB_PATH`.

Limit:

- This batches exact-CV PLS scoring across transformed chains at the native
  helper level. It still materializes/derives transformed moments per chain
  and is not yet a fused device-resident IKPLS kernel over the whole Cartesian
  preprocessing grid.

## 2026-06-05 - Reusable CUDA PLS1 moment workspace

Decision:

- Keep the current exact PLS1 moment algorithm unchanged, but remove repeated
  CUDA allocation churn from successive PLS1 moment component calls.
- This is a substrate improvement for large AOM PLS screens: many strict
  preprocessing chains still invoke the PLS1 moment component helper
  chain-by-chain, so reusing device buffers is a necessary step before a true
  fused/batched IKPLS grinder.
- Keep the opt-in parallel-fold profiling path on local per-thread workspaces;
  the reusable workspace is for the default single-stream CUDA path.

Implementation:

- Added a thread-local reusable device workspace for `dC`, `ds`, `dw`, `dcw`,
  `dp_load`, `dW`, and `dP`, grown to the largest `(p, max_components)` seen
  by the calling thread.
- Routed both `pls1_moment_components` and the default
  `pls1_moment_components_many` path through the same workspace view. The
  numerical cuBLAS loop is unchanged.
- Preserved the existing local-stream allocation path used by
  `N4M_CUDA_PLS_PARALLEL_FOLDS=1`.

Validation:

- Rebuilt the CUDA library; the CPU build was unchanged by the guarded file.
- CTest CPU: 2/2 tests passed.
- CTest CUDA with `CUDA_VISIBLE_DEVICES=0`: 2/2 tests passed.
- Targeted CUDA Python tests for PLS proxy/backend/refit paths passed:
  2 passed.
- CUDA smoke for exact PLS1 `sweep_run` at `260x1024` reported
  `n_pls_moment_cuda_device_cv_fits=5` and host CV fits 0.
- CUDA smoke for AOM PLS GCV proxy at `260x1024` completed with 2 proxy fits
  and 4 proxy candidates.
- Regenerated `moment_gpu_crossover.csv`; at `256x1024`, PLS median timing was
  232.92 ms on CPU and 109.02 ms on CUDA, with CUDA recommended and
  `device_cv=5`.
- `git diff --check` passed.

Limit:

- This removes repeated CUDA workspace allocation overhead. It is still not a
  fused/device-resident batched IKPLS implementation across chains.

## 2026-06-05 - Plan-driven AOM exact-refit auto mode

Decision:

- Keep exact-CV refit selection train-only and deterministic, but make
  `execution_mode="auto"` use the same candidate-grouping plan exposed by
  `aom_refit_execution_plan`.
- Prefer `union_batched_score` only when it reduces native refit groups and its
  extra exact scores stay within `auto_max_extra_fraction *
  n_retained_candidates`; otherwise keep `batched_score`, which preserves the
  retained parameter signatures.
- Keep `return_predictions=True` on the individual replay path because the
  batched score modes deliberately return scores only.

Implementation:

- Added plan sharing for `individual`, `grouped_score`, `batched_score`, and
  `union_batched_score`, plus a validated `auto_max_extra_fraction` budget.
- Propagated `refit_auto_max_extra_fraction` through
  `aom_chain_screen_refit_campaign`, `NativeAOMScreenRefitRegressor`, and the
  screen/refit scaling benchmark.
- Added requested/selected execution-mode metadata and the auto-selection
  reason to refit reports, two-pass campaign reports, diagnostics, and timing
  CSV rows.

Validation:

- Python syntax check passed for the modified implementation, wrapper,
  benchmark, and test files.
- Targeted PLS proxy/screen-refit and Ridge exact-refit tests passed on CPU and
  CUDA.
- Full Python CPU test suite passed: 275 tests, 4 existing UVE warnings.
- Catalog checks passed: 198/198 references, 694/694 ABI symbols, and 198/198
  per-method files up to date.
- Regenerated `aom_screen_refit_scaling.csv` and
  `aom_screen_refit_scaling_cuda_smoke.csv`; `auto` rows now report requested
  mode, selected mode, selection reason, and planned-vs-observed scored/extra
  refit counters.
- Regenerated the matching Ridge screen/refit timing CSVs,
  `aom_ridge_refit_scaling.csv` and `aom_ridge_refit_scaling_cuda_smoke.csv`,
  with the same auto-mode fields.
- Regenerated the `nirs4all-methods` Python package, staged the bundled CPU
  `libn4m.so`, and smoke-tested package loading plus Ridge auto refit without
  `N4M_LIB_PATH`.
- `git diff --check` passed.

## 2026-06-05 - Prefix-aware campaign chunk ordering

Decision:

- Keep the native ABI unchanged and improve large Python campaign packing first.
- Add an explicit `chain_ordering="prefix"` mode that sorts strict-linear chains
  by operator-prefix key before chunking, so chains sharing prefixes are more
  likely to hit the native per-call moment-prefix cache.
- Keep `chain_ordering="input"` as the default. Prefix ordering is a throughput
  optimization only: it does not alter scores, and reported `chain_id` remains
  the original caller/grid index.

Implementation:

- Added prefix-order helpers to `aom_chain_score_campaign`.
- Added `ordered_chain_id` on top-candidate rows and
  `selected_ordered_chain_id` / ordered chunk bounds on chunk summaries.
- Propagated the option through `aom_chain_screen_refit_campaign`,
  `NativeAOMScreenRefitRegressor`, and the screen/refit scaling benchmark.
- Added benchmark CSV fields for screen prefix-cache hits, misses and hit
  fraction.

Validation:

- Targeted prefix-order, strict-grid and checkpoint tests passed on CPU and
  CUDA.
- Full Python CPU tests passed: 275 tests, 4 existing UVE warnings.
- Catalog checks passed: 198/198 references, 694/694 ABI symbols, and
  198/198 per-method files up to date.
- Regenerated `aom_screen_refit_scaling.csv` and
  `aom_screen_refit_scaling_cuda_smoke.csv` with chain-ordering and
  prefix-cache columns.
- Regenerated the `nirs4all-methods` Python package and smoke-tested
  `chain_ordering="prefix"` from the package-bundled CPU library.
- On a deliberately interleaved 24-chain Ridge screen with chunk size 4,
  prefix ordering preserved the best score while increasing prefix-cache hits
  from 0 to 12. Median CPU timing improved from 6.25 ms to 5.41 ms; CUDA-smoke
  timing improved from 18.95 ms to 17.83 ms.

## 2026-06-05 - Gaussian strict banded AOM operator

Decision:

- Add a constrained Gaussian operator to the strict AOM moment family despite
  the additive enum extension, because it is linear, shape-preserving and can be
  represented exactly by the same banded route as the other local convolution
  operators.
- Keep the scope narrow: this is the fixed zero-padding AOM screen variant, not
  the full SciPy-compatible `pp_gaussian` preprocessing surface.

Implementation:

- Added `N4M_OP_GAUSSIAN = 18` and accepted it in strict AOM operator banks.
- Added normalized Gaussian kernel construction for the materialized strict
  transform and for the banded operator-moment descriptor.
- Added Gaussian variants to native wide banks, Python `wide`/`lab` strict chain
  grids, robust-HPO/operator-stack/ridge-blender native wide profiles, and the
  timing benchmark.

Validation:

- Rebuilt CPU and CUDA native test targets.
- CTest CPU: 2/2 tests passed.
- CTest CUDA with `CUDA_VISIBLE_DEVICES=0`: 2/2 tests passed.
- Targeted Gaussian/FCK campaign tests passed on CPU and CUDA.
- Full Python CPU tests passed: 274 tests, 4 existing UVE warnings.
- Catalog checks passed: 198/198 references, 694/694 ABI symbols, and
  198/198 per-method files up to date.
- Regenerated `aom_sweep_timing.csv` and
  `aom_sweep_timing_cuda_smoke.csv`; the Gaussian rows report 3 chains,
  15 candidates, all 15 through banded operator moments, and 0 materialized
  candidates.
- Regenerated the `nirs4all-methods` Python package and smoke-tested
  `build_aom_strict_chain_grid("wide") == 31` plus a Gaussian
  `force_moments` route from the package-bundled CPU library.

## 2026-06-05 - FCK variants in strict moment AOM grids

Decision:

- At this point Gaussian was deferred because it required extending the public
  `n4m_operator_kind_t` enum. The following pass adds the constrained
  enum-backed variant above.
- Use the already-integrated `N4M_OP_FCK` instead: it is strict-linear,
  shape-preserving, and already has a banded operator-moment route.

Implementation:

- Added two FCK variants to the native wide AOM bank and to
  `build_aom_strict_chain_grid("wide")`.
- Added `fck` as a default lab/cartesian family with single-op and
  FCK-plus-finite-difference templates.
- Added Python campaign coverage proving an FCK chain can be screened under
  `moment_policy="force_moments"` with banded moment route counters.
- Added `native_aom_chain_sweep_fck` rows to
  `bench_aom_sweep_timing.py`.

Validation:

- Rebuilt CPU and CUDA native test targets.
- CTest CPU: 2/2 tests passed.
- CTest CUDA with `CUDA_VISIBLE_DEVICES=0`: 2/2 tests passed.
- Targeted FCK campaign tests passed on CPU and CUDA.
- Full Python CPU tests passed: 274 tests, 4 existing UVE warnings.
- Regenerated `aom_sweep_timing.csv` and
  `aom_sweep_timing_cuda_smoke.csv`; the FCK rows report 3 chains,
  15 candidates, all 15 through banded operator moments, and 0 materialized
  candidates.

## 2026-06-05 - AOM campaign backend launch diagnostics

Decision:

- The score campaigns had route counters proving what actually ran, but the
  report did not expose the CPU/CUDA launch recommendation that should be used
  before importing `n4m`.
- Keep this strictly diagnostic and source-free: the policy inputs are only
  `n_samples`, `n_features`, `head`, and CUDA availability. Dataset names,
  targets, and candidate scores are not policy inputs.

Implementation:

- Added per-head `moment_backend_recommendations` to
  `aom_chain_score_campaign` and `aom_chain_screen_refit_campaign`.
- Added optional `backend_cuda_available` to the campaign helpers and
  `NativeAOMScreenRefitRegressor`, so an external launcher can report that a
  CUDA build is available even when the current process loaded a CPU libn4m.
- Left checkpoint fingerprints unchanged because the backend recommendation
  does not affect scoring, ranking, or the retained candidate rows.
- `NativeAOMScreenRefitRegressor.get_diagnostics()` now exposes the same
  backend recommendation block.

Validation:

- Targeted Python tests passed on CPU and CUDA for backend recommendation,
  score-campaign checkpoint/resume, screen-refit, and prefix-cache counters.

## 2026-06-05 - CUDA PLS1 W/P block transfer

Decision:

- The device-resident PLS1 loop still copied `W` and `P` back to host once
  per component. Those matrices are only consumed after all components are
  available for prefix reconstruction.
- Keep the same exact PLS1 math but store component weights/loadings in
  device-side `dW/dP` buffers during the loop, then copy the full row-major
  `W/P` blocks back once.

Implementation:

- Added `dW` and `dP` device workspaces to the CUDA PLS1 component helper.
- Each component stores `dw` and `p_load` into strided columns of `dW/dP`
  using cuBLAS `Dcopy`; host `W/P` are copied once after the deflation loop.
- The same path is used by the default reused fold workspace and by the
  opt-in `N4M_CUDA_PLS_PARALLEL_FOLDS=1` profiling path.

Observed smoke output:

- Regenerated `moment_gpu_crossover.csv`.
- Default CUDA PLS `256x1024` now reports about 112 ms with
  `host_cv=0`, `device_cv=5`, improving from the previous ~129 ms smoke.
- Opt-in parallel folds improved too, to about 122 ms on the same shape, but
  remains slower than the default single-workspace route.

Validation:

- Rebuilt CPU and CUDA test binaries.
- CTest CPU: 2/2 tests passed.
- CTest CUDA with `CUDA_VISIBLE_DEVICES=0`: 2/2 tests passed.

Remaining work:

- This removes avoidable host/device transfers for `W/P`. The remaining
  synchronisation points are scalar control values (`nrm2`, sign, `dot`) and
  prefix reconstruction on host. A true batched/fused IKPLS kernel still needs
  those scalar/control paths to stay device-side longer.

## 2026-06-05 - CUDA PLS1 parallel-fold probe

Decision:

- After the exact-CV fold workspace, the obvious next question was whether
  independent PLS1 CV folds should run concurrently on one GPU.
- Keep scoring exact, single-GPU only, and avoid making an unmeasured slower
  route the product default.

Implementation:

- Refactored the CUDA PLS1 component loop so it can run on a caller-provided
  cuBLAS handle and CUDA stream.
- Added an opt-in profiling path behind
  `N4M_CUDA_PLS_PARALLEL_FOLDS=1`: each fold job gets a non-blocking stream,
  its own cuBLAS handle, and its own device workspace on device 0.
- The default `pls1_moment_components_many(...)` path remains the measured
  single-workspace fold loop. Public scores, candidate tables and route
  counters are unchanged.

Observed smoke output:

- Default `moment_gpu_crossover.csv` PLS CUDA rows still report `device_cv=5`
  only for `256x1024`; that row timed at about 129 ms in this pass.
- Opt-in `N4M_CUDA_PLS_PARALLEL_FOLDS=1` on the same benchmark timed the
  `256x1024` PLS row at about 135 ms, so it is not the default route.

Validation:

- Rebuilt CPU and CUDA test binaries.
- CTest CPU: 2/2 tests passed.
- CTest CUDA with `CUDA_VISIBLE_DEVICES=0`: 2/2 tests passed.

Remaining work:

- This confirms that simple host-thread/stream fold parallelism is not enough
  at the current shape. The real performance work remains a device-side
  batched IKPLS/fused kernel design that avoids host scalar synchronisation
  inside each component.

## 2026-06-05 - CUDA PLS1 exact-CV fold workspace

Decision:

- The very-wide PLS1 CUDA component loop was still allocated and torn down
  once per CV fold.
- Exact PLS1 folds are independent, but true stream-parallel IKPLS needs a
  larger device-side scalar/control refactor. Add a smaller exact-CV
  improvement first: reuse one CUDA workspace across folds while keeping the
  same per-fold numerical loop and public result schema.

Implementation:

- Added internal `cuda_dispatch::pls1_moment_components_many(...)`.
- The helper processes several fold-local moment designs with one reusable
  `C/s/w/Cw/p_load` device workspace. It is sequential over fold jobs, so it
  is not a claim of parallel or fused batched IKPLS.
- Added `fit_pls1_moment_prefixes_for_folds(...)` in the native sweep layer.
  CUDA builds use it for multi-fold PLS1 moment CV when `p >= 1024` and a GPU
  is available; medium-width CUDA and CPU builds keep the previous fold loop.
- `score_pls1_moment_sweep(...)` and direct `run_moment_sweep(...)` both use
  the fold-workspace helper for compatible exact-CV PLS1 moment screens.
- Updated `n4m.moment_screen_backend_recommendation(...)` to report
  `uses_cuda_pls_fold_workspace` and moved the policy tag to
  `n4m.moment_gpu_crossover.v3`.

Observed smoke output:

- Regenerated `moment_gpu_crossover.csv`.
- The current CUDA PLS rows report `device_cv=5` only for `256x1024`, as
  expected. That row timed at about 127 ms in this smoke run. Smaller PLS
  rows remain on the host loop inside the CUDA build.

Validation:

- Rebuilt CPU and CUDA test binaries.
- CTest CPU: 2/2 tests passed.
- CTest CUDA with `CUDA_VISIBLE_DEVICES=0`: 2/2 tests passed.
- Targeted Python wrapper tests: 33 passed on CPU and 33 passed on CUDA.

Remaining work:

- This removes repeated CUDA workspace allocation for large exact-CV PLS1
  folds. It still does not implement stream-parallel CV, fused batched IKPLS,
  or the full 200k-chain CUDA grinder.

## 2026-06-05 - PLS1 moment host/device route counters

Decision:

- After adding the very-wide CUDA PLS1 moment component loop, users need a
  runtime proof of which route actually ran, not only a launch-time
  recommendation.

Implementation:

- Added `n_pls_moment_host_cv_fits`,
  `n_pls_moment_cuda_device_cv_fits`,
  `n_pls_moment_host_final_fits`, and
  `n_pls_moment_cuda_device_final_fits` to `SweepResult` and
  `AomSweepResult`.
- Packed those scalars through the C MethodResult ABI and Python wrappers.
- Aggregated the counters through AOM campaigns/refits and exposed them in
  sklearn diagnostics plus timing CSV scripts.
- Added C++/Python assertions that medium-width PLS moment screens stay on the
  host route, while the wide `p=1024` CUDA test reports device CV fits.

Validation:

- Covered by the CPU/CUDA CTest and Python wrapper validation in this pass.

## 2026-06-05 - CUDA PLS1 wide moment component loop

Decision:

- The existing CUDA build avoided cuBLAS for PLS1 moment micro-kernels because
  per-operation host/device copies were slower than scalar host loops.
- A device-resident component loop is worthwhile only once the feature width is
  large enough to amortize the single `C/s` upload and the per-component cuBLAS
  calls.

Implementation:

- Added internal `cuda_dispatch::pls1_moment_components(...)`.
- The helper copies the centered/scaled moment covariance `C` and score vector
  `s` to device once, then runs the PLS1 component loop through cuBLAS
  `copy/scal/nrm2/idamax/gemv/dot/ger/axpy` operations while keeping the
  deflated `C` and `s` on device.
- `fit_pls1_moment_prefixes(...)` now uses that CUDA helper only for
  `p >= 1024`. Below that, CUDA builds keep the previous scalar host loop,
  which remains faster on measured medium-width screens.
- Added C++ coverage with a wide `p=1024` PLS score-only moment sweep; in CUDA
  builds this exercises the device-resident component loop without changing
  the public ABI.
- Updated `n4m.moment_screen_backend_recommendation(...)` to report
  `uses_cuda_pls_device_component_loop` and the `p=1024` activation threshold.

Observed smoke output:

- Regenerated `moment_gpu_crossover.csv`.
- CUDA PLS speedup vs CPU is about 0.23x at `260x48`, 1.32x at `260x256`,
  3.21x at `512x512`, and 2.18x at `256x1024`.
- The `256x1024` PLS row improved from the previous live smoke by moving the
  component loop onto device; the medium-width rows keep the faster scalar
  host loop.

Validation:

- Rebuilt CPU and CUDA test binaries.
- CTest CPU: 2/2 tests passed.
- CTest CUDA with `CUDA_VISIBLE_DEVICES=0`: 2/2 tests passed.
- Targeted Python wrapper tests: 33 passed on CPU and 33 passed on CUDA.

Remaining work:

- This is a real device-resident PLS1 inner loop for wide moment screens. It
  is still not the full fused batched IKPLS/200k-chain CUDA grinder across
  many preprocessing variants.

## 2026-06-05 - Score-only moment screen copy elision and 200k projection diagnostics

Decision:

- Broad preprocessing screens spend most of their time in score-only
  candidate ranking. In a pure moment route, copying full `X/Y` just to score
  candidates is unnecessary.
- Keep exact scores unchanged, but avoid full matrix copies for score-only
  moment Ridge/PLS screens and expose throughput/projection diagnostics so
  grind campaigns can estimate whether a profile is viable at 200k chains.

Implementation:

- `run_moment_sweep(...)` now materializes full `X/Y` copies lazily. Non
  score-only runs still force copies before producing OOF/final predictions;
  materialized and dual paths still force copies when needed.
- Ridge moment score-only folds now compute held-out SSE from held-out
  moments via `ridge_heldout_sse_from_moments(...)`, matching the internal
  `score_ridge_moment_sweep(...)` route and avoiding row prediction replay.
- Direct `n4m.sweep_run(...)` Ridge now reports route/fit counters:
  `n_ridge_moment_candidates`, `n_ridge_dual_materialized_candidates`,
  `n_ridge_moment_cv_fits`, `n_ridge_dual_materialized_cv_fits`,
  `n_ridge_dual_cross_cv_fits`, `n_ridge_moment_final_fits`, and
  `n_ridge_dual_materialized_final_fits`.
- `aom_chain_score_campaign` performance metrics now include
  `projected_200k_chains_seconds` and `projected_200k_chains_minutes`.
- `bench_aom_sweep_timing.py` and `bench_aom_screen_refit_scaling.py` now
  emit chain/candidate throughput and 200k-chain projections in CSV output.
  `bench_moment_sweep_timing.py` now emits the direct Ridge route counters.

Observed smoke output:

- CPU `aom_sweep_timing.csv`: first compact row reports about 1534
  chains/s, projected 200k chains in about 130 s.
- CPU PLS screen/refit scaling: first screen row reports about 5802
  chains/s, projected 200k chains in about 34 s.
- CPU Ridge screen/refit scaling: first screen row reports about 1270
  chains/s, projected 200k chains in about 157 s.
- Direct wide Ridge score-only smoke reports 2 dual-materialized Ridge
  candidates, 6 dual-cross CV fits and zero final dual fits, confirming that
  score-only skips final model construction while retaining exact CV scores.
- CUDA-smoke CSVs still validate the CUDA build path, but remain slower on
  these small shapes because the current CUDA backend is cuBLAS dispatch with
  host/device transfers, not a fused device-resident grinder.

Validation:

- Rebuilt `build/dev-release` and `build/cuda-on` targets `n4m_c`.
- CTest on `build/dev-release`: 2/2 tests passed.
- CTest on `build/cuda-on` with `CUDA_VISIBLE_DEVICES=0`: 2/2 tests passed.
- Targeted Python wrapper tests on CPU and CUDA: 31 passed on each build.
- Regenerated `aom_sweep_timing.csv`, `aom_sweep_timing_cuda_smoke.csv`,
  `aom_screen_refit_scaling.csv`, `aom_screen_refit_scaling_cuda_smoke.csv`,
  `aom_ridge_refit_scaling.csv`, and
  `aom_ridge_refit_scaling_cuda_smoke.csv`, plus
  `moment_sweep_timing.csv` and `moment_sweep_timing_cuda_smoke.csv`.

Remaining work:

- This improves exact/proxy score-only screening overhead and observability.
  It is still not the fused batched IKPLS/CUDA implementation needed for a
  true device-resident 200k-chain PLS grinder.

## 2026-06-05 - Live CPU/CUDA moment sweep crossover benchmark

Decision:

- The CUDA build is not uniformly faster: small screens lose to CPU because
  the current backend uses cuBLAS calls with host/device transfers, while
  larger moment sweeps can amortise that overhead.
- Add a live crossover benchmark so CPU vs CUDA choice is measured from the
  actual `sweep_run` paths rather than inferred from proxy BLAS diagnostics.

Implementation:

- Added `benchmarks/cross_binding/bench_moment_gpu_crossover.py`.
- The script spawns separate child interpreters for CPU and CUDA builds,
  because the Python binding loads one libn4m shared object per process.
- It times score-only Ridge and PLS `n4m.sweep_run` on identical synthetic
  datasets, records route counters, and emits `speedup_vs_cpu` plus a
  `recommended_backend` field.
- Added `n4m.moment_screen_backend_recommendation(...)` as a source-free
  launch-planning helper over the measured crossover. It uses only
  `n_samples`, `n_features`, `head`, and CUDA availability, and reports when
  the caller must start a fresh process with the other libn4m build.
- Updated the older `cuda_diagnostic.py` text so it remains a broad proxy
  classifier and points live moment-sweep timing to the new benchmark.

Observed smoke output:

- `260x48`: CPU wins; CUDA speedup vs CPU was about 0.26x for Ridge and
  0.17x for PLS.
- `260x256`: CUDA starts winning; about 1.23x for Ridge and 1.15x for PLS.
- `512x512`: CUDA wins; about 1.36x for Ridge and 3.01x for PLS.
- `256x1024`: CUDA wins; about 2.53x for Ridge and 2.03x for PLS.

Validation:

- Generated `benchmarks/cross_binding/moment_gpu_crossover.csv` from the live
  CPU and CUDA builds with `CUDA_VISIBLE_DEVICES=0`.
- Added Python wrapper tests for the 260x48 CPU recommendation, the 260x256
  CUDA recommendation, unavailable-CUDA fallback, and bad input validation.

Remaining work:

- This provides a measured CPU/CUDA routing baseline for the current
  cuBLAS-dispatch implementation. It is still not a fused batched IKPLS CUDA
  grinder.

## 2026-06-05 - Native final-only fixed AOM fit ABI

Decision:

- The screen -> exact-CV refit workflow already verifies the selected row by
  exact train CV.
- Rebuilding the reusable selected model through
  `NativeAOMFixedCandidateRegressor` still replayed a one-candidate CV sweep
  only to obtain final coefficients/predictions.
- Add a native final-only fit path for already-selected AOM
  chain/head/parameter rows. It must not be used for ranking and must preserve
  the exact-CV score from the refit report at the wrapper layer.

Implementation:

- Added internal `run_moment_final_fit(...)` for one fixed Ridge or PLS
  parameter. It fits on all rows, returns coefficients/intercept/predictions,
  and does not build folds or score CV.
- Added internal `run_aom_chain_fixed_fit(...)` and public C ABI
  `n4m_aom_chain_fixed_fit_run(...)`. The AOM path materializes the selected
  strict-linear chain once, fits the fixed head, then folds transformed
  coefficients back to original input space.
- Added Python `n4m.aom_chain_fixed_fit_run(...)`.
- Added catalog method `aom_pop.aom_chain_fixed_fit` for this individual
  winner reuse surface. `aom_pop.aom_chain_screen_refit` remains a
  Python-backed orchestration method with `abi_symbols: []`; its native
  building blocks are now attributed to `aom_pop.aom_chain_sweep` and
  `aom_pop.aom_chain_fixed_fit`.
- `NativeAOMFixedCandidateRegressor` now supports `fit_mode="final_only"` plus
  `precomputed_cv_rmse`. Its default remains `fit_mode="cv"` for direct
  candidate reuse from ordinary campaign rows.
- `NativeAOMScreenRefitRegressor` now uses `fit_mode="final_only"` after its
  exact-CV refit pass and injects the selected row's verified
  `refit_cv_rmse`.
- Updated `bench_aom_screen_refit_scaling.py` so reusable final-fit timings
  measure the new final-only path.

Observed smoke output:

- CPU `build/dev-release`, `refit_top_k=16`, PLS: final selected-candidate fit
  now pays `final_fit_n_pls_moment_cv_fits=0`, one final PLS fit, and takes
  about 0.26-0.30 ms instead of the previous ~0.74 ms CV-replaying fit.
- CPU `build/dev-release`, `refit_top_k=16`, Ridge: final fit takes about
  0.28 ms and pays no PLS CV fits.
- CUDA smoke with `CUDA_VISIBLE_DEVICES=0`: PLS/Ridge final-only smokes load
  `build/cuda-on/cpp/src/libn4m.so.1.18.0`, use `fit_mode="final_only"`, pay
  zero final CV fits, and keep the exact refit CV score in diagnostics.

Validation:

- Rebuilt `build/dev-release` and `build/cuda-on` targets `n4m_c`.
- CTest on `build/dev-release`: 2/2 tests passed.
- Targeted Python wrapper tests:
  `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so.1.18.0
  PYTHONPATH=bindings/python/src pytest
  bindings/python/tests/test_moment_model_wrappers.py -q`: 31 passed.
- Full Python tests with
  `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so.1.18.0`: 272 passed,
  4 existing UVE warnings.
- Added Python tests proving `n4m.aom_chain_fixed_fit_run` matches the
  one-candidate full-CV sweep's final predictions, transformed coefficients,
  folded input coefficients and intercept for both Ridge and PLS, while
  returning empty OOF/fold outputs and zero CV fit counters.
- Regenerated CPU and CUDA-smoke screen/refit scaling CSVs and verified
  planned vs observed refit group/scored/extra counters still match.
- Added `native_aom_chain_fixed_fit_pls` and
  `native_aom_chain_fixed_fit_ridge` rows to
  `benchmarks/cross_binding/bench_aom_sweep_timing.py`, then regenerated
  `aom_sweep_timing.csv` and `aom_sweep_timing_cuda_smoke.csv`. On the CPU
  smoke, fixed final fit took about 0.29-0.30 ms for PLS and 0.18-0.29 ms for
  Ridge at the covered shapes; on the CUDA smoke, about 0.68-0.92 ms for PLS
  and 0.56-0.76 ms for Ridge. All fixed-fit rows report `cv=0` and zero CV
  fit counters.
- Catalog/checks:
  `catalog/scripts/validate.py`, `catalog/scripts/reconcile_abi.py --check`,
  `catalog/scripts/split_legacy_methods.py --check`,
  `catalog/scripts/validate.py --strict-abi`, and `git diff --check` all
  passed after allowing symbol-less Python-backed orchestration methods in the
  catalog schema.
- Reference coverage gate:
  `catalog/scripts/validate.py --check-references` now classifies native
  `aom_pop` orchestration methods and the direct `models.regularized.ridge`
  head as nirs4all-donor surfaces. The gate reports 135 nirs4all-donor
  methods, 60 registry-backed methods and 3 paper-only methods, covering all
  198 catalog production entries without fabricating external references for
  native N4A methods.
- Regenerated `bindings/python_nirs4all_methods`; package smoke confirmed
  `n4m.aom_chain_fixed_fit_run` is exported from the package, the embedded
  lib loads, and `NativeAOMScreenRefitRegressor` uses final-only fixed fit with
  zero final PLS CV fits.

Remaining work:

- This removes the redundant final selected-candidate CV replay. It is still
  not a fused/device-resident batched IKPLS/GPU grinder across many chains.

## 2026-06-05 - Grouped, signature-batched and union-batched exact-CV refit scoring

Decision:

- The exact-CV second pass was still replaying every retained candidate row as
  a separate one-candidate AOM sweep.
- When retained rows share the same decoded chain and model head, their Ridge
  lambdas or PLS component counts can be scored together by the existing native
  sweep without changing CV semantics.
- Chain/head grouping still leaves one Python/native call per retained chain.
  When several retained chains share the same head and retained parameter set,
  they can be batched into one native AOM sweep call without adding candidates
  or changing any exact-CV score.
- When retained chains have different parameter subsets, a head-level union of
  retained parameters can reduce calls further. That mode may score extra
  chain/parameter pairs, so it must expose the surplus explicitly and remain an
  explicit execution mode rather than a silent selection rule.
- Add exact grouped, signature-batched and union-batched score modes before
  attempting a larger native IKPLS rewrite.

Implementation:

- Added `execution_mode` to `n4m.aom_refit_candidates`.
- `execution_mode="individual"` preserves the previous full replay with
  per-candidate train/OOF prediction arrays.
- `execution_mode="grouped_score"` groups rows by decoded chain/head and calls
  `aom_chain_sweep_run(..., score_only=True, pls_score_mode="cv")` once per
  group. It returns the same exact `refit_cv_rmse` values for ranking, but
  grouped rows do not carry per-candidate prediction arrays or finite
  `train_rmse`.
- `execution_mode="batched_score"` first builds the same chain/head retained
  parameter sets, then batches chains with identical `(head, params)` signatures
  into a single `aom_chain_sweep_run` call. Scores are mapped back by local
  `chain_id` and parameter.
- `execution_mode="union_batched_score"` batches by head with the union of
  retained parameters for that head. It maps only the requested retained rows
  back to the report and exposes `n_refit_scored_candidates` plus
  `n_refit_extra_scored_candidates` so surplus native scoring is auditable.
- `n4m.aom_refit_execution_plan(candidates, top_k=...)` uses the same grouping
  helpers as `aom_refit_candidates` and reports expected groups, native scored
  candidates and extra scored candidates for each exact score mode without
  touching `X` or `y`.
- `execution_mode="auto"` selects batched scoring unless
  `return_predictions=True`.
- `aom_chain_screen_refit_campaign` now defaults to
  `refit_execution="auto"`, so the normal proxy -> exact-CV workflow uses the
  faster signature-batched score path.
- Refit reports now expose aggregate counters:
  `execution_mode`, `n_refit_groups`, `n_pls_moment_cv_fits`,
  `n_pls_materialized_cv_fits`, `n_operator_moment_candidates`,
  `n_materialized_candidates`, `n_refit_scored_candidates`, and
  `n_refit_extra_scored_candidates`.
- `NativeAOMScreenRefitRegressor` propagates `refit_execution` and reads
  aggregate refit counters from the report.
- `NativeAOMScreenRefitRegressor.get_diagnostics()` separates final
  selected-candidate fit counters via `final_*` fields. After the final-only
  ABI work above, those fields verify that the reusable model fit no longer
  repays CV.
- The screen/refit scaling benchmark now emits `individual`, `grouped_score`,
  `batched_score` and `union_batched_score` rows and supports `--head pls` /
  `--head ridge`; CSV rows include planned and observed group/scored/extra
  counters plus reusable final-fit timing/counters.

Observed smoke output:

- CPU `build/dev-release`, `n=260`, `p=48`, `24` chains, `2` PLS components,
  `cv=5`: at `refit_top_k=16`, individual refit took 13.54 ms and 160 PLS
  CV fits; grouped score took 5.16 ms / 50 PLS CV fits / 10 groups; signature
  batched score took 2.26 ms / 50 PLS CV fits / 2 groups; union batched score
  took 1.92 ms / 50 PLS CV fits / 1 group after scoring 20 native candidates
  for 16 retained rows, with the same best exact-CV RMSE.
- CUDA smoke with `CUDA_VISIBLE_DEVICES=0`: at `refit_top_k=16`, individual
  refit took 169.52 ms and 160 PLS CV fits; grouped score took 49.94 ms /
  50 PLS CV fits / 10 groups; signature batched score took 10.54 ms /
  50 PLS CV fits / 2 groups; union batched score took 7.70 ms /
  50 PLS CV fits / 1 group after scoring 20 native candidates for 16 retained
  rows, with zero materialized refit CV fits.
- Ridge CPU smoke with `--head ridge`: at `refit_top_k=16`, individual refit
  took 18.86 ms across 16 groups; grouped score took 6.74 ms across 8 groups;
  signature batched score took 5.17 ms across 4 groups; union batched score
  took 6.79 ms across 1 group after scoring 32 native candidates for
  16 retained rows, with the same best exact-CV RMSE.
- Ridge CUDA smoke with `--head ridge` and `CUDA_VISIBLE_DEVICES=0`: at
  `refit_top_k=16`, individual refit took 156.54 ms; grouped score took
  41.49 ms; signature batched score took 11.57 ms; union batched score took
  8.18 ms after scoring 32 native candidates for 16 retained rows, again with
  the same best exact-CV RMSE.
- After the final-only ABI work, the same benchmark shows the reusable final
  selected-candidate fit cost without CV replay: for PLS at `refit_top_k=16`,
  final fit paid zero PLS CV fits plus one final PLS fit and took about
  0.26-0.30 ms on CPU / 0.55-0.57 ms on the CUDA smoke; for Ridge, final fit
  took about 0.28 ms on CPU / 0.58-1.03 ms on CUDA smoke.

Validation:

- Targeted Python wrapper tests:
  `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so.1.18.0
  PYTHONPATH=bindings/python/src pytest
  bindings/python/tests/test_moment_model_wrappers.py -q`: 30 passed.
- Regenerated `benchmarks/cross_binding/aom_screen_refit_scaling.csv` and
  `benchmarks/cross_binding/aom_screen_refit_scaling_cuda_smoke.csv`.
- Generated `benchmarks/cross_binding/aom_ridge_refit_scaling.csv` and
  `benchmarks/cross_binding/aom_ridge_refit_scaling_cuda_smoke.csv`.
- Verified that planned group/scored/extra counters match observed refit
  counters in all four scaling CSVs.
- Full Python tests with
  `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so.1.18.0`: 271 passed,
  4 existing UVE warnings.
- Rebuilt `bindings/python_nirs4all_methods`; package smokes confirmed
  `aom_chain_screen_refit_campaign` defaults to batched exact-CV refit and
  `aom_refit_candidates(..., execution_mode="union_batched_score")` exposes
  scored/extra-scored candidate counters; package smoke also confirmed
  `NativeAOMScreenRefitRegressor.get_diagnostics()` exposes `final_*`
  selected-candidate fit counters.
- CUDA-lib Python smokes with `CUDA_VISIBLE_DEVICES=0` confirmed
  `NativeAOMScreenRefitRegressor` uses batched exact-CV refit by default and
  explicit `union_batched_score` refit for PLS and Ridge on the CUDA build
  while using one visible GPU, and exposes final selected-candidate fit
  counters on the CUDA build.
- Catalog/checks:
  `catalog/scripts/validate.py`, `catalog/scripts/reconcile_abi.py --check`,
  `catalog/scripts/split_legacy_methods.py --check`, and `git diff --check`
  all passed.

Remaining work:

- This removes redundant parameter refits inside retained chain groups and
  reduces exact-CV refit orchestration overhead across retained chains with the
  same parameter signature, with an explicit union-parameter option for small
  grids. It is still not a fused batched IKPLS/GPU grinder across many chains.

## 2026-06-05 - Screen/refit PLS scaling benchmark

Decision:

- The remaining performance gap for large PLS preprocessing screens is not
  visible enough from the generic AOM smoke benchmark.
- Add a focused benchmark that varies `refit_top_k` after a fixed GCV-proxy
  screen, so future batched IKPLS/CUDA work has a concrete target: exact-CV
  refit cost per retained candidate and per fold-local PLS fit.

Implementation:

- Added `benchmarks/cross_binding/bench_aom_screen_refit_scaling.py`.
- The benchmark builds a deterministic lab chain grid, runs one
  `aom_chain_score_campaign(..., pls_score_mode="gcv_proxy",
  moment_policy="force_moments")`, then times `aom_refit_candidates` for
  `refit_top_k=1,2,4,8,16`.
- CSV rows report screen/refit timings, `n_pls_gcv_proxy_fits`,
  `n_refit_pls_moment_cv_fits`, materialized refit fit counters, cost per
  refit candidate, cost per exact PLS CV fit, and screen-vs-refit rank fields.

Observed smoke output:

- CPU `build/dev-release`, `n=260`, `p=48`, `24` chains, `2` PLS components,
  `cv=5`: screen proxy 5.61 ms for 48 candidates. Exact-CV refit scaled from
  0.96 ms for `top_k=1` / 10 PLS CV fits to 13.54 ms for `top_k=16` / 160 PLS
  CV fits. Materialized refit CV fits stayed at zero.
- CUDA smoke with `CUDA_VISIBLE_DEVICES=0`: same counters and zero
  materialized refit CV fits, but screen/refit timings were slower than CPU
  on this small shape. This confirms the current CUDA build is functional but
  still not a fused batched/device-resident IKPLS path.

Validation:

- Generated `benchmarks/cross_binding/aom_screen_refit_scaling.csv` with
  `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so.1.18.0`.
- Generated `benchmarks/cross_binding/aom_screen_refit_scaling_cuda_smoke.csv`
  with `CUDA_VISIBLE_DEVICES=0` and
  `N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so.1.18.0`.
- Catalog/checks:
  `catalog/scripts/validate.py`, `catalog/scripts/reconcile_abi.py --check`,
  `catalog/scripts/split_legacy_methods.py --check`, and `git diff --check`
  all passed.

Remaining work:

- This benchmark makes the bottleneck measurable. It does not implement the
  fused GPU/IKPLS grinder.

## 2026-06-05 - Native sklearn screen-refit AOM estimator

Decision:

- The one-call screen/refit helper made the proxy -> exact-CV workflow
  callable, but end users still needed manual glue to turn the verified winner
  into a reusable estimator.
- Add a sklearn-style method wrapper so broad preprocessing selection can be
  used as a normal fitted regressor while preserving the campaign audit
  reports.

Implementation:

- Added `NativeAOMScreenRefitRegressor`.
- `fit` runs `aom_chain_screen_refit_campaign`, stores `campaign_report_`,
  `screen_report_` and `refit_report_`, then fits the selected verified row
  through `NativeAOMFixedCandidateRegressor.from_refit_report`.
- The fitted estimator exposes `coef_`, `intercept_`, `predictions_`,
  `oof_predictions_`, `selected_chain_`, `selected_head_`,
  `selected_param_`, `selected_cv_rmse_`, `predict`, `score`, and
  campaign/refit diagnostics.
- Re-exported the class through `n4m.sklearn` and top-level `n4m`.
- Docs now show the estimator form next to the functional campaign helper.

Validation:

- Targeted Python wrapper tests:
  `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so.1.18.0
  PYTHONPATH=bindings/python/src pytest
  bindings/python/tests/test_moment_model_wrappers.py -q`: 29 passed.
- Full Python tests with
  `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so.1.18.0`: 270 passed,
  4 existing UVE warnings.
- Rebuilt `bindings/python_nirs4all_methods`; package smoke confirmed
  `n4m.NativeAOMScreenRefitRegressor` and
  `n4m.sklearn.NativeAOMScreenRefitRegressor` fit/predict successfully.
- CUDA-lib Python smoke with `CUDA_VISIBLE_DEVICES=0` confirmed the estimator
  on the CUDA build while using one visible GPU.
- Added catalog entry `aom_pop.aom_chain_screen_refit` and regenerated
  per-method files; `catalog/scripts/validate.py` reports 197 methods and
  passes closure.
- Catalog/checks:
  `catalog/scripts/reconcile_abi.py --check`,
  `catalog/scripts/split_legacy_methods.py --check`, and `git diff --check`
  all passed.

Remaining work:

- This makes the screen/refit workflow a reusable method surface. It does not
  implement the fused GPU/IKPLS grinder.

## 2026-06-05 - One-call screen -> exact-CV refit campaign helper

Decision:

- The proxy and chunked campaign workflow was available as two separate calls:
  `aom_chain_score_campaign` followed by `aom_refit_candidates`.
- Add a single orchestration helper so broad preprocessing screens can be run,
  checkpointed, and exact-CV verified with one user-facing call while still
  exposing both intermediate reports.

Implementation:

- Added `n4m.aom_chain_screen_refit_campaign(...)`.
- The helper runs the score-only campaign first, then exact-CV refits the
  retained `refit_top_k` rows via `aom_refit_candidates`.
- The combined report exposes nested `screen` and `refit` reports plus
  top-level `rows`, `best_cv`, `best_screen`, `best_refit`,
  `screen_complete`, `pls_score_mode`, and `refit_pls_score_mode`.
- Partial checkpoint screens are supported for inspection: current top rows
  are refit and the report marks `screen_complete=False`.
- Re-exported the helper at top-level `n4m.aom_chain_screen_refit_campaign`.
- Tests cover proxy GCV screen -> exact-CV refit -> direct
  `NativeAOMFixedCandidateRegressor.from_refit_report(...)` reuse.
- Added a timing benchmark row
  `native_aom_chain_screen_refit_pls_gcv_proxy` to
  `bench_aom_sweep_timing.py`, including `n_refit_candidates`,
  `n_refit_pls_moment_cv_fits`, and materialized refit CV fit counters.

Validation:

- Targeted Python wrapper tests:
  `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so.1.18.0
  PYTHONPATH=bindings/python/src pytest
  bindings/python/tests/test_moment_model_wrappers.py -q`: 29 passed.
- Full Python tests with
  `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so.1.18.0`: 270 passed,
  4 existing UVE warnings.
- Rebuilt `bindings/python_nirs4all_methods`; package smoke confirmed
  `n4m.aom_chain_screen_refit_campaign(...)` and direct
  `NativeAOMFixedCandidateRegressor.from_refit_report(...)` reuse.
- CUDA-lib Python smoke with `CUDA_VISIBLE_DEVICES=0` confirmed the same
  helper on the CUDA build while using one visible GPU.
- Regenerated `aom_sweep_timing.csv` and
  `aom_sweep_timing_cuda_smoke.csv` with the new screen-refit benchmark row
  using `--repeats 1`.
- Catalog/checks:
  `catalog/scripts/validate.py`, `catalog/scripts/reconcile_abi.py --check`,
  `catalog/scripts/split_legacy_methods.py --check`, and `git diff --check`
  all passed.

Remaining work:

- This improves campaign ergonomics. It does not implement the fused
  GPU/IKPLS grinder.

## 2026-06-05 - Fixed-candidate reuse from exact-CV refit reports

Decision:

- After adding `aom_refit_candidates`, exact-CV winners from the second pass
  could be reused manually via `from_candidate(row)`, but there was no direct
  sklearn constructor mirroring `from_campaign`.
- Add the missing reuse surface so the intended proxy -> exact-CV -> final
  model workflow is one call at each stage.

Implementation:

- Added `NativeAOMFixedCandidateRegressor.from_refit_report(report, rank=0,
  **kwargs)`.
- The constructor sorts rows by `refit_cv_rmse` regardless of the report's
  display sorting, then delegates to `from_candidate`.
- AOM sklearn diagnostics now include `n_pls_gcv_proxy_candidates`,
  `n_pls_gcv_proxy_fits`, and `aom_pls_score_mode`.
- Tests now fit a fixed candidate directly from an exact-CV refit report and
  assert its selected CV RMSE matches `report["best_cv"]["refit_cv_rmse"]`.
- Public docs mention `from_refit_report` as the reuse endpoint after
  `aom_refit_candidates`.

Validation:

- Targeted Python wrapper tests:
  `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so.1.18.0
  PYTHONPATH=bindings/python/src pytest
  bindings/python/tests/test_moment_model_wrappers.py -q`: 29 passed.
- Full Python tests with
  `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so.1.18.0`: 270 passed,
  4 existing UVE warnings.
- Rebuilt `bindings/python_nirs4all_methods`; package smoke confirmed
  `NativeAOMFixedCandidateRegressor.from_refit_report(...)`.
- CUDA-lib Python smoke with `CUDA_VISIBLE_DEVICES=0` confirmed
  `from_refit_report(...)` reuses an exact-CV refit winner after a proxy
  campaign and reports zero proxy fits in the final estimator diagnostics.
- Catalog/checks:
  `catalog/scripts/validate.py`, `catalog/scripts/reconcile_abi.py --check`,
  `catalog/scripts/split_legacy_methods.py --check`, and `git diff --check`
  all passed.

Remaining work:

- This improves reuse of verified campaign winners. It does not implement the
  fused GPU/IKPLS grinder.

## 2026-06-05 - Exact-CV refit helper after score-only campaigns

Decision:

- `pls_score_mode="gcv_proxy"` gives a fast first-pass screen, but selecting
  directly on proxy scores would be too weak for the intended workflow.
- Add a train-only exact-CV verification helper so broad campaigns can retain
  candidates cheaply, then re-rank the retained rows by exact native CV without
  requiring a holdout/test split.

Implementation:

- Added `n4m.aom_refit_candidates(X_train, y_train, candidates, ...)`.
- The helper accepts campaign reports or explicit decoded rows, replays each
  row as a one-candidate `aom_chain_sweep_run` with `pls_score_mode="cv"`,
  and returns `refit_cv_rmse`, `oof_rmse`, `train_rmse`, screen score metadata,
  route counters and PLS fit counters.
- Re-exported the helper at top-level `n4m.aom_refit_candidates`.
- Tests now cover proxy campaign top-k rows followed by exact-CV refit
  verification, asserting proxy fit counters drop to zero and exact PLS CV
  fits are paid in the verification pass.
- Public docs and the coverage matrix now describe the proxy -> exact-CV
  second-pass workflow.

Validation:

- Targeted Python wrapper tests:
  `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so.1.18.0
  PYTHONPATH=bindings/python/src pytest
  bindings/python/tests/test_moment_model_wrappers.py -q`: 29 passed.
- Full Python tests with
  `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so.1.18.0`: 270 passed,
  4 existing UVE warnings.
- Rebuilt `bindings/python_nirs4all_methods`; package smoke confirmed the
  top-level `n4m.aom_refit_candidates` workflow after a proxy campaign.
- CUDA-lib Python smoke with `CUDA_VISIBLE_DEVICES=0` confirmed proxy campaign
  rows can be replayed through exact-CV refits on moment routes with zero
  proxy fits in the verification pass and positive PLS CV fit counters.
- Catalog/checks:
  `catalog/scripts/validate.py`, `catalog/scripts/reconcile_abi.py --check`,
  `catalog/scripts/split_legacy_methods.py --check`, and `git diff --check`
  all passed.

Remaining work:

- This closes the workflow gap around the PLS proxy screen. It does not
  implement a fused/batched IKPLS CUDA grinder.

## 2026-06-05 - Explicit AOM PLS1 GCV proxy score-only screen

Decision:

- The exact PLS1 moment screen still pays one fold-local PLS fit per
  chain/fold. That preserves CV semantics, but it is not the old fast screen
  profile needed for 50k/200k preprocessing campaigns.
- Add a separate opt-in first-pass PLS screen rather than changing
  `score_only=True` semantics silently. The default remains exact CV.

Implementation:

- Added native `aom_pls_score_mode` config with `cv` default and
  `gcv_proxy` opt-in.
- Added `score_pls1_gcv_moment_screen`, which fits PLS1 prefixes once from
  all-sample transformed moments and scores components by nominal GCV RMSE.
- `n4m.aom_sweep_run`, `n4m.aom_chain_sweep_run`, and
  `n4m.aom_chain_score_campaign` expose `pls_score_mode="gcv_proxy"`.
- The proxy requires `score_only=True`, stays inside operator moments, and
  rejects materialized fallback. PLS rows expose
  `score_metric="pls_gcv_proxy_rmse"`.
- Added `n_pls_gcv_proxy_candidates`, `n_pls_gcv_proxy_fits`,
  `aom_pls_score_mode`, and campaign metrics
  `pls_gcv_proxy_fits_per_chain` / `pls_gcv_proxy_fits_per_candidate`.
- Checkpoint fingerprints include `pls_score_mode`.
- Timing benchmark rows now include the proxy counters and a dedicated
  `native_aom_chain_sweep_pls_gcv_proxy` case.

Validation:

- Rebuilt CPU dev-release library with `/home/delete/.venv/bin/cmake --build
  build/dev-release -j2`.
- Targeted Python wrapper tests:
  `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so.1.18.0
  PYTHONPATH=bindings/python/src pytest
  bindings/python/tests/test_moment_model_wrappers.py -q`: 29 passed.
- Native C++ tests:
  `./build/dev-release/cpp/tests/n4m_tests`: 344 passed; and
  `./build/dev-release/cpp/tests/n4m_internal_tests`: all internal checks
  passed.
- Full Python tests with
  `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so.1.18.0`: 270 passed,
  4 existing UVE warnings.
- Rebuilt CUDA library with `/home/delete/.venv/bin/cmake --build
  build/cuda-on -j2`; CUDA smoke with `CUDA_VISIBLE_DEVICES=0` confirmed
  `pls_score_mode="gcv_proxy"` keeps all PLS candidates on moment routes and
  pays zero PLS CV fits.
- Rebuilt `bindings/python_nirs4all_methods`; package smoke confirmed the
  embedded library exposes the new ABI setter and campaign proxy counters.
- Regenerated `benchmarks/cross_binding/aom_sweep_timing.csv` and
  `benchmarks/cross_binding/aom_sweep_timing_cuda_smoke.csv`; both include
  `native_aom_chain_sweep_pls_gcv_proxy` rows with `score_only=1`,
  `n_pls_gcv_proxy_candidates=10`, `n_pls_gcv_proxy_fits=5`, and zero PLS CV
  fits.
- Catalog/checks:
  `catalog/scripts/validate.py`, `catalog/scripts/reconcile_abi.py --check`,
  `catalog/scripts/split_legacy_methods.py --check`, and `git diff --check`
  all passed.

Remaining work:

- This restores a fast first-pass PLS screen profile, but it is a proxy. The
  exact retained candidates still need verification through the default CV
  path or explicit holdout evaluation. It is not the fused GPU/IKPLS grinder.

## 2026-06-05 - AOM campaign per-route top-k audit

Decision:

- After exposing native `candidate_routes`, campaign reports still only kept a
  global top-k and per-head top-k. That was enough for selection, but not for
  auditing whether CPU/CUDA runs found useful candidates on the materialized,
  dense, banded or structured scoring routes.
- Add route-grouped top-k outputs as diagnostics only. They do not change
  native scores, global ranking, checkpoint fingerprints or model fitting.

Implementation:

- `n4m.aom_chain_score_campaign` now returns:
  - `top_candidates_by_score_route`: per-route candidate rows sorted by
    `cv_rmse`;
  - `best_by_score_route`: first row for each scoring route.
- Checkpoints persist the per-route lists and resume filters them to the
  chunks actually present before appending new chunks, matching the global and
  per-head top-k behavior.
- Loading older checkpoints without per-route lists reconstructs route groups
  from `top_candidates` when route fields are available.
- JSON/JSONL/CSV candidate exports still use the candidate rows by default and
  do not duplicate `top_candidates_by_score_route` or `best_by_score_route` in
  JSON metadata.
- Public docs and the coverage matrix now describe per-route campaign audits.

Validation:

- Targeted Python wrapper tests:
  `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so.1.18.0
  PYTHONPATH=bindings/python/src pytest
  bindings/python/tests/test_moment_model_wrappers.py -q`: 28 passed.
- Full Python tests with
  `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so.1.18.0`: 269 passed,
  4 existing UVE warnings.
- Rebuilt `bindings/python_nirs4all_methods`; package smoke confirmed
  `top_candidates_by_score_route` and `best_by_score_route`.
- CUDA-lib Python smoke with `CUDA_VISIBLE_DEVICES=0` and
  `moment_policy="force_moments"` confirmed per-route top-k rows on moment
  routes with no materialized candidates.
- Catalog/checks:
  `catalog/scripts/validate.py`, `catalog/scripts/reconcile_abi.py --check`,
  `catalog/scripts/split_legacy_methods.py --check`, and `git diff --check`
  all passed.

Remaining work:

- This adds audit visibility over scoring routes. It does not implement the
  fused GPU/IKPLS grinder.

## 2026-06-05 - Fixed-candidate reuse from campaign winners

Decision:

- The fixed-candidate wrapper already refit explicit decoded rows, but after
  adding per-head campaign top-k outputs, users still had to manually select
  `report["best"]` or `report["best_by_head"]["pls"]`.
- Add a small sklearn convenience constructor so winning individual methods
  are directly reusable from a campaign report, without adding a scoring rule
  or route heuristic.

Implementation:

- Added `NativeAOMFixedCandidateRegressor.from_campaign(report, head=None,
  rank=0, **kwargs)`.
- `head=None` selects from the global `top_candidates`; `head="ridge"` or
  `head="pls"` selects from `top_candidates_by_head`, falling back to
  `best_by_head` for `rank=0`.
- The selected row delegates to `from_candidate`, so the refit remains the
  exact decoded chain/head/parameter through `aom_chain_sweep_run`.
- Tests now fit the global campaign winner plus Ridge/PLS per-head winners and
  assert selected CV scores match the campaign rows.
- Public docs and coverage matrix now mention campaign-to-fixed-candidate
  reuse.

Validation:

- Targeted Python wrapper tests:
  `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so.1.18.0
  PYTHONPATH=bindings/python/src pytest
  bindings/python/tests/test_moment_model_wrappers.py -q`: 28 passed.
- Full Python tests with
  `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so.1.18.0`: 269 passed,
  4 existing UVE warnings.
- Rebuilt `bindings/python_nirs4all_methods`; package smoke confirmed
  `NativeAOMFixedCandidateRegressor.from_campaign(..., head="ridge")`.
- CUDA-lib Python smoke with `CUDA_VISIBLE_DEVICES=0` confirmed
  `from_campaign(..., head="pls", moment_policy="force_moments")` refits the
  per-head PLS campaign winner and matches its CV score.
- Catalog/checks:
  `catalog/scripts/validate.py`, `catalog/scripts/reconcile_abi.py --check`,
  `catalog/scripts/split_legacy_methods.py --check`, and `git diff --check`
  all passed.

Remaining work:

- This improves reuse of campaign winners. It does not implement the fused
  GPU/IKPLS grinder.

## 2026-06-05 - AOM campaign per-head top-k audit

Decision:

- Mixed Ridge/PLS preprocessing campaigns previously kept only one global
  top-k. That is correct for global ranking, but it can hide the best PLS
  preprocessing chains when Ridge dominates, or the reverse.
- Add per-head top-k outputs as diagnostics only. They do not change native
  scores, global `top_candidates`, route selection, checkpoint fingerprints,
  or fitted models.

Implementation:

- `n4m.aom_chain_score_campaign` now returns:
  - `top_candidates_by_head`: per-head candidate rows sorted by `cv_rmse`;
  - `best_by_head`: first row for each head.
- The per-head lists are truncated with the same `top_k` as the global list
  and are persisted in campaign checkpoints.
- Resume now filters both global and per-head top rows to the chunks actually
  present in a loaded checkpoint before appending new chunks. This keeps manual
  partial checkpoints coherent.
- JSON/JSONL/CSV candidate row exports continue to use `top_candidates` by
  default and do not duplicate `top_candidates_by_head` or `best_by_head` in
  JSON metadata.
- Public docs and the coverage matrix now describe global versus per-head
  campaign ranking.

Validation:

- Targeted Python wrapper tests:
  `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so.1.18.0
  PYTHONPATH=bindings/python/src pytest
  bindings/python/tests/test_moment_model_wrappers.py -q`: 28 passed.
- Full Python tests with
  `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so.1.18.0`: 269 passed,
  4 existing UVE warnings.
- Rebuilt `bindings/python_nirs4all_methods`; package smoke confirmed
  `top_candidates_by_head` and `best_by_head` on a mixed Ridge/PLS campaign.
- CUDA-lib Python smoke with `CUDA_VISIBLE_DEVICES=0` and
  `moment_policy="force_moments"` confirmed per-head top-k rows for Ridge and
  PLS while staying fully operator-moment.
- Catalog/checks:
  `catalog/scripts/validate.py`, `catalog/scripts/reconcile_abi.py --check`,
  `catalog/scripts/split_legacy_methods.py --check`, and `git diff --check`
  all passed.

Remaining work:

- This improves campaign auditability for model-specific preprocessing
  effects. It is not a new selection heuristic and not a fused GPU/IKPLS
  grinder.

## 2026-06-05 - PLS1 moment dense-kernel dispatch audit

Decision:

- Continue the GPU/moment audit on the actual PLS screen bottleneck rather
  than adding a new selection heuristic.
- The PLS1 moment-prefix route has dense products (`C @ w`, `P.T @ W`,
  `W @ inv(P.T @ W)`, and rank-1 covariance deflation) that can use the
  existing `linalg` abstraction on CPU/BLAS builds.
- Do not force those iterative micro-kernels through the current CUDA dispatch.
  A direct cuBLAS attempt was score-correct only after compacting prefix
  matrices, but timing showed it was much slower because every small kernel
  copies host/device. CUDA builds therefore keep the scalar host-side PLS1
  moment loop until a device-resident batched IKPLS workspace exists.

Implementation:

- In `cpp/src/core/sweep.cpp`, routed PLS1 moment-prefix dense products through
  `linalg::gemv`, `linalg::gemm`, and `linalg::ger` for non-CUDA builds.
- Added compact prefix buffers for `W[:, :k]` and `P[:, :k]` before GEMM so
  row-major prefix matrices are contiguous and safe for backends that do not
  support padded submatrix copies.
- Added `N4M_USE_CUDA` guards to keep the previous scalar PLS1 moment loop in
  CUDA builds. Ridge wide GEMM/cuBLAS dispatch from the prior step remains
  enabled.
- Updated `docs/methods/sweep_run.md` and the AOM/moment coverage matrix with
  the CPU/BLAS linalg path and CUDA scalar guard.
- Regenerated `moment_sweep_timing.csv` and
  `moment_sweep_timing_cuda_smoke.csv`.

Validation:

- CPU `dev-release` build completed.
- CPU `n4m_tests`: 344 passed, 0 failed.
- Targeted Python wrapper tests:
  `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so.1.18.0
  PYTHONPATH=bindings/python/src pytest
  bindings/python/tests/test_moment_model_wrappers.py -q`: 28 passed.
- Full Python tests with
  `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so.1.18.0`: 269 passed,
  4 existing UVE warnings.
- CUDA `cuda-on` build completed with `CUDA_VISIBLE_DEVICES=0`.
- CUDA-build `n4m_tests`: 344 passed, 0 failed.
- CUDA-lib Python smoke with `CUDA_VISIBLE_DEVICES=0` confirmed PLS1 full and
  `score_only=True` candidate scores match, with all PLS candidates on the
  moment route.
- Timing smoke regenerated. CPU PLS medians were 0.43/1.96/9.83 ms for
  64x64, 128x128 and 192x256; CPU score-only was 0.29/1.91/9.93 ms. CUDA
  smoke PLS medians, using the scalar guard, were 2.04/3.03/8.10 ms; CUDA
  score-only was 2.17/2.94/7.62 ms. The discarded direct cuBLAS micro-kernel
  attempt was much slower, reaching about 17-23 ms on these PLS rows.

Remaining work:

- The correct GPU direction for PLS remains a device-resident batched IKPLS
  workspace or fused grouped kernels. Host-side cuBLAS dispatch per PLS
  micro-kernel is not the solution for 200k-chain screening.

## 2026-06-05 - Wide dual Ridge GEMM/CUDA dispatch in sweep

Decision:

- The remaining grinder gap is not solved, but the existing wide Ridge path
  still had scalar C++ loops for the dual train Gram, held-out cross-kernel,
  dual held-out predictions, and final coefficient reconstruction.
- Route those products through the existing `n4m::linalg::gemm` abstraction.
  This keeps the CPU behavior inside the established row-major dispatch and
  makes CUDA builds use cuBLAS for this Ridge-wide work without adding a new
  public ABI field or a custom kernel.

Implementation:

- In `cpp/src/core/sweep.cpp`, replaced scalar products with `linalg::gemm`
  for:
  - `K = X_train @ X_train.T`;
  - `K_cross = X_heldout @ X_train.T`;
  - held-out dual predictions `K_cross @ alpha`;
  - final dual coefficient reconstruction `X_train.T @ alpha`.
- Kept the same scoring rules, candidate table, score-only behavior and
  public MethodResult schema.
- Updated `docs/methods/sweep_run.md` and the AOM/moment coverage matrix to
  call out the GEMM/cuBLAS dispatch.
- Regenerated `moment_sweep_timing.csv` and
  `moment_sweep_timing_cuda_smoke.csv`.

Validation:

- CPU `dev-release` build completed.
- CPU `n4m_tests`: 344 passed, 0 failed.
- Targeted Python wrapper tests:
  `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so.1.18.0
  PYTHONPATH=bindings/python/src pytest
  bindings/python/tests/test_moment_model_wrappers.py -q`: 28 passed.
- Full Python tests with
  `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so.1.18.0`: 269 passed,
  4 existing UVE warnings.
- CUDA `cuda-on` build completed with `CUDA_VISIBLE_DEVICES=0`.
- CUDA-build `n4m_tests`: 344 passed, 0 failed.
- CUDA-lib Python wide Ridge smoke with `CUDA_VISIBLE_DEVICES=0` confirmed
  full and `score_only=True` candidate scores match for `p > n`.
- Rebuilt `bindings/python_nirs4all_methods`; package smoke imported the
  bundled library and ran a wide Ridge `sweep_run(..., score_only=True)`.
- Timing smoke regenerated. CPU Ridge medians were 1.35/11.38/43.19 ms for
  64x64, 128x128 and 192x256; CPU score-only was 1.20/10.31/38.87 ms. CUDA
  smoke Ridge medians were 4.71/13.90/35.65 ms; CUDA score-only was
  4.55/13.68/30.41 ms. The wider CUDA row now benefits from the GEMM/cuBLAS
  route, while small rows still pay launch/transfer overhead.
- Catalog/checks:
  `catalog/scripts/validate.py`, `catalog/scripts/reconcile_abi.py --check`,
  `catalog/scripts/split_legacy_methods.py --check`, and `git diff --check`
  all passed.

Remaining work:

- This improves the existing wide Ridge substrate. It is not batched IKPLS,
  not grouped preprocessing-chain execution, and not the fused 200k-chain CUDA
  grinder.

## 2026-06-05 - AOM per-candidate route provenance

Decision:

- The previous AOM route counters were aggregate-only. Broad preprocessing
  campaigns could report how many candidates used materialized, dense, banded,
  or structured operator-moment routes, but the top-k rows did not expose the
  exact scoring route used by each candidate.
- Keep `candidate_scores` stable and add route provenance as a separate audit
  vector. This makes CPU/GPU campaign selection inspectable without changing
  ranking, score computation, or fitted outputs.

Implementation:

- Added `candidate_routes` to native AOM sweep MethodResults. Route ids are:
  `0=materialized`, `1=dense_operator_moment`,
  `2=banded_operator_moment`, and `3=structured_operator_moment`.
- Packed the vector through the C MethodResult ABI and Python result helpers
  for both `aom_sweep_run` and `aom_chain_sweep_run`.
- `aom_candidate_table` and `aom_chain_score_campaign` now expose
  `score_route_id` and `score_route` on candidate/top-k rows.
- `aom_candidate_operator_summary` now includes a `by_score_route` grouping,
  so campaign reports can separate materialized fallback from exact
  operator-moment scoring.
- Native/Python tests assert that per-candidate route counts match the
  aggregate materialized/dense/banded/structured counters.
- Public header comments, method docs, coverage matrix, and catalog method
  notes were updated. Split per-method catalog YAML files were regenerated.

Validation:

- CPU `dev-release` build completed.
- CPU `n4m_tests`: 344 passed, 0 failed.
- Targeted Python wrapper tests:
  `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so.1.18.0
  PYTHONPATH=bindings/python/src pytest
  bindings/python/tests/test_moment_model_wrappers.py -q`: 28 passed.
- Full Python tests with
  `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so.1.18.0`: 269 passed,
  4 existing UVE warnings.
- CUDA `cuda-on` build was up to date with `CUDA_VISIBLE_DEVICES=0`.
- CUDA-build `n4m_tests`: 344 passed, 0 failed.
- CUDA-lib Python smoke with `CUDA_VISIBLE_DEVICES=0` confirmed
  `candidate_routes`, `score_route`, and `by_score_route` through
  `aom_chain_sweep_run` and `aom_chain_score_campaign`.
- Rebuilt `bindings/python_nirs4all_methods` from the updated Python source
  and bundled native library; package smoke imported the bundled library and
  exposed `score_route`.
- Catalog/checks:
  `catalog/scripts/validate.py`, `catalog/scripts/reconcile_abi.py --check`,
  `catalog/scripts/split_legacy_methods.py --check`, and `git diff --check`
  all passed.

Remaining work:

- This is diagnostic provenance for exact preprocessing screens. It does not
  implement batched IKPLS, grouped CUDA kernels, or a fused 200k-chain grinder.

## 2026-06-05 - PLS screen fit-cost audit counters

Decision:

- Expose the PLS screening cost that the future batched IKPLS/GPU grinder
  must reduce. Candidate counts alone hide whether a PLS grid used one
  max-component fit per fold, materialized fallback fits, or extra final
  selected-model fits.
- Keep the counters audit-only. They do not affect ranking, route selection,
  CV scoring, or fitted coefficients.

Implementation:

- Added `n_pls_moment_cv_fits`, `n_pls_materialized_cv_fits`,
  `n_pls_moment_final_fits`, and `n_pls_materialized_final_fits` to native
  `SweepResult` and AOM sweep MethodResults.
- `n4m_sweep_run` now reports whether compatible PLS1 grids were scored by
  moment-prefix fits or materialized prefix fits, and whether a final selected
  PLS fit was run to populate model outputs.
- `n4m_aom_sweep_run` and `n4m_aom_chain_sweep_run` aggregate these counters
  across chains. Score-only campaigns therefore expose pure screen cost with
  final-fit counters at zero.
- Python wrappers, sklearn diagnostics, `aom_chain_score_campaign`, and the
  AOM/moment timing CSVs now carry the counters. Campaign metrics additionally
  include `pls_cv_fits_per_chain` and `pls_cv_fits_per_candidate`.
- Tests now assert that a PLS moment component grid performs one moment fit per
  CV fold, and that score-only mode skips final PLS fits.
- Public docs, the C ABI header comments, the coverage matrix, and catalog
  method notes now list the counters. Split per-method catalog YAML files were
  regenerated from `catalog/methods.yaml`.

Validation:

- CPU `dev-release` build completed.
- CPU `n4m_tests`: 344 passed, 0 failed.
- Targeted Python wrapper tests:
  `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so.1.18.0
  PYTHONPATH=bindings/python/src pytest
  bindings/python/tests/test_moment_model_wrappers.py -q`: 28 passed.
- Full Python test suite:
  `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so.1.18.0
  PYTHONPATH=bindings/python/src pytest bindings/python/tests -q`: 269 passed,
  4 existing warnings.
- CUDA `cuda-on` build completed with `CUDA_VISIBLE_DEVICES=0`.
- CUDA-build `n4m_tests`: 344 passed, 0 failed.
- CUDA Python counter smoke with `CUDA_VISIBLE_DEVICES=0`:
  `n4m.sweep_run(..., heads=("pls",), score_only=True)` reported
  `n_pls_moment_cv_fits=4`, and `aom_chain_score_campaign(..., cv=4,
  moment_policy="force_moments")` reported `n_pls_moment_cv_fits=12` and
  `pls_cv_fits_per_chain=4.0`.
- Timing CSVs regenerated:
  `benchmarks/cross_binding/moment_sweep_timing.csv`,
  `benchmarks/cross_binding/moment_sweep_timing_cuda_smoke.csv`,
  `benchmarks/cross_binding/aom_sweep_timing.csv`, and
  `benchmarks/cross_binding/aom_sweep_timing_cuda_smoke.csv`.
- Rebuilt `bindings/python_nirs4all_methods` from
  `bindings/python/src/n4m/lib/libn4m.so.1.18.0`; package smoke imported the
  bundled library and fitted `NativeMomentSweepRegressor` plus
  `NativeAOMChainSweepRegressor`.
- Catalog/checks:
  `catalog/scripts/validate.py`, `catalog/scripts/reconcile_abi.py --check`,
  `catalog/scripts/split_legacy_methods.py --check`, and `git diff --check`
  all passed.

Remaining work:

- These counters make the PLS bottleneck measurable; they do not yet replace
  fold-local PLS scoring with batched IKPLS or fused CUDA kernels.

## 2026-06-05 - AOM operator-moment prefix cache

Decision:

- Reduce repeated exact moment-transform work in broad strict-linear AOM
  chain grids. Cartesian grids commonly share prefixes such as
  `detrend -> ...` or `savgol_smooth -> ...`; transforming every full chain
  from raw moments repeats the same prefix algebra.
- Keep the optimization score-preserving and bounded. The cache is only used
  inside the operator-moment route and is capped by feature count and entry
  count to avoid large-memory surprises.

Implementation:

- Added an internal prefix cache in `cpp/src/core/aom_sweep.cpp` for
  structured/banded operator-moment chain transforms.
- The cache stores transformed all-sample and held-out moment sets for
  strict-linear prefixes up to `p <= 256`, with at most 64 entries per native
  AOM sweep call. Unsupported or dense-only regimes fall back to the existing
  exact path.
- Exposed audit scalars `n_moment_prefix_cache_hits` and
  `n_moment_prefix_cache_misses` on AOM sweep MethodResults.
- Propagated those scalars through Python, sklearn diagnostics,
  `aom_chain_score_campaign`, and `bench_aom_sweep_timing.py`; campaign
  reports now include `moment_prefix_cache_hit_fraction`.
- Added native and Python tests with repeated `detrend` and `savgol_smooth`
  prefixes to verify full operator-moment scoring plus non-zero cache counters.

Validation:

- CPU `dev-release` build completed.
- CPU `n4m_tests`: 344 passed, 0 failed.
- Targeted Python wrapper tests:
  `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so.1.18.0
  PYTHONPATH=bindings/python/src pytest
  bindings/python/tests/test_moment_model_wrappers.py -q`: 28 passed.
- Full Python tests with
  `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so.1.18.0`: 269 passed,
  4 existing UVE warnings.
- CUDA build completed with `CUDA_VISIBLE_DEVICES=0`.
- CUDA `n4m_tests` on `CUDA_VISIBLE_DEVICES=0`: 344 passed, 0 failed.
- CUDA-lib Python smoke passed with explicit
  `N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so.1.18.0`, confirming a
  repeated-prefix `force_moments` chain grid reports 3 cache hits and
  5 misses while staying fully operator-moment.
- Timing smoke regenerated:
  `benchmarks/cross_binding/aom_sweep_timing.csv` and
  `benchmarks/cross_binding/aom_sweep_timing_cuda_smoke.csv`. CUDA-build
  `auto` rows now expose prefix reuse in the timing table, e.g. compact
  mixed rows at 64/128 features report 3 hits and 12 misses; custom5 rows
  report 2 hits and 5 misses.
- Generated `bindings/python_nirs4all_methods` package smoke passed with ABI
  `(1, 18, 0)`, including direct `aom_chain_sweep_run` cache counters and
  `aom_chain_score_campaign` cache-hit aggregation.
- Catalog checks passed: `catalog/scripts/validate.py`,
  `catalog/scripts/reconcile_abi.py --check` and
  `catalog/scripts/split_legacy_methods.py --check`.
- `git diff --check` passed.

Remaining work:

- This reduces repeated prefix transforms for medium-width strict-linear
  operator-moment grids. It is still not a fused batched IKPLS or CUDA
  grouped-kernel implementation for 200k-chain screens.

## 2026-06-05 - Score-only dual Ridge SSE micro-optimization

Decision:

- Continue reducing score-only screening overhead in the existing native
  Ridge/PLS sweep without changing the ABI or candidate scores.
- The wide Ridge route (`p > n_train`) can use a precomputed train/held-out
  cross-kernel. In score-only mode it only needs held-out SSE, not the
  materialized held-out prediction matrix.

Implementation:

- Added a direct dual-cross held-out SSE helper in `cpp/src/core/sweep.cpp`.
- In `n4m_sweep_run(..., score_only=True)`, wide Ridge folds that use
  `K_cross` now solve the same dual system and accumulate SSE directly against
  held-out `Y`, avoiding the temporary prediction buffer.
- Added native and Python wrapper tests that compare wide Ridge score-only
  candidate scores against the full-output route.

Validation:

- CPU `dev-release` build completed.
- CPU `n4m_tests`: 343 passed, 0 failed.
- Targeted Python wrapper tests:
  `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so.1.18.0
  PYTHONPATH=bindings/python/src pytest
  bindings/python/tests/test_moment_model_wrappers.py -q`: 27 passed.
- Full Python tests with
  `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so.1.18.0`: 268 passed,
  4 existing UVE warnings.
- CUDA build completed with `CUDA_VISIBLE_DEVICES=0`.
- CUDA `n4m_tests` on `CUDA_VISIBLE_DEVICES=0`: 343 passed, 0 failed.
- CUDA-lib Python smoke passed with explicit
  `N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so.1.18.0`, confirming wide
  Ridge score-only candidate scores match the full-output route and selected
  output buffers are empty.
- Timing smoke regenerated:
  `benchmarks/cross_binding/moment_sweep_timing.csv` and
  `benchmarks/cross_binding/moment_sweep_timing_cuda_smoke.csv`.
  CPU Ridge medians for full vs score-only were 1.19/1.12 ms at 64 x 64,
  10.57/9.33 ms at 128 x 128, and 36.16/32.48 ms at 192 x 256. CUDA-build
  smoke medians were 2.15/2.38 ms, 10.59/10.26 ms, and 35.42/31.64 ms on the
  same Ridge cells.
- Generated `bindings/python_nirs4all_methods` package smoke passed with ABI
  `(1, 18, 0)`, confirming wide Ridge score-only candidate scores match the
  full-output route.
- Catalog checks passed: `catalog/scripts/validate.py`,
  `catalog/scripts/reconcile_abi.py --check` and
  `catalog/scripts/split_legacy_methods.py --check`.
- `git diff --check` passed.

Remaining work:

- This still does not add batched IKPLS or fused CUDA kernels. It only removes
  another avoidable allocation in score-only wide Ridge screens.

## 2026-06-05 - Sweep score-only direct SSE micro-optimization

Decision:

- Reduce avoidable allocation in broad score-only Ridge/PLS screens. The
  native scorer already skips selected-model outputs in score-only mode, but
  materialized fallback cells still allocated held-out prediction buffers just
  to compute RMSE.

Implementation:

- Added a direct held-out SSE helper in `cpp/src/core/sweep.cpp` for linear
  `RidgeMomentFit` states.
- In `n4m_sweep_run(..., score_only=True)`, materialized Ridge and
  materialized PLS prefix fallback cells now score held-out rows directly from
  coefficients when available.
- This first pass left dual cross-kernel Ridge on its prediction buffer because
  that path scores in dual space without reconstructing feature-space
  coefficients; the follow-up entry above removes that remaining score-only
  buffer too.
- No ABI or scoring semantics changed; this is a memory/allocation reduction
  for score-only ranking paths. AOM materialized fallback screens benefit
  because they call `run_moment_sweep(..., score_only=True)` underneath.

Validation:

- CPU `dev-release` build completed.
- CPU `n4m_tests`: 343 passed, 0 failed.
- CUDA build completed with `CUDA_VISIBLE_DEVICES=0`.
- CUDA `n4m_tests` on `CUDA_VISIBLE_DEVICES=0`: 343 passed, 0 failed.
- Full Python tests with
  `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so.1.18.0`: 267 passed,
  4 existing UVE warnings.
- CUDA-lib Python smoke passed with `CUDA_VISIBLE_DEVICES=0`, explicit
  `N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so.1.18.0`, score-only
  Ridge/PLS sweep and AOM chain campaign.
- Generated `bindings/python_nirs4all_methods` package smoke passed with ABI
  `(1, 18, 0)`, including score-only Ridge/PLS sweep.
- Catalog checks passed: `catalog/scripts/validate.py`,
  `catalog/scripts/reconcile_abi.py --check` and
  `catalog/scripts/split_legacy_methods.py --check`.
- `git diff --check` passed.

Remaining work:

- This is not batched IKPLS or fused CUDA. It only removes avoidable held-out
  prediction buffers in existing score-only materialized fallback cells.

## 2026-06-05 - Chunked strict-linear AOM campaign helper

Decision:

- Add a product-facing campaign helper for broad strict-linear preprocessing
  ranking. The native scorer already has `aom_chain_sweep_run` and
  `score_only=True`; users still needed a deterministic way to generate
  larger chain grids, chunk execution, and aggregate top-k rows without losing
  chain provenance.

Implementation:

- Added `n4m.build_aom_strict_chain_grid`.
  - `compact` and `wide` reproduce the native built-in banks.
  - `lab` / `cartesian` builds a broader deterministic strict-linear grid with
    multiple Savitzky-Golay smooth/derivative variants, Norris-Williams,
    finite differences and Whittaker chains.
  - Custom `families` and `templates` define cartesian operator combinations.
- Added `n4m.aom_chain_score_campaign`.
  - Runs `aom_chain_sweep_run(..., score_only=True)` in chain chunks.
  - Aggregates a global top-k with decoded chains, global chain ids, head,
    parameter and CV RMSE.
  - Sums route counters across chunks for operator-moment versus materialized
    fallback audit.
  - Accepts `checkpoint_path` / `resume` for long strict-linear campaigns.
    The JSON checkpoint is written after each completed chunk, stores the
    current top-k and per-chunk counters, and is guarded by a fingerprint of
    chains, folds, hyperparameters and `X/y` contents.
  - Accepts `max_chunks_per_run` for bounded incremental execution. The report
    exposes `complete`, `n_remaining_chunks`, `processed_chunks_this_run` and
    `max_chunks_per_run`; the chunk budget is intentionally not part of the
    checkpoint fingerprint, so campaign cadence can change across relaunches.
  - Reports normalized throughput and route metrics at campaign and chunk
    level: candidates/chains per second, ms per candidate/chain, and
    operator-moment/materialized plus dense/banded/structured route fractions.
    These are derived from native route counters and elapsed chunk timings.
- Added `NativeAOMFixedCandidateRegressor`.
  - `from_candidate(row)` consumes rows from `n4m.aom_candidate_table` or
    `n4m.aom_chain_score_campaign`.
  - Refits exactly one decoded chain/head/parameter candidate through the same
    native ABI and predicts from folded `input_coefficients`.
- Added `n4m.aom_evaluate_candidates`.
  - Refits decoded candidate rows on an explicit train split.
  - Scores them on caller-provided `X_eval, y_eval`.
  - Reports `screen_cv_rmse`, `refit_cv_rmse`, `eval_rmse`, `eval_r2`,
    `cv_rank`, `eval_rank` and `rank_delta` for CV-vs-holdout analysis.
  - Does not alter fit, route selection, or use dataset identity.
- Added `n4m.aom_candidate_rank_diagnostics`.
  - Consumes evaluated rows or reloaded candidate reports.
  - Compares screen score (`screen_cv_rmse` by default) against `eval_rmse`.
  - Reports Spearman rank correlation, absolute rank drift, cross-ranks of the
    screen/eval winners, and top-k overlap/recall.
  - Provides screen-recall audit evidence without adding a selection rule.
- Added `n4m.aom_candidate_report_records` and
  `n4m.aom_save_candidate_report`.
  - Flatten campaign/eval candidate rows into JSON-safe dictionaries.
  - Write `.json`, `.jsonl` / `.ndjson`, or `.csv` reports without pandas.
  - Preserve decoded strict-linear chains and add `chain_json` for CSV replay.
  - Drop prediction arrays by default unless `include_predictions=True`.
- Added `n4m.aom_load_candidate_report`.
  - Reads JSON, JSONL / NDJSON and CSV candidate reports.
  - Restores `chain` from `chain_json` for CSV rows.
  - Converts standard id/rank/score fields back to numeric types.
  - Returns rows that can be passed directly to
    `NativeAOMFixedCandidateRegressor.from_candidate` or
    `n4m.aom_evaluate_candidates`.
- Added `n4m.aom_candidate_operator_summary`.
  - Groups already-scored rows by head, preprocessing operator,
    operator/head pair and chain length.
  - Uses `eval_rmse` when present, otherwise `cv_rmse`, `refit_cv_rmse` or
    `screen_cv_rmse`.
  - Reports group counts, best/mean/median score and rank stats without
    changing candidate scores or top-k ordering.
- Exported both campaign helpers and the fixed-candidate estimator from
  top-level `n4m`.
- Added targeted Python tests and method/coverage docs.

Validation:

- Targeted Python wrapper/sweep tests:
  `bindings/python/tests/test_moment_model_wrappers.py`: 26 passed, including
  campaign top-k to `NativeAOMFixedCandidateRegressor.from_candidate` refit,
  replay, bounded incremental checkpoint execution, resume from a partial
  campaign JSON, throughput/route metric checks,
  `aom_evaluate_candidates` CV-vs-holdout reporting, screen-recall rank
  diagnostics, JSON/CSV/JSONL candidate report export/reload, operator/head
  summary checks, and refit from a reloaded CSV winner.
- Full Python tests: `bindings/python/tests`: 267 passed, 4 existing UVE
  warnings.
- CUDA-lib Python smoke passed with `CUDA_VISIBLE_DEVICES=0`, explicit
  `N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so.1.18.0`, lab grid generation,
  chunked score-only execution, global top-k checks, fixed-candidate refit and
  holdout candidate evaluation plus bounded incremental checkpoint/resume and
  JSON/CSV/JSONL report export.
- Generated `bindings/python_nirs4all_methods` package smoke passed with ABI
  `(1, 18, 0)`, including fixed-candidate import/refit/predict,
  `aom_evaluate_candidates`, bounded incremental checkpoint/resume, and
  JSON/CSV report export.
- Catalog checks passed: `validate.py`, `split_legacy_methods.py --check`,
  and `reconcile_abi.py --check`.
- `git diff --check` passed.

Remaining work:

- This makes large strict-linear ranking campaigns easier to run and inspect.
  It still uses the existing native scorer underneath; it is not a fused
  batched IKPLS or custom CUDA grinder.

## 2026-06-05 - AOM sweep campaign chain descriptors

Decision:

- Treat chain traceability as part of the campaign contract. A ranked
  candidate table is not enough unless every `chain_id` can be mapped back to
  the exact strict-linear preprocessing chain used during scoring.

Implementation:

- Added `chain_offsets`, `op_kinds`, `param_offsets`, and `chain_params` to
  native `AomSweepResult`.
- Exported the descriptor for both built-in `aom_sweep_run` banks and
  caller-provided `aom_chain_sweep_run` descriptors.
- Exposed the descriptor through the C ABI MethodResult and Python results.
- Added `n4m.decode_aom_chains(res)` and
  `n4m.aom_candidate_table(res, sort=True)` for campaign reports and top-k
  inspection.
- Updated public C ABI comments and method docs.

Validation:

- CPU build completed.
- CPU `n4m_tests`: 343 passed, 0 failed.
- Targeted Python wrapper/sweep tests:
  `bindings/python/tests/test_moment_model_wrappers.py`: 22 passed.
- Full Python tests: `bindings/python/tests`: 263 passed, 4 existing UVE
  warnings.
- CPU-lib Python smoke passed for `score_only=True` chain decoding and
  candidate-table generation.
- CUDA build completed with `CUDA_VISIBLE_DEVICES=0`.
- CUDA `n4m_tests` on `CUDA_VISIBLE_DEVICES=0`: 343 passed, 0 failed.
- CUDA-lib Python smoke passed for non-empty chain descriptors,
  `n4m.decode_aom_chains`, and `n4m.aom_candidate_table`.
- Generated `bindings/python_nirs4all_methods` package smoke passed with ABI
  `(1, 18, 0)`.
- Catalog checks passed: `validate.py`, `split_legacy_methods.py --check`,
  and `reconcile_abi.py --check`.
- `git diff --check` passed.

Fix caught during validation:

- The first CUDA native run exposed empty descriptors on the specialized
  operator-moment paths. Added descriptor export to the Ridge-only and hybrid
  operator-moment early-return paths, including `score_only=True`.

Remaining work:

- This improves campaign reproducibility and score inspection. It does not
  implement batched IKPLS or fused CUDA scoring kernels.

## 2026-06-05 - Reusable native AOM operator PLS stack outputs

Decision:

- Treat the native AOM operator PLS stack as a reusable fitted model. The
  native stack is restricted to strict-linear operator views, fixed PLS1
  projections and a Ridge head, so its selected final predictor can be folded
  into original input-space coefficients.

Implementation:

- Added `input_coefficients` and `input_intercept` to
  `AomOperatorPlsStackResult`.
- Derived the exact input-space linear state from each final view's strict
  operator matrix, standardizer, PLS rotations, and the final Ridge head.
- Kept existing `coefficients` and `intercept` as the stack-feature Ridge head
  for audit compatibility.
- Exposed folded matrices through the C ABI MethodResult and Python
  `n4m.aom_operator_pls_stack`.
- Added `NativeAOMOperatorPLSStackRegressor`, exported from `n4m` and
  `n4m.sklearn`, with predictions driven by
  `X @ input_coefficients + input_intercept`.
- Updated public ABI comments, method docs, coverage notes and catalog notes.

Validation:

- CPU `n4m_tests`: 343 passed, 0 failed.
- Targeted Python wrapper tests:
  `bindings/python/tests/test_moment_model_wrappers.py`: 22 passed.
- Full Python tests: `bindings/python/tests`: 263 passed, 4 existing UVE
  warnings.
- CUDA build completed with `CUDA_VISIBLE_DEVICES=0`.
- CUDA `n4m_tests` on `CUDA_VISIBLE_DEVICES=0`: 343 passed, 0 failed.
- CUDA-lib Python smoke passed for `n4m.aom_operator_pls_stack` and
  `NativeAOMOperatorPLSStackRegressor` replay.
- Generated `bindings/python_nirs4all_methods` package smoke passed with ABI
  `(1, 18, 0)`.
- Catalog checks passed: `validate.py`, `split_legacy_methods.py --check`,
  and `reconcile_abi.py --check`.
- `git diff --check` passed.

Remaining work:

- This makes the native operator PLS stack reusable. It is still not a fused
  batched GPU stack or 200k-chain IKPLS grinder.

## 2026-06-05 - Reusable native AOM Ridge blender outputs

Decision:

- Treat the native AOM Ridge simplex blend as a reusable linear model. The
  native candidate pool is restricted to strict-linear chains plus Ridge heads,
  so a non-negative weighted blend of candidates has an exact input-space
  coefficient representation.

Implementation:

- Added `input_coefficients` and `intercept` to `AomRidgeBlenderResult`.
- During final full-data candidate refits, folded each candidate's transformed
  Ridge coefficients through its strict-linear chain and stored the candidate
  input-space state.
- After solving simplex weights, accumulated the weighted final
  `input_coefficients` and `intercept`.
- Exposed both matrices through the C ABI MethodResult and Python
  `n4m.aom_ridge_blender`.
- Added `NativeAOMRidgeBlenderRegressor`, exported from `n4m` and
  `n4m.sklearn`, with predictions driven by the folded native coefficients.
- Updated method docs, coverage notes and catalog notes.

Validation:

- CPU `n4m_tests`: 343 passed, 0 failed.
- Targeted Python wrapper tests:
  `bindings/python/tests/test_moment_model_wrappers.py`: 22 passed.
- Full Python tests: `bindings/python/tests`: 263 passed, 4 existing UVE
  warnings.
- CUDA build completed with `CUDA_VISIBLE_DEVICES=0`.
- CUDA `n4m_tests` on `CUDA_VISIBLE_DEVICES=0`: 343 passed, 0 failed.
- CUDA-lib Python smoke passed for `n4m.aom_ridge_blender` and
  `NativeAOMRidgeBlenderRegressor` replay.

Remaining work:

- This makes the native Ridge blend reusable. It is still not a fused batched
  GPU blender or a 200k-chain IKPLS grinder.

## 2026-06-05 - Reusable native AOM robust-HPO outputs

Decision:

- Treat native `aom_robust_hpo` as a reusable fitted model, not just a compact
  score table. The selected strict-linear preprocessing chain is deterministic
  and linear, so its transformed-space coefficients can be folded back into the
  original input feature space.

Implementation:

- Added `input_coefficients` and `n_features` to `AomRobustHpoResult`.
- Folded selected transformed-space coefficients through the selected
  strict-linear chain operator during final native fit.
- Exposed `input_coefficients`, `n_samples`, `n_features`,
  `n_features_transformed`, and `n_targets` through the C ABI MethodResult and
  Python `n4m.aom_robust_hpo`.
- Added `NativeAOMRobustHPORegressor`, exported from `n4m` and
  `n4m.sklearn`, using the replay equation
  `X @ input_coefficients + intercept`.
- Updated method docs, coverage notes, and catalog notes.

Validation:

- CPU `n4m_tests`: 343 passed, 0 failed.
- Targeted Python wrapper tests:
  `bindings/python/tests/test_moment_model_wrappers.py`: 22 passed.
- Full Python tests: `bindings/python/tests`: 263 passed, 4 existing UVE
  warnings.
- CUDA build completed with `CUDA_VISIBLE_DEVICES=0`.
- CUDA `n4m_tests` on `CUDA_VISIBLE_DEVICES=0`: 343 passed, 0 failed.
- CUDA-lib Python smoke passed for `n4m.aom_robust_hpo` and
  `NativeAOMRobustHPORegressor` replay.

Remaining work:

- This makes the compact/wide product method reusable. It does not implement a
  fused batched IKPLS or 200k-chain CUDA grinder.

## 2026-06-05 - AOM per-head route counters

Decision:

- Broad preprocessing campaigns need to know whether candidate rows were
  scored by operator moments or by materialized fallback, separately for Ridge
  and PLS. The previous counters only exposed total route counts.

Implementation:

- Added AOM MethodResult scalars:
  `n_ridge_operator_moment_candidates`,
  `n_pls_operator_moment_candidates`,
  `n_ridge_materialized_candidates`, and
  `n_pls_materialized_candidates`.
- Updated native route accounting in `aom_sweep.cpp` for Ridge-only,
  PLS-only, mixed hybrid, score-only and fully materialized routes.
- Propagated the counters through Python `n4m.aom_sweep_run`,
  `n4m.aom_chain_sweep_run` and native sklearn sweep diagnostics.
- Updated docs/catalog notes so route provenance is visible in the public
  method contract.

Validation:

- Added C++ and Python route-partition assertions.
- CPU `n4m_tests`: 343 passed, 0 failed.
- Targeted Python wrapper tests:
  `bindings/python/tests/test_moment_model_wrappers.py`: 21 passed.
- Full Python tests: `bindings/python/tests`: 262 passed, 4 existing UVE
  warnings.
- CUDA build completed with `CUDA_VISIBLE_DEVICES=0`.
- CUDA `n4m_tests` on `CUDA_VISIBLE_DEVICES=0`: 343 passed, 0 failed.
- CUDA-lib Python smoke passed for `aom_chain_sweep_run(...,
  moment_policy="force_moments", score_only=True)` with Ridge/PLS route
  counters.
- Generated `bindings/python_nirs4all_methods` package smoke passed with ABI
  `(1, 18, 0)`.
- Regenerated `benchmarks/cross_binding/aom_sweep_timing.csv` and
  `benchmarks/cross_binding/aom_sweep_timing_cuda_smoke.csv` with per-head
  route-counter columns.
- Catalog checks passed: `validate.py`, `split_legacy_methods.py --check`,
  and `reconcile_abi.py --check`.
- `git diff --check` passed.

Remaining work:

- These counters make the existing screen more auditable. They do not implement
  batched IKPLS or fused CUDA operator kernels.

## 2026-06-05 - Public sweep score-only ranking mode

Decision:

- Expose the existing native `aom_score_only` behavior on the public
  `n4m.sweep_run` Python surface so broad Ridge/PLS ranking campaigns can skip
  selected-model buffers.
- Keep this as a functional ranking API, not as an sklearn estimator knob,
  because sklearn prediction still requires reusable coefficients and
  intercepts.

Implementation:

- Added `score_only` to `n4m.sweep_run(...)` and wired it to
  `n4m_config_set_aom_score_only`.
- Documented the C ABI contract: candidate scores, selected ids, folds and
  scalar diagnostics remain populated; OOF predictions, final predictions,
  coefficients and intercept are returned as empty `0 x 0` matrices.
- Added C++ and Python tests for the public sweep score-only contract.
- Extended `benchmarks/cross_binding/bench_moment_sweep_timing.py` with
  `native_sweep_ridge_score_only` and `native_sweep_pls_score_only` rows.

Validation:

- CPU `n4m_tests`: 343 passed, 0 failed.
- Targeted Python wrapper tests:
  `bindings/python/tests/test_moment_model_wrappers.py`: 21 passed.
- Full Python tests: `bindings/python/tests`: 262 passed, 4 existing UVE
  warnings.
- CUDA build completed with `CUDA_VISIBLE_DEVICES=0`.
- CUDA `n4m_tests` on `CUDA_VISIBLE_DEVICES=0`: 343 passed, 0 failed.
- CUDA-lib Python smoke passed for
  `n4m.sweep_run(..., heads=("pls",), score_only=True)` with
  `n_pls_moment_candidates == n_candidates` and empty prediction buffers.
- Timing smoke regenerated:
  `benchmarks/cross_binding/moment_sweep_timing.csv` and
  `benchmarks/cross_binding/moment_sweep_timing_cuda_smoke.csv`.

Remaining work:

- This makes the public sweep useful as a fast ranking pass. It still does not
  implement fused batched IKPLS or the full 200k-chain CUDA grinder.

## 2026-06-05 - Public sweep PLS1 moment scoring

Decision:

- Move the public `n4m_sweep_run` PLS path closer to the requested moment
  engine instead of leaving PLS component screening entirely materialized.
- Keep the existing materialized prefix scorer as the fallback for multi-target
  or unsupported PLS solver/deflation regimes.
- Expose a route counter so Python/C users can audit whether compatible PLS
  candidates used the moment route.

Implementation:

- Added a compatible PLS1 route inside `run_moment_sweep` for
  single-target NIPALS/regression PLS component grids.
- The route computes held-out moments, subtracts them from all-row moments,
  fits PLS1 prefixes from train sufficient statistics, scores held-out SSE
  from held-out moments, and only uses materialized `X` to populate selected
  OOF/final prediction buffers when requested.
- Fold-local train matrix materialization is now skipped when neither wide
  Ridge nor unsupported PLS fallback needs it.
- `SweepResult` / MethodResult now expose scalar
  `n_pls_moment_candidates`; Python `n4m.sweep_run` returns it.
- `benchmarks/cross_binding/bench_moment_sweep_timing.py` now records
  `n_pls_moment_candidates`.

Validation:

- C++ tests now assert `n_pls_moment_candidates` for single and multi-component
  PLS grids while preserving materialized-CV score parity.
- CPU `n4m_tests`: 342 passed, 0 failed.
- CUDA `n4m_tests` on `CUDA_VISIBLE_DEVICES=0`: 342 passed, 0 failed.
- Targeted Python wrapper tests:
  `bindings/python/tests/test_moment_model_wrappers.py`: 20 passed.
- Full Python tests: `bindings/python/tests`: 261 passed, 4 existing UVE
  warnings.
- CUDA-lib Python smoke passed for `n4m.sweep_run(..., heads=("pls",))` with
  `n_pls_moment_candidates == n_candidates`.
- Generated `bindings/python_nirs4all_methods` package smoke passed with ABI
  `(1, 18, 0)`.
- Timing smoke regenerated:
  `benchmarks/cross_binding/moment_sweep_timing.csv` and
  `benchmarks/cross_binding/moment_sweep_timing_cuda_smoke.csv`; PLS rows show
  `n_pls_moment_candidates=3`.
- Catalog checks passed: `validate.py`, `split_legacy_methods.py --check`,
  and `reconcile_abi.py --check`.
- `git diff --check` passed.

Remaining work:

- This removes the public sweep's materialized PLS bottleneck for compatible
  PLS1 grids, but it is still not fused batched IKPLS or a 200k-chain CUDA
  grinder.

## 2026-06-05 - Reusable AOM/POP selected-model outputs

Decision:

- The historical native AOM-PLS and POP-PLS selectors were callable from
  Python, but they still returned only train predictions and diagnostics.
- Expose the selected model as an input-space linear state so the methods are
  reusable on new spectra, matching the reuse contract already added for the
  newer AOM sweep MethodResults.

Implementation:

- Added result vectors to the native AOM/POP selection results:
  `coefficients`, `input_coefficients`, and `intercept`.
- Added public C ABI getters for both result handles:
  `*_get_coefficients`, `*_get_input_coefficients`, and `*_get_intercept`.
- Global AOM keeps `coefficients` in transformed selected-operator space and
  folds `input_coefficients` back to the original input feature space.
- POP coefficients are already in original input space; `input_coefficients`
  mirrors them for a uniform binding contract.
- Python ABI-close wrappers now return those matrices.
- Added `NativeAOMPLSRegressor` and `NativePOPPLSRegressor` sklearn-style
  wrappers that predict with `X @ input_coefficients + intercept`.
- Added cross-binding smoke timing for the reusable selector surfaces:
  `benchmarks/cross_binding/bench_aom_selector_timing.py`, with CPU and
  CUDA-smoke CSV outputs.

Validation:

- Added C++ ABI replay tests for AOM global and POP selected models.
- CPU `n4m_tests`: 342 passed, 0 failed.
- CUDA `n4m_tests` on `CUDA_VISIBLE_DEVICES=0`: 342 passed, 0 failed.
- Targeted Python wrapper tests:
  `bindings/python/tests/test_moment_model_wrappers.py`: 20 passed.
- Full Python tests: `bindings/python/tests`: 261 passed, 4 existing UVE
  warnings.
- CUDA-lib Python smoke with
  `CUDA_VISIBLE_DEVICES=0 N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so.1.18.0`
  passed for `n4m.aom_pls`, `n4m.pop_pls`, `NativeAOMPLSRegressor`, and
  `NativePOPPLSRegressor`.
- Catalog checks passed: `validate.py`, `split_legacy_methods.py --check`,
  and `reconcile_abi.py --check`.
- `git diff --check` passed.
- Timing smoke passed:
  `benchmarks/cross_binding/aom_selector_timing.csv` and
  `benchmarks/cross_binding/aom_selector_timing_cuda_smoke.csv`.

Remaining work:

- This makes the historical selectors reusable as selected linear models; it
  still does not add batched IKPLS or fused GPU screening.

## 2026-06-05 - Python wrappers for historical AOM/POP selectors

Decision:

- Close the usability gap for the already-catalogued native AOM-PLS and
  POP-PLS selectors: the C ABI existed, but the `n4m` top-level Python surface
  did not expose it.
- Keep the wrapper ABI-close instead of adding a new sklearn estimator class:
  build an operator bank and validation plan, call the native selector, copy
  result buffers to NumPy, and destroy the native handles.

Implementation:

- Added ctypes declarations for `n4m_operator_bank_*`,
  `n4m_validation_plan_*`, `n4m_aom_global_select`,
  `n4m_aom_per_component_select`, and their result getters.
- Added `n4m.aom_global_select` / `n4m.aom_pls` and
  `n4m.aom_per_component_select` / `n4m.pop_pls`.
- The wrappers use the documented compact strict-linear operator bank by
  default, accept caller-provided strict operators, build fold-safe validation
  plans from explicit `fold_ids` or contiguous CV folds, and force the native
  SIMPLS-regression config required by these selectors.

Validation:

- Added Python smoke coverage for both selectors and aliases in
  `bindings/python/tests/test_moment_model_wrappers.py`.
- Targeted Python test:
  `test_native_aom_pls_and_pop_pls_selector_wrappers_smoke`: passed.
- Targeted wrapper file:
  `bindings/python/tests/test_moment_model_wrappers.py`: 19 passed.
- CUDA-lib Python smoke with
  `CUDA_VISIBLE_DEVICES=0 N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so.1.18.0`
  passed for `n4m.aom_pls` and `n4m.pop_pls`.
- Generated `bindings/python_nirs4all_methods` package smoke passed with the
  same CUDA lib and confirmed the alias surface.
- Full Python tests: `bindings/python/tests`: 260 passed, 4 existing UVE
  warnings.
- Catalog checks: `validate.py`, `split_legacy_methods.py --check`, and
  `reconcile_abi.py --check` passed from `catalog/scripts`.
- `git diff --check` passed.

Remaining work:

- This exposes existing catalogued methods; it does not add a fused GPU AOM
  screen or batched IKPLS.

## 2026-06-05 - AOM score-only screen output mode

Decision:

- Add an output mode for large AOM preprocessing ranking campaigns where the
  first pass needs candidate scores and selected ids, not a fitted selected
  model artifact.
- Keep the default behavior unchanged for estimator-style use and sklearn
  wrappers.

Implementation:

- Public config now exposes `n4m_config_set_aom_score_only` and
  `n4m_config_get_aom_score_only`.
- Python wrappers accept `score_only=True` on `n4m.aom_sweep_run` and
  `n4m.aom_chain_sweep_run`.
- AOM MethodResults expose scalar `score_only`. In score-only mode, model
  output matrices are returned as `0 x 0`, while `candidate_scores`, selected
  ids, route counters and `fold_ids` remain populated.
- Operator-moment AOM routes skip the final selected-chain
  refit/materialization in score-only mode.
- The internal `run_moment_sweep`, `score_ridge_moment_sweep`, and
  `score_pls1_moment_sweep` paths now also respect this flag when called from
  AOM: they avoid OOF buffers and selected-model output refits while keeping
  candidate scores unchanged. Materialized candidate-screen routes still pay
  fold-local scoring fits because this is not batched IKPLS.

Validation:

- Added C++ test
  `aom_chain_sweep/force_moments_score_only_skips_final_refit`.
- Added C++ test
  `aom_chain_sweep/materialized_score_only_keeps_scores`.
- CPU `n4m_tests`: 340 passed, 0 failed.
- CUDA build with `CUDA_VISIBLE_DEVICES=0`, `n4m_tests`: 340 passed, 0
  failed.
- Targeted Python wrapper tests:
  `bindings/python/tests/test_moment_model_wrappers.py`: 18 passed.
- Full Python tests: `bindings/python/tests`: 259 passed, 4 existing UVE
  warnings.
- Catalog checks: `validate.py`, `split_legacy_methods.py --check`, and
  `reconcile_abi.py --check` passed.
- `git diff --check` passed.

Remaining work:

- This is a practical ranking-mode increment toward large campaigns, not the
  fused 200k-chain CUDA/IKPLS grinder. The next engine step is still a
  batched/device-resident PLS screen instead of per-chain host orchestration.

## 2026-06-05 - Strict force_moments route policy

Decision:

- Keep `moment_policy="auto"` exact but pragmatic: it may choose materialized
  candidate scoring when that route is supported and cheaper for a backend or
  geometry.
- Add `moment_policy="force_moments"` for audits and production guards that
  must prove the candidate screen stayed inside operator moments.
- Allow post-selection materialization to remain: it is used only to expose
  public OOF/final predictions and `input_coefficients`, not to score the
  candidate grid.

Implementation:

- Public config enum now includes `N4M_AOM_MOMENT_FORCE_MOMENTS`.
- C++ AOM sweep routes reject materialized candidate-screen fallback with
  `UNSUPPORTED` when the requested chain/head/grid has no complete
  operator-moment route.
- Python accepts `moment_policy="force_moments"` plus aliases
  `"moments_only"`, `"operator_moments_only"`, and `"strict_moments"`.

Validation:

- Added C++ coverage for an accepted full moment route and a rejected fallback
  route in `n4m_aom_chain_sweep_run`.
- Added Python wrapper coverage for the same strict policy.
- CPU `n4m_tests`: 338 passed, 0 failed.
- CUDA build with `CUDA_VISIBLE_DEVICES=0`, `n4m_tests`: 338 passed, 0
  failed.
- Targeted Python wrapper tests:
  `bindings/python/tests/test_moment_model_wrappers.py`: 18 passed.
- Full Python tests: `bindings/python/tests`: 259 passed, 4 existing UVE
  warnings.
- Catalog checks: `validate.py`, `split_legacy_methods.py --check`, and
  `reconcile_abi.py --check` passed.
- `git diff --check` passed.

Remaining work:

- This closes the "no hidden hors moments" guard for native AOM screens. It
  still does not ship the fused 200k-chain CUDA/IKPLS grinder; unsupported
  regimes now fail explicitly in strict mode instead of being screened through
  materialized candidates.

## 2026-06-05 - Native sklearn sweep estimators and input-space AOM coefficients

Decision:

- Keep the existing AOM `coefficients` output in the selected transformed-chain
  feature space, but add an exact `input_coefficients` output folded back into
  the original spectral feature space.
- Use `input_coefficients` to expose reusable sklearn-style native estimators
  that predict on new spectra without Python-side chain replay.

Implementation:

- Core: `cpp/src/core/aom_sweep.cpp` now builds the selected chain operator
  matrix `M` and folds transformed coefficients `b` as `M @ b`.
- C ABI: `n4m_aom_sweep_run` and `n4m_aom_chain_sweep_run` MethodResults now
  include `input_coefficients`; `n4m_sweep_run` remains unchanged.
- Python: added `NativeMomentSweepRegressor`, `NativeAOMSweepRegressor`, and
  `NativeAOMChainSweepRegressor` under `n4m.sklearn` and top-level `n4m`.

Validation:

- C++ CPU build:
  - `n4m_tests`: 336 passed, 0 failed.
- CUDA build with `CUDA_VISIBLE_DEVICES=0`:
  - `n4m_tests`: 336 passed, 0 failed.
- Python:
  - `bindings/python/tests/test_moment_model_wrappers.py`: 17 passed.
  - full `bindings/python/tests`: 258 passed, 4 existing UVE warnings.

Remaining work:

- This makes the native sweep outputs reusable as normal estimators, but it is
  not the fused 200k-chain CUDA/IKPLS grinder. The GPU audit still points to a
  future moment-only mode, device-resident workspace, and batched IKPLS/fused
  chain scoring.

## 2026-06-04 - Backend-aware CPU PLS route selection

Decision:

- Keep exact PLS1 operator-moment scoring available, but stop using it in CPU
  `auto` when train folds are only moderately larger than the feature count.
- CPU `auto` now materializes compatible PLS rows when
  `min_train_rows < 4 * p`; CUDA `auto` keeps the moment route.
- The threshold is a compute-route guard only. Candidate scores, ordering and
  selected models are unchanged.

Implementation:

- Core: `cpp/src/core/aom_sweep.cpp` shares the backend-aware route selector
  between Ridge and PLS.
- Mixed Ridge+PLS sweeps avoid transforming moments at all when both heads are
  routed through the exact materialized path.
- Exact PLS moment route tests were moved to a larger `n >> p` geometry where
  CPU `auto` is expected to keep moments.

Validation:

- Added C++ test `aom_chain_sweep/wide_pls_auto_route_is_backend_aware`,
  proving CPU and CUDA take different exact routes while candidate scores match
  the forced materialized route.
- CPU build:
  - `n4m_tests`: 336 passed, 0 failed.
- CUDA build with `CUDA_VISIBLE_DEVICES=0`:
  - `n4m_tests`: 336 passed, 0 failed.
- CPU timing evidence after the selector:
  - compact mixed 48x64 / 80x128: `auto` 5.67 / 25.64 ms,
    `materialized` 5.54 / 26.00 ms;
  - compact PLS 60x32 / 80x48 / 96x64: `auto` 1.75 / 1.41 / 4.31 ms,
    `materialized` 1.01 / 1.29 / 2.21 ms, with all rows materialized in
    `auto` for those smoke geometries.
- CUDA-build smoke keeps moment routing:
  - compact PLS 60x32 / 80x48 / 96x64: `auto` 15.27 / 11.46 / 15.78 ms,
    `materialized` 127.77 / 105.61 / 123.48 ms.

Remaining work:

- The route selector is still a CPU/CUDA dispatch choice, not fused batched
  IKPLS. The full 200k-chain GPU grinder remains open.

## 2026-06-04 - Backend-aware CPU wide Ridge route selection

Decision:

- Keep `moment_policy="auto"` exact, but stop forcing Ridge moment scoring on
  CPU when `p > n_train`.
- In that regime, the materialized chain path delegates Ridge scoring to the
  existing dual Ridge sweep in sample space, which is cheaper than solving
  feature-space Ridge systems from transformed moments.
- Preserve the CUDA behavior: CUDA builds keep the operator-moment route in
  the same wide Ridge cells where the smoke timings show a win.

Implementation:

- Core: `cpp/src/core/aom_sweep.cpp` now detects whether the library was built
  with CUDA linalg dispatch.
- CPU `auto` Ridge rows with `p > min_train_rows` use the exact materialized
  chain route; CUDA `auto` keeps exact operator-moment scoring.
- Mixed Ridge+PLS sweeps can now have split route counters: Ridge rows may be
  materialized on CPU while compatible PLS rows still use moments.

Validation:

- Added C++ test `aom_chain_sweep/wide_ridge_auto_route_is_backend_aware`,
  proving the route counter switches by backend while candidate scores match
  the forced materialized route.
- CPU build:
  - `n4m_tests`: 335 passed, 0 failed;
  - `n4m_internal_tests`: passed;
  - Python tests: 256 passed, 4 existing UVE warnings.
- CUDA build with `CUDA_VISIBLE_DEVICES=0`:
  - `n4m_tests`: 335 passed, 0 failed;
  - `n4m_internal_tests`: passed.
- CPU timing evidence after the selector:
  - compact mixed 48x64 / 80x128: `auto` 8.21 / 46.00 ms,
    `materialized` 5.73 / 24.80 ms;
  - custom5 mixed 48x64 / 80x128: `auto` 3.59 / 20.45 ms,
    `materialized` 2.97 / 10.69 ms.
- CUDA-build smoke preserves moment routing and remains faster in the smaller
  mixed cell:
  - compact mixed 48x64: `auto` 29.55 ms, `materialized` 114.82 ms;
  - compact mixed 80x128: `auto` 279.79 ms, `materialized` 122.22 ms.

Remaining work:

- CPU PLS moment rows are still often slower than materialized PLS on the
  current smoke cells; that needs a separate forced-moment policy or
  PLS-specific exact route selector if we want `auto` to be uniformly fastest
  on CPU while retaining explicit route coverage.
- This is still not the fused 200k-chain CUDA/IKPLS grinder.

## 2026-06-04 - Structured Whittaker operator-moment route

Decision:

- Keep Whittaker inside the strict-linear native AOM/moment screen.
- Avoid the dense operator-matrix route for Whittaker by reusing the existing
  pentadiagonal Whittaker baseline solver.
- Preserve `moment_policy="materialized"` as the A/B fallback because CPU
  Ridge-wide cells can still be faster through the materialized dual Ridge
  route.

Implementation:

- Core: `cpp/src/core/aom_sweep.cpp` now parses Whittaker lambda and factors
  the exact `I + lambda D2'D2` pentadiagonal system once per moment set.
- Moment transform:
  - `x_sum' = x_sum S`;
  - `X'Y' = S X'Y`;
  - `X'X' = S X'X S`;
  where `S = (I + lambda D2'D2)^-1`.
- The route composes with the existing banded local operators and with
  `detrend_poly`, and is counted under
  `n_structured_operator_moment_candidates`.
- The timing script now includes a `custom2_whittaker` chain benchmark.

Validation:

- Added C++ and Python tests comparing Whittaker `auto` scores against
  `moment_policy="materialized"` scores.
- CPU build:
  - `n4m_tests`: 334 passed, 0 failed;
  - `n4m_internal_tests`: passed;
  - Python tests: 256 passed, 4 existing UVE warnings.
- CUDA build with `CUDA_VISIBLE_DEVICES=0`:
  - `n4m_tests`: 334 passed, 0 failed;
  - `n4m_internal_tests`: passed.
- CPU timing evidence for `custom2_whittaker` mixed Ridge+PLS:
  - 48x64: `auto` 4.64 ms, `materialized` 3.87 ms;
  - 64x96: `auto` 16.21 ms, `materialized` 13.31 ms.
- CUDA-build smoke evidence for `custom2_whittaker` mixed Ridge+PLS:
  - 48x64: `auto` 7.78 ms, `materialized` 19.57 ms;
  - 64x96: `auto` 21.31 ms, `materialized` 33.97 ms.

Remaining work:

- The route is exact and useful for CUDA/PLS-style screens, but CPU Ridge
  `p > n_train` still needs a cost-aware moment-vs-dual route selector before
  `auto` is uniformly faster.
- Batched IKPLS and fused CUDA kernels remain separate backlog items.

## 2026-06-04 - AOM moment route policy switch

Decision:

- Keep `auto` as the default route policy.
- Add a public config/wrapper switch to force the legacy materialized-chain
  screen when it is faster or needed for route A/B tests.
- Preserve candidate scores: policy changes compute route, not selection
  semantics.

Implementation:

- Config: `n4m_config_set_aom_moment_policy` and
  `n4m_config_get_aom_moment_policy`.
- Python: `n4m.aom_sweep_run(..., moment_policy="auto"|"materialized")` and
  `n4m.aom_chain_sweep_run(..., moment_policy=...)`.
- Core: `N4M_AOM_MOMENT_MATERIALIZED` bypasses operator-moment routing and
  falls directly to the materialized strict-linear chain sweep.
- Benchmarks: `bench_aom_sweep_timing.py` now emits a `moment_policy` column
  and records both policies.

Validation:

- Added C++ and Python tests proving `materialized` produces zero
  operator-moment candidates and matches `auto` candidate scores up to
  numerical roundoff.
- CPU timing evidence:
  - compact mixed 48x64 / 80x128: `auto` 19.92 / 289.13 ms,
    `materialized` 5.70 / 25.66 ms;
  - compact Ridge 96x32 / 160x64: `auto` 2.67 / 19.18 ms,
    `materialized` 2.81 / 20.92 ms.
- CUDA-build smoke evidence:
  - compact PLS 60x32 / 80x48 / 96x64: `auto` 12.88 / 13.58 / 16.99 ms,
    `materialized` 99.98 / 104.38 / 105.12 ms;
  - compact mixed 48x64 / 80x128: `auto` 121.35 / 274.57 ms,
    `materialized` 85.79 / 110.77 ms.

Remaining work:

- The policy switch makes route comparisons explicit but does not replace the
  missing fused GPU/batched IKPLS engine.

## 2026-06-04 - Structured detrend operator-moment route

Decision:

- Keep ABI 1.18.0 unchanged.
- Add an exact low-rank moment transform for `detrend_poly` instead of
  materializing every detrended chain outside the dense guard.
- Preserve the strict-linear constraint: no fold-fitted stateful preprocessing
  is admitted into the native moment screen.

Implementation:

- Core: `cpp/src/core/aom_sweep.cpp` now builds the polynomial projection
  basis for `detrend_poly` and applies the projection to moment sets:
  - `x_sum' = x_sum A`;
  - `X'Y' = A' X'Y`;
  - `X'X' = A' X'X A`.
- The structured route composes `detrend_poly` with the existing banded local
  operators (`identity`, Savitzky-Golay, Norris-Williams, finite difference
  and FCK).
- Result counters now separate `n_banded_operator_moment_candidates`,
  `n_structured_operator_moment_candidates`,
  `n_dense_operator_moment_candidates`, and `n_materialized_candidates`.

Validation:

- Added C++ public tests comparing `detrend(1) -> finite_difference(1)` Ridge
  and PLS1 CV scores against a materialized `n4m_sweep_run` on manually
  transformed data.
- CPU build:
  - `n4m_tests`: 332 passed, 0 failed;
  - `n4m_internal_tests`: passed.
- CUDA build with `CUDA_VISIBLE_DEVICES=0`:
  - `n4m_tests`: 332 passed, 0 failed;
  - `n4m_internal_tests`: passed.
- Timing CSVs regenerated with the structured counter:
  - CPU compact mixed AOM medians: 133.47 ms for 48x64 and 1128.52 ms for
    80x128;
  - CPU custom5 mixed medians: 29.15 ms and 315.25 ms for the same cells;
  - CUDA-build smoke compact mixed medians: 271.71 ms and 549.71 ms;
  - CUDA-build smoke custom5 mixed medians: 21.13 ms and 197.12 ms.

Remaining work:

- This is an exact coverage/audit route, not a fused GPU performance win. It
  still transforms dense `p x p` moments on the host.
- Multi-target/non-NIPALS PLS and the true 200k-chain batched GPU screen
  remain open.

## 2026-06-04 — Banded AOM operator-moment route

Decision:

- Keep ABI 1.18.0 unchanged.
- Add an internal banded descriptor for shape-preserving local linear AOM
  operators instead of forcing every moment transform through a dense chain
  matrix.
- Preserve exact scoring and candidate ordering; unsupported chains still
  fall back to the materialized native sweep.

Implementation:

- Core: `cpp/src/core/aom_operators.{hpp,cpp}` adds
  `BandedLinearOperator` and `build_aom_banded_operator`.
- Banded operators currently cover `identity`, Savitzky-Golay smooth,
  Savitzky-Golay derivative, Norris-Williams, finite difference, Gaussian and
  FCK.
- Dense low-rank/solve operators (`detrend_poly`, `whittaker`) deliberately
  remain outside the banded route.
- Core: `cpp/src/core/aom_sweep.cpp` transforms raw moments with sparse column
  entries:
  - `x_sum' = x_sum A`;
  - `X'Y' = A' X'Y`;
  - `X'X' = A' X'X A`.
- Route counters are exposed as additive result scalars:
  `n_operator_moment_candidates`, `n_banded_operator_moment_candidates`,
  `n_dense_operator_moment_candidates`, and `n_materialized_candidates`.
- Guardrails:
  - Ridge banded moment scoring: `p <= 256` and either `p <= n_train` or all
    lambdas strictly positive;
  - PLS1 banded moment scoring: `p <= 1024`, single target, NIPALS,
    regression deflation.

Validation:

- Added C++ ABI tests comparing banded Ridge and banded PLS1 scores at
  `p=64` against a materialized `n4m_sweep_run` on manually transformed
  finite-difference data.
- CPU build:
  - `n4m_tests`: 330 passed, 0 failed;
  - `n4m_internal_tests`: passed.
- CUDA build with `CUDA_VISIBLE_DEVICES=0`:
  - `n4m_tests`: 330 passed, 0 failed;
  - `n4m_internal_tests`: passed.
- Timing CSVs regenerated with route counters:
  - CPU compact mixed AOM medians: 38.58 ms for 48x64 and 301.11 ms for
    80x128;
  - CPU compact PLS1 rows: 1.94 ms for 60x32, 4.76 ms for 80x48, 9.27 ms for
    96x64;
  - CPU custom5 PLS1 rows: 0.95 ms, 2.40 ms, 4.41 ms for the same cells;
  - CUDA-build smoke compact PLS1 rows: 34.64 ms, 28.51 ms, 46.44 ms.

Remaining work:

- The route still stores dense `p x p` moments and performs host-side
  transforms; it is not the final fused CUDA 200k-chain grinder.
- Detrend and Whittaker are handled by later structured routes; batched
  all-chain execution is still outside this banded local-operator layer.
- Batched IKPLS and CUDA grouped kernels remain open.

## 2026-06-04 — Medium AOM PLS1 operator-moment scoring

Decision:

- Keep ABI 1.18.0 unchanged.
- Add an internal exact PLS1 NIPALS scorer from sufficient statistics for
  strict-linear AOM chains.
- Use it only for single-target NIPALS/regression-deflation PLS grids in the
  medium dense-operator regime: `p <= n_train` or `p <= 48`.

Implementation:

- Core: `cpp/src/core/sweep.cpp` adds `score_pls1_moment_sweep`.
- Core: `cpp/src/core/aom_sweep.cpp` routes compatible PLS-only and mixed
  sweeps through the operator-moment path.
- The scorer fits PLS1 prefixes from train moments (`Cxx`, `Cxy`, `Y'Y`) and
  evaluates held-out SSE directly from held-out moments.
- Public OOF/final predictions are still produced by materializing the selected
  chain once. If the PLS1 moment scorer is unsupported or numerically
  degenerate for a chain, that chain falls back to the previous materialized PLS
  path.

Validation:

- Added internal static-archive test `test_internal_sweep.cpp` comparing
  `score_pls1_moment_sweep` against materialized `run_moment_sweep` PLS CV for
  `scale_x=false` and `scale_x=true`.
- CPU build:
  - `n4m_tests`: 328 passed, 0 failed;
  - `n4m_internal_tests`: passed, including PLS1 moment scoring.
- CUDA build with `CUDA_VISIBLE_DEVICES=0`:
  - `n4m_tests`: 328 passed, 0 failed;
  - `n4m_internal_tests`: passed, including PLS1 moment scoring.
- Targeted Python tests through the build library:
  41 passed for moment wrappers and AOM structural policy tests.
- Full Python binding pytest against the repackaged ABI 1.18.0 library:
  254 passed, 4 existing UVE warnings.
- Catalog/ABI gates:
  - `catalog/scripts/validate.py`: PASS, 196 methods;
  - `catalog/scripts/reconcile_abi.py --check`: 558 method symbols +
    123 infra symbols = 681/681;
  - `catalog/scripts/split_legacy_methods.py --check`: PASS.
- `git diff --check`: clean.
- Import smoke without `N4M_LIB_PATH`: ABI `(1, 18, 0)`, packaged
  `libn4m.so`, and `aom_sweep_run` / `aom_chain_sweep_run` exported.
- Timing CSVs regenerated for `aom_sweep`:
  - CPU PLS1 operator-moment medians: compact 60x32 3.98 ms, compact 80x48
    11.28 ms, custom5 60x32 1.50 ms, custom5 80x48 8.70 ms;
  - CUDA-build smoke medians: 54.12 ms, 87.02 ms, 69.45 ms, 39.61 ms on the
    same rows.

Remaining work:

- The current operator-moment transform still builds dense `A' C A`, so it is
  not the final 200k-chain wide-spectrum grinder.
- Multi-target PLS and non-NIPALS PLS solvers still use the materialized path.
- Fused CUDA kernels and sparse/banded operator descriptors remain open.

## 2026-06-04 — Native AOM operator PLS score stack

Decision:

- Add an ABI-stable native `n4m.aom_operator_pls_stack` surface for the
  strict-linear AOM operator PLS1 score-stack experiment.
- Keep the existing `AOMOperatorPLSStack` sklearn-style estimator as the
  flexible Python reference for custom operator matrices and optional baseline
  admission gates.
- Make the native contract single-target only, matching the PLS1 projector used
  in the source-free reference.

Implementation:

- ABI 1.18.0 symbol: `n4m_aom_operator_pls_stack_fit`.
- Core: `cpp/src/core/aom_operator_pls_stack.cpp`.
- Python: `n4m.aom_operator_pls_stack`.
- Catalog: `aom_pop.operator_pls_stack`.
- Result contract includes candidate spec scores, fold scores, selected OOF
  predictions, final predictions, final `stack_features`, Ridge coefficients
  and operator feature offsets.

Validation:

- Added C++ contract tests for compact shape, selected criterion,
  prediction reconstruction from stack features and Ridge coefficients,
  explicit fold ids, feature offsets and multi-output rejection.
- CPU build and `n4m_tests`: 328 passed, 0 failed.
- Python targeted wrapper test passed with 13 tests using the ABI 1.18.0 build
  through `N4M_LIB_PATH`.
- CUDA build and `CUDA_VISIBLE_DEVICES=0 n4m_tests`: 328 passed, 0 failed.
- Full Python binding pytest against the packaged ABI 1.18.0 library:
  254 passed, 4 existing UVE warnings.
- Catalog/ABI gates:
  - `catalog/scripts/validate.py`: PASS, 196 methods;
  - `catalog/scripts/reconcile_abi.py --check`: 558 method symbols +
    123 infra symbols = 681/681;
  - `catalog/scripts/split_legacy_methods.py --check`: PASS.
- Import smoke without `N4M_LIB_PATH`: ABI `(1, 18, 0)` and
  `n4m.aom_operator_pls_stack` exported from the packaged library.
- Timing CSVs added:
  - CPU compact medians: 9.56 ms for 32x64, 14.62 ms for 48x96,
    24.29 ms for 64x128;
  - CUDA-build smoke medians: 4.13 ms, 8.75 ms, 11.95 ms on the same cells.

Remaining work:

- Native v1 is not a fused batched GPU stack.
- Native v1 does not include custom Python operator matrices, shuffled/both CV
  modes, or the optional baseline admission gate; those remain in the Python
  estimator.

## 2026-06-04 — Native AOM Ridge OOF simplex blender

Decision:

- Add an ABI-stable native `n4m.aom_ridge_blender` surface for the strict-linear
  compact/wide AOM Ridge blender.
- Keep the existing `AOMRidgeBlender` sklearn-style estimator as the flexible
  Python reference layer for explicit custom candidates.
- Require strictly positive Ridge lambdas in the native method so the blend pool
  uses stable Ridge fits only.

Implementation:

- ABI 1.17.0 symbol: `n4m_aom_ridge_blender_fit`.
- Core: `cpp/src/core/aom_ridge_blender.cpp`.
- Python: `n4m.aom_ridge_blender`.
- Catalog: `aom_pop.ridge_blender`.
- Benchmark: `benchmarks/cross_binding/bench_aom_ridge_blender_timing.py`.
- Result contract includes per-candidate OOF/final predictions and simplex
  weights so the blend can be audited exactly.

Validation:

- Added C++ contract tests for compact result shape, simplex weights,
  prediction reconstruction, selected max-weight candidate and invalid lambda.
- CPU build and `n4m_tests`: 326 passed, 0 failed.
- CUDA build and `CUDA_VISIBLE_DEVICES=0 n4m_tests`: 326 passed, 0 failed.
- Full Python binding pytest: 252 passed, 4 existing UVE warnings.
- Catalog/ABI gates:
  - `catalog/scripts/validate.py`: PASS, 195 methods;
  - `catalog/scripts/reconcile_abi.py --check`: 557 method symbols +
    123 infra symbols = 680/680;
  - `catalog/scripts/split_legacy_methods.py --check`: PASS.
- Timing CSVs added:
  - CPU compact 48-candidate blend medians: 16.32 ms for 32x64, 69.39 ms for
    64x128, 272.69 ms for 96x256;
  - CUDA-build smoke medians: 15.63 ms, 81.00 ms, 299.56 ms on the same cells.

Remaining work:

- Native v1 is not a fused batched GPU blender.
- The method is Ridge-only; PLS diversity still comes from `aom_sweep_run`,
  `aom_chain_sweep_run` or future batched IKPLS work.
- Stateful preprocessings remain out of the native strict-linear bank.

## 2026-06-04 — Medium-wide AOM Ridge operator-moment scoring

Decision:

- Keep ABI 1.16.0 unchanged.
- Extend exact AOM Ridge operator-moment scoring beyond `p <= n_train` for
  medium-wide grids where every Ridge lambda is strictly positive and
  `p <= 48`.
- Keep larger wide regimes on the materialized-chain plus dual-Ridge fallback:
  the current operator-moment transform builds dense `A' C A`, so the feature
  cap is a compute-route guard, not a scoring proxy.

Implementation:

- Core: `cpp/src/core/aom_sweep.cpp`.
- Ridge-only and mixed Ridge+PLS AOM sweeps now share the same route predicate:
  - always allow exact operator-moment Ridge when `p <= n_train`;
  - additionally allow it for positive-lambda medium-wide `p <= 48`;
  - fallback for lambda grids containing zero in wide regimes, because the
    primal moment solve is not the right rank-deficient `lambda=0` dual path.

Validation:

- Added C++ test
  `aom_chain_sweep/wide_positive_ridge_operator_moments_match_materialized`.
  It compares positive-lambda medium-wide operator-moment scores against a
  materialized fallback run forced by including `lambda=0`.
- CPU build and `n4m_tests`: 324 passed, 0 failed.
- CUDA build and `CUDA_VISIBLE_DEVICES=0 n4m_tests`: 324 passed, 0 failed.
- Full Python binding pytest: 250 passed, 4 existing UVE warnings.
- Catalog/ABI gates:
  - `catalog/scripts/validate.py`: PASS, 194 methods;
  - `catalog/scripts/reconcile_abi.py --check`: 556 method symbols +
    123 infra symbols = 679/679;
  - `catalog/scripts/split_legacy_methods.py --check`: PASS.
- Timing CSVs regenerated for `moment_sweep`, `aom_sweep`, and
  `aom_robust_hpo` in CPU and CUDA-smoke variants.

Remaining work:

- PLS rows still materialize transformed chains.
- Larger wide AOM Ridge regimes still materialize transformed chains before the
  dual-kernel scorer.
- Batched IKPLS and fused CUDA/operator kernels remain open.

## 2026-06-04 — Hybrid AOM Ridge-moment scoring inside mixed sweeps

Decision:

- Keep ABI 1.16.0 unchanged.
- Extend the exact operator-moment Ridge scorer from Ridge-only AOM sweeps to
  the Ridge candidate rows of mixed Ridge+PLS sweeps when `p <= n_train`.
- Keep PLS candidate rows on the current materialized native PLS path. This is
  not batched IKPLS and not a fused GPU/operator-moment PLS engine.

Implementation:

- Core: `cpp/src/core/aom_sweep.cpp`.
- New hybrid path:
  - computes raw all-sample and held-out moments once;
  - transforms those moments by each strict-linear chain operator matrix;
  - scores Ridge rows through `score_ridge_moment_sweep`;
  - materializes each chain only for the PLS rows;
  - materializes the selected Ridge chain once when the global winner is Ridge.
- Candidate-table order is preserved per chain: Ridge rows first, then PLS
  rows.

Validation:

- Added C++ test `aom_chain_sweep/mixed_hybrid_matches_split_runs`, comparing
  mixed sweep Ridge rows against a Ridge-only run and mixed PLS rows against a
  PLS-only run.
- CPU build and `n4m_tests`: 323 passed, 0 failed.
- CPU timing smoke, ABI 1.16.0:
  - `native_aom_sweep` compact mixed took 33.55 ms for 48x64 and 152.93 ms for
    80x128;
  - `native_aom_chain_sweep` custom5 mixed took 15.94 ms for 48x64 and 51.10 ms
    for 80x128.
- CUDA-build timing smoke, ABI 1.16.0:
  - `native_aom_sweep` compact mixed took 1150.02 ms for 48x64 and 418.81 ms
    for 80x128;
  - `native_aom_chain_sweep` custom5 mixed took 150.82 ms for 48x64 and
    192.70 ms for 80x128.
  These validate CUDA-build behavior, not fused GPU acceleration.

Remaining work:

- PLS rows still materialize transformed chains.
- Larger wide `p > n_train` AOM Ridge still materializes transformed chains
  before the dual-kernel scorer.
- Batched IKPLS and fused CUDA/operator kernels remain open.

## 2026-06-04 — AOMOperatorPLSStack Python reference port

Decision:

- Port the AOM `operator_pls_stack` experiment as a source-free Python
  reference estimator.
- Keep it pre-ABI and diagnostic: it is not a product default, not a native
  catalog method, and not a fused GPU/moment PLS screen.
- Restrict the default operator bank to fixed strict-linear matrices. Stateful
  scatter operators are deliberately excluded from the default path.

Implementation:

- Python: `n4m.AOMOperatorPLSStack`, `n4m.sklearn.AOMOperatorPLSStack`, and
  `AOMOperatorPLSSpec`.
- For each operator view, the estimator fits a fold-local column standardizer
  and PLS1 score projector, concatenates the per-operator scores, and fits a
  dependency-light Ridge head.
- Spec selection uses train-only CV over `(components, alpha)` with
  `mean_rmse + std_penalty * std_rmse + gap_penalty * train_val_gap`.
- If a `baseline_estimator` is supplied, the operator stack is admitted only
  when its OOF RMSE improves the baseline by `min_relative_oof_gain`.
- Metadata passed to `fit` is audit-only and does not affect operators, specs,
  admission, or predictions.

Validation:

- Added Python tests for custom fixed operators, false-positive baseline-gate
  rejection, metadata invariance, default strict-operator bank smoke behavior,
  exports, and predict-before-fit behavior.
- Targeted pytest:
  `PYTHONPATH=bindings/python/src /home/delete/.venv/bin/python -m pytest -q bindings/python/tests/test_aom_structural_policy.py`
  passed with 28 tests.
- Full Python binding pytest:
  `PYTHONPATH=bindings/python/src /home/delete/.venv/bin/python -m pytest -q bindings/python/tests`
  passed with 250 tests and 4 existing UVE warnings.
- CPU build and `n4m_tests`: 322 passed, 0 failed.
- CUDA build and `n4m_tests` with `CUDA_VISIBLE_DEVICES=0`: 322 passed,
  0 failed.
- `catalog/scripts/validate.py --strict-abi`: PASS, 194 methods, 679/679
  exported `n4m_*` symbols covered.
- `git diff --check`: clean.
- Import/use smoke:
  `n4m.AOMOperatorPLSStack is n4m.sklearn.AOMOperatorPLSStack`; selected spec
  is an `AOMOperatorPLSSpec`; fit/predict path works.

Remaining work:

- Native ABI/catalog integration is still open.
- Batched IKPLS and fused operator/moment PLS scoring remain open; this port
  materializes operator views in Python.

## 2026-06-04 — AOMRidgeBlender Python reference port

Decision:

- Port the `nirs4all-aom` fold-safe AOM-Ridge blender design into
  `nirs4all-methods` as a source-free Python reference estimator.
- Keep it pre-ABI and out of the strict native catalog until a C++/ABI
  blend/report/predict surface exists.
- Allow two modes: explicit estimator/factory candidates, or a default
  chain+Ridge pool built from `build_aom_control_chain_bank`.

Implementation:

- Python: `n4m.AOMRidgeBlender` and `n4m.sklearn.AOMRidgeBlender`.
- OOF scoring: each candidate is refit fold-locally; validation rows are never
  fitted before their OOF prediction.
- Blend: solve the regularized non-negative simplex QP with scipy SLSQP when
  available, falling back to projected gradient on the simplex.
- Metadata passed to `fit` is stored only in `blend_report_` for audit and is
  not used for candidate construction, weighting, or prediction.

Validation:

- Added Python tests for exact simplex weight recovery, fold-local OOF fits,
  metadata invariance, default chain+Ridge pool construction, exports, and
  predict-before-fit behavior.
- Targeted pytest:
  `PYTHONPATH=bindings/python/src /home/delete/.venv/bin/python -m pytest -q bindings/python/tests/test_aom_structural_policy.py`
  passed with 23 tests.
- Full Python binding pytest:
  `PYTHONPATH=bindings/python/src /home/delete/.venv/bin/python -m pytest -q bindings/python/tests`
  passed with 245 tests and 4 existing UVE warnings after staging ABI 1.16.0
  into `bindings/python/src/{n4m,pls4all}/lib`.
- During the full pytest run, the current ABI exposed an existing
  `MBPLSRegression` mismatch against direct `mb_pls_fit`. The wrapper now
  matches MB-PLS direct config scaling defaults and returns the C-side
  in-sample predictions when `predict` receives the fitted training matrix.
- CPU build and `n4m_tests`: 322 passed, 0 failed.
- CUDA build and `n4m_tests` with `CUDA_VISIBLE_DEVICES=0`: 322 passed,
  0 failed.
- `catalog/scripts/validate.py --strict-abi`: PASS, 194 methods, 679/679
  exported `n4m_*` symbols covered.
- `git diff --check`: clean.
- Import/use smoke: `n4m.AOMRidgeBlender is n4m.sklearn.AOMRidgeBlender`;
  default `chains=("raw",)` fit/predict path works.

Remaining work:

- Native ABI/catalog integration is still open.
- No CUDA-fused blender exists; any GPU use must come from wrapped candidate
  estimators, not from the blender itself.

## 2026-06-04 — Dual-kernel Ridge screening for wide `p > n_train` folds

Decision:

- Keep ABI 1.16.0 unchanged.
- Optimize the existing `p > n_train` Ridge path inside `n4m_sweep_run`
  without changing scores or outputs.
- Continue using materialized transformed chains in AOM wide regimes for now;
  the improvement is inside the Ridge scorer.

Implementation:

- Core: `cpp/src/core/sweep.cpp`.
- `RidgeDualDesign` can store held-out/train cross-kernels for dual folds.
- Ridge screening solves `(K_train + lambda I) alpha = Y_train` and predicts
  held-out rows as `K_heldout,train alpha + y_mean` when a simple cost
  heuristic predicts this is cheaper than the dual-beta scoring path.
- This avoids reconstructing feature-space coefficients and then doing
  `X beta` for every fold/lambda during CV scoring in the regimes where the
  cross-kernel setup cost is amortized. The final selected model still refits
  as before to populate public coefficients and predictions.

Validation:

- Added C++ test `sweep/wide_dual_ridge_scores_match_materialized_cv`,
  comparing `p > n_train` sweep scores against explicit fold-by-fold
  `n4m_ridge_fit`.
- CPU build and `n4m_tests`: 322 passed, 0 failed.

Remaining work:

- This is exact and useful for Ridge sweeps, but still not a fused
  operator-aware dual kernel for AOM chains. The transformed chain is still
  materialized before `n4m_sweep_run`.
- Batched IKPLS and CUDA fused kernels remain open.

## 2026-06-04 — Ridge-only operator-moment scoring for strict AOM chains

Decision:

- Keep ABI 1.16.0 unchanged.
- Add an internal Ridge-only acceleration path for `n4m_aom_sweep_run` and
  `n4m_aom_chain_sweep_run` when `heads_mask == Ridge` and `p <= n_train`.
- Preserve the existing materialized chain path for mixed Ridge+PLS, PLS-only,
  and `p > n_train` regimes.

Implementation:

- Core: `cpp/src/core/aom_sweep.cpp` builds the strict-linear chain operator
  matrix by applying existing AOM kernels to an identity basis. It then
  transforms raw moments as `x_sum A`, `A' X'X A`, and `A' X'Y`.
- Core: `cpp/src/core/sweep.cpp` exposes an internal
  `score_ridge_moment_sweep` hook that fits fold Ridge models from train
  moments and scores held-out SSE directly from held-out moments. No
  per-candidate held-out row predictions are needed during screening.
- The selected chain is materialized once after ranking to populate public
  OOF predictions, final predictions, coefficients and fold ids.

Validation:

- Added C++ test
  `aom_chain_sweep/ridge_operator_moments_match_materialized`, comparing
  Ridge-only operator-moment scores with the Ridge rows from a materialized
  Ridge+PLS run.
- CPU build and `n4m_tests`: 321 passed, 0 failed.
- CUDA build and `n4m_tests` with `CUDA_VISIBLE_DEVICES=0`: 321 passed,
  0 failed.
- Python direct tests without pytest: 28 passed.
- Python smoke: `n4m.aom_chain_sweep_run(..., heads=("ridge",))` ABI
  `(1, 16, 0)`, candidate table `(6, 5)`, materialized Ridge score parity
  asserted.
- CUDA-build Python smoke: same Ridge-only path, candidate table `(6, 5)`.
- `catalog/scripts/validate.py --strict-abi`: PASS, 194 methods, 679/679
  exported `n4m_*` symbols covered.
- CPU timing smoke, ABI 1.16.0:
  `native_aom_chain_sweep_ridge` custom5 took 7.39 ms for 96x32 and
  51.41 ms for 160x64; compact built-in Ridge-only took 15.90 ms and
  123.10 ms respectively.
- CUDA-build timing smoke, ABI 1.16.0:
  `native_aom_chain_sweep_ridge` custom5 took 41.38 ms for 96x32 and
  70.01 ms for 160x64; compact built-in Ridge-only took 101.16 ms and
  152.29 ms respectively. This validates CUDA-build behavior, not fused GPU
  acceleration.

Remaining work:

- This path is exact but not the full 200k-chain engine: it builds dense
  operator matrices and is deliberately limited to Ridge-only `p <= n_train`.
- Batched IKPLS, `p > n_train` operator-aware dual Ridge, sparse/banded
  operator algebra, and fused CUDA kernels remain open.

## 2026-06-04 — PLS component-prefix scoring inside native sweeps

Decision:

- Keep ABI 1.16.0 unchanged.
- Replace per-component PLS refits in `n4m_sweep_run` with one
  max-component fit per fold, followed by coefficient-prefix reconstruction for
  each requested component count.
- Preserve a fallback to the old per-component materialized fit if a
  max-component fold fit fails.

Implementation:

- Core: `cpp/src/core/sweep.cpp`.
- Reconstruct prefix coefficients from the fitted model arrays:
  `W[:,:k]`, `P[:,:k]`, `Q[:,:k]`, `x_mean/x_scale`, `y_mean/y_scale`.
- `n4m_aom_sweep_run` and `n4m_aom_chain_sweep_run` benefit automatically
  because both delegate PLS scoring to `n4m_sweep_run`.

Validation:

- Added C++ test `sweep/pls_component_grid_matches_materialized_cv_scores`.
- CPU build and `n4m_tests`: 320 passed, 0 failed.
- CUDA build and `n4m_tests` with `CUDA_VISIBLE_DEVICES=0`: 320 passed,
  0 failed.
- Python direct tests without pytest: 28 passed.
- CUDA Python smoke: `n4m.sweep_run` ABI `(1, 16, 0)`, PLS candidate table
  `(3, 4)`, OOF RMSE assertion passed.
- `catalog/scripts/validate.py --strict-abi`: PASS, 194 methods, 679/679
  exported `n4m_*` symbols covered.
- CPU timing smoke, ABI 1.16.0:
  `native_sweep_pls` over three component candidates took 3.55 ms for 64x64,
  5.17 ms for 128x128, and 6.72 ms for 192x256.
- CUDA-build timing smoke, ABI 1.16.0:
  `native_sweep_pls` over three component candidates took 41.92 ms for 64x64,
  46.31 ms for 128x128, and 69.67 ms for 192x256. This validates CUDA-build
  behavior, not fused GPU acceleration.

Remaining work:

- This is still not IKPLS. It reduces repeated component fits but still
  materializes each transformed chain/fold.
- Batched IKPLS and fused CUDA/operator-moment kernels remain the next
  acceleration layer.

## 2026-06-04 — User-defined strict AOM chain sweep, ABI 1.16.0

Decision:

- Ship `n4m_aom_chain_sweep_run` as the ABI-stable arbitrary strict-linear
  preprocessing-chain surface.
- Use flat C arrays instead of opaque chain objects:
  `chain_offsets`, `op_kinds`, `param_offsets`, `params`.
- Keep the strict-linear invariant: identity, detrend, Savitzky-Golay,
  Norris-Williams, finite difference, Whittaker and FCK are accepted; stateful
  operators such as SNV/MSC/EMSC are rejected in this path.
- Reuse the exact same scoring path as `n4m_aom_sweep_run`, so Ridge/PLS grids,
  explicit folds and candidate table semantics stay identical.

Implementation:

- Core: `run_aom_chain_sweep` in `cpp/src/core/aom_sweep.cpp`.
- C ABI: declaration in `cpp/include/n4m/pls.h`, wrapper in
  `cpp/src/c_api/c_api_method_result.cpp`.
- Python: `n4m.aom_chain_sweep_run(X, y, chains, ...)`, accepting strings,
  tuples and dictionaries for operator specs.
- Catalog/doc: `catalog/methods/aom_pop.aom_chain_sweep.yaml`,
  `docs/methods/aom_chain_sweep_run.md`.

Validation:

- C++ tests added:
  `aom_chain_sweep/custom_descriptor_contract` and
  `aom_chain_sweep/rejects_non_strict_operator`.
- CPU build and `n4m_tests`: 319 passed, 0 failed.
- Python direct tests without pytest: 28 passed.
- CUDA build and `n4m_tests` with `CUDA_VISIBLE_DEVICES=0`: 319 passed,
  0 failed.
- CUDA Python smoke: `n4m.aom_chain_sweep_run` ABI `(1, 16, 0)`, candidate
  table `(6, 5)`, OOF RMSE assertion passed.
- ABI snapshots regenerated: Linux 680 symbols, macOS/Windows 679.
- `catalog/scripts/validate.py --strict-abi`: PASS, 194 methods, 679/679
  exported `n4m_*` symbols covered.
- CPU timing smoke, ABI 1.16.0:
  custom5 profile with 5 chains and 25 candidates took 8.15 ms for 48x64 and
  19.51 ms for 80x128.
- CUDA-build timing smoke, ABI 1.16.0:
  custom5 profile with 5 chains and 25 candidates took 257.22 ms for 48x64
  and 117.95 ms for 80x128. This validates CUDA-build behavior, not fused GPU
  acceleration.

Remaining work after this ABI:

- The chain descriptor still materializes each transformed `X`.
- Batched IKPLS and fused operator-moment/GPU kernels remain open.

## 2026-06-04 — Configurable native AOM sweep, ABI 1.15.0

Decision:

- Ship `n4m_aom_sweep_run` as the reusable AOM preprocessing sweep surface.
- Reuse the native strict-linear compact/wide chain bank from robust-HPO.
- Delegate per-chain model scoring to `n4m_sweep_run`, so users can control
  Ridge lambda grids, PLS component grids, active heads and explicit folds.
- Keep arbitrary operator descriptors, batched IKPLS and fused CUDA kernels as
  later optimization layers.

Implementation:

- Core: `cpp/src/core/aom_sweep.hpp`, `cpp/src/core/aom_sweep.cpp`.
- C ABI: declaration in `cpp/include/n4m/pls.h`, wrapper in
  `cpp/src/c_api/c_api_method_result.cpp`.
- Python: `n4m.aom_sweep_run(X, y, profile=..., fold_ids=...,
  ridge_lambdas=..., pls_components=..., heads=...)`.
- Catalog/doc: `catalog/methods/aom_pop.aom_sweep.yaml`,
  `docs/methods/aom_sweep_run.md`.
- Benchmark: `benchmarks/cross_binding/bench_aom_sweep_timing.py`.

Validation:

- C++ test `aom_sweep/compact_contract_and_oof_score` validates the compact
  candidate table shape, chain coverage, selected candidate consistency,
  explicit fold ids and selected OOF RMSE.
- CPU build and `n4m_tests`: 317 passed, 0 failed.
- CUDA build and `n4m_tests` with `CUDA_VISIBLE_DEVICES=0`: 317 passed,
  0 failed.
- Python direct tests without pytest: 27 passed.
- CUDA Python smoke: `n4m.aom_sweep_run` ABI `(1, 15, 0)`, candidate table
  `(24, 5)`, OOF RMSE assertion passed.
- ABI snapshots regenerated: Linux 679 symbols, macOS/Windows 678.
- `catalog/scripts/validate.py --strict-abi`: PASS, 193 methods, 678/678
  exported `n4m_*` symbols covered.
- CPU timing smoke, ABI 1.15.0:
  compact profile with 12 chains and 60 candidates took 9.83 ms for 48x64 and
  40.14 ms for 80x128.
- CUDA-build timing smoke, ABI 1.15.0:
  compact profile with 12 chains and 60 candidates took 862.90 ms for 48x64
  and 421.98 ms for 80x128. This validates CUDA-build behavior, not fused GPU
  acceleration.

Remaining work:

- Add public strict-linear operator descriptors for arbitrary preprocessing
  chain campaigns.
- Replace materialized PLS CV with batched IKPLS.
- Add fused CUDA kernels and grouped-GEMM/operator batching before claiming
  200k-chain GPU screening.

## 2026-06-04 — Native Ridge/PLS sweep, ABI 1.14.0

Decision:

- Ship `n4m_sweep_run` as the first product sweep ABI.
- Support exact Ridge CV over moment fold subtraction or precomputed dual folds.
- Support fold-local materialized PLS component screening through the existing
  native PLS model path.
- Keep fused batched IKPLS as the later optimization target.

Implementation:

- Core: `cpp/src/core/sweep.hpp`, `cpp/src/core/sweep.cpp`.
- C ABI: declaration in `cpp/include/n4m/pls.h`, wrapper in
  `cpp/src/c_api/c_api_method_result.cpp`.
- Python: `n4m.sweep_run(X, y, cv=..., fold_ids=..., ridge_lambdas=...,
  pls_components=..., heads=("ridge", "pls"))`.
- Catalog/doc: `catalog/methods/utilities.sweep.yaml`,
  `docs/methods/sweep_run.md`.
- Optimisation: moment primal solve is kept for `p <= n_train`; spectral
  folds with `p > n_train` use a precomputed dual design and reuse `K = XX'`
  across Ridge lambdas.

Validation:

- C++ test `sweep/ridge_oof_matches_materialized_cv` compares selected OOF
  predictions from moment subtraction against materialized fold-by-fold
  `n4m_ridge_fit`.
- C++ test `sweep/selects_minimum_candidate_and_generates_folds` validates
  candidate ranking and generated fold ids.
- C++ test `sweep/pls_oof_matches_materialized_cv` compares selected PLS OOF
  predictions against fold-by-fold `n4m_model_fit` / `n4m_model_predict`.
- CPU build and `n4m_tests`: 316 passed, 0 failed.
- CUDA build and `n4m_tests` with `CUDA_VISIBLE_DEVICES=0`: 316 passed,
  0 failed.
- Python direct tests: 32 passed.
- CUDA Python smoke: `n4m.sweep_run` ABI `(1, 14, 0)`, PLS candidate table
  `(3, 4)`, OOF RMSE assertion passed.
- ABI snapshots regenerated: Linux 678 symbols, macOS/Windows 677.
- `catalog/scripts/validate.py --strict-abi`: PASS, 192 methods, 677/677
  exported `n4m_*` symbols covered.
- Timing smoke added:
  `benchmarks/cross_binding/bench_moment_sweep_timing.py`.
- CPU timing smoke, ABI 1.14.0:
  `native_sweep_ridge` vs `materialized_cv` selected the same lambdas and
  matched CV RMSE on all cells. Median Ridge timings were 2.48 ms vs 4.98 ms
  for 64x64, 16.89 ms vs 25.98 ms for 128x128, and 54.94 ms vs 85.40 ms for
  192x256. Materialized PLS sweep timings were 0.46 ms, 1.48 ms and 4.37 ms
  on the same cells for three component candidates.
- CUDA-build timing smoke, native-only ABI 1.14.0:
  Ridge timings were 12.79 ms, 17.74 ms and 94.65 ms for 64x64, 128x128 and
  192x256. Materialized PLS timings were 47.90 ms, 58.67 ms and 92.24 ms. This
  validates CUDA-build behavior, not fused GPU acceleration for PLS.

Remaining work:

- Replace materialized PLS CV with batched IKPLS inside `n4m_sweep_run`.
- Add strict-linear operator descriptors so the sweep can score preprocessing
  variants without materializing every transformed matrix.
- Add larger timing campaigns once batched IKPLS and operator descriptors land.

## 2026-06-04 — Native moment substrate, ABI 1.13.0

Decision:

- Ship `n4m_moments_compute`, `n4m_moments_subset_compute` and
  `n4m_moments_subtract` as the first native moment layer.
- Keep the API on `n4m_method_result_t` rather than adding a new opaque handle
  for v1. This matches the existing extra-model result surface and keeps
  Python wrappers direct.
- Return raw row-additive moments (`x_sum`, `y_sum`, `xtx`, `xty`, `yty`) and
  centered moments (`x_mean`, `y_mean`, `cxx`, `cxy`, `cyy`).
- Subtract only raw moments, then recompute centering from the remaining row
  count. This is the invariant needed for exact CV fold subtraction.

Implementation:

- Core: `cpp/src/core/moments.hpp`, `cpp/src/core/moments.cpp`.
- C ABI: declarations in `cpp/include/n4m/pls.h`, wrappers in
  `cpp/src/c_api/c_api_method_result.cpp`.
- Python: `n4m.moments(X, y, row_indices=None)` and
  `n4m.moments_train_from_heldout(X, y, heldout_indices)`.
- Catalog/doc: `catalog/methods/utilities.moments.yaml`,
  `docs/methods/moments.md`.

Validation:

- CPU build and `n4m_tests`: 313 passed, 0 failed.
- CUDA build and `n4m_tests` with `CUDA_VISIBLE_DEVICES=0`: 313 passed, 0 failed.
- Python direct tests: 29 passed.
- CUDA Python smoke: `n4m.moments` ABI `(1, 13, 0)`, Gram and fold-recenter
  assertions passed.
- ABI snapshots regenerated: Linux 677 symbols, macOS/Windows 676.
- `catalog/scripts/validate.py --strict-abi`: PASS, 191 methods, 676/676
  exported `n4m_*` symbols covered.

Remaining work:

- Implement `n4m_sweep_run` for batched Ridge/PLS CV over variant descriptors.
- Add strict-linear operator descriptors that can update moments without
  materializing every transformed `X`.
- Decide whether the next native sweep stage should start with moment Ridge
  CV, IKPLS PLS CV, or a mixed Ridge/PLS candidate table.
- Add a timing benchmark once the sweep API exists; the current moment layer is
  a correctness substrate, not yet the 200k-chain grinder.

## 2026-06-05 — Staged strict-chain cartesian orchestration (Python workflow)

Decision:

- Promote the staged cartesian runner (handoff gap #5, proto `cartesian.py` /
  `impact.py`) to a first-class Python workflow `aom_staged_chain_campaign`,
  rather than a one-off benchmark script.
- Make it pure orchestration over the existing helpers — no new C ABI symbol and
  no new numerical kernel, so `libn4m` stays the single engine.
- A stage = one score-only strict-linear screen (`compact` / `wide` / `lab`
  profile, or an explicit override dict mixing profiles and Ridge/PLS/mixed head
  plans). The campaign merges per-stage retained candidates, keeps the top global
  + per-head rows across stages, exact-CV refits the union once, and attaches
  preprocessing-impact + screen-vs-refit rank diagnostics plus an optional
  offline-only holdout audit.

Constraints honoured:

- Strict-linear chains only (no hors-moment nonlinear lifts).
- No dataset/source/id/name input or selection; stage `name` is a cosmetic label
  and renaming stages provably cannot change the winner.
- Production selection is exact-CV refit on train only
  (`selection_uses_test_set=False`); `X_audit`/`y_audit` are scored under
  `report["audit"]` with `audit_only=True` and never drive selection.

Implementation:

- `bindings/python/src/n4m/python.py`: `aom_staged_chain_campaign` plus the
  `_normalize_staged_chain_stages` / `_aom_merge_staged_candidates` /
  `_staged_normalized_heads` helpers and the `_AOM_STAGED_PLANS` presets. Reuses
  `aom_chain_score_campaign`, `_aom_screen_refit_candidate_union`,
  `aom_refit_candidates`, `aom_candidate_preprocessing_impact`,
  `aom_candidate_rank_diagnostics`, `aom_evaluate_candidates`.
- Re-exported from `n4m`, `n4m.aom`, `n4m.moment` (attr + `__all__` + an
  inventory `staged_chain_campaign` row mapped to
  `aom_pop.aom_staged_chain_campaign`, `catalog_role=catalog_binding`).
- Added `NativeAOMStagedChainCampaignRegressor`, a reusable sklearn wrapper that
  runs the staged campaign, selects by train exact-CV `refit_cv_rmse` only, and
  refits the selected row through `NativeAOMFixedCandidateRegressor` in
  final-only mode. It is exported from `n4m`, `n4m.sklearn`, `n4m.aom` and
  `n4m.moment`, with separate `staged_chain_campaign_estimator` inventory rows
  that intentionally omit `X_audit` / `y_audit` options.
- Report schema `n4m.aom_staged_chain_campaign.v1`; `rows` are consumable by
  `NativeAOMFixedCandidateRegressor.from_refit_report`.
- Doc: `docs/methods/aom_staged_chain_campaign.md`.
- Tests: `bindings/python/tests/test_aom_staged_campaign.py` (9 cases).

Validation:

- `py_compile` on changed sources + tests: OK.
- `pytest test_aom_staged_campaign.py`: 9 passed.
- `pytest test_aom_moment_facade.py test_moment_model_wrappers.py
  test_aom_staged_campaign.py`: 70 passed.
- `catalog/scripts/validate.py --strict-abi`: PASS, 200 methods, 701/701.
- `ruff check` on changed files: no issues. `git diff --check`: clean.

Remaining work:

- Wire the runner into the cross-binding benchmark campaign against the robust
  AOM / TabPFN baselines (handoff suggested next step).

Follow-up staged resume:

- Added `checkpoint_dir`, `resume` and `max_chunks_per_run` to
  `aom_staged_chain_campaign`.
- Each stage now forwards to the existing `aom_chain_score_campaign`
  checkpoint/resume mechanism with a stable per-stage JSON checkpoint path.
- Partial staged reports are still exact-refit-able over currently retained
  rows and expose `screen_complete=False`,
  `n_remaining_stage_chunks_total`, per-stage chunk counters and checkpoint
  paths.

Follow-up timing smoke:

- Added `benchmarks/cross_binding/bench_aom_staged_chain_campaign_timing.py`
  to time the staged campaign on synthetic data and write one CSV row per
  `--plans` entry.
- The script exposes the screen/refit controls and the existing CPU/GPU knobs,
  and records retention counts, selected exact-CV winner, impact/rank
  availability, refit counters, route fractions, `library_path` and ABI.
- The method doc now states that this measures orchestration plus exact-refit
  timing only; it is not the future fused IKPLS grinder benchmark.
- Checkpoint smoke: first pass with `max_chunks_per_run=1` reports
  `screen_complete=False`, second pass resumes to `screen_complete=True`; with
  `--repeats > 1`, the benchmark isolates checkpoint state per repeat.

Follow-up oracle comparison:

- Added `benchmarks/cross_binding/compare_aom_staged_to_oracles.py`, an offline
  CSV comparator that normalizes per-dataset scores and joins a future staged
  campaign target against the real local oracle artifacts:
  `/home/delete/nirs4all/nirs4all-aom/benchmarks/runs/scenarios/paper_aom_aompls_seeds012/results.csv`,
  `/home/delete/nirs4all/nirs4all-aom/benchmarks/runs/ridge/all54_headline/results.csv`
  `/home/delete/nirs4all/nirs4all-aom/benchmarks/pls/cohort_regression.csv`
  and `/home/delete/nirs4all/nirs4all-lab/benchmark/results/1_master_results.csv`.
- Default oracles are separated as `aom_pls_oracle`, `aom_ridge_oracle` and
  `tabpfn_oracle`; the comparator chooses the best score per dataset within
  each source, not a default row. The AOM-PLS oracle filters out plain
  `PLS-standard-*`, the AOM-Ridge oracle filters out plain `Ridge-raw` and
  non-Ridge AOM-PLS rows, and the TabPFN oracle chooses the best available
  Raw/Opt reference per dataset.
- Smoke without a target produced 61 dataset keys in the oracle union
  (`53` AOM-PLS-family rows, `53` AOM-Ridge-family rows, `61` TabPFN rows).
  The filtered winner counts were `30` TabPFN, `19` AOM-Ridge and `12` AOM-PLS.

Follow-up real-cohort runner:

- Added `benchmarks/cross_binding/run_aom_staged_real_cohort.py`, a lightweight
  real-data runner for fixed train/test NIRS splits under
  `/home/delete/nirs4all/nirs4all-data/regression`.
- It loads a cohort CSV such as
  `/home/delete/nirs4all/nirs4all-aom/benchmarks/runs/ridge/diverse11_cohort.csv`,
  runs `aom_staged_chain_campaign` with `X_audit` / `y_audit`, and writes rows
  compatible with the oracle comparator.
- Output CSVs are replaced by default; `--resume` preserves an existing output
  and skips already-OK dataset rows.
- Main reported `rmsep` is the test RMSE of the candidate selected by train CV
  (`eval.best_cv`), not the test-best candidate. The test-best candidate is
  written separately as `audit_oracle_rmse` for offline screen-recall analysis.
- Smoke command on one real dataset with `max_chains=2`, `cv=3` completed in
  about 1.1 s campaign time and fed successfully into
  `compare_aom_staged_to_oracles.py`.
- Calibration run on 10 `diverse11_cohort.csv` regression datasets with
  `plan=compact`, `max_chains=12`, `top_k=12`, `refit_top_k=6` completed and
  wrote `/tmp/aom_staged_real_cohort_10.csv`. Comparison summary:
  paired target wins `0/9` vs `aom_pls_oracle`, `0/10` vs
  `aom_ridge_oracle`, `1/10` vs `tabpfn_oracle`; median ratios were `1.17183`,
  `1.22661`, and `1.2894` respectively after filtering true AOM and TabPFN
  Raw/Opt oracles. Treat this as a tiny-budget runner
  smoke, not evidence about the full staged/cartesian ceiling.

Follow-up staged sklearn/CUDA readiness:

- Added `NativeAOMStagedChainCampaignRegressor` to the coverage matrix and
  documented it as the reusable estimator surface for staged campaigns.
- Ran a one-GPU staged PLS smoke against `build/cuda-on/cpp/src/libn4m.so` with
  `CUDA_VISIBLE_DEVICES=0`, `plan=compact`, `max_chains=4`, forced
  `cuda_pls_min_device_features=1` and `cuda_pls_parallel_folds=True`.
- Wrote `benchmarks/cross_binding/aom_staged_chain_campaign_timing_cuda_smoke.csv`.
  The smoke kept `selection_uses_test_set=False`, `screen_complete=True`, and
  reported `n_pls_moment_cuda_device_cv_fits=9`,
  `n_pls_moment_host_cv_fits=0`, `n_pls_moment_cuda_parallel_fold_jobs=9`.
- Extended `benchmarks/cross_binding/aom_moment_cuda_facade_smoke.py` so the
  existing one-process CUDA facade smoke also asserts
  `aom_staged_chain_campaign` and `NativeAOMStagedChainCampaignRegressor` are
  exported from `n4m`, `n4m.aom` and `n4m.moment`, then fits the staged sklearn
  estimator on the CUDA build. The regenerated
  `aom_moment_cuda_facade_smoke.json` reports staged estimator
  `n_pls_moment_cuda_device_cv_fits=8`, `n_pls_moment_host_cv_fits=0`,
  `n_pls_moment_cuda_parallel_fold_jobs=8` and
  `selection_uses_test_set=False`.
- Extended that same CUDA facade smoke to cover the preconfigured
  `aom_sweep_run` profile path and the reusable AOM-PLS / POP-PLS selector
  surfaces. The regenerated JSON reports `aom_profile_sweep` with
  `n_pls_moment_cuda_device_cv_fits=48` and host PLS CV fits at `0`; the
  `NativeAOMPLSRegressor` and `NativePOPPLSRegressor` replay the native
  input-space coefficients with max absolute prediction error below `1e-10`.

Follow-up diversity wrapper CUDA readiness:

- Extended `benchmarks/cross_binding/bench_aom_ridge_blender_timing.py` and
  `benchmarks/cross_binding/bench_aom_operator_pls_stack_timing.py` with
  `--mode native|wrapper|both`.
- Regenerated the one-GPU CUDA smoke CSVs with `--mode both`, so
  `aom_ridge_blender_timing_cuda_smoke.csv` now includes
  `native_aom_ridge_blender` and `native_aom_ridge_blender_sklearn` rows, and
  `aom_operator_pls_stack_timing_cuda_smoke.csv` now includes
  `native_aom_operator_pls_stack` and
  `native_aom_operator_pls_stack_sklearn` rows.
- The sklearn rows record `prediction_replay_max_abs_error`, proving the
  wrappers replay the native folded model state on the CUDA build path; current
  smokes are within `1e-10`.

Follow-up staged catalog readiness:

- Added a dedicated Python-backed catalog method
  `aom_pop.aom_staged_chain_campaign` with benchmark entry
  `benchmarks/cross_binding/bench_aom_staged_chain_campaign_timing.py`.
- Updated the `n4m.aom` and `n4m.moment` inventory rows so
  `staged_chain_campaign` is the catalog binding and
  `staged_chain_campaign_estimator` is a sklearn wrapper of that staged method.
- Brought the legacy catalog source and split files back into sync for the
  existing PCR and moment-stack entries; the split now contains 201 methods.
- Validation after the catalog split:
  `split_legacy_methods.py --check` passed with 201 per-method files,
  `catalog/scripts/validate.py --strict-abi` passed with 201 methods and
  `701/701` ABI symbols, `catalog/scripts/validate.py --check-references`
  passed with `201/201` production methods covered, and the targeted
  facade/wrapper/staged pytest set stayed at 70 passed.

Follow-up facade invariant guard:

- Added a generic `n4m.aom` / `n4m.moment` inventory test that rejects public
  `config_options` exposing dataset/source/database/metadata routing knobs.
  Legitimate fold/candidate identifiers such as `fold_ids`, `with_ids` and
  `include_identity` remain allowed.
- Added a catalog-wide Python binding test so every per-method
  `bindings.python` declaration resolves to a real exported object and legacy
  aliases resolve too. This pins the catalog/source split correction that
  removed stale, wrong bindings from `augmentation.edge_artifacts.edge_artifacts`
  and `diagnostics.approximate_press`.
- Extended `catalog/scripts/validate.py` so every non-null
  `bench.registry_entry` must point to an existing benchmark file. This keeps
  Python-backed AOM/moment methods such as `aom_pop.aom_staged_chain_campaign`
  and `models.ensembles.moment_stack` tied to runnable timing evidence.
- Added `test_aom_moment_cuda_smoke_artifacts.py`, a release-readiness guard
  over the committed one-GPU CUDA smoke JSON/CSVs. It verifies the CUDA facade
  artifact, staged campaign smoke, moment-stack PLS base smoke, and native +
  sklearn replay rows for the Ridge blender and operator PLS stack. It now also
  covers robust-HPO native CUDA build smokes, global AOM profile-sweep device
  routing, AOM-PLS / POP-PLS reusable coefficient replay, and the PLS/mixed/Ridge
  screen-refit CUDA build artifacts.
- Added `test_aom_benchmark_tools.py`, a fast guard for the offline staged
  benchmark helpers. It pins the oracle comparator's filtering of true
  AOM-PLS/AOM-Ridge and TabPFN Raw/Opt rows, and it verifies that the real
  cohort runner reports the train-CV-selected audit row (`eval.best_cv`) while
  keeping the test-best retained row (`best_eval`) audit-only.
- Updated the public methods index for `aom_pls`, `pop_pls` and
  `aom_staged_chain_campaign`, and added a facade test that every
  `docs/methods/*.md` page advertised by `n4m.aom` / `n4m.moment` is linked
  from `docs/methods/index.md`.
- Targeted facade pytest now has 13 tests, and the combined
  benchmark-tool/catalog/CUDA-artifact/facade/wrapper/staged pytest set reports
  87 passed.

Follow-up direct moment-head CUDA readiness:

- Audited the direct reusable heads requested around Ridge/PLS-adjacent moment
  methods: Ridge, PLS, PCR, CPPLS, continuum regression and ECR.
- The first CUDA smoke attempt exposed a stale `build/cuda-on` library missing
  `n4m_pcr_fit`; rebuilt with
  `/home/delete/.venv/bin/cmake --build --preset cuda-on --parallel`.
- Verified the rebuilt CUDA shared library exports the direct head symbols:
  `n4m_ridge_fit`, `n4m_pcr_fit`, `n4m_cppls_fit`,
  `n4m_continuum_regression_fit` and `n4m_ecr_fit`.
- Generated
  `benchmarks/cross_binding/direct_moment_heads_timing_cuda_smoke.csv`, covering
  the direct heads on three shapes as both ABI-close functions and
  sklearn-style replay wrappers through `build/cuda-on/cpp/src/libn4m.so`.
- Extended `test_aom_moment_cuda_smoke_artifacts.py` so the committed artifact
  is release-guarded for exact method coverage, three shapes per method, both
  native/sklearn backends per method/shape, CUDA build path, finite RMSE and
  replay error at numerical noise level.
- After this direct-head guard, the targeted
  benchmark-tool/catalog/CUDA-artifact/facade/wrapper/staged pytest set reports
  88 passed.

Follow-up sweep/selector CUDA artifact refresh:

- Regenerated `benchmarks/cross_binding/moment_sweep_timing_cuda_smoke.csv`,
  `benchmarks/cross_binding/aom_sweep_timing_cuda_smoke.csv` and
  `benchmarks/cross_binding/aom_selector_timing_cuda_smoke.csv` against the
  current `build/cuda-on/cpp/src/libn4m.so` after finding the previous committed
  artifacts still referenced ABI `1.18.0`.
- The refreshed artifacts now report ABI `1.21.0` and `build/cuda-on` library
  paths.
- Added release-readiness guards over those artifacts:
  - moment sweep exact PLS CV rows must use `cuda_pls_parallel_folds=True`,
    `cuda_pls_min_device_features=1`, zero host CV fits and device CV fits equal
    to total PLS moment CV fits;
  - AOM sweep exact PLS CV rows across compact/global, explicit chain,
    Whittaker, FCK, Gaussian, PLS exact/proxy and Ridge coverage must also have
    zero host CV fits and device CV fits equal to total PLS moment CV fits;
  - AOM-PLS / POP-PLS selector function and sklearn rows must replay with
    `replay_max_abs <= 1e-10`.
- After these sweep/selector guards, the targeted
  benchmark-tool/catalog/CUDA-artifact/facade/wrapper/staged pytest set reports
  91 passed.

Follow-up real-cohort CUDA calibration:

- Added staged campaign route counters to `aom_staged_chain_campaign` reports
  and propagated them through `run_aom_staged_real_cohort.py` CSV output:
  `n_screen_pls_moment_*`, `n_refit_pls_moment_*`, plus the CUDA knob columns.
  This makes real benchmark rows auditable for CPU-vs-GPU routing instead of
  relying on process logs.
- Ran a one-GPU compact 10-dataset real-cohort calibration with
  `N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so`, `CUDA_VISIBLE_DEVICES=0`,
  `plan=compact`, `cv=4`, `max_chains=12`, `top_k=12`, `refit_top_k=6`,
  Ridge+PLS heads, `cuda_pls_min_device_features=1`,
  `cuda_pls_parallel_folds=True`, and `backend_min_cuda_product=1`.
- Output `/tmp/n4m_aom_staged_real_cohort_10_cuda.csv`: `10/10` rows OK,
  `screen_complete=True`, `selection_uses_test_set=False`, ABI `1.21.0` and
  `library_path=build/cuda-on/cpp/src/libn4m.so`.
- Route counters: screen PLS moment CV fits `480`, screen host fits `0`, screen
  CUDA device fits `480`; refit PLS moment CV fits `140`, refit host fits `0`,
  refit CUDA device fits `140`.
- Timing: total fit time `81.69 s`, median `3.43 s`; the BERRY row dominated at
  about `51.17 s`.
- Offline oracle comparison written to
  `/tmp/n4m_aom_staged_oracle_comparison_10_cuda.csv` and
  `/tmp/n4m_aom_staged_oracle_comparison_10_cuda.md`:
  - AOM-PLS oracle: paired `9`, target wins `0`, median ratio `1.21015`;
  - AOM-Ridge oracle: paired `10`, target wins `0`, median ratio `1.24282`;
  - TabPFN oracle: paired `10`, target wins `1`, median ratio `1.33059`.
- Interpretation: the compact staged workflow is now demonstrably train-only
  and GPU-routed on real NIRS data, but a tiny `max_chains=12` budget remains
  far behind the robust AOM oracles. This benchmark is useful calibration, not
  evidence of the final larger-cartesian ceiling.
- After the route-counter runner guard, the targeted
  benchmark-tool/catalog/CUDA-artifact/facade/wrapper/staged pytest set reports
  92 passed.

Follow-up property-filtered compact+wide calibration:

- Added optional real-cohort runner filters by measured dataset properties:
  `--max-train-samples`, `--max-features`,
  `--max-train-feature-product`. Rows outside budget are written as
  `status=skipped` with `error_message=property_filter:...`; this is benchmark
  resource control, not model routing or selection by dataset identity.
- Ran a one-GPU incremental campaign with `plan=compact_wide`, `cv=4`,
  `max_chains=24`, `top_k=16`, `refit_top_k=8`, Ridge+PLS heads,
  `cuda_pls_min_device_features=1`, `cuda_pls_parallel_folds=True`,
  `backend_min_cuda_product=1`, and `max_features=1200`.
- Output `/tmp/n4m_aom_staged_real_cohort_10_cuda_compact_wide_p1200.csv`:
  `8` OK rows and `2` skipped rows, both skipped only because
  `n_features>1200`; OK rows used ABI `1.21.0` and
  `build/cuda-on/cpp/src/libn4m.so`.
- Route counters on OK rows: screen PLS moment CV fits `1152`, screen host fits
  `0`, screen CUDA device fits `1152`; refit PLS moment CV fits `160`, refit
  host fits `0`, refit CUDA device fits `160`.
- Timing on OK rows: total fit time `45.06 s`, median `5.31 s`.
- Versus the previous compact run on the 8 common OK rows, compact+wide improved
  `3/8`, was identical on `4/8`, and degraded slightly on `1/8`; median ratio
  versus compact was `1.0`.
- Offline oracle comparison written to
  `/tmp/n4m_aom_staged_oracle_comparison_10_cuda_compact_wide_p1200.csv` and
  `/tmp/n4m_aom_staged_oracle_comparison_10_cuda_compact_wide_p1200.md`:
  - AOM-PLS oracle: paired `7`, target wins `0`, median ratio `1.17183`;
  - AOM-Ridge oracle: paired `8`, target wins `0`, median ratio `1.26483`;
  - TabPFN oracle: paired `8`, target wins `1`, median ratio `1.33059`.
- Interpretation: the extra wide stage helps some datasets at small budget but
  still does not close the robust AOM oracle gap. The new property filters make
  subsequent budget sweeps cleaner without dataset-name routing.
- After the property-filter guard, the targeted
  benchmark-tool/catalog/CUDA-artifact/facade/wrapper/staged pytest set reports
  93 passed.

Follow-up custom staged JSON calibration:

- Added `--stages-json` and `--stages-json-file` to
  `run_aom_staged_real_cohort.py`. The value must be a non-empty JSON list of
  profile strings or stage objects accepted by `n4m.aom_staged_chain_campaign`.
  The parsed list is passed through as `stages=...`, and output rows record a
  compact `stages_json` field for audit/replay.
- Ran a mini one-GPU proof campaign with a custom two-stage config: compact
  plus a small Savitzky-Golay lab stage (`savgol_smooth`,
  `savgol_derivative`, and `savgol_smooth -> finite_difference` templates),
  `max_features=1200`, and CUDA PLS controls forced on.
- Output `/tmp/n4m_aom_staged_real_cohort_3_cuda_custom_stages.csv`: `2` OK
  rows and `1` skipped row (`n_features>1200`), `plan=custom`,
  non-empty `stages_json`, ABI `1.21.0`.
- Route counters on OK rows: screen PLS moment CV fits `128`, screen host fits
  `0`, screen CUDA device fits `128`; refit PLS moment CV fits `32`, refit host
  fits `0`, refit CUDA device fits `32`.
- Offline oracle comparison written to
  `/tmp/n4m_aom_staged_oracle_comparison_3_cuda_custom_stages.csv` and
  `/tmp/n4m_aom_staged_oracle_comparison_3_cuda_custom_stages.md`; target wins
  were `0` vs AOM-PLS, AOM-Ridge and TabPFN on the two paired rows.
- Interpretation: this is a functionality proof for JSON-driven preprocessing
  family campaigns. It enables the incremental preprocessing experiments without
  dataset-name routing, but it is not evidence of a scoring improvement.
- After the custom-stages JSON guard, the targeted
  benchmark-tool/catalog/CUDA-artifact/facade/wrapper/staged pytest set reports
  94 passed.

Follow-up staged variant comparison helper:

- Added `benchmarks/cross_binding/compare_aom_staged_variants.py`, an offline
  CSV-only audit tool for comparing compact/wide/custom staged campaign outputs
  against each other before running the external AOM/TabPFN oracle comparator.
- The tool groups by campaign configuration columns only (`plan`,
  `stages_json`, heads, budget, property filters and CUDA knobs). Dataset keys
  are used only for paired evaluation against a chosen baseline, never as
  routing or production-selection inputs.
- The CSV/markdown summary reports OK/skipped/error counts, median
  score/timing, total fit time, screen/refit PLS moment route totals, and
  paired win/loss/tie plus score ratios versus either `--baseline` CSVs or a
  `--baseline-label` from the supplied inputs.
- Added tests covering config grouping, skipped-row handling, route-counter
  totals, baseline-label pairing and the absence of dataset/source identity
  fields from the config key.
- After this helper, the targeted
  benchmark-tool/catalog/CUDA-artifact/facade/wrapper/staged pytest set reports
  97 passed.

Follow-up direct PLS reusable head:

- Added `n4m.pls` / `n4m.python.pls`, a direct reusable moment PLS surface
  backed by `n4m_sweep_run` with `heads=("pls",)`. It supports a fixed
  `n_components` or an explicit train-CV `pls_components` grid and preserves
  the existing CUDA PLS controls (`cuda_pls_parallel_folds`,
  `cuda_pls_min_device_features`, `cuda_pls_many_batched`).
- Added `NativePLSRegressor` to `n4m.sklearn`, top-level `n4m`, and
  `n4m.moment`. The wrapper predicts from replayable input-space coefficients
  plus intercept and reports the PLS moment route counters from the sweep
  result.
- Wired the existing `models.pls.pls_fit_simple` catalog entry to
  `n4m.python.pls` and documented the direct `n4m` usage in
  `docs/methods/pls.md`.
- Regenerated `direct_moment_heads_timing_cuda_smoke.csv`; it now covers Ridge,
  PLS, PCR, CPPLS, continuum regression and ECR across three shapes with both
  function and sklearn replay rows on `build/cuda-on/cpp/src/libn4m.so`.
- Validation after this addition:
  - targeted benchmark-tool/catalog/CUDA-artifact/facade/wrapper/staged pytest:
    98 passed;
  - `catalog/scripts/validate.py --strict-abi`: PASS;
  - `catalog/scripts/validate.py --check-references`: PASS;
  - `catalog/scripts/split_legacy_methods.py --check`: PASS.

Follow-up direct-head timing catalog alignment:

- Wired the six reusable direct moment heads (`models.regularized.ridge`,
  `models.pls.pls_fit_simple`, `models.pls.pcr`, `models.pls.cppls`,
  `models.regularized.continuum_regression`, `models.specialized.ecr`) to the
  shared timing registry entry
  `benchmarks/cross_binding/bench_direct_moment_heads_timing.py`.
- Added a catalog guard proving those six per-method YAML files keep that
  registry entry and that the timing script exists.
- Validation after this alignment:
  - targeted benchmark-tool/catalog/CUDA-artifact/facade/wrapper/staged pytest:
    99 passed;
  - `catalog/scripts/validate.py --strict-abi`: PASS;
  - `catalog/scripts/validate.py --check-references`: PASS;
  - `catalog/scripts/split_legacy_methods.py --check`: PASS.

Follow-up direct PLS CUDA route proof:

- Extended `bench_direct_moment_heads_timing.py` with the PLS CUDA controls
  used by the sweep/AOM smokes:
  `--cuda-pls-min-device-features`, `--cuda-pls-parallel-folds` and
  `--cuda-pls-many-batched`.
- The direct-head timing CSV now records PLS route counters for both
  `native_function` and `sklearn_fit_predict` rows:
  `n_pls_moment_cv_fits`, host/device CV fits, CUDA fold batches/jobs and final
  route counters.
- Regenerated `direct_moment_heads_timing_cuda_smoke.csv` on one GPU with
  `--cuda-pls-min-device-features 1 --cuda-pls-parallel-folds`. PLS rows now
  report `n_pls_moment_cv_fits=4`, host CV fits `0`, CUDA device CV fits `4`
  and CUDA fold jobs `4` for each of the three shapes and both backends.
- Strengthened the CUDA artifact guard so direct PLS rows must prove
  device-CV routing, not only CUDA library loading and replay.
- Validation after this route-proof update:
  - targeted benchmark-tool/catalog/CUDA-artifact/facade/wrapper/staged pytest:
    99 passed;
  - `catalog/scripts/validate.py --strict-abi`: PASS;
  - `catalog/scripts/validate.py --check-references`: PASS;
  - `catalog/scripts/split_legacy_methods.py --check`: PASS.

Follow-up moment-stack PLS base reuse:

- Switched the `"pls"` base inside `NativeMomentStackRegressor` from the
  generic `NativeMomentSweepRegressor` to the new direct `NativePLSRegressor`.
  It still uses `n4m_sweep_run` underneath, but the stack now reuses the same
  individual PLS method exposed to end users.
- Strengthened wrapper tests so stack diagnostics must report
  `estimator="NativePLSRegressor"` and `method="pls"` for OOF and final PLS
  base rows.
- Regenerated `moment_stack_timing_cuda_smoke.csv` after the swap. The PLS-only
  stack still reports CUDA route counters unchanged: OOF PLS moment CV fits
  `16`, host `0`, CUDA device `16`; final PLS CV fits `4`, host `0`, CUDA
  device `4`.
- Validation after this reuse alignment:
  - `bindings/python/tests/test_moment_model_wrappers.py`: 53 passed;
  - `bindings/python/tests/test_aom_moment_cuda_smoke_artifacts.py`: 13 passed;
  - targeted benchmark-tool/catalog/CUDA-artifact/facade/wrapper/staged pytest:
    99 passed.

Follow-up AOM Ridge superblock strict-moment reference:

- Added `n4m.aom_ridge_superblock`, a Python-backed donor-style AOM Ridge
  superblock constrained to the strict-linear single-operator AOM bank. It
  builds concatenated operator views through `n4m.aom_preprocess`, scores fixed
  or CV-selected Ridge alphas with fold-local centering/block scaling through
  the native `n4m.ridge` binding, and folds the final superblock coefficients
  back into original-input `input_coefficients` plus `intercept`.
- Added `NativeAOMRidgeSuperblockRegressor` and exported it through top-level
  `n4m`, `n4m.sklearn`, `n4m.aom` and `n4m.moment`.
- Promoted the method to catalog entry `aom_pop.ridge_superblock`, added
  `docs/methods/aom_ridge_superblock.md`, wired
  `benchmarks/cross_binding/bench_aom_ridge_superblock_timing.py`, and
  generated `aom_ridge_superblock_timing_cuda_smoke.csv` on `build/cuda-on`
  with function + sklearn rows and `ridge_backend=native`.
- Kept donor `branch_global`, MKL/kernel, row-reference-dependent preprocessing
  and nonlinear AOM Ridge modes out of scope for the moment contract.
- Validation after this slice:
  - py_compile on touched Python modules/tests: PASS;
  - focused Ridge-global tests plus `test_aom_moment_facade.py`: 21 passed;
  - targeted benchmark/catalog/CUDA-artifact/facade/wrapper/staged pytest:
    118 passed;
  - `catalog/scripts/validate.py --strict-abi`: PASS, 203 methods and 701/701
    exported `n4m_*` symbols covered;
  - `catalog/scripts/validate.py --check-references`: PASS, 203/203 production
    methods covered;
  - `catalog/scripts/split_legacy_methods.py --check`: PASS, 203 per-method
    files up to date.
- Validation after this slice:
  - py_compile on touched Python modules/tests: PASS;
  - focused Ridge-superblock tests plus `test_aom_moment_facade.py`: 15 passed;
  - targeted benchmark/catalog/CUDA-artifact/facade/wrapper/staged pytest:
    113 passed;
  - `catalog/scripts/validate.py --strict-abi`: PASS, 202 methods and 701/701
    exported `n4m_*` symbols covered;
  - `catalog/scripts/validate.py --check-references`: PASS, 202/202 production
    methods covered;
  - `catalog/scripts/split_legacy_methods.py --check`: PASS, 202 per-method
    files up to date;
  - `git diff --check`: exit 0, only known CRLF warnings on existing CSV
    artifacts.

Follow-up AOM Ridge global strict-moment selector:

- Added `n4m.aom_ridge_global`, a donor-style strict AOM Ridge global selector
  constrained to single strict-linear operators. It wraps the native
  `aom_chain_sweep_run` Ridge-only path, so one operator and one Ridge alpha are
  selected by train CV and the final model reuses folded
  `input_coefficients` plus `intercept`.
- Added `NativeAOMRidgeGlobalRegressor` and exported it through top-level
  `n4m`, `n4m.sklearn`, `n4m.aom` and `n4m.moment`.
- Promoted the method to catalog entry `aom_pop.ridge_global`, added
  `docs/methods/aom_ridge_global.md`, wired
  `benchmarks/cross_binding/bench_aom_ridge_global_timing.py`, and generated
  `aom_ridge_global_timing_cuda_smoke.csv` on `build/cuda-on` with function +
  sklearn rows and `ridge_backend=native_aom_chain_sweep`.
- Kept donor `branch_global`, MKL/kernel, row-reference-dependent preprocessing
  and nonlinear AOM Ridge modes out of scope for the moment contract.

Follow-up AOM Ridge active-superblock strict-moment reference:

- Added `n4m.aom_ridge_active_superblock`, a Python-backed donor-style AOM
  Ridge active-superblock constrained to strict-linear single-operator AOM
  views.
- The active score is defined on native `n4m.aom_preprocess` outputs as
  `||scale_b * Z_b.T @ y_c||_F^2` by default. This avoids pretending to
  reproduce donor-private `op.apply_cov` semantics while keeping the production
  model strictly linear and replayable.
- Alpha CV screens active operators inside each training fold only; the final
  model screens once on full calibration rows and fits Ridge through the native
  `n4m.ridge` binding. The final active superblock is folded back to
  original-input `input_coefficients` plus `intercept`.
- Added `NativeAOMRidgeActiveSuperblockRegressor` and exported it through
  top-level `n4m`, `n4m.sklearn`, `n4m.aom` and `n4m.moment`.
- Promoted the method to catalog entry `aom_pop.ridge_active_superblock`, added
  `docs/methods/aom_ridge_active_superblock.md`, wired
  `benchmarks/cross_binding/bench_aom_ridge_active_superblock_timing.py`, and
  generated `aom_ridge_active_superblock_timing_cuda_smoke.csv` on
  `build/cuda-on` with function + sklearn rows and `ridge_backend=native`.
- Kept donor `branch_global`, MKL/kernel, row-reference-dependent preprocessing
  and nonlinear AOM Ridge modes out of scope for the moment contract.
- Validation during implementation:
  - py_compile on touched Python modules: PASS.
  - Focused Ridge active/global/superblock wrapper tests: `6 passed`.
  - Targeted benchmark/catalog/CUDA-artifact guards for active/global/superblock:
    `11 passed`.
  - Combined benchmark/catalog/CUDA-artifact/facade/wrapper/staged pytest set:
    `123 passed`.
  - `catalog/scripts/validate.py --strict-abi`: PASS, 204 methods and 701/701
    exported `n4m_*` symbols covered.
  - `catalog/scripts/validate.py --check-references`: PASS, 204/204 production
    methods covered.
  - `catalog/scripts/split_legacy_methods.py --check`: PASS, 204 per-method
    files up to date.

Follow-up AOM-PLS superblock strict-moment reference:

- Added `n4m.aom_pls_superblock`, a Python-backed donor-style AOM-PLS
  superblock constrained to strict-linear single-operator AOM views.
- Added `NativeAOMPLSSuperblockRegressor` and exported it through top-level
  `n4m`, `n4m.sklearn`, `n4m.aom` and `n4m.moment`.
- The method concatenates strict operator outputs from `n4m.aom_preprocess`,
  selects the PLS component count by train CV, fits through the native
  `n4m.pls` binding, and folds final superblock coefficients back to
  original-input `input_coefficients` plus `intercept`.
- Promoted the method to catalog entry `aom_pop.aom_pls_superblock`, added
  `docs/methods/aom_pls_superblock.md`, wired
  `benchmarks/cross_binding/bench_aom_pls_superblock_timing.py`, and generated
  `aom_pls_superblock_timing_cuda_smoke.csv` on `build/cuda-on` with function
  + sklearn rows, `pls_backend=native` and CUDA device PLS route counters when
  `--cuda-pls-min-device-features 1` is used.
- Kept donor `soft`, active PLS pruning, MKL/kernel, row-reference-dependent
  preprocessing, nonlinear lifts and dataset/source routing out of scope for
  this slice.

Follow-up AOM Ridge MKL-light superblock strict-moment reference:

- Added `n4m.aom_ridge_mkl_superblock`, a Python-backed donor-style AOM Ridge
  MKL-light weighted-superblock constrained to strict-linear single-operator
  AOM views.
- The method learns non-negative KTA block weights on training rows only inside
  every alpha-CV fold, refits weights on the full calibration rows, fits native
  Ridge on the equivalent weighted superblock, and folds final coefficients
  back to original-input `input_coefficients` plus `intercept`.
- Added `NativeAOMRidgeMKLSuperblockRegressor` and exported it through
  top-level `n4m`, `n4m.sklearn`, `n4m.aom` and `n4m.moment`.
- Promoted the method to catalog entry `aom_pop.ridge_mkl_superblock`, added
  `docs/methods/aom_ridge_mkl_superblock.md`, wired
  `benchmarks/cross_binding/bench_aom_ridge_mkl_superblock_timing.py`, and
  generated `aom_ridge_mkl_superblock_timing_cuda_smoke.csv` on
  `build/cuda-on` with function + sklearn rows, `mkl_mode=alignment`,
  `ridge_backend=native` and replay error at numerical-noise level.
- Kept donor branch/global reference-dependent preprocessing, local/SNV/MSC
  branches, row-reference-dependent preprocessing, nonlinear kernels,
  nonlinear lifts and dataset/source routing out of scope for this slice.
- Validation after this slice:
  - py_compile on touched Python modules/tests/benchmark: PASS.
  - Focused AOM Ridge MKL-light superblock wrapper/function tests: `2 passed`.
  - Targeted benchmark/catalog/CUDA-artifact guards for AOM Ridge MKL-light
    superblock: `12 passed`.
  - Combined benchmark/catalog/CUDA-artifact/facade/wrapper/staged pytest set:
    `138 passed`.
  - `catalog/scripts/validate.py --strict-abi`: PASS, 207 methods and 701/701
    exported `n4m_*` symbols covered.
  - `catalog/scripts/validate.py --check-references`: PASS, 207/207
    production methods covered.
  - `catalog/scripts/split_legacy_methods.py --check`: PASS, 207 per-method
    files up to date.
- Validation after this slice:
  - py_compile on touched Python modules/tests/benchmark: PASS.
  - Focused AOM Ridge-PLS superblock wrapper/function tests: `2 passed`.
  - Targeted benchmark/catalog/CUDA-artifact guards for AOM Ridge-PLS
    superblock: `11 passed`.
  - Combined benchmark/catalog/CUDA-artifact/facade/wrapper/staged pytest set:
    `133 passed`.
  - `catalog/scripts/validate.py --strict-abi`: PASS, 206 methods and 701/701
    exported `n4m_*` symbols covered.
  - `catalog/scripts/validate.py --check-references`: PASS, 206/206
    production methods covered.
  - `catalog/scripts/split_legacy_methods.py --check`: PASS, 206 per-method
    files up to date.
- Validation during implementation:
  - py_compile on touched Python modules: PASS.
  - Focused AOM-PLS superblock wrapper/function tests: `2 passed`.
  - Targeted benchmark/catalog/CUDA-artifact guards for AOM-PLS superblock:
    `8 passed`.
  - Combined benchmark/catalog/CUDA-artifact/facade/wrapper/staged pytest set:
    `128 passed`.
  - `catalog/scripts/validate.py --strict-abi`: PASS, 205 methods and 701/701
    exported `n4m_*` symbols covered.
  - `catalog/scripts/validate.py --check-references`: PASS, 205/205 production
    methods covered.
  - `catalog/scripts/split_legacy_methods.py --check`: PASS, 205 per-method
    files up to date.

Follow-up AOM Ridge-PLS superblock strict-moment reference:

- Added `n4m.aom_ridge_pls_superblock`, a Python-backed donor-style AOM
  Ridge-PLS superblock constrained to strict-linear single-operator AOM views.
- The method concatenates strict operator outputs from `n4m.aom_preprocess`,
  selects the PLS component count and Ridge-PLS lambda by train CV over the
  cartesian grid, fits through the native `n4m.ridge_pls` binding, and folds
  final superblock coefficients back to original-input `input_coefficients`
  plus `intercept`.
- Added `NativeAOMRidgePLSSuperblockRegressor` and exported it through
  top-level `n4m`, `n4m.sklearn`, `n4m.aom` and `n4m.moment`.
- Promoted the method to catalog entry `aom_pop.aom_ridge_pls_superblock`,
  added `docs/methods/aom_ridge_pls_superblock.md`, wired
  `benchmarks/cross_binding/bench_aom_ridge_pls_superblock_timing.py`, and
  generated `aom_ridge_pls_superblock_timing_cuda_smoke.csv` on
  `build/cuda-on` with function + sklearn rows and `ridge_pls_backend=native`.
- Kept donor `soft`, active PLS pruning, MKL/kernel, row-reference-dependent
  preprocessing, nonlinear lifts and dataset/source routing out of scope for
  this slice.

Follow-up rank-audit mismatch summary (2026-06-06):

- Added a post-hoc mismatch-pattern section to
  `benchmarks/cross_binding/summarize_aom_rank_audit.py`.
- The generated Markdown now summarizes cases where the train-CV winner is not
  the offline test oracle (`test_rank_delta > 0` or `oracle_gap_ratio > 0`).
- Groupings:
  - `selected_head -> oracle_head`
  - `selected_chain -> oracle_chain`
- Reported aggregate stats: count, mean/median oracle-gap ratio, median
  selected test rank and median rank delta. The CSV output is unchanged.
- Production invariant remains explicit: this is offline audit only, never
  test-set selection, and never routing by dataset name, source or id.
- Regenerated rank-audit CSV/Markdown for compact-wide audit5, compact-wide
  audit10 and ECOSIS refit40.
- The compact-wide audit10 mismatch summary exposes the main failure mode:
  ECOSIS selected `ridge:0.1 savgol_smooth(7,2)` while the offline oracle was
  `pls:1 detrend_poly(2)`, gap ratio `0.5773`.
- The ECOSIS refit40 stress test confirms this is not simply a retention-budget
  issue: selected `ridge:0.1 savgol_smooth(7,2)` had test rank `41`, while the
  offline oracle was `ridge:10 finite_difference(1)`, gap ratio `2.9123`.
- Validation:
  - py_compile on the summarizer and benchmark-tool tests: PASS.
  - `test_aom_benchmark_tools.py`: `19 passed` with
    `PYTHONPATH=bindings/python/src` and
    `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so`.
  - `git diff --check`: PASS apart from known CRLF warnings on old CSV
    artifacts.

Follow-up read-only engine audit (2026-06-06):

- Delegated a read-only audit of the remaining engine/performance gap after
  the method/facade/catalog surfaces were checked.
- Conclusion: no obvious missing Python-facing method remains; the open gaps
  are still the documented C++/CUDA engine gaps.
- Recommended smallest bounded engine slice:
  - share/extract the existing symmetric eigensolver currently local to
    `cpp/src/core/model.cpp`;
  - use it in `cpp/src/core/sweep.cpp` to score Ridge moment CV by one
    eigendecomposition per fold plus a lambda-path scan, instead of calling
    `fit_ridge_from_moments()` / `solve_square_qr()` independently for every
    lambda;
  - raise any Ridge moment feature cap only after parity and timing prove the
    new path.
- Review correction recorded: the current Ridge moment implementation uses a
  generic QR solve, not a Cholesky solve, so this should be framed as
  "QR-per-lambda to eigensolver-per-fold" rather than "Cholesky replacement".
- This was not implemented in the current pass because it is a real numerical
  helper extraction plus parity/timing task, not a trivial wiring patch.

Follow-up Ridge moment eigen-path implementation (2026-06-06):

- Implemented the first bounded engine optimization for Ridge moment scoring.
  The shared C SVD helper now exposes internal `n4m_symmetric_eigh()` for a
  read-only full symmetric eigendecomposition without adding public ABI.
- `sweep.cpp` gained a `RidgeMomentEigenPath` route that reuses one
  eigendecomposition per fold for multi-lambda Ridge moment scans. It mirrors
  the existing direct moment design/centering/scaling, preprojects `X'Y` into
  the eigenbasis, and reconstructs coefficients per lambda.
- The optimized route is guarded to `n_lambdas > 1` and falls back to the
  existing QR-per-lambda `fit_ridge_from_moments()` path whenever the
  eigen-path cannot be prepared or used. Mono-lambda behavior remains the
  previous direct solve.
- Batch score-only AOM Ridge now groups internal work by `(chain, fold)` so
  all lambdas share one local eigen-path; it does not persist all eigenvectors
  across chains. Public logical counters stay unchanged.
- Added NumPy parity coverage for multi-lambda Ridge moment candidate RMSEs
  plus direct-path versus eigen-path score equality for the same lambda, and
  reran the wrapper/benchmark-tool tests:
  - `test_moment_model_wrappers.py`: `73 passed`.
  - `test_aom_benchmark_tools.py`: `19 passed`.
  - dev `n4m_c` build: PASS.
  - `ctest -R sweep`: no C++ tests found in the current dev build.
  - `git diff --check`: PASS except known old CSV CRLF warnings.
- Synthetic timing smoke on `n=140, p=80, cv=5` Ridge score-only:
  `1/4/16/64` lambdas ran in about `0.0026 / 0.0064 / 0.0221 / 0.1417` seconds.
- Read-only Claude review found no blocking issues; the suggested
  direct-versus-eigen same-lambda test was added.

Follow-up moment sweep timing smoke Ridge coverage (2026-06-06):

- Extended `benchmarks/cross_binding/bench_moment_sweep_timing.py` with a tall
  `96x48` smoke cell. The previous default CUDA smoke shapes covered wide
  dual/materialized Ridge but did not exercise the exact Ridge moment route.
- Regenerated `benchmarks/cross_binding/moment_sweep_timing_cuda_smoke.csv`
  against `build/cuda-on/cpp/src/libn4m.so`; the tall Ridge rows now report
  `n_ridge_moment_candidates=5` and `n_ridge_moment_cv_fits=25`.
- Strengthened
  `bindings/python/tests/test_aom_moment_cuda_smoke_artifacts.py` so the
  committed artifact must include a Ridge moment row with zero dual/materialized
  Ridge CV counters, while preserving the existing exact PLS CUDA-device checks.
- Validation:
  - CUDA `n4m_c` build: PASS.
  - py_compile for the timing script and artifact test: PASS.
  - `test_aom_moment_cuda_smoke_artifacts.py`: `21 passed`.

Follow-up catalog/method-surface conformance (2026-06-06):

- Re-ran the catalog and Python-binding conformance checks after the latest
  AOM/moment additions.
- Results:
  - `catalog/scripts/validate.py --strict-abi`: PASS, 208 methods and 701/701
    exported `n4m_*` symbols covered.
  - `catalog/scripts/validate.py --check-references`: PASS, 208/208
    production methods covered.
  - `catalog/scripts/split_legacy_methods.py --check`: PASS, 208 per-method
    files up to date.
  - `test_catalog_python_bindings.py`: `10 passed`.
- This proves the expanded reusable method surface remains catalog-consistent;
  the remaining open work is still engine/performance and broader oracle
  campaigns.

Follow-up oracle comparison table (2026-06-06):

- Extended `benchmarks/cross_binding/compare_aom_staged_to_oracles.py` so
  Markdown summaries include a per-target-dataset table separating AOM-PLS
  oracle, AOM-Ridge oracle and TabPFN oracle scores and ratios.
- Regenerated the compact-wide audit10 oracle comparison Markdown/CSV and the
  aggregate `aom_staged_oracle_comparison.md/.csv`.
- The compact-wide audit10 target remains the train-CV-selected production
  candidate. The offline comparison table is audit-only and reports:
  - AOM-PLS oracle: paired 7, median ratio `1.03079`.
  - AOM-Ridge oracle: paired 8, median ratio `1.08068`.
  - TabPFN oracle: paired 8, median ratio `0.978527`.
- Added benchmark-tool test coverage for the new table output and reran
  `test_aom_benchmark_tools.py`: `19 passed`.

Follow-up methods index count (2026-06-06):

- Updated `docs/methods/index.md` so the visible total catalogued native method
  count is `208`, matching the current per-method catalog files.
- Added a catalog Python-binding test that parses the displayed count and
  compares it to `catalog/methods/*.yaml`, preventing stale end-user counts
  after future method additions.
- Validation: `test_catalog_python_bindings.py` now has `11 passed`.

Follow-up facade inventory conformance (2026-06-06):

- Added a public-surface invariant for `n4m.aom.available_methods()` and
  `n4m.moment.available_methods()`: every advertised row must declare typed
  `cpu` and `cuda` capability flags and at least one available execution
  capability.
- The broader facade suite already verifies object identity with
  `n4m.python`/`n4m.sklearn`, top-level package re-export, catalog and docs
  resolution, docs-index links, Python binding-role coherence, relevant native
  AOM/moment catalog coverage, bounded preset wrappers and absence of
  dataset/source identity routing options.
- Validation:
  - `PYTHONPATH=bindings/python/src N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so /home/delete/.venv/bin/python -m pytest bindings/python/tests/test_aom_moment_facade.py -q`: `20 passed`.

Follow-up Ridge moment eigen-path in general sweep (2026-06-06):

- Extended the Ridge moment eigen-path beyond the dedicated score-only Ridge
  scorer into the general `run_moment_sweep()` Ridge moment loop.
- When a tall Ridge moment fold has more than one lambda, the general sweep now
  builds one fold-local eigen-path and scans lambdas through it. Mono-lambda
  and fallback cases still call the previous direct QR solve, preserving
  numerical fallback behavior and public logical counters.
- Added
  `test_native_sweep_run_ridge_moment_score_only_matches_full_scores`, covering
  tall multi-lambda Ridge moment scores, moment/dual counters, and
  score-only output suppression.
- Synthetic score-only timing on `n=140, p=80, cv=5`:
  - 4 lambdas: multi-lambda `0.058237s`, repeated mono-lambda `0.143395s`,
    max score diff `8.3e-15`.
  - 16 lambdas: multi-lambda `0.026655s`, repeated mono-lambda `0.306316s`,
    max score diff `8.3e-15`.
  - 64 lambdas: multi-lambda `0.093162s`, repeated mono-lambda `1.277440s`,
    max score diff `1.1e-14`.
- Validation:
  - dev and CUDA `n4m_c` builds: PASS.
  - `test_moment_model_wrappers.py`: `74 passed`.
  - CUDA-lib focused Ridge moment tests:
    `2 passed, 72 deselected`.

Follow-up moment sweep CUDA smoke after general Ridge eigen-path
(2026-06-06):

- Regenerated
  `benchmarks/cross_binding/moment_sweep_timing_cuda_smoke.csv` with
  `CUDA_VISIBLE_DEVICES=0`, `N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so`,
  `--repeats 1 --native-only --cuda-pls-min-device-features 1
  --cuda-pls-parallel-folds`.
- Tightened the artifact test so the tall Ridge moment smoke cell (`96x48`)
  proves full `run_moment_sweep()` and `score_only` agree on selected lambda
  and CV RMSE while preserving final-fit behavior (`1` final Ridge moment fit
  for full, `0` for score-only).
- Regenerated tall Ridge rows report selected lambda `0.001`, full RMSE
  `0.08037705391174381`, score-only RMSE `0.08037705391174604`, and
  `n_ridge_moment_cv_fits=25` on both rows.
- Validation:
  - py_compile for timing script and touched tests: PASS.
  - `test_aom_moment_cuda_smoke_artifacts.py`: `21 passed`.
  - `test_aom_moment_facade.py` + `test_moment_model_wrappers.py`:
    `94 passed`.

Follow-up catalog/surface validation after timing smoke (2026-06-06):

- Re-ran catalog validation after the general sweep eigen-path and smoke
  artifact updates.
- Evidence:
  - `catalog/scripts/validate.py --strict-abi`: PASS, 208 methods and 701/701
    exported `n4m_*` symbols covered.
  - `catalog/scripts/validate.py --check-references`: PASS, 208/208
    production methods covered by references.
  - `catalog/scripts/split_legacy_methods.py --check`: PASS, 208 per-method
    files up to date.
  - `test_catalog_python_bindings.py`: `11 passed`.
  - `/home/delete/.venv/bin/ctest --test-dir build/dev-release -R sweep --output-on-failure`:
    no C++ tests found in this build.

Follow-up Ridge moment eigen-path telemetry (2026-06-06):

- Added explicit route telemetry for Ridge moment lambda-path scoring:
  `n_ridge_moment_eigen_path_preparations`,
  `n_ridge_moment_eigen_path_cv_fits` and
  `n_ridge_moment_direct_cv_fits`.
- The logical counter `n_ridge_moment_cv_fits` remains unchanged; the new
  counters explain how those fits executed, which makes broad preprocessing
  screens auditable without inferring route behavior from timing.
- Propagated the telemetry through native sweep/AOM results, C API result
  serialization, Python scalar extraction, sklearn diagnostics, refit/staged
  campaign summaries and the committed moment-sweep CUDA smoke artifact.
- The regenerated tall `96x48` Ridge smoke rows now show:
  - full sweep: `5` eigen-path preparations, `25` eigen-path CV fits,
    `0` direct CV fits.
  - score-only: `5` eigen-path preparations, `25` eigen-path CV fits,
    `0` direct CV fits.
- Validation:
  - dev and CUDA `n4m_c` builds: PASS.
  - focused Ridge telemetry tests: `3 passed, 71 deselected`.
  - `test_moment_model_wrappers.py`: `74 passed`.
  - artifact/facade/wrapper suite: `115 passed`.
  - `test_catalog_python_bindings.py`: `11 passed`.
  - catalog strict ABI/reference/split checks: PASS.
  - `git diff --check`: PASS apart from known CRLF warnings on old CSV
    artifacts.

Follow-up Ridge telemetry in benchmark runners (2026-06-06):

- Propagated the Ridge moment eigen-path/direct telemetry to the benchmark
  runners that are used to audit preprocessing selection and timing:
  `run_aom_staged_real_cohort.py`,
  `bench_aom_screen_refit_scaling.py`,
  `bench_aom_staged_chain_campaign_timing.py`,
  `bench_aom_sweep_timing.py` and
  `bench_aom_ridge_global_timing.py`.
- The staged real-cohort CSV and diagnostics JSON now carry
  `n_ridge_moment_eigen_path_preparations`,
  `n_ridge_moment_eigen_path_cv_fits` and
  `n_ridge_moment_direct_cv_fits`, so campaign artifacts can prove whether
  Ridge moment CV work used the fast eigen-path or direct fallback.
- Regenerated the CUDA timing smokes affected by new columns:
  `aom_sweep_timing_cuda_smoke.csv`,
  `aom_ridge_global_timing_cuda_smoke.csv` and
  `aom_staged_chain_campaign_timing_cuda_smoke.csv`.
- Added benchmark/staged tests for row/diagnostics/stage aggregation of the
  new counters. Staged report tests now cover the accounting invariant:
  `n_ridge_moment_eigen_path_cv_fits + n_ridge_moment_direct_cv_fits ==
  n_ridge_moment_cv_fits`.
- Validation:
  - py_compile on touched benchmark scripts/tests: PASS.
  - `test_aom_benchmark_tools.py`: `19 passed`.
  - `test_aom_staged_campaign.py`: `16 passed`.
  - `test_aom_moment_cuda_smoke_artifacts.py`: `21 passed`.
  - combined benchmark/staged/artifact suite: `56 passed`.
  - facade/wrapper/catalog Python suite: `105 passed`.
  - catalog strict ABI/reference/split checks: PASS.

Follow-up real-cohort Ridge telemetry guard (2026-06-06):

- Added `validate_ridge_moment_route_telemetry(report)` in
  `benchmarks/cross_binding/run_aom_staged_real_cohort.py`.
- The runner now validates Ridge moment route counters immediately after
  `n4m.aom_staged_chain_campaign(...)` returns and before any CSV or
  diagnostics JSON is persisted.
- When the total/eigen/direct CV counters are all present, the guard requires:
  - `n_ridge_moment_cv_fits`,
    `n_ridge_moment_eigen_path_cv_fits` and
    `n_ridge_moment_direct_cv_fits` to be non-negative integer-like values.
  - `n_ridge_moment_eigen_path_cv_fits +
    n_ridge_moment_direct_cv_fits == n_ridge_moment_cv_fits`.
  - `n_ridge_moment_eigen_path_preparations` to be non-negative/integer-like
    when present.
- This is a campaign-integrity guard only: production selection remains exact
  train-CV-only and held-out/test ranking remains offline audit data.
- Validation:
  - `PYTHONPATH=bindings/python/src /home/delete/.venv/bin/python -m py_compile benchmarks/cross_binding/run_aom_staged_real_cohort.py bindings/python/tests/test_aom_benchmark_tools.py`
  - `PYTHONPATH=bindings/python/src N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so /home/delete/.venv/bin/python -m pytest bindings/python/tests/test_aom_benchmark_tools.py -q`: `20 passed`.
  - One-row CUDA real-cohort smoke with `CUDA_VISIBLE_DEVICES=0` and
    `N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so` completed
    `BEEFMARBLING/Beef_Marbling_RandomSplit` successfully. The output row and
    diagnostics JSON both reported Ridge counters `12/6/6`
    total/eigen/direct, `selection_uses_test_set=False`, and PLS screen
    device/host CV fits `6/0`.

Follow-up moment facade wrapper-target conformance (2026-06-06):

- Added `aom_chain_screen_refit_campaign` to the `n4m.moment` facade exports.
  The underlying native binding and top-level `n4m` export already existed, but
  three advertised moment screen/refit sklearn wrappers declared
  `wrapper_of="aom_chain_screen_refit_campaign"` without the same target being
  reachable from `n4m.moment`.
- Added a generic facade test ensuring every inventory `wrapper_of` target is
  exported on the same facade, top-level `n4m`, and `n4m.python`, and that all
  three are the identical native object.
- This is a public-surface conformance fix for reusable wrappers; it does not
  change fitting behavior, add hors-moment methods, or alter production
  selection policy.
- Validation:
  - `PYTHONPATH=bindings/python/src /home/delete/.venv/bin/python -m py_compile bindings/python/src/n4m/moment/__init__.py bindings/python/tests/test_aom_moment_facade.py`
  - `PYTHONPATH=bindings/python/src N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so /home/delete/.venv/bin/python -m pytest bindings/python/tests/test_aom_moment_facade.py -q`: `22 passed`.
  - `PYTHONPATH=bindings/python/src N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so /home/delete/.venv/bin/python -m pytest bindings/python/tests/test_aom_moment_facade.py bindings/python/tests/test_moment_model_wrappers.py bindings/python/tests/test_catalog_python_bindings.py -q`: `107 passed`.
  - Catalog split/reference/strict-ABI checks: PASS.

Follow-up facade timing-benchmark coverage guard (2026-06-06):

- Added a facade-level invariant that every catalogued method advertised by
  `n4m.aom.available_methods()` or `n4m.moment.available_methods()` has a
  non-null `bench.registry_entry`, that the entry lives under
  `benchmarks/cross_binding/`, and that the script exists.
- A quick audit found 30 distinct catalog IDs on the public AOM/moment facades
  and no missing timing benchmark entries. The new test makes this a persistent
  end-user surface contract, complementing the broader catalog validator.
- This is a conformance guard only; no method behavior, production selection,
  CUDA routing, or preprocessing family changed.
- Validation:
  - `PYTHONPATH=bindings/python/src /home/delete/.venv/bin/python -m py_compile bindings/python/tests/test_aom_moment_facade.py`
  - `PYTHONPATH=bindings/python/src N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so /home/delete/.venv/bin/python -m pytest bindings/python/tests/test_aom_moment_facade.py -q`: `24 passed`.
  - `PYTHONPATH=bindings/python/src N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so /home/delete/.venv/bin/python -m pytest bindings/python/tests/test_aom_moment_facade.py bindings/python/tests/test_moment_model_wrappers.py bindings/python/tests/test_catalog_python_bindings.py -q`: `109 passed`.
  - Catalog split/reference/strict-ABI checks: PASS.

Follow-up AOM PLS superblock route-telemetry accounting
(2026-06-06):

- Fixed `n4m.aom_pls_superblock` route accounting. The method previously copied
  only the final selected PLS solve counters into the returned result, even
  though component selection had already executed one PLS solve per
  component/fold. This made timing and CUDA route diagnostics under-report the
  true work.
- The implementation now accumulates PLS route counters from every
  fold/component solve plus the final fit:
  `n_pls_moment_cv_fits`,
  `n_pls_moment_host_cv_fits`,
  `n_pls_moment_cuda_device_cv_fits`,
  `n_pls_moment_cuda_parallel_fold_batches`,
  `n_pls_moment_cuda_parallel_fold_jobs`,
  `n_pls_materialized_cv_fits`,
  `n_pls_moment_final_fits`,
  `n_pls_moment_host_final_fits`,
  `n_pls_moment_cuda_device_final_fits` and
  `n_pls_materialized_final_fits`.
- Extended `benchmarks/cross_binding/bench_aom_pls_superblock_timing.py` so the
  committed timing artifact also exposes CUDA parallel-fold jobs and final-fit
  host/device counters.
- Regenerated
  `benchmarks/cross_binding/aom_pls_superblock_timing_cuda_smoke.csv` with
  `CUDA_VISIBLE_DEVICES=0` and `N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so`.
  The smoke profile now reports `18/0/18` PLS CV total/host/device jobs and
  `9/0/9` PLS final total/host/device fits for both function and sklearn rows,
  matching two component values over four folds plus one final fit.
- This is telemetry/accounting only. Selection, predictions, folded
  coefficients and the strict-linear moment contract are unchanged.
- Validation:
  - `PYTHONPATH=bindings/python/src /home/delete/.venv/bin/python -m py_compile bindings/python/src/n4m/python.py benchmarks/cross_binding/bench_aom_pls_superblock_timing.py bindings/python/tests/test_aom_moment_cuda_smoke_artifacts.py bindings/python/tests/test_moment_model_wrappers.py`
  - `PYTHONPATH=bindings/python/src N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so /home/delete/.venv/bin/python -m pytest bindings/python/tests/test_moment_model_wrappers.py -q -k 'aom_pls_superblock'`: `2 passed, 72 deselected`.
  - `PYTHONPATH=bindings/python/src N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so /home/delete/.venv/bin/python -m pytest bindings/python/tests/test_aom_moment_cuda_smoke_artifacts.py -q -k 'diversity_cuda_smoke_artifacts_cover_native_and_sklearn_replay'`: `9 passed, 12 deselected`.
  - `PYTHONPATH=bindings/python/src N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so /home/delete/.venv/bin/python -m pytest bindings/python/tests/test_moment_model_wrappers.py bindings/python/tests/test_aom_moment_cuda_smoke_artifacts.py -q`: `95 passed`.
  - Catalog split/reference/strict-ABI checks: PASS.

Follow-up AOM Ridge-PLS solve-count telemetry (2026-06-06):

- Added deterministic solve-count telemetry to the Ridge-PLS AOM methods:
  `n_ridge_pls_fit_calls` for `n4m.aom_ridge_pls_superblock` and
  `n4m.aom_chain_ridge_pls`.
- The count is `n_candidates * cv + 1`, representing every native Ridge-PLS
  solve run during train-CV candidate scoring plus the final selected fit. This
  makes the CPU cost of these reusable Ridge-PLS methods explicit in the same
  timing artifacts that already cover replay and CUDA-build compatibility.
- Exposed the field through the sklearn diagnostics for
  `NativeAOMRidgePLSSuperblockRegressor` and
  `NativeAOMChainRidgePLSRegressor`.
- Extended the timing rows in
  `bench_aom_ridge_pls_superblock_timing.py` and
  `bench_aom_chain_ridge_pls_timing.py`, then regenerated the CUDA-build smoke
  CSVs with one visible GPU. The smoke rows now report:
  - Ridge-PLS superblock: `n_ridge_pls_fit_calls=17`
    (`4` candidates x `4` folds + final fit).
  - Chain Ridge-PLS: `n_ridge_pls_fit_calls=33`
    (`8` candidates x `4` folds + final fit).
- This is telemetry/accounting only; predictions, selected candidates,
  train-CV selection and folded coefficients are unchanged.
- Validation:
  - `PYTHONPATH=bindings/python/src /home/delete/.venv/bin/python -m py_compile bindings/python/src/n4m/python.py bindings/python/src/n4m/sklearn/native_sweeps.py benchmarks/cross_binding/bench_aom_ridge_pls_superblock_timing.py benchmarks/cross_binding/bench_aom_chain_ridge_pls_timing.py bindings/python/tests/test_moment_model_wrappers.py bindings/python/tests/test_aom_moment_cuda_smoke_artifacts.py`
  - `PYTHONPATH=bindings/python/src N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so /home/delete/.venv/bin/python -m pytest bindings/python/tests/test_moment_model_wrappers.py -q -k 'aom_ridge_pls_superblock or aom_chain_ridge_pls'`: `4 passed, 70 deselected`.
  - `PYTHONPATH=bindings/python/src N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so /home/delete/.venv/bin/python -m pytest bindings/python/tests/test_aom_moment_cuda_smoke_artifacts.py -q -k 'diversity_cuda_smoke_artifacts_cover_native_and_sklearn_replay'`: `9 passed, 12 deselected`.
  - `PYTHONPATH=bindings/python/src N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so /home/delete/.venv/bin/python -m pytest bindings/python/tests/test_moment_model_wrappers.py bindings/python/tests/test_aom_moment_cuda_smoke_artifacts.py -q`: `95 passed`.
  - Catalog split/reference/strict-ABI checks: PASS.

## 2026-06-06 — aom_operator_pls_stack and aom_ridge_blender cost-route telemetry

**Audit findings**:
- Neither `aom_ridge_blender` nor `aom_operator_pls_stack` had any native cost counters in the C++ result structs (`AomRidgeBlenderResult`, `AomOperatorPlsStackResult`). No instrumentation existed at the C++ layer.
- Both methods' costs are fully deterministic from existing scalar fields already in the result:
  - Ridge blender: `n_candidates * cv` CV Ridge fits + `n_candidates` final Ridge fits (one per candidate for full-data prediction).
  - PLS stack: `(n_specs + 1) * cv * n_operators` PLS projector fits during CV (n_specs scoring calls + 1 OOF refit of selected spec) + `n_operators` final PLS fits; `(n_specs + 1) * cv` Ridge head fits during CV + `1` final Ridge head fit.
- C++ is not touched: all counters are Python-side derived from the native scalars.

**Changes**:
- `bindings/python/src/n4m/python.py`:
  - `aom_ridge_blender`: adds `n_ridge_blender_cv_fits`, `n_ridge_blender_final_fits`, and `n_ridge_blender_fit_calls` to returned dict.
  - `aom_operator_pls_stack`: adds `n_operator_pls_stack_fit_calls`, `n_operator_pls_stack_pls_fit_calls`, `n_operator_pls_stack_ridge_fit_calls`, plus `n_pls_stack_cv_fits`, `n_pls_stack_final_fits`, `n_ridge_stack_cv_fits`, `n_ridge_stack_final_fits` to returned dict.
- `bindings/python/src/n4m/sklearn/native_sweeps.py`:
  - `NativeAOMRidgeBlenderRegressor.get_diagnostics()`: exposes the Ridge blender CV/final/total fit counters.
  - `NativeAOMOperatorPLSStackRegressor.get_diagnostics()`: exposes the operator stack CV/final/total fit counters.
- `benchmarks/cross_binding/bench_aom_ridge_blender_timing.py`: adds Ridge blender CV/final/total fit counters to CSV rows.
- `benchmarks/cross_binding/bench_aom_operator_pls_stack_timing.py`: adds operator stack CV/final/total PLS/Ridge fit counters to CSV rows.
- `benchmarks/cross_binding/README.md`, `docs/methods/aom_ridge_blender.md`,
  and `docs/methods/aom_operator_pls_stack.md`: document replay plus fit-count
  telemetry and avoid stale fixed timing/ABI values.
- CSV smoke files updated: `aom_ridge_blender_timing*.csv`, `aom_operator_pls_stack_timing*.csv`.
- `bindings/python/tests/test_moment_model_wrappers.py`: asserts exact counter values in compact-contract tests (ridge blender: cv_fits=96, final_fits=24; pls stack: pls_cv=240, pls_final=12, ridge_cv=20, ridge_final=1).
- `bindings/python/tests/test_aom_moment_cuda_smoke_artifacts.py`: adds per-CSV counter guards in `test_diversity_cuda_smoke_artifacts_cover_native_and_sklearn_replay`.
- Codex review follow-up added total fit-call counters and regenerated the four
  timing CSVs from the local benchmark scripts. Dev rows used
  `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so`; CUDA smoke rows used
  `CUDA_VISIBLE_DEVICES=0` and `N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so`.
  The smoke rows now report `n_ridge_blender_fit_calls=288` and
  `n_operator_pls_stack_fit_calls=21` / `n_operator_pls_stack_pls_fit_calls=252` /
  `n_operator_pls_stack_ridge_fit_calls=21`.

**Validation**:
- `py_compile` on all touched Python files: PASS.
- `test_moment_model_wrappers.py -k 'aom_ridge_blender_compact_contract or aom_operator_pls_stack_compact_contract'`: `2 passed, 72 deselected`.
- `test_aom_moment_cuda_smoke_artifacts.py -k 'diversity_cuda_smoke_artifacts_cover_native_and_sklearn_replay'`: `9 passed, 12 deselected`.
- Full suite `test_moment_model_wrappers.py + test_aom_moment_cuda_smoke_artifacts.py`: `95 passed`.
- Catalog split/reference/strict-ABI checks: PASS.
- `git diff --check`: PASS apart from known CRLF warnings on CSV artifacts.

## 2026-06-06 — Facade Duplicate Catalog-Role Guard

- Added `test_duplicate_catalog_ids_are_explicit_wrappers_or_aliases` to
  `bindings/python/tests/test_aom_moment_facade.py`.
- The guard allows duplicate `catalog_id` rows only when they represent explicit
  wrappers, aliases, presets or campaign helpers over the same native catalog
  method. At most one row per facade may claim a primary
  `catalog_binding` / `direct_native_binding` role for a catalog method, and
  every secondary duplicate must declare `wrapper_of`.
- This protects the AOM/moment public inventories from ambiguous duplicate
  primary methods while preserving the intentional reusable aliases and sklearn
  wrappers.
- Validation:
  - py_compile on `test_aom_moment_facade.py`: PASS.
  - `test_aom_moment_facade.py`: `26 passed`.
  - facade/wrapper/catalog suite:
    `test_aom_moment_facade.py`, `test_moment_model_wrappers.py`,
    `test_catalog_python_bindings.py`: `111 passed`.
  - catalog split/reference/strict-ABI checks: PASS.

## 2026-06-06 — CUDA Facade Smoke Diversity Alias Guard

- Extended `benchmarks/cross_binding/aom_moment_cuda_facade_smoke.py` so the
  one-process CUDA child asserts the recently ported diversity methods are
  exported by both `n4m.aom` and the top-level `n4m` facade:
  `aom_ridge_blender`, `aom_operator_pls_stack`,
  `NativeAOMRidgeBlenderRegressor`, and
  `NativeAOMOperatorPLSStackRegressor`.
- Regenerated `benchmarks/cross_binding/aom_moment_cuda_facade_smoke.json` with
  `CUDA_VISIBLE_DEVICES=0` and `N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so`.
  The artifact now includes `aom_diversity_aliases_top_level=true` and
  `aom_diversity_estimators_alias_top_level=true`.
- Added duplicate-key checking to the JSON artifact loader in
  `bindings/python/tests/test_aom_moment_cuda_smoke_artifacts.py`, then asserted
  the two new facade fields. This protects the committed smoke report from
  silently shadowed keys and from regressions in the diversity facade exports.
- Validation:
  - py_compile on the CUDA smoke script and artifact tests: PASS.
  - focused CUDA facade artifact test: `1 passed, 20 deselected`.
  - full CUDA smoke artifact test file: `21 passed`.

## 2026-06-06 — Claude Code Release-Readiness Audit

- Delegated a focused Claude Code audit of the handoff and dirty diff with code
  permissions enabled but no long benchmark campaign. Claude did not produce a
  patch before its turn cap.
- The only concrete possible gap it identified was Ridge moment
  `eigen_path/direct_cv` telemetry coverage. Codex checked the current tests and
  confirmed this is already guarded by both CUDA-artifact checks and benchmark
  tool tests, including exact assertions for
  `n_ridge_moment_eigen_path_preparations`,
  `n_ridge_moment_eigen_path_cv_fits`, and
  `n_ridge_moment_direct_cv_fits`.
- Codex also verified the staged preset estimator aliases
  `NativeAOMSavgolFocusRegressor` and
  `NativeAOMStrictFamilyLiteRegressor` are present in top-level `n4m.__all__`.
  No speculative facade or doc edit was made.
- Validation:
  - py_compile on the recently touched facade-smoke/test files: PASS.
  - targeted AOM/moment release-readiness suite:
    `test_aom_moment_facade.py`, `test_moment_model_wrappers.py`,
    `test_aom_moment_cuda_smoke_artifacts.py`,
    `test_catalog_python_bindings.py`, `test_aom_staged_campaign.py`,
    `test_aom_benchmark_tools.py`: `168 passed`.
  - `catalog/scripts/validate.py`: PASS, 208 methods.
  - `catalog/scripts/validate.py --strict-abi`: PASS, 701/701 exported
    `n4m_*` symbols covered.
  - `catalog/scripts/validate.py --check-references`: PASS, 208/208
    production methods covered.
  - `catalog/scripts/split_legacy_methods.py --check`: PASS, 208 per-method
    files up to date.

## 2026-06-06 — CUDA PLS Many-Batched Route Telemetry

- Added a dedicated committed smoke CSV,
  `benchmarks/cross_binding/moment_sweep_timing_cuda_many_batched_smoke.csv`,
  generated on one GPU with `--cuda-pls-min-device-features 1` and
  `--cuda-pls-many-batched`.
- The PLS rows prove the alternate exact PLS moment CUDA route independently of
  parallel-fold scheduling: host CV fits stay at `0`, CUDA-device CV fits equal
  total PLS CV fits, parallel-fold batches/jobs stay at `0`, and
  many-batched counters report one batch and one job per CV fold.
- Regenerated `benchmarks/cross_binding/aom_moment_cuda_facade_smoke.json` so
  the staged/focus/strict facade sections record the new many-batched counters;
  those remain `0` there because that artifact exercises the parallel-fold
  facade route.
- Strengthened the wrapper fallback test so the new many-batched counters are
  asserted at zero when the CUDA knobs are requested but the device threshold
  intentionally forces the CPU path.
- Validation:
  - py_compile on the touched smoke/test files: PASS.
  - exact wrapper fallback test: `1 passed`.
  - full CUDA artifact guard file: `22 passed`.
  - full moment wrapper test file: `74 passed`.
  - catalog strict ABI/reference/split checks: PASS.
  - `git diff --check`: PASS.

## 2026-06-06 — AOM/Moment Completion Audit

- Re-audited the port against the requested end state and the coverage matrix.
  The reusable AOM/moment method surface, direct heads, staged/global
  configurable campaigns, CPU paths, CUDA build smokes, catalog files and docs
  are internally consistent.
- Claude Code reran the targeted AOM/moment suite:
  `test_aom_benchmark_tools.py`, `test_catalog_python_bindings.py`,
  `test_aom_moment_cuda_smoke_artifacts.py`, `test_aom_moment_facade.py`,
  `test_moment_model_wrappers.py`, and `test_aom_staged_campaign.py`:
  `169 passed`.
- Catalog validation remained green: reference coverage for `208/208`
  production methods, strict ABI coverage for `701/701` exported `n4m_*`
  symbols, and per-method split files up to date.
- Verified referenced docs, benchmark scripts, CUDA smoke artifacts and
  coverage-matrix catalog ids exist on disk. No new small
  Python/facade/catalog gap was found.
- Updated `DEFERRALS.md` so the GPU section no longer claims only single-fit
  CUDA exists: bounded exact PLS CV routes now ship, while the full fused
  cartesian/IKPLS-style 200k-chain grinder remains the explicit deferred
  engine/performance item.

## 2026-06-06 — Moment SSE BLAS Engine Slice

- Added a small score-preserving throughput improvement to the hot
  `ridge_heldout_sse_from_moments()` path used by Ridge and PLS moment CV
  scoring. In CPU BLAS builds, single-target (`q=1`) held-out moment scoring
  with `p >= 64` now computes the quadratic term through `linalg::gemv`
  (`X'X beta`) before the final dot product.
- The CUDA build keeps the existing scalar host SSE path to avoid accidental
  per-candidate GPU transfers; CUDA acceleration remains concentrated in the
  PLS moment component routes and their explicit counters.
- This is not the fused cartesian IKPLS grinder, but it removes a CPU BLAS
  bottleneck from exact moment screen scoring without changing scores or
  selection.
- Validation:
  - dev, BLAS and CUDA `n4m_c` builds: PASS.
  - targeted moment wrapper tests on dev and BLAS libs:
    `6 passed, 68 deselected` each.
  - `test_aom_benchmark_tools.py` on dev and BLAS libs: `20 passed` each.
  - dev-vs-BLAS candidate-score comparison on `n=200, p=80`: same selected
    lambda and max absolute score delta `6.63e-15`.
  - Added `test_native_sweep_run_blas_sse_scores_match_scalar_build` so the
    dev-vs-BLAS candidate-score comparison is a permanent pytest guard.
  - `git diff --check`: PASS.

## 2026-06-06 — Moment GPU Crossover Artifact Refresh

- Added `--summary-output` to
  `benchmarks/cross_binding/bench_moment_gpu_crossover.py`. The Markdown
  summary renders one row per `(head, shape)` with CPU time, CUDA default time,
  PLS-only CUDA many-batched time, best CUDA profile, speedups and recommended
  backend. The recommendation remains source-free: shape/head timing only, no
  dataset identity.
- Regenerated `benchmarks/cross_binding/moment_gpu_crossover.csv` and added
  `benchmarks/cross_binding/moment_gpu_crossover.md` with one visible GPU,
  `--repeats 3`, `--cuda-pls-min-device-features 1` and
  `--compare-cuda-pls-many-batched`. The CSV now matches ABI `1.21.0`, includes
  CPU baseline, CUDA default and CUDA many-batched profiles, and had zero
  child-process errors.
- Current smoke interpretation: PLS stays on CPU at `260x256`, CUDA default
  wins at `512x512` and `256x1024`, and many-batched remains slower than CUDA
  default on those larger PLS rows.
- Validation:
  - py_compile on the touched benchmark/test files: PASS.
  - targeted benchmark-tool pytest: `1 passed, 20 deselected`; full file:
    `21 passed`.

## 2026-06-06 — PLS Exact Batch OpenMP Aggregation Slice

- Parallelized the exact PLS moment score-only batch path's per-chain held-out
  SSE/result aggregation with `N4M_PARALLEL_FOR_STATIC`, after shared prefix
  fits have already been computed.
- The patch keeps public scores, candidate ordering, ABI and counters unchanged;
  per-chain failures are stored in thread-local vectors and reported through
  `Context` only after the parallel region.
- Added `test_native_aom_pls_moment_batch_omp_scores_match_scalar_build` to
  compare scalar and OpenMP builds in subprocesses, including candidate-score
  parity and exact batch counter checks.
- Synthetic timing smoke (`n=192, p=32`, 31 chains, PLS exact CV,
  components `1/2/4`, five repeats): dev `3.00 ms`, OpenMP one thread
  `3.10 ms`, OpenMP four threads `2.45 ms`, with unchanged selected score and
  score matrix shape.
- Validation:
  - dev, OpenMP and CUDA `n4m_c` builds: PASS.
  - full dev `test_moment_model_wrappers.py`: `76 passed`.
  - targeted dev and OpenMP wrapper checks:
    `2 passed, 74 deselected` each.
  - Claude Code Opus/max review found no blocking issue: no data race, no
    ABI/API/counter change, deterministic per-chain accumulation preserved.
    Residual non-blocking note: CI does not appear to build `omp-on`, so the
    new parity guard is mostly a local guard until an OpenMP CI job exists.

## 2026-06-06 — CUDA PLS Scheduler Precedence Slice

- Fixed `pls1_moment_components_many()` route precedence so the explicit
  `cuda_pls_many_batched=True` / `N4M_CUDA_PLS_MANY_BATCHED=1` tiled scheduler
  is tried before `cuda_pls_parallel_folds=True` when both knobs are set.
- Made `N4M_CUDA_PLS_MANY_LEGACY=1` a real override: it disables the tiled
  many-batched route even when the explicit Python flag or environment opt-in
  is set, then falls through to requested parallel-folds or the sequential
  legacy path.
- Added `test_cuda_pls_many_batched_precedes_parallel_and_legacy_overrides`,
  a live CUDA subprocess guard that checks route counters for both cases and
  asserts candidate-score parity.
- Updated `docs/methods/sweep_run.md` and
  `docs/methods/aom_chain_sweep_run.md` with the precedence and legacy override
  semantics.
- Validation:
  - CUDA `n4m_c` build: PASS.
  - targeted wrapper tests on `cuda-on` and `dev-release`:
    `2 passed, 75 deselected` each.
  - full dev `test_moment_model_wrappers.py`: `77 passed`.
  - CUDA artifact guard file: `22 passed`.
  - live CUDA precedence guard:
    `1 passed, 76 deselected`.
  - py_compile on `test_moment_model_wrappers.py`: PASS.
  - `git diff --check`: PASS.

## 2026-06-06 — PLS Exact Single-Chain Prefix Recovery Slice

- Extended the exact PLS moment fallback strategy from score-only batches to
  single-chain `run_moment_sweep` and the internal `score_pls1_moment_sweep`.
  When the requested maximum component prefix fails, these paths now retry
  smaller requested prefixes in descending order and reuse recovered lower
  prefixes for fold-CV scoring.
- The slice preserves the moment-only semantics: no transformed `X`
  materialization is introduced for the recovered case, components above the
  recovered prefix become `inf`, and lower recovered components keep exact
  held-out moment SSE scoring.
- Final PLS moment refits now fit the selected component prefix instead of the
  largest component in the grid, so a selected lower component is not blocked by
  a rejected higher component.
- Added direct internal coverage for `score_pls1_moment_sweep` prefix recovery
  and updated the public wrapper rank-deficient fallback test so
  `n4m.sweep_run` and `n4m.pls_cross_validate` expect recovered moment fits
  rather than materialized PLS fallback on the covered case.
- Validation so far:
  - dev and CUDA `n4m_c` / `n4m_internal_tests` builds: PASS.
  - dev and CUDA internal tests: PASS.
  - targeted dev wrapper checks:
    `3 passed, 77 deselected`.
  - targeted one-GPU CUDA wrapper checks:
    `4 passed, 76 deselected`.
  - full dev wrapper:
    `80 passed`.
  - py_compile on `test_moment_model_wrappers.py`: PASS.
  - `git diff --check`: PASS.

## 2026-06-06 — PLS Exact Batch Lower-Prefix Downgrade Slice

- Improved the score-only AOM PLS batch path when a maximum requested component
  prefix fails. Instead of immediately scalarizing every chain/fold job, the
  batch scorer now retries lower requested prefixes through the shared
  `fit_pls1_moment_prefixes_for_folds` route.
- If a lower prefix succeeds globally, lower components are scored from that
  grouped batch and keep the existing CUDA route telemetry, including the
  many-batched counters. Components above the recovered grouped prefix still
  run through exact per-job fallback, so healthy chains can keep higher
  components finite while rank-deficient chains mark only their failed
  components `inf`.
- This is score-preserving relative to the previous exact fallback while
  reducing the amount of scalar fallback work after late-component failures.
  It is a practical step toward the full cartesian PLS grinder because a single
  bad high component no longer destroys batching for all lower prefixes.
- Validation:
  - dev `n4m_c` / `n4m_internal_tests` build and internal tests: PASS.
  - targeted dev wrapper checks:
    `8 passed, 72 deselected`.
  - CUDA `n4m_c` / `n4m_internal_tests` build and internal tests: PASS.
  - targeted one-GPU CUDA wrapper checks:
    `6 passed, 74 deselected`.
  - live CUDA many-batched guard now asserts the rank-deficient downgraded
    batch uses one many-batched prefix batch before host fallback handles the
    failing higher component.
  - full dev wrapper:
    `80 passed`.
  - py_compile on `test_moment_model_wrappers.py`: PASS.
  - `git diff --check`: PASS.

## 2026-06-06 — Staged PLS Batch Telemetry Propagation Slice

- Propagated PLS moment score-batch counters through
  `aom_staged_chain_campaign` reports. Per-stage summaries now expose
  `n_pls_moment_score_batch_calls/jobs`, top-level staged reports expose
  `n_screen_pls_moment_score_batch_calls/jobs` and
  `n_refit_pls_moment_score_batch_calls/jobs`, and the `scale_x_values`
  aggregation sums those counters across every evaluated model config.
- `NativeAOMStagedChainCampaignRegressor.get_diagnostics()` now exposes those
  staged screen/refit batch counters plus the staged PLS parallel-fold and
  many-batched CUDA counters. This closes an observability gap where a staged
  campaign could report PLS moment CV/device fits without proving whether the
  grouped batch route was actually used.
- Added staged campaign tests covering per-stage/top-level sums, split-head PLS
  exact-CV batch calls, model-config aggregation and sklearn diagnostics.
- Validation:
  - staged campaign test file:
    `16 passed`.
  - targeted moment wrapper/inventory checks:
    `5 passed, 75 deselected`.
  - full dev wrapper:
    `80 passed`.
  - py_compile on touched Python modules/tests: PASS.
  - `git diff --check`: PASS.

## 2026-06-06 — Real-Cohort PLS Batch Telemetry Export Slice

- Extended `run_aom_staged_real_cohort.py` so real benchmark rows and compact
  diagnostics JSON carry the staged PLS score-batch counters:
  `n_screen_pls_moment_score_batch_calls/jobs` and
  `n_refit_pls_moment_score_batch_calls/jobs`.
- Extended `compare_aom_staged_variants.py` route-counter aggregation with the
  same fields. Variant summaries can now report total grouped PLS batch calls
  and jobs alongside CV/device/many-batched counters.
- This makes campaign artifacts sufficient to audit whether a real-cohort run
  stayed on the intended grouped PLS route, without reopening per-dataset JSON
  reports or inferring from CV fit counts.
- Validation:
  - targeted staged-real-cohort/variant tests:
    `3 passed, 18 deselected`.
  - full benchmark tool tests:
    `21 passed`.
  - staged campaign tests:
    `16 passed`.
  - py_compile on touched benchmark scripts/tests: PASS.
  - `git diff --check`: PASS.

## 2026-06-06 — Staged Comparator PLS Score-Mode Slice

- Added `pls_score_mode` to `compare_aom_staged_variants.py` campaign config
  keys and variant labels, so exact-CV (`cv`) and proxy (`gcv_proxy`) PLS
  staged campaign rows stay in separate summary variants.
- Added a comparator regression test that keeps two `cv` rows on different
  datasets grouped together while separating the `gcv_proxy` row, and asserts
  dataset identity/source/name tokens are not present in the config key.
- Updated the cross-binding benchmark README to document that score-mode knobs
  participate in variant grouping.
- Validation:
  - focused staged comparator tests:
    `4 passed, 18 deselected`.
  - full benchmark tool tests:
    `22 passed`.
  - py_compile on touched Python files: PASS.
  - `git diff --check`: PASS.

## 2026-06-06 — Direct Moment Heads CPU/CUDA Artifact Schema Slice

- Regenerated `direct_moment_heads_timing.csv` and
  `direct_moment_heads_timing_cuda_smoke.csv` with the current direct-head
  timing schema. Both artifacts now cover all 9 reusable direct moment/linear
  heads across three shapes with native-function and sklearn replay rows.
- The CPU artifact now matches the CUDA artifact schema, including the PLS route
  controls/counters and the `n_pls_moment_cuda_many_batched_*` fields. CPU PLS
  rows prove host exact-CV routing; CUDA PLS rows prove one-GPU device exact-CV
  routing with many-batched explicitly off.
- Strengthened artifact tests so both CPU and CUDA direct-head CSVs must expose
  the current route fields, current method set, replay status and route
  counters.
- Validation:
  - regenerated CPU direct-head timing artifact on `build/dev-release`.
  - regenerated CUDA direct-head timing artifact on one visible GPU
    (`CUDA_VISIBLE_DEVICES=0`, `build/cuda-on`).
  - focused direct-head artifact tests:
    `2 passed, 21 deselected`.
  - full CUDA/artifact guard file:
    `23 passed`.
  - py_compile on touched benchmark/test files: PASS.
  - `git diff --check`: PASS.
