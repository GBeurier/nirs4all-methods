# Handoff - AOM / moment portage

Date: 2026-06-06

## Current state

This branch has a broad AOM/moment integration in `nirs4all-methods`.

Completed and validated in the latest pass:

- Staged benchmark PLS score-mode switch:
  - `run_aom_staged_real_cohort.py` and
    `bench_aom_staged_chain_campaign_timing.py` now expose
    `--pls-score-mode {cv,gcv_proxy}` and forward it to
    `n4m.aom_staged_chain_campaign`.
  - The selected mode is persisted in real-cohort CSV rows, diagnostics JSON
    runner metadata and staged timing CSV rows. Default remains exact-CV
    (`cv`); `gcv_proxy` is explicit for proxy-vs-exact recall/timing
    campaigns, while refit remains exact-CV.
  - Validation: `test_aom_benchmark_tools.py` passed (`21 passed`);
    a synthetic staged timing smoke with `--pls-score-mode gcv_proxy` wrote
    `pls_score_mode=gcv_proxy`; py_compile and `git diff --check` passed.

- PLS exact-CV reusable screen/refit preset:
  - Commit `570ef01 feat(aom): add exact PLS screen-refit preset` adds
    `NativeAOMMomentPLSExactScreenRefitRegressor`, a PLS-only
    `NativeAOMScreenRefitRegressor` preset with `heads=("pls",)`,
    `ridge_lambdas=()`, `moment_policy="force_moments"`,
    `chain_ordering="prefix"` and first-pass `pls_score_mode="cv"`.
  - The existing `NativeAOMMomentPLSScreenRefitRegressor` remains the
    GCV-proxy -> exact-refit preset; the new exact preset is additive and is
    intended for screen-recall audits and reusable exact train-CV PLS screens.
  - Public exports now cover `n4m`, `n4m.sklearn`, `n4m.aom` and
    `n4m.moment`; inventories expose `moment_pls_exact_screen_refit`.
    Method docs, coverage matrix and catalog notes distinguish PLS GCV,
    PLS exact-CV and Ridge exact presets.
  - Validation after this Python-surface/docs slice: full dev-release
    `test_moment_model_wrappers.py` passed (`80 passed`); py_compile on
    touched Python modules/tests passed; catalog validation and split-method
    checks passed; `git diff --check` passed. Branch was pushed to
    `origin/release-readiness-fixes`.
  - Follow-up CUDA facade smoke coverage added after the commit: the
    regenerated `benchmarks/cross_binding/aom_moment_cuda_facade_smoke.json`
    now has `pls_exact_screen_refit_estimator` with exact-CV screen/refit
    (`pls_score_mode=cv`, `refit_pls_score_mode=cv`), zero GCV-proxy fits,
    screen/refit CUDA device CV fits `8/4` and host PLS CV fits `0`.
    `test_aom_moment_cuda_smoke_artifacts.py` passed (`22 passed`).

- Real-cohort benchmark PLS batch telemetry export:
  - `benchmarks/cross_binding/run_aom_staged_real_cohort.py` now persists the
    staged PLS score-batch counters in result CSV rows and diagnostics JSON:
    screen/refit `n_*_pls_moment_score_batch_calls` and
    `n_*_pls_moment_score_batch_jobs`.
  - `compare_aom_staged_variants.py` now includes those counters in variant
    route totals, so benchmark summaries can report how much of a run actually
    used grouped PLS batch scoring.
  - This closes the loop from native/staged telemetry to campaign artifacts:
    real-cohort benchmark CSVs now prove both CV/device counts and batch-route
    engagement.
  - Validation so far: targeted staged-real-cohort/variant tests passed
    (`3 passed`); full `test_aom_benchmark_tools.py` passed (`21 passed`);
    staged campaign tests passed (`16 passed`); py_compile on touched
    benchmark scripts/tests and `git diff --check` passed.

- Staged PLS batch telemetry propagation:
  - `aom_staged_chain_campaign` now carries PLS moment score-batch
    calls/jobs through every level where the other PLS CV/CUDA counters were
    already visible: per-stage summaries, top-level screen aggregation,
    refit aggregation, `scale_x_values` model-config aggregation and the
    `NativeAOMStagedChainCampaignRegressor.get_diagnostics()` payload.
  - The staged reports now distinguish “PLS CV fits happened” from “the screen
    actually stayed on the grouped score-batch path”, which is necessary for
    evaluating whether large cartesian campaigns are using the intended grinder
    route.
  - The sklearn diagnostics also expose the staged PLS parallel-fold and
    many-batched CUDA counters alongside the existing host/device counts.
  - Validation: staged campaign tests passed (`16 passed`); targeted
    wrapper/inventory tests passed (`5 passed`); full dev-release
    `test_moment_model_wrappers.py` passed (`80 passed`); `py_compile` on
    touched Python modules/tests and `git diff --check` passed.

- PLS exact batch lower-prefix downgrade:
  - `score_pls1_moment_sweeps_score_only` now keeps a shared batch/GPU route
    after a maximum-component prefix failure when a lower requested PLS prefix
    is still valid across all chain/fold jobs.
  - The path first retries lower requested prefixes as one grouped
    `fit_pls1_moment_prefixes_for_folds` job. Recovered lower components are
    scored from that batch, preserving many-batched / parallel-fold CUDA
    counters when those routes are active.
  - Components above the recovered grouped prefix are not blindly marked
    failed. They continue through the exact per-job fallback, so healthy
    chains/folds can still keep higher components finite while rank-deficient
    jobs mark only the failing components `inf`.
  - This is another step toward the 200k-chain PLS grinder: a late
    rank-deficient component no longer forces the whole screen down to scalar
    per-job fits for lower prefixes.
  - Validation so far: dev internal tests passed; targeted dev wrapper checks
    passed (`8 passed`); CUDA `n4m_c`/`n4m_internal_tests` build and internal
    tests passed; targeted one-GPU CUDA wrapper checks passed (`6 passed`);
    the live CUDA many-batched guard now covers the downgraded rank-deficient
    batch route; full dev-release `test_moment_model_wrappers.py` passed
    (`80 passed`); `py_compile` and `git diff --check` passed.

- PLS exact single-chain fallback prefix recovery:
  - `run_moment_sweep` and the internal `score_pls1_moment_sweep` no longer
    have to abandon the exact moment route just because the requested maximum
    PLS component prefix is numerically/rank deficient. After a max-prefix
    failure, they now try the largest still-needed requested prefix, descend
    only when needed, and reuse recovered lower prefixes for exact fold-CV
    scoring.
  - Components above the recovered prefix are marked `inf`; smaller recovered
    components keep their exact moment scores. `n4m.sweep_run` and
    `n4m.pls_cross_validate` therefore avoid the old materialized fallback on
    the covered rank-deficient case, while preserving train-CV semantics and
    not introducing transformed `X` materialization.
  - Final PLS moment refits now request only the selected component prefix
    instead of the largest grid component, so a valid lower selected component
    can still complete after a higher component was rejected during CV.
  - No ABI or public-method surface changed. The recovery counters count actual
    prefix-fit attempts; components not recovered remain failed/`inf`.
  - Validation after this C++ fallback fix: rebuilt dev-release and CUDA
    `n4m_c`/`n4m_internal_tests`; dev and CUDA internal tests passed; targeted
    dev pytest passed (`3 passed`); targeted one-GPU CUDA pytest passed
    (`4 passed`); full dev-release `test_moment_model_wrappers.py` passed
    (`80 passed`); `py_compile` and `git diff --check` passed.

- PLS exact batch fallback prefix reuse:
  - `score_pls1_moment_sweeps_score_only` no longer refits every requested
    component independently after a global batched prefix failure. The fallback
    now tries the largest still-needed requested prefix per chain/fold job,
    descends only if that prefix fails, and reuses every recovered lower prefix
    for scoring.
  - This preserves the exact fold-CV scores and the existing failure semantics:
    a component that fails on any fold is marked `inf`, smaller recovered
    prefixes stay finite, and no transformed `X` materialization is introduced.
  - The fallback fit counters now count actual prefix-fit attempts, so healthy
    jobs in a mixed healthy/degenerate batch fall back to one PLS fit per fold
    instead of one fit per component per fold.
  - Validation after this C++ fallback fix: rebuilt dev-release and CUDA
    `n4m_c`/`n4m_internal_tests`; dev and CUDA internal tests passed; targeted
    dev pytest passed (`3 passed`); targeted one-GPU CUDA pytest passed
    (`4 passed`); full dev-release `test_moment_model_wrappers.py` passed
    (`80 passed`).

- Fixed-candidate CUDA option surface alignment:
  - `n4m.aom_chain_fixed_fit_run` and `NativeAOMFixedCandidateRegressor` now
    accept the same public PLS CUDA option names advertised by the AOM/moment
    facade inventories: `cuda_pls_parallel_folds`,
    `cuda_pls_min_device_features` and `cuda_pls_many_batched`.
  - Final-only fixed fits currently consume the threshold knob on the native
    PLS component path; fold/many-batch knobs are forwarded for API symmetry and
    for `fit_mode="cv"`, where the underlying `aom_chain_sweep_run` can use
    them.
  - Validation after this Python-surface fix: targeted fixed-candidate/facade
    pytest passed; full dev-release `test_moment_model_wrappers.py` passed
    (`80 passed`); `py_compile` and `git diff --check` passed.

- AOM PLS exact batch partial-failure guard:
  - `score_pls1_moment_sweeps_score_only` now keeps broad force-moment screens
    alive when a rank-deficient late component makes the global batched prefix
    fit fail. It falls back to slower per-chain/fold/component moment fits,
    keeps scoreable components finite and marks failed components as `inf`
    instead of aborting the whole AOM screen.
  - This remains a moment-only fallback: no transformed `X` materialization is
    introduced, and normal successful batch runs still report the existing
    `n_pls_moment_score_batch_*` counters.
  - Added
    `test_aom_pls_moment_batch_degenerate_components_do_not_abort_screen`.
  - Validation after this fix: rebuilt dev-release and CUDA `n4m_c`; manual
    CPU/CUDA reproduction on a rank-deficient identity-chain AOM PLS screen now
    returns finite component-1 scores and `inf` component-2 scores with
    `n_materialized_candidates=0`; targeted dev/CUDA pytest passed; full
    dev-release `test_moment_model_wrappers.py` passed (`80 passed`);
    `py_compile` and `git diff --check` passed.

- Rank-deficient PLS moment fallback segfault fix:
  - Fixed `run_moment_sweep` so PLS moment-route failures can lazily build
    fold-local materialized designs before falling back. Previously, compatible
    PLS1 moment screens could decide that fold designs were unnecessary, then
    hit a numerical/rank-deficient moment fit failure and dereference an empty
    `fold_designs` vector in the fallback path.
  - Added `test_pls_moment_fallback_builds_fold_designs_on_demand`, covering
    the tiny rank-deficient fixture in both `score_only=True` and `False`, plus
    the public `n4m.pls_cross_validate(..., score_only=True)` path.
  - Validation after this fix: rebuilt dev-release and CUDA `n4m_c`; the
    previous segfault fixture now returns finite/inf candidate scores with
    `n_pls_moment_cv_fits=0` and materialized fallback counters; targeted
    dev/CUDA pytest passed; full dev-release `test_moment_model_wrappers.py`
    passed (`79 passed`); `py_compile` and `git diff --check` passed.

- PLS CV ABI reference surface:
  - Added ABI 1.22.0 symbol
    `n4m_pls_cross_validate(ctx, cfg, X, Y, fold_ids, n_fold_ids, n_folds,
    component_grid, n_component_grid, out_result)`.
  - The symbol now provides an exact PLS-only reference path by delegating to
    `n4m_sweep_run` with `heads_mask=PLS`; candidate scores and route counters
    match the existing sweep PLS branch. It is catalogued as ABI infrastructure,
    not as a production method.
  - Exposed the reference path as `n4m.pls_cross_validate` and
    `n4m.moment.pls_cross_validate`; the moment facade inventory now advertises
    the PLS CV reference entry and its CUDA PLS knobs.
  - Updated Python ctypes declarations, ABI floor, Linux/macOS/Windows expected
    symbol snapshots, ABI changelog, DEFERRALS, release-readiness notes,
    worklog and the moment coverage matrix.
  - Validation after this slice: rebuilt dev-release and CUDA `n4m_c` to
    `libn4m.so.1.22.0`; targeted dev/CUDA equivalence tests passed; full
    `test_moment_model_wrappers.py` passed (`78 passed`); `py_compile` on
    touched Python modules/tests passed; catalog strict ABI/reference/split
    checks, `reconcile_abi.py --check`, `git diff --check` and
    `scripts/bump_version.sh --check` passed.
  - Follow-up status: the tiny rank-deficient PLS fallback crash found during
    validation is now fixed and covered by the latest test above.
  - Remaining true gap: actual grouped/fused host/device executor,
    score-equivalence tests and timings for the 200k-chain PLS grinder.

- Compact-wide audit10 benchmark follow-up:
  - Ran `plan=compact_wide` on the first 10 runnable diverse11 cohort rows in
    two chunks (`audit5` + `audit10_tail5`) on one GPU, then combined them into
    `benchmarks/cross_binding/aom_staged_real_cohort_compact_wide_audit10_20260606.csv`.
  - Output: 10 rows total, 8 OK and 2 property-skipped rows
    (`BERRY`, `FUSARIUM`, both `n_features>1200`), all with
    `selection_uses_test_set=False`.
  - OK rows all selected `ridge` and `selected_campaign_stage=compact`; the
    extra wide stage did not win under this retained/refit budget.
  - PLS routing stayed on CUDA: screen PLS moment CV fits `1920/0/1920`
    total/host/device; refit PLS moment CV fits `220/0/220`.
    Split-head counters were `64` split chunks and `128` score calls; Ridge
    screen counters were `2160/24/2160`.
  - Timing: total OK fit time `1562.03s`, median OK fit time `155.04s`.
  - Baseline comparison versus
    `aom_staged_real_cohort_compact10_split_head_auto_20260606.csv`: 8 paired,
    `0` wins, `0` losses, `8` ties, median ratio `1`. This shows
    `compact_wide` currently adds cost but no selected improvement on this
    cohort/budget.
  - Oracle comparison:
    `aom_staged_compact_wide_audit10_20260606_oracle_summary.md` reports
    AOM-PLS oracle median ratio `1.03079` (1/7 target wins), AOM-Ridge oracle
    median ratio `1.08068` (0/8 wins), TabPFN oracle median ratio `0.978527`
    (4/8 target wins), matching the compact split-head-auto score profile.
  - New rank-audit summary:
    `aom_staged_compact_wide_audit10_20260606_rank_audit.md` has 8 audit rows.
    Median test-rank delta is `2.5`, max `5`; median oracle-gap ratio is
    `0.02354`, max `0.57732` on ECOSIS; median CV/test Spearman is `0.64286`,
    min `-0.73810` on ECOSIS. This is the first concrete evidence from the new
    audit payload that recall failure is dataset-specific even when overall
    compact-vs-wide scores tie.
    `summarize_aom_rank_audit.py` now exposes selected/oracle head, parameter
    and preprocessing chain plus audit top-1/top-3/top-5 recall columns. The
    ECOSIS row shows the missed test-best candidate directly:
    selected `ridge:0.1 savgol_smooth(7,2)` vs oracle
    `pls:1 detrend_poly(2)`, oracle CV-rank `7`, selected test-rank `5`,
    top-1/top-3 recall `0/0`.
  - Impact summary:
    `aom_staged_compact_wide_audit10_20260606_impact_summary.md` ranks
    `detrend_poly` as the top operator by dataset wins (`5/7`), identity next
    (`2/5`), and SavGol smooth one win (`1/8`); derivatives did not win under
    this profile.
  - ECOSIS retention stress test:
    `aom_staged_real_cohort_ecosis_compact_wide_refit40_audit_20260606.csv`
    reran only ECOSIS with `top_k=60`, `refit_top_k=40` and
    `refit_per_head_top_k=10`, producing `44` retained/refit candidates.
    Production selection still chose the same `ridge:0.1 savgol_smooth(7,2)`
    candidate and tied the audit10 score (`RMSEP=41.3984`), but the offline
    audit oracle moved to `ridge:10 finite_difference(1)` with
    `eval_rmse=10.5817`, `cv_rank=11`, selected test-rank `41/44`,
    top-1/top-3/top-5 recall `0/0/0`, and CV/test Spearman `0.0667`.
    This proves that, at least for ECOSIS, the observed failure is not just
    insufficient retained-candidate budget; the train-CV ranking itself is not
    aligned with held-out behavior.

- Audit/test-rank diagnostics persistence and summarizer:
  - `run_aom_staged_real_cohort.py` now persists a compact `audit` section in
    each per-dataset diagnostics JSON when `report["audit"]` is present.
    The payload keeps `audit_only`, `n_candidates`, the train-CV selected
    candidate scored on test as `selected_cv`, the best-by-test-rank candidate
    as `oracle`, and `audit_rank_diagnostics`; prediction arrays are stripped.
  - This is offline audit only. Production selection remains train-CV-only and
    diagnostics keep `selection_uses_test_set=False`.
  - Added `benchmarks/cross_binding/summarize_aom_rank_audit.py`, which reads a
    diagnostics directory or glob and writes CSV plus optional Markdown with
    selected CV RMSE/rank, selected test RMSE/rank, test-rank delta, oracle test
    RMSE, oracle CV rank, candidate count and CV-vs-test Spearman.
  - Existing compact10 diagnostics in
    `benchmarks/cross_binding/aom_staged_compact10_split_head_auto_20260606/`
    pre-date this payload, so the summarizer reports `8` files without audit
    and writes a header-only CSV until the campaign is rerun with
    `--diagnostics-dir`.
  - Validation: py_compile on touched scripts/tests; local
    `test_aom_benchmark_tools.py` `19 passed`; targeted
    benchmark/catalog/facade/staged suite `63 passed`; synthetic summarizer
    smoke wrote `1` audit row and correctly reported `1` pre-feature file;
    existing compact10 smoke wrote `0` audit rows and reported `8` pre-feature
    files.

- Real-cohort split-head scoring audit path:
  - `run_aom_staged_real_cohort.py` now exposes
    `--split-head-scoring {auto,off,force}` and defaults to `auto`.
  - This is score-preserving for mixed `ridge,pls` chunks: the screen runs
    separate Ridge-only and PLS-only native score calls, then merges the same
    `(chain, head, param)` candidate rows. `off` keeps the legacy single-call
    launch shape for timing comparisons.
  - Output CSVs now include `split_head_scoring`,
    `n_screen_split_head_chunks` and `n_screen_chunk_score_calls`.
  - `n4m.aom_staged_chain_campaign` aggregates those counters across stages,
    stage summaries expose the per-stage values, and
    `NativeAOMStagedChainCampaignRegressor.get_diagnostics()` exposes the
    top-level counters.
  - The staged report now also surfaces the Ridge screen counters expected by
    the real-cohort diagnostics payload:
    `n_ridge_moment_cv_fits`, `n_ridge_moment_score_batch_calls` and
    `n_ridge_moment_score_batch_jobs`.
  - For `scale_x_values` model-config grids, the staged report now aggregates
    split-head and Ridge screen counters across every evaluated config instead
    of inheriting those counters only from the selected config.
  - `NativeAOMStagedChainCampaignRegressor`,
    `NativeAOMSavgolFocusRegressor` and
    `NativeAOMStrictFamilyLiteRegressor` now default
    `split_head_scoring="auto"` for mixed-head reusable sklearn use, while the
    lower-level `aom_staged_chain_campaign` helper keeps the historical
    explicit `off` default.
  - `benchmarks/cross_binding/aom_moment_cuda_facade_smoke.py` and its JSON
    artifact now include `staged_mixed_default_estimator`, a mixed Ridge+PLS
    sklearn smoke proving the reusable default actually splits mixed chunks on
    CUDA (`n_screen_split_head_chunks=1`, `n_screen_chunk_score_calls=2`),
    routes the PLS screen through CUDA (`8/0` device/host CV fits), records
    Ridge screen counters (`8/1/8`) and keeps `selection_uses_test_set=False`.
  - `compare_aom_staged_variants.py` includes `split_head_scoring` in the
    config key so `auto` and `off` runs are not grouped together.
  - Validation: py_compile on touched Python modules/tests; benchmark-tool +
    staged pytest files `33 passed`; synthetic score-preserving smoke confirmed
    identical top candidates for `off` vs `auto`; staged smoke reported
    `n_screen_split_head_chunks=2`, `n_screen_chunk_score_calls=4`,
    `selection_uses_test_set=False`; CUDA facade artifact test passed after
    regenerating the smoke JSON; full targeted pytest suite `154 passed`;
    catalog split/reference/strict-ABI validations passed.
  - Real one-row CUDA CLI smoke also passed to `/tmp`: BEEFMARBLING reported
    `split_head_scoring=auto`, `n_screen_split_head_chunks=2`,
    `n_screen_chunk_score_calls=4`, PLS screen CUDA/host `12/0`, and
    `selection_uses_test_set=False`.
  - Real compact10 split-head follow-up:
    `benchmarks/cross_binding/aom_staged_real_cohort_compact10_split_head_auto_20260606.csv`,
    diagnostics dir
    `benchmarks/cross_binding/aom_staged_compact10_split_head_auto_20260606/`,
    baseline comparison
    `benchmarks/cross_binding/aom_staged_compact10_split_head_auto_vs_compact_20260606.md`,
    oracle summary
    `benchmarks/cross_binding/aom_staged_compact10_split_head_auto_20260606_oracle_summary.md`,
    and impact summary
    `benchmarks/cross_binding/aom_staged_compact10_split_head_auto_20260606_impact_summary.md`.
    It produced 8 OK rows and 2 property-skipped rows, all with
    `selection_uses_test_set=False`; versus the compact mixed diagnostic
    baseline it was score-identical on the 8 paired datasets (0 wins, 0 losses,
    8 ties, median ratio `1`). PLS stayed on CUDA for screen/refit (`960/0`
    and `220/0`), and the screen counters reported 32 split chunks and 64
    score calls across the 8 OK rows after summing both `scale_x` configs.
    The CSV now includes Ridge screen-counter columns as well:
    total Ridge moment/batch counters across OK rows were `1080/12/1080`.
    The diagnostics were regenerated after the Ridge/config-grid counter fix;
    e.g. BEEFMARBLING now records Ridge moment/batch counters `360/4/360`.

- Focused preprocessing diagnostic campaign:
  - Ran a one-GPU compact10 follow-up using custom stages derived from the
    previous diagnostics: identity/detrend/Norris plus a SavGol-variety branch
    with six smooth windows and three derivatives.
  - Outputs:
    `benchmarks/cross_binding/aom_staged_real_cohort_compact10_diag_focused_20260606.csv`,
    diagnostics dir
    `benchmarks/cross_binding/aom_staged_compact10_diag_focused_20260606/`,
    oracle comparison
    `benchmarks/cross_binding/aom_staged_real_cohort_compact10_diag_focused_20260606_oracle_compare.csv`,
    variant comparison
    `benchmarks/cross_binding/aom_staged_compact10_diag_focused_vs_compact_20260606.md`,
    and impact summary
    `benchmarks/cross_binding/aom_staged_compact10_diag_focused_20260606_impact_summary.md`.
  - Results: 8 OK rows, 2 property-skipped rows (`n_features>1200`), all OK
    rows with `selection_uses_test_set=False`.
  - The selected production head was Ridge on all 8 OK rows; selected stages
    were `identity_detrend_norris` on 5/8 and `savgol_variety` on 3/8.
  - PLS GPU routing stayed clean: screen CUDA/host `1280/0`, refit CUDA/host
    `215/0`.
  - Versus the compact mixed diagnostic baseline: 2 wins, 4 losses, 2 ties,
    median ratio `1.00125`; median fit time `8.55s`.
  - Oracle ratios: AOM-PLS oracle median `1.05178` (1/7 target wins),
    AOM-Ridge oracle median `1.09528` (1/8 target wins), TabPFN oracle median
    `0.992611` (4/8 target wins).
  - Interpretation: focused SavGol/detrend diversity is selectable and audited,
    but this small follow-up did not beat the compact baseline in aggregate.

- Real-cohort staged diagnostics output:
  - `run_aom_staged_real_cohort.py` now accepts `--diagnostics-dir DIR`.
  - Default behavior is unchanged when the flag is omitted.
  - For each `ok` dataset row it writes
    `<safe_dataset_key>.diagnostics.json` with `best`, `impact`,
    `rank_diagnostics`, selected model config and route/counter fields.
  - It also appends `impact_groups.csv` rows for `by_operator`,
    `by_stage_family`, `by_stage_option` and `by_head_stage_option`.
  - Added `summarize_aom_impact_groups.py` to convert `impact_groups.csv` into
    aggregate CSV/Markdown ranked by dataset wins, rank-1 occurrences and
    train-CV rank.
  - This is offline audit output only; production selection still uses the
    train-CV refit winner and `selection_uses_test_set` is persisted.
  - Follow-up real compact10 mixed diagnostics run:
    `benchmarks/cross_binding/aom_staged_real_cohort_compact10_mixed_diag_20260606.csv`
    plus diagnostics dir
    `benchmarks/cross_binding/aom_staged_compact10_mixed_diag_20260606/`.
  - Summary:
    `benchmarks/cross_binding/aom_staged_compact10_mixed_diag_20260606_summary.md`.
    Automated impact summary:
    `benchmarks/cross_binding/aom_staged_compact10_mixed_diag_20260606_impact_summary.md`
    plus the matching `.csv`.
    Best train-CV impact groups were `detrend_poly` on 5/8 OK datasets,
    `identity` on 2/8 and `savgol_smooth` on 1/8; selected production head was
    Ridge on all 8 OK rows.
  - Validation: py_compile pass; runner tests
    `5 passed, 11 deselected`.

- Compact10 staged CUDA benchmark against local oracles:
  - Ran `run_aom_staged_real_cohort.py` on one GPU with `--limit 10`,
    `--max-features 1200`, `--plan compact`, `--max-chains 12`,
    `--top-k 12`, `--refit-top-k 6`, `--refit-per-head-top-k 2`,
    `--scale-x-grid false,true`, `--cuda-pls-parallel-folds` and
    `--cuda-pls-min-device-features 1`.
  - Ran three variants: mixed Ridge+PLS, Ridge-only and PLS-only.
  - Each variant produced 8 OK rows and 2 property-skipped rows; all OK rows
    have `selection_uses_test_set=False`.
  - Mixed selected `ridge` on every OK row and exactly matched Ridge-only
    RMSEP; the compact mixed profile currently adds no PLS diversity on this
    cohort.
  - Mixed/Ridge ratios: AOM-PLS oracle median `1.03079` (1/7 target wins),
    AOM-Ridge oracle median `1.08068` (0/8 wins), TabPFN median `0.978527`
    (4/8 wins).
  - PLS-only ratios: AOM-PLS oracle median `1.22592` (0/7 wins), AOM-Ridge
    oracle median `1.21773` (0/8 wins), TabPFN median `1.27884` (1/8 wins).
  - PLS CUDA routing was clean: mixed screen/refit PLS CUDA/host `960/0` and
    `220/0`; PLS-only `960/0` and `450/0`.
  - Summary artifact:
    `benchmarks/cross_binding/aom_staged_compact10_cuda_20260606_summary.md`.

- Robust-HPO native sklearn CUDA smoke artifact:
  - Regenerated `benchmarks/cross_binding/aom_robust_hpo_timing_cuda_smoke.csv`
    without `--native-only`, adding the `native_sklearn` backend rows.
  - New artifact: 6 rows (3 shapes x 2 backends:
    `native_abi` + `native_sklearn`),
    `profile=compact`, `library_path=build/cuda-on/cpp/src/libn4m.so`.
  - `prediction_replay_max_abs_error` for `native_sklearn` rows: <= 4.5e-15,
    proving `NativeAOMRobustHPORegressor.predict(X)` replays native fitted
    state (`X @ input_coefficients + intercept`) on the CUDA build path.
  - `test_aom_moment_cuda_smoke_artifacts.py -k robust_hpo`: `1 passed`.
  - Full targeted pytest set: `152 passed`.
  - Catalog split/reference/strict-ABI: all green.
  - Docs updated: README.md (new "AOM Robust-HPO Timing Smoke" section),
    `aom_moment_coverage_matrix.md` (robust-HPO row, artifact note),
    `aom_moment_worklog.md` and `handoff.md` (this entry).

- SavGol-focus reusable sklearn preset:
  - `NativeAOMSavgolFocusRegressor`
  - exported from `n4m`, `n4m.sklearn`, `n4m.aom` and `n4m.moment`
  - preset wrapper over `NativeAOMStagedChainCampaignRegressor` with
    `plan="savgol_focus"`, `max_chains=6`, `top_k=10`, `refit_top_k=8`,
    `refit_per_head_top_k=2`, `scale_x_values=(False, True)` and the one-GPU
    CUDA PLS route knobs used in the benchmark
  - no held-out/test inputs, no dataset/source routing, and final prediction
    still reuses the selected final-only fixed candidate
  - facade inventories expose it as `savgol_focus_regressor` with
    `catalog_role="preset_sklearn_wrapper"` over
    `aom_pop.aom_staged_chain_campaign`
  - `test_aom_staged_campaign.py` now fits the preset on a tiny synthetic
    dataset and verifies `plan="savgol_focus"`, selected scale config,
    train-CV-only selection and replay through `selected_model_`
  - `aom_moment_cuda_facade_smoke.py` and
    `aom_moment_cuda_facade_smoke.json` now cover this preset on the CUDA
    build; the JSON section `savgol_focus_estimator` reports screen PLS CUDA CV
    fits `64` / host `0`, refit PLS CUDA CV fits `8` / host `0`, and
    `selection_uses_test_set=False`
- Strict-family lite reusable sklearn audit preset:
  - `NativeAOMStrictFamilyLiteRegressor`
  - exported from `n4m`, `n4m.sklearn`, `n4m.aom` and `n4m.moment`
  - preset wrapper over `NativeAOMStagedChainCampaignRegressor` with
    `plan="strict_family_focus"`, but deliberately low defaults:
    `max_chains=2`, `chain_chunk_size=2`, `top_k=6`, `refit_top_k=4`,
    `refit_per_head_top_k=1`, `scale_x=False` and no `scale_x_values` grid
  - intended as a cost-safe family audit over SavGol, Norris-Williams,
    finite-difference, Gaussian, FCK and Whittaker stages; it is not the heavy
    benchmark profile and still performs train-CV-only production selection
  - facade inventories expose it as `strict_family_lite_regressor` with
    `catalog_role="preset_sklearn_wrapper"` over
    `aom_pop.aom_staged_chain_campaign`
  - `test_aom_staged_campaign.py` now fits the preset on a tiny synthetic
    dataset and verifies family stage coverage, no dataset/source/id stage
    metadata, `selection_uses_test_set=False`, selected `scale_x=False`, and
    replay through `selected_model_`
  - `aom_moment_cuda_facade_smoke.py` and
    `aom_moment_cuda_facade_smoke.json` now cover this preset on the CUDA
    build; the JSON section `strict_family_lite_estimator` reports screen PLS
    CUDA CV fits `36` / host `0`, refit PLS CUDA CV fits `4` / host `0`, and
    `selection_uses_test_set=False`
- Focused strict-family staged campaign plans:
  - `n4m.aom_staged_chain_campaign(..., plan="savgol_focus")`
  - `n4m.aom_staged_chain_campaign(..., plan="strict_family_focus")`
  - both are source-free, strict-linear recipes over the existing lab families;
    existing plans and explicit `stages=[...]` behavior are unchanged
  - `savgol_focus` runs compact, SavGol smooth, SavGol derivative and
    SavGol-combination stages so SavGol variety is reached with small
    `max_chains`
  - `strict_family_focus` adds separate Norris-Williams, finite-difference,
    Gaussian, FCK and Whittaker stages so `max_chains` is applied per family
    instead of cutting off late `lab` families
  - production selection remains train exact-CV only and no dataset/source/name
    metadata is read
  - one-GPU CLI smoke on one real cohort row passed with
    `plan=strict_family_focus`; it wrote 9 focused stage checkpoints, selected
    `strict_combinations`, kept `selection_uses_test_set=False`, and routed PLS
    screen/refit CV through CUDA with host counters at `0`
  - follow-up `savgol_focus --max-chains 6 --scale-x-grid false,true` on the
    local diverse-10 cohort with `--max-features 1200` produced 8 OK rows and 2
    property-skipped rows; against compact scale-grid on the 8 paired rows it
    had 5 wins, 2 losses and 1 tie, median paired ratio `0.995259`, mean ratio
    `0.996638`, median fit time `16.06s`, screen PLS CUDA CV fits `1920` / host
    `0`, refit PLS CUDA CV fits `245` / host `0`
  - a broader `strict_family_focus --max-chains 4` partial run showed family
    wins but stalled during a MANURE refit/finalization after all focused stage
    checkpoints were written; keep it as a heavier audit profile, not the fast
    default campaign
- Staged campaign model-config scaling grid:
  - `n4m.aom_staged_chain_campaign(..., scale_x_values=[False, True])`
  - `NativeAOMStagedChainCampaignRegressor(scale_x_values=[False, True])`
  - runs one normal staged sub-campaign per `scale_x` value, scopes checkpoints
    by config, selects the config by train exact-CV `best.refit_cv_rmse`, and
    fits the final reusable candidate with the selected `scale_x`
  - reports `model_config_grid`, `model_config_summaries`,
    `selected_model_config_id`, `selected_model_config`, selected `scale_x`, and
    cost counters aggregated across evaluated configs
  - `run_aom_staged_real_cohort.py` now accepts `--scale-x-grid false,true` and
    writes `scale_x`, `scale_x_values`, `selected_model_config_id`
  - compact 10-dataset one-GPU run selected `scale_x=True` on 8/10 rows and
    kept all PLS screen/refit CV fits on CUDA (`screen=1200`, `refit=280`, host
    `0`)
  - oracle summary for that run: AOM-PLS paired median ratio `1.03079` with 1
    target win, AOM-Ridge paired median ratio `1.05918` with 0 target wins,
    TabPFN paired median ratio `1.05956` with 4 target wins
  - follow-up `compact_wide` scale-grid run with `--max-features 1200` produced
    8 OK rows and 2 property-skipped rows; against the compact scale-grid
    baseline on the 8 paired rows it had 2 wins, 1 loss and 5 ties with median
    paired ratio `1.0` and mean ratio `0.994179`; it paid more work
    (`screen CUDA PLS CV fits=2880`, host `0`) for marginal score movement
- Strict single-chain AOM Ridge-PLS selector:
  - `n4m.aom_chain_ridge_pls(...)`
  - `NativeAOMChainRidgePLSRegressor`
  - catalog method `aom_pop.aom_chain_ridge_pls`
  - exposed through `n4m`, `n4m.sklearn`, `n4m.aom` and `n4m.moment`
  - applies strict-linear AOM chains sequentially, scores
    `(chain, n_components, ridge_lambda)` by train-only CV, final-fits through
    native `ridge_pls`, and folds selected-chain coefficients back to raw
    `input_coefficients` plus `intercept`
  - keeps the full candidate score table but tracks only the current best OOF
    prediction buffer, avoiding an `n_candidates x n_samples x n_targets`
    allocation
  - has docs and `bench_aom_chain_ridge_pls_timing.py`
  - has one-GPU CUDA-build smoke artifact
    `benchmarks/cross_binding/aom_chain_ridge_pls_timing_cuda_smoke.csv`
  - intentionally excludes SNV, MSC, EMSC, OSC, row-reference-dependent
    preprocessing, nonlinear lifts, kernels, trees, TabPFN residuals and
    dataset/source routing
- Direct PCR head:
  - C ABI `n4m_pcr_fit`
  - Python `n4m.pcr`
  - Facade `n4m.moment.pcr`
  - sklearn wrapper `NativePCRRegressor`
  - catalog method `models.pls.pcr`
  - docs and timing smoke
- Moment-only OOF stack:
  - `n4m.moment_stack(...)`
  - `NativeMomentStackRegressor`
  - catalog method `models.ensembles.moment_stack`
  - uses train-only OOF predictions from Ridge / PLS sweep / PCR / continuum / ECR / CPPLS
  - fitted diagnostics expose OOF/final base route counters, including PLS CUDA device vs host CV fits
  - no nonlinear lift, no transformed-spectrum stacking, no dataset-name routing
- Candidate preprocessing impact audit:
  - `n4m.aom_candidate_preprocessing_impact`
  - exposed through `n4m`, `n4m.aom`, and `n4m.moment`
  - groups scored candidate reports by stage, operator, option, chain position and head/stage option
  - reports best/mean/median score, rank stats and improvement vs identity baseline when present
- Staged strict-chain cartesian campaign:
  - `n4m.aom_staged_chain_campaign(...)`
  - `n4m.NativeAOMStagedChainCampaignRegressor`
  - catalog method `aom_pop.aom_staged_chain_campaign`
  - exposed through `n4m`, `n4m.aom`, and `n4m.moment`
  - runs compact/wide/lab or custom strict-linear score-only stages
  - merges global + per-head retained candidates, exact-CV refits the union once
  - attaches preprocessing-impact, screen-vs-refit rank diagnostics and optional audit-only holdout scoring
  - supports per-stage checkpoint/resume via `checkpoint_dir`, `resume`, `max_chunks_per_run`
  - the sklearn estimator intentionally has no `X_audit` / `y_audit` inputs and fits the selected train-CV refit row final-only
  - no nonlinear lifts, no dataset-name routing, production selection remains train-CV only
- Existing broader surfaces in this branch:
  - `n4m.aom` and `n4m.moment` facades
  - strict-chain grid builder and streaming iterator
  - AOM chain screen/refit campaign helpers
  - fixed selected-candidate reuse
  - route summaries, rank diagnostics, candidate report save/load
  - CPU/CUDA backend recommendation knobs, including `backend_min_cuda_product`
  - timing smoke scripts for direct heads, moment stack, moment sweep, AOM sweep/screen/refit and CUDA facade smoke

## Validation just run

From `/home/delete/nirs4all/nirs4all-methods`:

```bash
PYTHONPATH=bindings/python/src /home/delete/.venv/bin/python -m py_compile \
  bindings/python/src/n4m/python.py \
  bindings/python/src/n4m/sklearn/native_sweeps.py \
  bindings/python/src/n4m/__init__.py \
  bindings/python/src/n4m/sklearn/__init__.py \
  bindings/python/src/n4m/aom/__init__.py \
  bindings/python/src/n4m/moment/__init__.py \
  bindings/python/tests/test_moment_model_wrappers.py \
  bindings/python/tests/test_aom_moment_cuda_smoke_artifacts.py \
  benchmarks/cross_binding/bench_aom_chain_ridge_pls_timing.py
```

```bash
PYTHONPATH=bindings/python/src N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python -m pytest \
  bindings/python/tests/test_aom_benchmark_tools.py \
  bindings/python/tests/test_catalog_python_bindings.py \
  bindings/python/tests/test_aom_moment_cuda_smoke_artifacts.py \
  bindings/python/tests/test_aom_moment_facade.py \
  bindings/python/tests/test_moment_model_wrappers.py \
  bindings/python/tests/test_aom_staged_campaign.py -q
```

Result after `aom_chain_ridge_pls`: `141 passed`.

After adding staged `scale_x_values`, the same targeted suite is `143 passed`.

After adding `NativeAOMSavgolFocusRegressor`, the focused staged campaign test
file is `14 passed`.

After adding `NativeAOMStrictFamilyLiteRegressor`:

- `bindings/python/tests/test_aom_staged_campaign.py`: `15 passed`
- `bindings/python/tests/test_aom_moment_facade.py`: `13 passed`
- `bindings/python/tests/test_aom_moment_facade.py` now also has a generic
  guard for every `catalog_role="preset_sklearn_wrapper"` row: presets must
  advertise a `preset_plan`, wrap a catalog binding, resolve to the sklearn
  class object, and hide `plan`, `stages`, `families` and `templates` from
  their config surface.
- `bindings/python/tests/test_aom_moment_facade.py` also guards that every
  AOM/moment facade entry is the exact top-level `n4m` object, and that every
  relevant catalogued `n4m.python` binding under `aom_pop`, `utilities`,
  direct PLS/regularized/specialized heads and `moment_stack` is exposed by at
  least one of `n4m.aom` / `n4m.moment`.
- one-GPU CUDA facade smoke regenerated
  `benchmarks/cross_binding/aom_moment_cuda_facade_smoke.json` with
  `strict_family_lite_estimator`: screen CUDA `36`, host `0`; refit CUDA `4`,
  host `0`
- `bindings/python/tests/test_aom_moment_cuda_smoke_artifacts.py -q -k facade`:
  `1 passed, 20 deselected`
- targeted benchmark-tool/catalog/CUDA-artifact/facade/wrapper/staged pytest
  set: `152 passed`
- `catalog/scripts/split_legacy_methods.py --check`: PASS, `208` per-method
  files up to date
- `catalog/scripts/validate.py --check-references`: PASS, `208/208`
  production methods covered
- `catalog/scripts/validate.py --strict-abi`: PASS, ABI coverage `701/701`

After adding CUDA facade coverage for the preset:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/aom_moment_cuda_facade_smoke.py \
  --cuda-lib build/cuda-on/cpp/src/libn4m.so \
  --output benchmarks/cross_binding/aom_moment_cuda_facade_smoke.json \
  --cuda-visible-devices 0 --n-samples 80 --n-features 1024 --cv 4
```

Result: JSON regenerated with `savgol_focus_estimator` device route counters:
screen CUDA `64`, host `0`; refit CUDA `8`, host `0`.

```bash
PYTHONPATH=bindings/python/src N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python -m pytest \
  bindings/python/tests/test_aom_moment_cuda_smoke_artifacts.py -q -k facade
```

Result: `1 passed, 20 deselected`.

After adding the focused strict-family plans:

```bash
PYTHONPATH=bindings/python/src /home/delete/.venv/bin/python -m py_compile \
  bindings/python/src/n4m/python.py \
  bindings/python/tests/test_aom_staged_campaign.py \
  benchmarks/cross_binding/run_aom_staged_real_cohort.py
```

```bash
PYTHONPATH=bindings/python/src N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python -m pytest \
  bindings/python/tests/test_aom_staged_campaign.py -q
```

Result: `13 passed`.

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/run_aom_staged_real_cohort.py \
  --output /tmp/aom_staged_strict_family_focus_cli_smoke.csv \
  --limit 1 --plan strict_family_focus --cv 3 --max-chains 2 \
  --chain-chunk-size 2 --top-k 4 --refit-top-k 3 \
  --refit-per-head-top-k 1 --heads ridge,pls --components 1 \
  --ridge-lambdas 0.1 \
  --checkpoint-dir /tmp/aom_staged_strict_family_focus_cli_smoke_ckpt \
  --no-resume --cuda-pls-parallel-folds \
  --cuda-pls-min-device-features 1 --backend-min-cuda-product 1
```

Result: one OK row,
`BEEFMARBLING/Beef_Marbling_RandomSplit`, `plan=strict_family_focus`,
`selected_campaign_stage=strict_combinations`, `screen_complete=True`,
`selection_uses_test_set=False`, screen PLS CUDA CV fits `54` / host `0`,
refit PLS CUDA CV fits `3` / host `0`.

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/run_aom_staged_real_cohort.py \
  --output /tmp/aom_staged_real_cohort_savgol_focus10_scalex_grid_p1200_cuda.csv \
  --limit 10 --plan savgol_focus --cv 5 --max-chains 6 \
  --chain-chunk-size 6 --top-k 10 --refit-top-k 8 \
  --refit-per-head-top-k 2 --heads ridge,pls --components 1,2 \
  --ridge-lambdas 0.1,1.0,10.0 \
  --checkpoint-dir /tmp/aom_staged_real_cohort_savgol_focus10_scalex_grid_p1200_ckpt \
  --scale-x-grid false,true --cuda-pls-parallel-folds \
  --cuda-pls-min-device-features 1 --backend-min-cuda-product 1 \
  --max-features 1200
```

Result: 8 OK rows, 2 property-skipped rows. Variant comparison versus compact
scale-grid on the 8 paired rows: 5 wins, 2 losses, 1 tie, median paired ratio
`0.995259`, mean ratio `0.996638`. Oracle summary: AOM-PLS median ratio
`1.03051` with 1 target win; AOM-Ridge median ratio `1.07519` with 1 target
win; TabPFN median ratio `0.971371` with 4 target wins. Output artifacts:
`/tmp/aom_staged_savgol_focus_p1200_oracle_summary.md` and
`/tmp/aom_staged_variant_savgol_focus_p1200.md`.

```bash
PYTHONPATH=bindings/python/src /home/delete/.venv/bin/python catalog/scripts/split_legacy_methods.py
PYTHONPATH=bindings/python/src /home/delete/.venv/bin/python catalog/scripts/validate.py --check-references
PYTHONPATH=bindings/python/src /home/delete/.venv/bin/python catalog/scripts/validate.py --strict-abi
```

Result: split wrote `208` per-method files; reference validation PASS
`208/208`; strict ABI validation PASS `701/701`.

```bash
PYTHONPATH=bindings/python/src /home/delete/.venv/bin/python catalog/scripts/validate.py --check-references
```

Result after `aom_chain_ridge_pls`: PASS, `208/208` production methods covered.

```bash
PYTHONPATH=bindings/python/src /home/delete/.venv/bin/python catalog/scripts/validate.py --strict-abi
```

Result after `aom_chain_ridge_pls`: PASS, ABI coverage `701/701`.

```bash
PYTHONPATH=bindings/python/src /home/delete/.venv/bin/python catalog/scripts/split_legacy_methods.py --check
```

Result: PASS, `208` per-method files up to date.

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/bench_aom_chain_ridge_pls_timing.py \
  --output benchmarks/cross_binding/aom_chain_ridge_pls_timing_cuda_smoke.csv \
  --repeats 1 --mode both
```

Result: 6 rows. `library_path=build/cuda-on/cpp/src/libn4m.so`,
`selection_mode=chain_ridge_pls`, `ridge_pls_backend=native`,
`prediction_replay_max_abs_error=0.0`.

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/run_aom_staged_real_cohort.py \
  --output /tmp/aom_staged_real_cohort_compact10_scalex_grid_cuda.csv \
  --limit 10 --plan compact --max-chains 12 --top-k 12 --refit-top-k 6 \
  --refit-per-head-top-k 2 \
  --checkpoint-dir /tmp/aom_staged_real_cohort_compact10_scalex_grid_ckpt \
  --scale-x-grid false,true \
  --cuda-pls-parallel-folds --cuda-pls-min-device-features 1 \
  --backend-min-cuda-product 1
```

Result: 10/10 rows OK, selected `scale_x=True` on 8/10 rows, screen CUDA PLS CV
fits `1200` and host fits `0`, refit CUDA PLS CV fits `280` and host fits `0`.

```bash
PYTHONPATH=bindings/python/src /home/delete/.venv/bin/python \
  benchmarks/cross_binding/compare_aom_staged_to_oracles.py \
  --target /tmp/aom_staged_real_cohort_compact10_scalex_grid_cuda.csv \
  --target-label n4m_staged_compact_scalex_grid_cuda \
  --output /tmp/aom_staged_compact10_scalex_grid_cuda_oracle_comparison.csv \
  --summary-output /tmp/aom_staged_compact10_scalex_grid_cuda_oracle_summary.md
```

Summary: AOM-PLS paired median ratio `1.03079` with 1 target win, AOM-Ridge
paired median ratio `1.05918` with 0 target wins, TabPFN paired median ratio
`1.05956` with 4 target wins.

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/run_aom_staged_real_cohort.py \
  --output /tmp/aom_staged_real_cohort_compact_wide10_scalex_grid_p1200_cuda.csv \
  --limit 10 --plan compact_wide --max-chains 24 --chain-chunk-size 8 \
  --top-k 16 --refit-top-k 8 --refit-per-head-top-k 3 \
  --checkpoint-dir /tmp/aom_staged_real_cohort_compact_wide10_scalex_grid_p1200_ckpt \
  --scale-x-grid false,true \
  --cuda-pls-parallel-folds --cuda-pls-min-device-features 1 \
  --backend-min-cuda-product 1 --max-features 1200
```

Result: 8 OK, 2 skipped by feature-count property. Against the compact
scale-grid baseline on common rows: 2 wins, 1 loss, 5 ties, median ratio `1.0`,
mean ratio `0.994179`. Oracle summary:
AOM-PLS median ratio `1.03051`, AOM-Ridge `1.08068`, TabPFN `0.968453`.

```bash
PYTHONPATH=bindings/python/src /home/delete/.venv/bin/python \
  benchmarks/cross_binding/compare_aom_staged_variants.py \
  --input compact=/tmp/aom_staged_real_cohort_compact10_scalex_grid_cuda.csv \
  --input compact_wide_p1200=/tmp/aom_staged_real_cohort_compact_wide10_scalex_grid_p1200_cuda.csv \
  --baseline-label compact \
  --output /tmp/aom_staged_variant_compact_vs_compact_wide_p1200.csv \
  --summary-output /tmp/aom_staged_variant_compact_vs_compact_wide_p1200.md
```

```bash
git diff --check
```

Result: exit 0; only the known CRLF warnings on existing CSV artifacts.

```bash
PYTHONPATH=bindings/python/src /home/delete/.venv/bin/python -m py_compile \
  bindings/python/src/n4m/python.py \
  bindings/python/src/n4m/__init__.py \
  bindings/python/src/n4m/aom/__init__.py \
  bindings/python/src/n4m/moment/__init__.py \
  bindings/python/src/n4m/sklearn/native_sweeps.py \
  bindings/python/src/n4m/sklearn/__init__.py \
  bindings/python/tests/test_aom_staged_campaign.py \
  bindings/python/tests/test_moment_model_wrappers.py \
  bindings/python/tests/test_aom_moment_facade.py \
  benchmarks/cross_binding/bench_moment_stack_timing.py \
  benchmarks/cross_binding/bench_direct_moment_heads_timing.py
```

```bash
PYTHONPATH=bindings/python/src N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python -m pytest \
  bindings/python/tests/test_aom_moment_facade.py \
  bindings/python/tests/test_moment_model_wrappers.py \
  bindings/python/tests/test_aom_staged_campaign.py -q
```

Result after staged campaign: `70 passed`.

```bash
PYTHONPATH=bindings/python/src /home/delete/.venv/bin/python catalog/scripts/validate.py --strict-abi
```

Result after adding `aom_pop.aom_staged_chain_campaign`: `PASS`, 201 methods,
ABI coverage `701/701`.

```bash
PYTHONPATH=bindings/python/src /home/delete/.venv/bin/python catalog/scripts/validate.py --check-references
```

Result: `PASS`, reference coverage `137 nirs4all-donor + 61 registry + 3
paper_only = 201/201 production methods`.

The AOM/moment facade guard now also verifies that advertised
`config_options` do not expose dataset/source/database/metadata routing knobs.
The catalog Python-binding guard verifies that every per-method
`bindings.python` declaration resolves to a real export. The targeted
catalog/facade/wrapper/staged pytest set is now `73 passed`.

`catalog/scripts/validate.py` now also checks non-null
`bench.registry_entry` paths, so catalogued AOM/moment Python-backed methods
cannot advertise missing timing scripts.

`bindings/python/tests/test_aom_moment_cuda_smoke_artifacts.py` now checks the
committed one-GPU CUDA readiness artifacts for the facade smoke, staged
campaign, moment stack PLS base, Ridge blender wrapper and operator PLS stack
wrapper. The targeted catalog/CUDA-artifact/facade/wrapper/staged pytest set is
now `78 passed`.

The same CUDA-artifact guard now also covers the global AOM profile sweep,
native AOM-PLS / POP-PLS reusable coefficient replay, robust-HPO native CUDA
build smokes, and the PLS/mixed/Ridge screen-refit CUDA build artifacts.
The regenerated facade JSON reports `aom_profile_sweep` with
`n_pls_moment_cuda_device_cv_fits=48` and host PLS CV fits at `0`; AOM-PLS and
POP-PLS sklearn replay errors are both below `1e-10`.

`docs/methods/index.md` now links the AOM/POP and staged campaign method docs
advertised by `n4m.aom` / `n4m.moment`; the facade test guards that link
coverage. After this doc-index guard, the targeted
catalog/CUDA-artifact/facade/wrapper/staged pytest set is `80 passed`.

`bindings/python/tests/test_aom_benchmark_tools.py` now guards the two offline
benchmark helpers: the oracle comparator must filter true AOM-PLS/AOM-Ridge and
TabPFN Raw/Opt rows separately, and the real-cohort runner must report the
train-CV-selected audit row (`eval.best_cv`) rather than the test-best retained
candidate (`best_eval`). With this guard included, the targeted
benchmark-tool/catalog/CUDA-artifact/facade/wrapper/staged pytest set is
`87 passed`.

```bash
git diff --check
```

Result: pass.

```bash
PYTHONPATH=bindings/python/src /home/delete/.venv/bin/python - <<'PY'
import csv
row = next(csv.DictReader(open(
    "benchmarks/cross_binding/aom_staged_chain_campaign_timing_cuda_smoke.csv"
)))
assert row["selection_uses_test_set"] == "False"
assert row["screen_complete"] == "True"
assert int(row["n_pls_moment_cuda_device_cv_fits"]) > 0
assert int(row["n_pls_moment_host_cv_fits"]) == 0
PY
```

Result: staged CUDA smoke CSV counters pass.

Earlier in the same pass, `catalog/scripts/validate.py --check-references` also passed after adding `models.ensembles.moment_stack` as a nirs4all-donor method.

Moment stack CUDA smoke:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/bench_moment_stack_timing.py \
  --output benchmarks/cross_binding/moment_stack_timing_cuda_smoke.csv \
  --repeats 1 --shapes 80x1024 --base-models pls --cv 4 --inner-cv 4 \
  --n-components 1 --cuda-pls-min-device-features 1 \
  --cuda-pls-parallel-folds
```

Observed readiness counters: `n_base_oof_pls_moment_cuda_device_cv_fits=16`,
`n_base_oof_pls_moment_host_cv_fits=0`,
`n_base_final_pls_moment_cuda_device_cv_fits=4`,
`n_base_final_pls_moment_host_cv_fits=0`.

CSV assertion:

```bash
PYTHONPATH=bindings/python/src /home/delete/.venv/bin/python - <<'PY'
import csv
row = next(csv.DictReader(open(
    "benchmarks/cross_binding/moment_stack_timing_cuda_smoke.csv"
)))
assert int(row["n_base_oof_pls_moment_cuda_device_cv_fits"]) == 16
assert int(row["n_base_oof_pls_moment_host_cv_fits"]) == 0
assert int(row["n_base_final_pls_moment_cuda_device_cv_fits"]) == 4
assert int(row["n_base_final_pls_moment_host_cv_fits"]) == 0
PY
```

## Important files

- `bindings/python/src/n4m/python.py`
- `bindings/python/src/n4m/aom/__init__.py`
- `bindings/python/src/n4m/moment/__init__.py`
- `bindings/python/src/n4m/sklearn/native_sweeps.py`
- `bindings/python/tests/test_aom_moment_facade.py`
- `bindings/python/tests/test_aom_benchmark_tools.py`
- `bindings/python/tests/test_moment_model_wrappers.py`
- `bindings/python/tests/test_aom_staged_campaign.py`
- `bindings/python/tests/test_aom_moment_cuda_smoke_artifacts.py`
- `benchmarks/cross_binding/bench_aom_chain_ridge_pls_timing.py`
- `benchmarks/cross_binding/aom_chain_ridge_pls_timing_cuda_smoke.csv`
- `benchmarks/cross_binding/bench_aom_staged_chain_campaign_timing.py`
- `benchmarks/cross_binding/run_aom_staged_real_cohort.py`
- `benchmarks/cross_binding/compare_aom_staged_to_oracles.py`
- `cpp/src/c_api/c_api_method_result.cpp`
- `cpp/include/n4m/pls.h`
- `catalog/methods/models.pls.pcr.yaml`
- `catalog/methods/models.ensembles.moment_stack.yaml`
- `catalog/methods/aom_pop.aom_chain_ridge_pls.yaml`
- `docs/architecture/aom_moment_coverage_matrix.md`
- `docs/architecture/aom_moment_worklog.md`
- `docs/methods/aom_chain_ridge_pls.md`
- `docs/methods/aom_chain_sweep_run.md`
- `docs/methods/aom_staged_chain_campaign.md`
- `docs/methods/pcr.md`
- `docs/methods/moment_stack.md`

## Remaining work

The remaining gaps are engine/performance work, not simple method wiring:

1. Fused/batched IKPLS grinder.
   - Current code has an exact moment route for compatible single-target PLS1 and optional CUDA fold scheduling/many-batched knobs.
   - It is still not a fully fused many-chain/many-fold/many-candidate IKPLS engine.

2. Full arbitrary-chain moment screen.
   - Current strict-linear coverage includes the practical local/banded/structured operators used in the AOM grids.
   - Non-compatible regimes still fall back or materialize.

3. CUDA fused sweep kernels.
   - CUDA builds work and there are device routes/counters.
   - There is no final grouped fused CUDA kernel suite for the full cartesian.

4. Broad benchmark/recall campaigns.
   - Need a controlled campaign against robust AOM oracle / AOM Ridge oracle / AOM PLS oracle / TabPFN baselines.
   - The new `aom_candidate_preprocessing_impact` helper should be used to analyze which preprocessing families/options justify more budget.

## Constraints to keep

- Do not add hors-moment nonlinear lifts in this pass.
- Do not select by dataset name, source name, dataset id or equivalent identity.
- Do not use test-set selection for production model choice. Test ranking can be an offline audit only.
- Keep `libn4m` as the single C++/binding infrastructure rather than extracting AOM into a separate runtime.

## Suggested next step

Stop expanding features and run one focused benchmark campaign:

1. Pick 10 datasets.
2. Run compact/wide/lab strict-chain screens with Ridge, PLS and mixed heads.
3. Retain top global + per-head candidates.
4. Exact-refit retained rows.
5. Use `aom_candidate_preprocessing_impact` and `aom_candidate_rank_diagnostics` to decide whether more cartesian budget is justified.
6. Compare against the real robust AOM oracle and TabPFN baselines from `n4a-lab` / `n4a-paper` / `n4a-aom`.

The staged runner, synthetic timing smoke, real-cohort runner, and offline
oracle comparator are now wired. The comparator defaults to the real local
artifacts:

- AOM-PLS oracle:
  `/home/delete/nirs4all/nirs4all-aom/benchmarks/runs/scenarios/paper_aom_aompls_seeds012/results.csv`
- AOM-Ridge oracle:
  `/home/delete/nirs4all/nirs4all-aom/benchmarks/runs/ridge/all54_headline/results.csv`
- TabPFN oracle:
  `/home/delete/nirs4all/nirs4all-aom/benchmarks/pls/cohort_regression.csv`
  plus fallback `/home/delete/nirs4all/nirs4all-lab/benchmark/results/1_master_results.csv`

The comparator filters plain PLS/Ridge baselines out of the AOM-PLS and
AOM-Ridge oracles, and uses the best available TabPFN Raw/Opt reference per
dataset.

To produce target rows, run for example:

```bash
PYTHONPATH=bindings/python/src N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/run_aom_staged_real_cohort.py \
  --output benchmarks/cross_binding/aom_staged_real_cohort_results.csv \
  --limit 10 --plan compact --max-chains 12 --top-k 12 --refit-top-k 6
```

Without `--resume`, the runner replaces the output CSV. With `--resume`, it
keeps existing rows and skips dataset rows already marked `ok`.

Then compare:

```bash
/home/delete/.venv/bin/python benchmarks/cross_binding/compare_aom_staged_to_oracles.py \
  --target benchmarks/cross_binding/aom_staged_real_cohort_results.csv \
  --target-score-column rmsep
```

The runner's `rmsep` is the held-out score of the candidate selected by train
CV. The test-best retained candidate is recorded separately as
`audit_oracle_rmse` and must stay audit-only.

Staged CUDA smoke:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/bench_aom_staged_chain_campaign_timing.py \
  --output benchmarks/cross_binding/aom_staged_chain_campaign_timing_cuda_smoke.csv \
  --repeats 1 --plans compact --n-samples 96 --n-features 128 --cv 3 \
  --heads pls --components 1 --ridge-lambdas 0.1 --max-chains 4 \
  --chain-chunk-size 2 --top-k 4 --refit-top-k 3 \
  --refit-per-head-top-k 1 --moment-policy auto \
  --cuda-pls-min-device-features 1 --cuda-pls-parallel-folds
```

Observed readiness counters: `selection_uses_test_set=False`,
`screen_complete=True`, `n_pls_moment_cuda_device_cv_fits=9`,
`n_pls_moment_host_cv_fits=0`,
`n_pls_moment_cuda_parallel_fold_jobs=9`.

The broader CUDA facade smoke also covers the staged sklearn estimator:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/aom_moment_cuda_facade_smoke.py \
  --cuda-lib build/cuda-on/cpp/src/libn4m.so \
  --output benchmarks/cross_binding/aom_moment_cuda_facade_smoke.json \
  --cuda-visible-devices 0 --n-samples 80 --n-features 1024 --cv 4
```

Observed staged estimator counters in the JSON:
`n_pls_moment_cuda_device_cv_fits=8`,
`n_pls_moment_host_cv_fits=0`,
`n_pls_moment_cuda_parallel_fold_jobs=8`,
`selection_uses_test_set=False`.

Native AOM diversity wrapper CUDA smokes:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/bench_aom_ridge_blender_timing.py \
  --output benchmarks/cross_binding/aom_ridge_blender_timing_cuda_smoke.csv \
  --repeats 1 --profile compact --cv 5 --mode both

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/bench_aom_operator_pls_stack_timing.py \
  --output benchmarks/cross_binding/aom_operator_pls_stack_timing_cuda_smoke.csv \
  --repeats 1 --profile compact --cv 4 --mode both
```

Both CSVs now include native function and sklearn replay-wrapper rows:
`native_aom_ridge_blender_sklearn` and
`native_aom_operator_pls_stack_sklearn`. The wrapper rows report
`prediction_replay_max_abs_error <= 1e-10`, proving `predict(X)` replays the
native folded model state through the CUDA build path.

Direct moment-head CUDA smoke:

- A stale `build/cuda-on/cpp/src/libn4m.so` initially missed the
  `n4m_pcr_fit` symbol even though the CPU dev-release build exposed the direct
  head ABI. Rebuilt the CUDA preset with
  `/home/delete/.venv/bin/cmake --build --preset cuda-on --parallel`.
- Verified the rebuilt CUDA shared library exports `n4m_ridge_fit`,
  `n4m_pcr_fit`, `n4m_cppls_fit`, `n4m_continuum_regression_fit` and
  `n4m_ecr_fit`; direct `n4m.pls` is Python-backed over `n4m_sweep_run`.
- Generated
  `benchmarks/cross_binding/direct_moment_heads_timing_cuda_smoke.csv` with one
  GPU. It contains Ridge, PLS, PCR, CPPLS, continuum regression and ECR across
  three shapes, with both `native_function` and `sklearn_fit_predict` rows on
  `build/cuda-on/cpp/src/libn4m.so`.
- Added a release-readiness guard asserting the exact method set, three shapes
  per method, both backends per method/shape, finite RMSE, CUDA build path and
  `replay_max_abs_error <= 1e-10`.
- After this direct-head guard, the targeted
  benchmark-tool/catalog/CUDA-artifact/facade/wrapper/staged pytest set reports
  `88 passed`.

Sweep/selector CUDA artifact refresh:

- Regenerated `moment_sweep_timing_cuda_smoke.csv`,
  `aom_sweep_timing_cuda_smoke.csv` and `aom_selector_timing_cuda_smoke.csv`
  against `build/cuda-on/cpp/src/libn4m.so` after the older committed artifacts
  were still on ABI `1.18.0`.
- The refreshed artifacts report ABI `1.21.0` and a `build/cuda-on` library
  path.
- Added release-readiness guards proving:
  - moment sweep exact PLS CV rows have `cuda_pls_parallel_folds=True`,
    `cuda_pls_min_device_features=1`, host CV fits at zero, and device CV fits
    equal to total PLS moment CV fits;
  - AOM sweep exact PLS CV rows across compact/global, custom chain,
    Whittaker, FCK, Gaussian, PLS exact/proxy and Ridge rows have host CV fits at
    zero and device CV fits equal to total PLS moment CV fits;
  - AOM-PLS and POP-PLS selector function/sklearn rows replay with
    `replay_max_abs <= 1e-10`.
- After these sweep/selector guards, the targeted
  benchmark-tool/catalog/CUDA-artifact/facade/wrapper/staged pytest set reports
  `91 passed`.

Real 10-dataset staged CUDA calibration:

- Added staged campaign route counters to `aom_staged_chain_campaign` reports
  and to `run_aom_staged_real_cohort.py` output rows:
  `n_screen_pls_moment_*`, `n_refit_pls_moment_*`, plus the CUDA knob columns.
- Ran a one-GPU compact real-cohort calibration:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/run_aom_staged_real_cohort.py \
  --output /tmp/n4m_aom_staged_real_cohort_10_cuda.csv \
  --limit 10 --plan compact --cv 4 --max-chains 12 \
  --chain-chunk-size 6 --top-k 12 --refit-top-k 6 \
  --refit-per-head-top-k 2 --heads ridge,pls --components 1,2 \
  --ridge-lambdas 0.1,1.0,10.0 --moment-policy auto \
  --cuda-pls-min-device-features 1 --cuda-pls-parallel-folds \
  --backend-min-cuda-product 1
```

- Result: `10/10` rows OK, `screen_complete=True`,
  `selection_uses_test_set=False`, ABI `1.21.0`,
  `library_path=build/cuda-on/cpp/src/libn4m.so`.
- GPU route evidence: screen PLS moment CV fits `480`, screen host fits `0`,
  screen CUDA device fits `480`; refit PLS moment CV fits `140`, refit host fits
  `0`, refit CUDA device fits `140`.
- Fit time: total `81.69 s`, median `3.43 s`; one BERRY dataset dominated at
  about `51.17 s`.
- Oracle comparison:

```bash
/home/delete/.venv/bin/python benchmarks/cross_binding/compare_aom_staged_to_oracles.py \
  --target /tmp/n4m_aom_staged_real_cohort_10_cuda.csv \
  --target-score-column rmsep \
  --output /tmp/n4m_aom_staged_oracle_comparison_10_cuda.csv \
  --summary-output /tmp/n4m_aom_staged_oracle_comparison_10_cuda.md
```

- Paired results: vs AOM-PLS oracle `0/9` target wins, median ratio `1.21015`;
  vs AOM-Ridge oracle `0/10` target wins, median ratio `1.24282`; vs TabPFN
  oracle `1/10` target wins, median ratio `1.33059`.
- Interpretation: this proves the compact staged workflow runs train-only
  selection on real data with PLS CV on GPU, but the tiny `max_chains=12` budget
  is still far behind the robust AOM oracles. It is calibration evidence, not a
  ceiling estimate for larger cartesian/preprocessing budgets.
- After the route-counter runner guard, the targeted
  benchmark-tool/catalog/CUDA-artifact/facade/wrapper/staged pytest set reports
  `92 passed`.

Property-filtered compact+wide calibration:

- Added optional real-cohort runner filters by measured dataset properties:
  `--max-train-samples`, `--max-features`,
  `--max-train-feature-product`. Rows outside the budget are written as
  `status=skipped` with `error_message=property_filter:...`; no model selection
  ever uses dataset/source/name identity.
- Ran a one-GPU incremental campaign:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/run_aom_staged_real_cohort.py \
  --output /tmp/n4m_aom_staged_real_cohort_10_cuda_compact_wide_p1200.csv \
  --limit 10 --plan compact_wide --cv 4 --max-chains 24 \
  --chain-chunk-size 8 --top-k 16 --refit-top-k 8 \
  --refit-per-head-top-k 3 --heads ridge,pls --components 1,2 \
  --ridge-lambdas 0.1,1.0,10.0 --moment-policy auto \
  --cuda-pls-min-device-features 1 --cuda-pls-parallel-folds \
  --backend-min-cuda-product 1 --max-features 1200
```

- Result: `8` OK rows and `2` skipped rows, both skipped only because
  `n_features>1200`. OK rows used ABI `1.21.0` and
  `build/cuda-on/cpp/src/libn4m.so`.
- GPU route evidence on OK rows: screen PLS moment CV fits `1152`, screen host
  fits `0`, screen CUDA device fits `1152`; refit PLS moment CV fits `160`,
  refit host fits `0`, refit CUDA device fits `160`.
- Timing: total fit time on OK rows `45.06 s`, median `5.31 s`.
- Compared with the previous compact run on the 8 common OK datasets:
  compact+wide improved `3/8`, was identical on `4/8`, and degraded slightly on
  `1/8`; median ratio versus compact was `1.0`.
- Oracle comparison:
  - AOM-PLS oracle: paired `7`, target wins `0`, median ratio `1.17183`;
  - AOM-Ridge oracle: paired `8`, target wins `0`, median ratio `1.26483`;
  - TabPFN oracle: paired `8`, target wins `1`, median ratio `1.33059`.
- Interpretation: adding a wide stage at this small budget improves a few rows
  but still does not close the robust AOM oracle gap. The property filters give
  a clean way to iterate budget without dataset-name selection.
- After the property-filter guard, the targeted
  benchmark-tool/catalog/CUDA-artifact/facade/wrapper/staged pytest set reports
  `93 passed`.

Custom staged JSON calibration:

- Added `--stages-json` and `--stages-json-file` to
  `run_aom_staged_real_cohort.py`. The value must be a non-empty JSON list of
  profile strings or stage objects accepted by `n4m.aom_staged_chain_campaign`.
  The runner passes it through as `stages=...` and records a compact
  `stages_json` field in each output row.
- Mini one-GPU proof run:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=bindings/python/src \
N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python benchmarks/cross_binding/run_aom_staged_real_cohort.py \
  --output /tmp/n4m_aom_staged_real_cohort_3_cuda_custom_stages.csv \
  --limit 3 --plan compact \
  --stages-json '[{"name":"compact","profile":"compact","max_chains":8,"top_k":8},{"name":"sg_lab","profile":"lab","families":{"identity":[["identity",[]]],"savgol_smooth":[["savgol_smooth",[5,2]],["savgol_smooth",[9,2]],["savgol_smooth",[15,3]]],"savgol_derivative":[["savgol_derivative",[7,2,1]],["savgol_derivative",[15,3,1]]],"finite_difference":[["finite_difference",[1]]]},"templates":[["identity"],["savgol_smooth"],["savgol_derivative"],["savgol_smooth","finite_difference"]],"max_chains":8,"top_k":8}]' \
  --cv 4 --max-chains 8 --chain-chunk-size 4 --top-k 8 \
  --refit-top-k 6 --refit-per-head-top-k 2 \
  --heads ridge,pls --components 1,2 --ridge-lambdas 0.1,1.0,10.0 \
  --moment-policy auto --cuda-pls-min-device-features 1 \
  --cuda-pls-parallel-folds --backend-min-cuda-product 1 \
  --max-features 1200
```

- Result: `2` OK rows, `1` skipped row (`n_features>1200`), `plan=custom`,
  `stages_json` non-empty, ABI `1.21.0`.
- GPU route evidence on OK rows: screen PLS moment CV fits `128`, screen host
  fits `0`, screen CUDA device fits `128`; refit PLS moment CV fits `32`,
  refit host fits `0`, refit CUDA device fits `32`.
- Oracle comparison for those two rows: target wins `0` vs AOM-PLS, AOM-Ridge
  and TabPFN. This is a functionality proof for custom preprocessing-family
  campaign wiring, not a scoring improvement.
- After the custom-stages JSON guard, the targeted
  benchmark-tool/catalog/CUDA-artifact/facade/wrapper/staged pytest set reports
  `94 passed`.

Staged variant comparison helper:

- Added `benchmarks/cross_binding/compare_aom_staged_variants.py` to compare
  several staged campaign result CSVs before comparing to AOM/TabPFN oracles.
  It is offline only: no fitting, no production selection, and no dataset-name
  routing. Dataset keys are used only to pair evaluation scores against a
  baseline.
- Example:

```bash
python benchmarks/cross_binding/compare_aom_staged_variants.py \
  --input compact=/tmp/n4m_aom_staged_real_cohort_10_cuda.csv \
  --input wide=/tmp/n4m_aom_staged_real_cohort_10_cuda_compact_wide_p1200.csv \
  --baseline-label compact \
  --output /tmp/n4m_aom_staged_variant_summary.csv \
  --summary-output /tmp/n4m_aom_staged_variant_summary.md
```

- The summary groups by campaign config (`plan`, `stages_json`, heads, budget,
  property filters and CUDA knobs) and reports OK/skipped/error counts, median
  score/timing, route-counter totals, and paired win/loss/tie ratios.
- With this helper included, the targeted
  benchmark-tool/catalog/CUDA-artifact/facade/wrapper/staged pytest set reports
  `97 passed`.

Direct reusable PLS head:

- Added `n4m.pls` / `n4m.python.pls`, backed by `n4m_sweep_run` in PLS-only
  mode. It supports fixed `n_components` or an explicit train-CV
  `pls_components` grid and passes through the existing CUDA PLS controls.
- Added `NativePLSRegressor` to `n4m.sklearn`, top-level `n4m`, and
  `n4m.moment`. It predicts from replayable input-space coefficients plus
  intercept and exposes the sweep PLS route counters.
- Wired `catalog/methods/models.pls.pls_fit_simple.yaml` and
  `catalog/methods.yaml` to the new `n4m.python.pls` binding, and updated
  `docs/methods/pls.md`.
- Regenerated
  `benchmarks/cross_binding/direct_moment_heads_timing_cuda_smoke.csv`; it now
  covers Ridge, PLS, PCR, CPPLS, continuum regression and ECR on the CUDA build,
  with function and sklearn replay rows.
- With this direct PLS head included, the targeted
  benchmark-tool/catalog/CUDA-artifact/facade/wrapper/staged pytest set reports
  `98 passed`; strict ABI, reference validation and split-method checks pass.

Linear PLS variants (weighted, robust, ridge-augmented):

- `n4m.weighted_pls`: FFI wired to `n4m_weighted_pls_fit`; exports
  replayable `coefficients`, `predictions`, `x_mean`, `y_mean`;
  `NativeWeightedPLSRegressor` reconstructs the intercept from the native means;
  present in `direct_moment_heads_timing_cuda_smoke.csv` across all 3 shapes
  with native and sklearn rows.
- `n4m.robust_pls`: FFI wired to `n4m_robust_pls_fit`; Huber IRLS over weighted
  PLS fits; Python helper default `max_irls_iter=5`; exports `coefficients`,
  `predictions`, `x_mean`, `y_mean`; `NativeRobustPLSRegressor` replays native
  training predictions; present in the smoke CSV with native and sklearn rows.
- `n4m.ridge_pls`: FFI wired to `n4m_ridge_pls_fit`; ridge-regularised augmented
  SIMPLS; `NativeRidgePLSRegressor` replays native training predictions; present
  in the smoke CSV with native and sklearn rows.
- All three have catalog YAML entries and are part of the `n4m.moment` facade
  inventory.
- `models.pls.kernel` / `kernel_pls` is **not wired** and is explicitly excluded:
  non-linear kernel PLS requires a separate kernel-moment substrate outside the
  strict-linear AOM/operator-moment architecture.
- Tested: coefficient replay was verified at numerical noise level on CPU before
  adding wrappers; all nine direct heads are present in the regenerated 54-row
  CUDA smoke CSV and every row reports
  `surface_status=function_and_sklearn_replay`.
- Validation after this wrapper pass:
  - Combined benchmark/catalog/CUDA-artifact/facade/wrapper/staged pytest set:
    `107 passed`.
  - `catalog/scripts/validate.py --strict-abi`: PASS, 201 methods and 701/701
    exported `n4m_*` symbols covered.
  - `catalog/scripts/validate.py --check-references`: PASS, 201/201 production
    methods covered.
  - `git diff --check`: exit 0, only the known CRLF warnings on existing CSV
    artifacts.

Direct-head timing catalog alignment:

- The nine reusable direct moment heads now all advertise
  `benchmarks/cross_binding/bench_direct_moment_heads_timing.py` as their
  `bench.registry_entry`: Ridge, PLS, PCR, CPPLS, weighted PLS, robust PLS,
  ridge-augmented PLS, continuum regression and ECR.
- Added a catalog guard for that exact registry entry across the direct-head
  per-method YAML files.
- With this guard included, the targeted
  benchmark-tool/catalog/CUDA-artifact/facade/wrapper/staged pytest set reports
  `99 passed`; strict ABI, reference validation and split-method checks pass.

Direct PLS CUDA route proof:

- `bench_direct_moment_heads_timing.py` now accepts
  `--cuda-pls-min-device-features`, `--cuda-pls-parallel-folds` and
  `--cuda-pls-many-batched`, and records PLS route counters in its CSV.
- Regenerated
  `benchmarks/cross_binding/direct_moment_heads_timing_cuda_smoke.csv` with one
  GPU and `--cuda-pls-min-device-features 1 --cuda-pls-parallel-folds`.
  PLS rows now show `n_pls_moment_cv_fits=4`, host CV fits `0`, CUDA device CV
  fits `4` and CUDA fold jobs `4` for all three shapes and both function/wrapper
  backends.
- The CUDA artifact guard now requires that direct PLS rows prove device-CV
  routing, not only CUDA library loading and replay.
- Validation remains `99 passed` for the targeted
  benchmark-tool/catalog/CUDA-artifact/facade/wrapper/staged pytest set; strict
  ABI, reference validation and split-method checks pass.

Moment-stack PLS base reuse:

- `NativeMomentStackRegressor` now uses `NativePLSRegressor` for its `"pls"`
  base instead of the generic `NativeMomentSweepRegressor`. This keeps the same
  underlying `n4m_sweep_run` route but reuses the direct individual PLS method
  exposed to users.
- Wrapper tests now require stack diagnostics to report
  `estimator="NativePLSRegressor"` and `method="pls"` for OOF/final PLS base
  rows.
- Regenerated `benchmarks/cross_binding/moment_stack_timing_cuda_smoke.csv`;
  PLS route counters remain OOF `16/0/16` total/host/device CV fits and final
  `4/0/4`.
- Validation: `test_moment_model_wrappers.py` reports `53 passed`,
  `test_aom_moment_cuda_smoke_artifacts.py` reports `13 passed`, and the
  targeted combined suite remains `99 passed`.

AOM preprocessing timing gap:

- Added `benchmarks/cross_binding/bench_aom_preprocess_timing.py` for the
  currently supported reusable `n4m.aom_preprocess` primitive contract:
  direct single-operator `identity`, degree-1 detrend,
  `savgol_smooth(window=5, poly=2)` and
  `savgol_derivative(window=5, poly=2, deriv=1)`, Norris-Williams,
  finite-difference, Gaussian, Whittaker and FCK in both `soft` and `hard`
  gating modes.
- Wired `aom_pop.aom_preprocessing` to that script as its
  `bench.registry_entry` in both `catalog/methods.yaml` and the split YAML.
- The benchmark asserts exact single-operator replay against
  `operator_outputs`, expected operator kinds `0/7/8/9/10/15/16/17/18`,
  `weight_shape=1x1`, and `weight_sum=1`. This is a timing/API smoke for the
  direct strict-linear single-operator bank; richer strict-chain/model-scoring
  diversity remains in AOM sweep/campaign helpers, and SNV-style banks remain
  outside the moment contract.
- Generated
  `benchmarks/cross_binding/aom_preprocess_timing_cuda_smoke.csv` with
  `CUDA_VISIBLE_DEVICES=0`, `N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so` and
  ABI `1.21.0`; it contains 3 shapes x 2 gating modes x 9 direct operators
  with replay error at numerical-noise level.
- Validation after this pass:
  - Rebuilt both `dev-release` and `cuda-on` CMake presets after switching
    `aom_preprocess` to the shared AOM strict-linear operator engine.
  - Added C ABI coverage in `build/dev-release/cpp/tests/n4m_tests` for the
    9-operator direct strict-linear bank, hard/soft weights and replay of
    identity plus finite-difference operator outputs.
  - `test_aom_moment_cuda_smoke_artifacts.py`: `14 passed`.
  - Combined benchmark/catalog/CUDA-artifact/facade/wrapper/staged pytest set:
    `108 passed`.
  - `build/dev-release/cpp/tests/n4m_tests`: `351 passed, 0 failed`.
  - `catalog/scripts/validate.py --strict-abi`: PASS, 201 methods and 701/701
    exported `n4m_*` symbols covered.
  - `catalog/scripts/validate.py --check-references`: PASS, 201/201 production
    methods covered.
  - `catalog/scripts/split_legacy_methods.py --check`: PASS.
  - `git diff --check`: exit 0, only the known CRLF warnings on existing CSV
    artifacts.

AOM Ridge superblock strict-moment reference:

- Added `n4m.aom_ridge_superblock`, a Python-backed donor-style AOM Ridge
  superblock constrained to strict-linear single-operator AOM views.
- Added `NativeAOMRidgeSuperblockRegressor` and exported it through top-level
  `n4m`, `n4m.sklearn`, `n4m.aom` and `n4m.moment`.
- The method concatenates strict operator outputs from `n4m.aom_preprocess`,
  uses the native `n4m.ridge` binding for fold-local Ridge alpha scoring/fitting,
  applies train-fold centering and optional block RMS scaling to validation
  folds, and folds final superblock coefficients back to original-input
  `input_coefficients` plus `intercept`.
- Promoted the method to catalog entry `aom_pop.ridge_superblock`, added
  `docs/methods/aom_ridge_superblock.md`, wired
  `benchmarks/cross_binding/bench_aom_ridge_superblock_timing.py`, and
  generated `aom_ridge_superblock_timing_cuda_smoke.csv` on `build/cuda-on`
  with function + sklearn rows and `ridge_backend=native`.
- The method is CUDA-build compatible and timed, but still not a fused GPU
  superblock grinder. It
  intentionally excludes donor `branch_global`, MKL/kernel,
  row-reference-dependent preprocessing and nonlinear AOM Ridge modes.
- Validation after this slice:
  - py_compile on touched Python modules/tests: PASS.
  - Focused Ridge-superblock tests plus `test_aom_moment_facade.py`: `15 passed`.
  - Targeted benchmark/catalog/CUDA-artifact/facade/wrapper/staged pytest:
    `113 passed`.
  - `catalog/scripts/validate.py --strict-abi`: PASS, 202 methods and 701/701
    exported `n4m_*` symbols covered.
  - `catalog/scripts/validate.py --check-references`: PASS, 202/202 production
    methods covered.
  - `catalog/scripts/split_legacy_methods.py --check`: PASS, 202 per-method
    files up to date.
  - `git diff --check`: exit 0, only known CRLF warnings on existing CSV
    artifacts.

AOM Ridge global strict-moment selector:

- Added `n4m.aom_ridge_global`, a donor-style strict AOM Ridge global selector
  constrained to single strict-linear operators.
- Added `NativeAOMRidgeGlobalRegressor` and exported it through top-level
  `n4m`, `n4m.sklearn`, `n4m.aom` and `n4m.moment`.
- The method wraps native `aom_chain_sweep_run` in Ridge-only mode, selects one
  operator plus one Ridge alpha by train CV, and returns folded
  `input_coefficients` plus `intercept` for reuse.
- Promoted the method to catalog entry `aom_pop.ridge_global`, added
  `docs/methods/aom_ridge_global.md`, wired
  `benchmarks/cross_binding/bench_aom_ridge_global_timing.py`, and generated
  `aom_ridge_global_timing_cuda_smoke.csv` on `build/cuda-on` with function +
  sklearn rows and `ridge_backend=native_aom_chain_sweep`.
- It intentionally excludes donor `branch_global`, MKL/kernel,
  row-reference-dependent preprocessing and nonlinear AOM Ridge modes.
- Validation after this slice:
  - py_compile on touched Python modules/tests: PASS.
  - Focused Ridge-global tests plus `test_aom_moment_facade.py`: `21 passed`.
  - Targeted benchmark/catalog/CUDA-artifact/facade/wrapper/staged pytest:
    `118 passed`.
  - `catalog/scripts/validate.py --strict-abi`: PASS, 203 methods and 701/701
    exported `n4m_*` symbols covered.
  - `catalog/scripts/validate.py --check-references`: PASS, 203/203 production
    methods covered.
  - `catalog/scripts/split_legacy_methods.py --check`: PASS, 203 per-method
    files up to date.

AOM Ridge active-superblock strict-moment reference:

- Added `n4m.aom_ridge_active_superblock`, a Python-backed donor-style AOM
  Ridge active-superblock constrained to strict-linear single-operator AOM
  views.
- Added `NativeAOMRidgeActiveSuperblockRegressor` and exported it through
  top-level `n4m`, `n4m.sklearn`, `n4m.aom` and `n4m.moment`.
- The active score is defined on native `n4m.aom_preprocess` outputs as
  `||scale_b * Z_b.T @ y_c||_F^2` by default; optional `kta` and `blend`
  modes use the same train-only outputs. This is donor-style, not a hidden
  claim of exact donor-private `op.apply_cov` reproduction.
- Alpha CV screens active operators inside each training fold only; the final
  model screens once on full calibration rows and fits Ridge through native
  `n4m.ridge`. The final active superblock folds back to
  `input_coefficients` plus `intercept`.
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

AOM-PLS superblock strict-moment reference:

- Added `n4m.aom_pls_superblock`, a Python-backed donor-style AOM-PLS
  superblock constrained to strict-linear single-operator AOM views.
- Added `NativeAOMPLSSuperblockRegressor` and exported it through top-level
  `n4m`, `n4m.sklearn`, `n4m.aom` and `n4m.moment`.
- The method concatenates strict operator outputs from `n4m.aom_preprocess`,
  selects the PLS component count by train CV, fits through native `n4m.pls`,
  and folds final superblock coefficients back to original-input
  `input_coefficients` plus `intercept`.
- Promoted the method to catalog entry `aom_pop.aom_pls_superblock`, added
  `docs/methods/aom_pls_superblock.md`, wired
  `benchmarks/cross_binding/bench_aom_pls_superblock_timing.py`, and generated
  `aom_pls_superblock_timing_cuda_smoke.csv` on `build/cuda-on` with function
  + sklearn rows, `pls_backend=native` and CUDA device PLS route counters when
  `--cuda-pls-min-device-features 1` is used.
- Kept donor `soft`, active PLS pruning, MKL/kernel, row-reference-dependent
  preprocessing, nonlinear lifts and dataset/source routing out of scope for
  this slice.

AOM Ridge MKL-light superblock strict-moment reference:

- Added `n4m.aom_ridge_mkl_superblock`, a Python-backed donor-style AOM Ridge
  MKL-light weighted-superblock constrained to strict-linear single-operator
  AOM views.
- Added `NativeAOMRidgeMKLSuperblockRegressor` and exported it through
  top-level `n4m`, `n4m.sklearn`, `n4m.aom` and `n4m.moment`.
- The method learns non-negative KTA block weights on training rows only inside
  every alpha-CV fold, refits weights on the full calibration rows, fits native
  Ridge on the equivalent weighted superblock, and folds final coefficients
  back to original-input `input_coefficients` plus `intercept`.
- Promoted the method to catalog entry `aom_pop.ridge_mkl_superblock`, added
  `docs/methods/aom_ridge_mkl_superblock.md`, wired
  `benchmarks/cross_binding/bench_aom_ridge_mkl_superblock_timing.py`, and
  generated `aom_ridge_mkl_superblock_timing_cuda_smoke.csv` on `build/cuda-on`
  with function + sklearn rows, `mkl_mode=alignment`, `ridge_backend=native`
  and replay error at numerical-noise level.
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

AOM Ridge-PLS superblock strict-moment reference:

- Added `n4m.aom_ridge_pls_superblock`, a Python-backed donor-style AOM
  Ridge-PLS superblock constrained to strict-linear single-operator AOM views.
- Added `NativeAOMRidgePLSSuperblockRegressor` and exported it through
  top-level `n4m`, `n4m.sklearn`, `n4m.aom` and `n4m.moment`.
- The method concatenates strict operator outputs from `n4m.aom_preprocess`,
  selects the PLS component count and Ridge-PLS lambda by train CV, fits
  through native `n4m.ridge_pls`, and folds final superblock coefficients back
  to original-input `input_coefficients` plus `intercept`.
- Promoted the method to catalog entry `aom_pop.aom_ridge_pls_superblock`,
  added `docs/methods/aom_ridge_pls_superblock.md`, wired
  `benchmarks/cross_binding/bench_aom_ridge_pls_superblock_timing.py`, and
  generated `aom_ridge_pls_superblock_timing_cuda_smoke.csv` on `build/cuda-on`
  with function + sklearn rows and `ridge_pls_backend=native`.
- Kept donor `soft`, active PLS pruning, MKL/kernel, row-reference-dependent
  preprocessing, nonlinear lifts and dataset/source routing out of scope for
  this slice.

Rank-audit mismatch summary follow-up (2026-06-06):

- Added a post-hoc mismatch-pattern section to
  `benchmarks/cross_binding/summarize_aom_rank_audit.py`.
- The Markdown summary now aggregates rows where the train-CV winner was not
  the offline test oracle (`test_rank_delta > 0` or `oracle_gap_ratio > 0`),
  grouped by `selected_head -> oracle_head` and
  `selected_chain -> oracle_chain`.
- Aggregates include count, mean/median oracle-gap ratio, median selected test
  rank and median rank delta. CSV schema is unchanged.
- The section explicitly preserves the invariant: production selection remains
  train-CV only, and the audit must not be used to route or select by dataset
  name, source or id.
- Regenerated:
  - `benchmarks/cross_binding/aom_staged_compact_wide_audit5_20260606_rank_audit.csv/.md`
  - `benchmarks/cross_binding/aom_staged_compact_wide_audit10_20260606_rank_audit.csv/.md`
  - `benchmarks/cross_binding/aom_staged_ecosis_compact_wide_refit40_audit_20260606_rank_audit.csv/.md`
- Notable audit10 patterns:
  - ECOSIS selected `ridge:0.1 savgol_smooth(7,2)` while offline oracle was
    `pls:1 detrend_poly(2)`, gap ratio `0.5773`.
  - ECOSIS refit40 still selected `ridge:0.1 savgol_smooth(7,2)`, while the
    offline oracle was `ridge:10 finite_difference(1)`, gap ratio `2.9123` and
    selected test rank `41`.
- Validation:
  - `/home/delete/.venv/bin/python -m py_compile benchmarks/cross_binding/summarize_aom_rank_audit.py bindings/python/tests/test_aom_benchmark_tools.py`: PASS.
  - `PYTHONPATH=bindings/python/src N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so /home/delete/.venv/bin/python -m pytest bindings/python/tests/test_aom_benchmark_tools.py -q`: `19 passed`.
  - `git diff --check`: PASS apart from pre-existing CRLF warnings on old CSV
    artifacts.

Read-only engine audit follow-up (2026-06-06):

- Delegated a read-only Claude Code audit of the remaining engine/performance
  gap. It found no remaining Python/facade/catalog method wiring gap; the
  open work is C++/CUDA engine work.
- Smallest plausible PR-sized engine slice:
  - accelerate Ridge moment scoring by scanning the lambda path from one
    symmetric eigendecomposition per fold instead of solving the Ridge system
    independently for every lambda;
  - then raise the Ridge moment feature route cap only after timing/parity
    proves the crossover is favorable.
- Important review correction: current `sweep.cpp` uses the generic
  `solve_square_qr()` path, not Cholesky, and the existing performant
  `symmetric_eigh()` helper is local/static in `cpp/src/core/model.cpp`.
  A clean implementation should first extract or share that eigensolver rather
  than duplicate a large numerical routine in `sweep.cpp`.
- Likely touch list for that future slice:
  - `cpp/src/core/common/*` or equivalent shared eigensolver helper,
  - `cpp/src/core/model.cpp` to consume the shared helper,
  - `cpp/src/core/sweep.cpp` for a Ridge lambda-path scorer,
  - focused C++/Python parity tests plus one timing smoke for p around
    500-1000.
- Not implemented in this pass because doing it safely requires numerical
  helper extraction and parity/timing evidence; the rank-audit tooling change
  above is complete and validated.

Ridge moment eigen-path implementation follow-up (2026-06-06):

- Implemented the bounded Ridge moment engine slice from the audit:
  `cpp/src/core/common/svd.c` now exposes an internal read-only
  `n4m_symmetric_eigh()` wrapper over the existing symmetric eigensolver, with
  overflow-guarded workspace allocation; `cpp/src/core/common/svd.h` documents
  it as internal/non-ABI.
- `cpp/src/core/sweep.cpp` now prepares a `RidgeMomentEigenPath` for
  multi-lambda Ridge moment scoring: it builds the same centered/scaled
  cross-moments as `fit_ridge_from_moments()`, eigendecomposes `X'X` once per
  fold, preprojects `X'Y`, then scans lambdas by diagonal division.
- The path is only attempted when `n_lambdas > 1`; mono-lambda calls keep the
  previous direct QR solve. If eigen preparation or an eigen solve is not
  usable, the code falls back to the existing direct moment solve for that
  fold/lambda, preserving behavior.
- Batch score-only Ridge AOM now computes one local eigen-path per
  `(chain, fold)` worker and scores all lambdas from it without storing
  eigenvectors for every chain/fold globally. Public counters remain unchanged:
  `n_ridge_moment_cv_fits` and `n_ridge_moment_score_batch_jobs` still report
  the logical lambda-by-fold candidate work.
- Added `test_native_sweep_run_ridge_moment_scores_match_numpy_path`, which
  compares Ridge moment CV candidate RMSEs against explicit NumPy solves across
  five lambdas and four folds with centering/scaling disabled, and checks that
  the same lambda scores identically in the mono-lambda direct path and the
  multi-lambda eigen path.
- Validation:
  - `/home/delete/.venv/bin/cmake --build build/dev-release --target n4m_c -j2`: PASS.
  - `PYTHONPATH=bindings/python/src N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so /home/delete/.venv/bin/python -m pytest bindings/python/tests/test_moment_model_wrappers.py -q`: `73 passed`.
  - `PYTHONPATH=bindings/python/src N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so /home/delete/.venv/bin/python -m pytest bindings/python/tests/test_aom_benchmark_tools.py -q`: `19 passed`.
  - `/home/delete/.venv/bin/ctest --test-dir build/dev-release -R sweep --output-on-failure`: no C++ tests found in this build.
  - `git diff --check`: PASS apart from known CRLF warnings on old CSV artifacts.
- Microbench smoke on synthetic `n=140, p=80, cv=5`, Ridge score-only:
  `1/4/16/64` lambdas took approximately `0.0026 / 0.0064 / 0.0221 / 0.1417`
  seconds respectively on the dev build.
- Read-only Claude review found no blocking correctness, memory, counter or
  parallelism issue. Its only actionable test gap was the mono-lambda direct
  versus multi-lambda eigen-path equality check, which is now covered.

Moment sweep timing smoke Ridge-moment coverage follow-up (2026-06-06):

- Fixed a coverage blind spot in `benchmarks/cross_binding/bench_moment_sweep_timing.py`:
  the previous default CUDA smoke shapes all routed Ridge through the
  wide dual/materialized path, despite the script documenting a native moment
  Ridge sweep timing smoke.
- Added a tall `96x48` cell to the smoke shapes so Ridge reports
  `n_ridge_moment_candidates=5` and `n_ridge_moment_cv_fits=25`, covering the
  new multi-lambda Ridge moment scorer separately from the wide Ridge route.
- Updated `bindings/python/tests/test_aom_moment_cuda_smoke_artifacts.py` to
  require that the committed `moment_sweep_timing_cuda_smoke.csv` includes at
  least one Ridge moment row and that those rows have zero dual/materialized
  Ridge CV counters.
- Regenerated `benchmarks/cross_binding/moment_sweep_timing_cuda_smoke.csv`
  with one visible GPU and `N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so`.
  The tall `96x48` Ridge rows now show the expected moment counters, and the
  PLS rows still route exact PLS CV through CUDA device counters.
- Validation:
  - `/home/delete/.venv/bin/cmake --build build/cuda-on --target n4m_c -j2`: PASS.
  - `/home/delete/.venv/bin/python -m py_compile benchmarks/cross_binding/bench_moment_sweep_timing.py bindings/python/tests/test_aom_moment_cuda_smoke_artifacts.py`: PASS.
  - `PYTHONPATH=bindings/python/src N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so /home/delete/.venv/bin/python -m pytest bindings/python/tests/test_aom_moment_cuda_smoke_artifacts.py -q`: `21 passed`.

Catalog/method-surface conformance follow-up (2026-06-06):

- Revalidated the broadened AOM/moment method surface after the Ridge
  eigen-path and timing-smoke updates.
- Evidence:
  - `/home/delete/.venv/bin/python catalog/scripts/validate.py --strict-abi`:
    PASS, 208 methods and 701/701 exported `n4m_*` symbols covered.
  - `/home/delete/.venv/bin/python catalog/scripts/validate.py --check-references`:
    PASS, 208/208 production methods covered by donor/registry/paper-only
    references.
  - `/home/delete/.venv/bin/python catalog/scripts/split_legacy_methods.py --check`:
    PASS, 208 per-method files up to date.
  - `PYTHONPATH=bindings/python/src N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so /home/delete/.venv/bin/python -m pytest bindings/python/tests/test_catalog_python_bindings.py -q`:
    `10 passed`.
- This confirms the current method count and Python/catalog exposure are
  internally consistent; it does not close the remaining engine gaps
  (batched IKPLS, fused CUDA kernels, broader oracle campaign).

Oracle comparison table follow-up (2026-06-06):

- Improved `benchmarks/cross_binding/compare_aom_staged_to_oracles.py` so the
  Markdown summary now includes a per-target-dataset table with target RMSEP,
  AOM-PLS oracle RMSEP/ratio, AOM-Ridge oracle RMSEP/ratio, TabPFN
  oracle RMSEP/ratio and overall winner.
- This keeps AOM-PLS oracle, AOM-Ridge oracle and TabPFN oracle separated in
  the report instead of only exposing aggregate median ratios.
- Regenerated:
  - `benchmarks/cross_binding/aom_staged_compact_wide_audit10_20260606_oracle_summary.md`
  - `benchmarks/cross_binding/aom_staged_real_cohort_compact_wide_audit10_20260606_oracle_compare.csv`
  - `benchmarks/cross_binding/aom_staged_oracle_comparison.md`
  - `benchmarks/cross_binding/aom_staged_oracle_comparison.csv`
- Current compact-wide audit10 summary remains train-CV-selected target only,
  never test-set selected. It reports 8 target datasets, AOM-PLS paired=7
  median ratio `1.03079`, AOM-Ridge paired=8 median ratio `1.08068`, and
  TabPFN paired=8 median ratio `0.978527`.
- Validation:
  - `/home/delete/.venv/bin/python -m py_compile benchmarks/cross_binding/compare_aom_staged_to_oracles.py bindings/python/tests/test_aom_benchmark_tools.py`: PASS.
  - `PYTHONPATH=bindings/python/src N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so /home/delete/.venv/bin/python -m pytest bindings/python/tests/test_aom_benchmark_tools.py -q`: `19 passed`.

Methods index count follow-up (2026-06-06):

- Fixed `docs/methods/index.md`: the displayed total catalogued native method
  count was stale at `201`; it now matches the current catalog count `208`.
- Added `test_methods_index_total_matches_catalog_file_count` in
  `bindings/python/tests/test_catalog_python_bindings.py` so future catalog
  additions cannot leave the end-user methods index count stale silently.
- Validation:
  - `/home/delete/.venv/bin/python -m py_compile bindings/python/tests/test_catalog_python_bindings.py`: PASS.
  - `PYTHONPATH=bindings/python/src N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so /home/delete/.venv/bin/python -m pytest bindings/python/tests/test_catalog_python_bindings.py -q`: `11 passed`.

Facade inventory conformance follow-up (2026-06-06):

- Added an explicit facade guard that every `n4m.aom.available_methods()` and
  `n4m.moment.available_methods()` row declares typed `cpu` and `cuda`
  capability flags and at least one execution capability.
- Existing facade checks already cover object identity, top-level re-export,
  catalog/doc resolution, public docs-index links, binding-role coherence,
  complete relevant native AOM/moment catalog exposure, preset boundedness and
  the ban on dataset/source-name routing options.
- Validation:
  - `PYTHONPATH=bindings/python/src N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so /home/delete/.venv/bin/python -m pytest bindings/python/tests/test_aom_moment_facade.py -q`: `20 passed`.

Ridge moment eigen-path general sweep follow-up (2026-06-06):

- Extended the already-validated Ridge moment eigen-path to the general
  `run_moment_sweep()` Ridge moment route, not only the score-only
  `score_ridge_moment_sweep()` / batch AOM scorer.
- For `p <= n_train` moment folds and `len(ridge_lambdas) > 1`, the sweep now
  prepares one eigen-path per fold and scans lambdas from it. Mono-lambda and
  fallback behavior still use the previous direct QR solve, and public logical
  counters remain unchanged.
- Added `test_native_sweep_run_ridge_moment_score_only_matches_full_scores` so
  the tall Ridge moment multi-lambda route must match full sweep scores,
  preserve moment-vs-dual counters, and skip final outputs when `score_only`.
- Synthetic `n=140, p=80, cv=5` score-only smoke comparing one multi-lambda
  call to repeated mono-lambda calls:
  - 4 lambdas: `0.058237s` vs `0.143395s`, `2.46x`, max score diff `8.3e-15`.
  - 16 lambdas: `0.026655s` vs `0.306316s`, `11.49x`, max score diff `8.3e-15`.
  - 64 lambdas: `0.093162s` vs `1.277440s`, `13.71x`, max score diff `1.1e-14`.
- Validation:
  - `/home/delete/.venv/bin/cmake --build build/dev-release --target n4m_c -j2`: PASS.
  - `/home/delete/.venv/bin/cmake --build build/cuda-on --target n4m_c -j2`: PASS.
  - `PYTHONPATH=bindings/python/src N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so /home/delete/.venv/bin/python -m pytest bindings/python/tests/test_moment_model_wrappers.py -q`: `74 passed`.
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=bindings/python/src N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so /home/delete/.venv/bin/python -m pytest bindings/python/tests/test_moment_model_wrappers.py -q -k 'ridge_moment_score_only_matches_full_scores or ridge_moment_scores_match_numpy_path'`: `2 passed, 72 deselected`.

Moment sweep CUDA smoke artifact follow-up after general eigen-path
(2026-06-06):

- Regenerated
  `benchmarks/cross_binding/moment_sweep_timing_cuda_smoke.csv` against
  `build/cuda-on/cpp/src/libn4m.so` with one visible GPU and the same smoke
  command documented in `benchmarks/cross_binding/README.md`.
- Strengthened the artifact guard so the tall Ridge moment cell (`96x48`) must
  include both `native_sweep_ridge` and `native_sweep_ridge_score_only`,
  both with five Ridge moment candidates; the full row must perform one final
  Ridge moment fit, the score-only row must perform none, and both rows must
  agree on selected lambda and CV RMSE.
- Regenerated tall Ridge values:
  - full: selected lambda `0.001`, RMSE `0.08037705391174381`,
    elapsed `13.025 ms`, `n_ridge_moment_cv_fits=25`, final fits `1`.
  - score-only: selected lambda `0.001`, RMSE `0.08037705391174604`,
    elapsed `20.925 ms`, `n_ridge_moment_cv_fits=25`, final fits `0`.
- Validation:
  - py_compile on the timing script and touched tests: PASS.
  - `test_aom_moment_cuda_smoke_artifacts.py`: `21 passed`.
  - `test_aom_moment_facade.py` + `test_moment_model_wrappers.py`: `94 passed`.

Post-tranche catalog/surface validation (2026-06-06):

- Re-ran the catalog and split checks after the general sweep eigen-path and
  CUDA smoke artifact updates.
- Results:
  - `catalog/scripts/validate.py --strict-abi`: PASS, 208 methods and 701/701
    exported `n4m_*` symbols covered.
  - `catalog/scripts/validate.py --check-references`: PASS, 208/208
    production methods covered by references.
  - `catalog/scripts/split_legacy_methods.py --check`: PASS, 208 per-method
    files up to date.
  - `test_catalog_python_bindings.py`: `11 passed`.
  - `/home/delete/.venv/bin/ctest --test-dir build/dev-release -R sweep --output-on-failure`:
    no C++ tests found in this build.

Ridge moment eigen-path telemetry follow-up (2026-06-06):

- Added route counters to `SweepResult` and `AomSweepResult`:
  `n_ridge_moment_eigen_path_preparations`,
  `n_ridge_moment_eigen_path_cv_fits` and
  `n_ridge_moment_direct_cv_fits`.
- The counters distinguish logical Ridge moment CV fits from the physical
  execution route. Existing `n_ridge_moment_cv_fits` remains the logical
  lambda-by-fold candidate count; the new counters show whether those fits
  used the fold-local eigen-path or the direct QR fallback.
- Propagated the counters through the C API, `n4m.python`, sklearn
  diagnostics, AOM sweep aggregation, refit reports, staged campaign reports,
  and the moment sweep timing smoke CSV.
- Regenerated `moment_sweep_timing_cuda_smoke.csv`; the tall `96x48` Ridge
  full and score-only rows now both report `5` eigen-path preparations, `25`
  eigen-path CV fits and `0` direct CV fits.
- Validation:
  - dev `n4m_c` build: PASS.
  - CUDA `n4m_c` build: PASS.
  - focused Ridge moment route tests: `3 passed, 71 deselected`.
  - `test_moment_model_wrappers.py`: `74 passed`.
  - `test_aom_moment_cuda_smoke_artifacts.py` + facade + wrappers:
    `115 passed`.
  - `test_catalog_python_bindings.py`: `11 passed`.
  - `catalog/scripts/validate.py --strict-abi`: PASS, 208 methods and 701/701
    exported `n4m_*` symbols covered.
  - `catalog/scripts/validate.py --check-references`: PASS, 208/208
    production methods covered.
  - `catalog/scripts/split_legacy_methods.py --check`: PASS.
  - `git diff --check`: PASS apart from known CRLF warnings on old CSV
    artifacts.

Ridge telemetry benchmark-runner propagation follow-up (2026-06-06):

- Propagated the Ridge moment eigen-path/direct counters into the benchmark
  surfaces used for preprocessing-selection campaigns:
  - `run_aom_staged_real_cohort.py` result rows and diagnostics JSON counters.
  - `bench_aom_screen_refit_scaling.py` screen/refit/final-fit timing rows.
  - `bench_aom_staged_chain_campaign_timing.py` orchestration timing rows.
  - `bench_aom_sweep_timing.py` and `bench_aom_ridge_global_timing.py` smoke
    rows.
- Regenerated CUDA smoke artifacts whose scripts gained columns:
  - `aom_sweep_timing_cuda_smoke.csv`
  - `aom_ridge_global_timing_cuda_smoke.csv`
  - `aom_staged_chain_campaign_timing_cuda_smoke.csv`
- Strengthened benchmark/staged tests so runner rows, diagnostics payloads and
  stage/model-config aggregates preserve the new counters; the invariant
  `eigen_path_cv_fits + direct_cv_fits == n_ridge_moment_cv_fits` is covered
  on staged reports.
- Validation:
  - py_compile on touched benchmark scripts/tests: PASS.
  - `test_aom_benchmark_tools.py`: `19 passed`.
  - `test_aom_staged_campaign.py`: `16 passed`.
  - `test_aom_moment_cuda_smoke_artifacts.py`: `21 passed`.
  - combined benchmark/staged/artifact suite: `56 passed`.
  - facade/wrapper/catalog Python suite: `105 passed`.
  - catalog strict ABI/reference/split checks: PASS.

Real-cohort Ridge telemetry guard follow-up (2026-06-06):

- Added a staged real-cohort runner guard:
  `validate_ridge_moment_route_telemetry(report)`.
- The guard runs immediately after `n4m.aom_staged_chain_campaign(...)`
  returns and before CSV/diagnostics persistence. When
  `n_ridge_moment_cv_fits`,
  `n_ridge_moment_eigen_path_cv_fits` and
  `n_ridge_moment_direct_cv_fits` are present, all route counters must be
  non-negative integer-like values and
  `eigen_path_cv_fits + direct_cv_fits == n_ridge_moment_cv_fits`.
- This keeps real preprocessing-selection campaign artifacts from silently
  recording impossible Ridge moment route telemetry. It does not affect
  production selection policy; held-out/test ranking remains audit-only and
  `selection_uses_test_set=False`.
- Validation:
  - py_compile on `run_aom_staged_real_cohort.py` and
    `test_aom_benchmark_tools.py`: PASS.
  - `test_aom_benchmark_tools.py`: `20 passed`.
  - One-row one-GPU real-cohort smoke to `/tmp`:
    `BEEFMARBLING/Beef_Marbling_RandomSplit` completed `ok`,
    `selection_uses_test_set=False`, Ridge counters `12/6/6`
    total/eigen/direct, and diagnostics JSON carried the same counters.

Moment facade wrapper-target follow-up (2026-06-06):

- Added the missing `n4m.moment.aom_chain_screen_refit_campaign` export. Three
  moment screen/refit sklearn wrappers already declared
  `wrapper_of="aom_chain_screen_refit_campaign"`, and the function existed on
  `n4m` / `n4m.python`, but not on the `n4m.moment` facade itself.
- Added a generic facade guard so every advertised `wrapper_of` target must be
  exported on the same facade, top-level `n4m`, and `n4m.python`, and must be
  the exact same native object. This protects the reusable AOM/moment wrappers,
  including the superblock and staged presets, from dangling inventory targets.
- Validation:
  - py_compile on `bindings/python/src/n4m/moment/__init__.py` and
    `test_aom_moment_facade.py`: PASS.
  - `test_aom_moment_facade.py`: `22 passed`.
  - facade/wrapper/catalog suite:
    `test_aom_moment_facade.py`, `test_moment_model_wrappers.py`,
    `test_catalog_python_bindings.py`: `107 passed`.
  - catalog split/reference/strict-ABI checks: PASS.

Facade timing-benchmark coverage guard follow-up (2026-06-06):

- Added a generic facade guard requiring every catalogued method advertised by
  `n4m.aom.available_methods()` or `n4m.moment.available_methods()` to declare
  an existing `bench.registry_entry` under `benchmarks/cross_binding/`.
- The current public AOM/moment surface exposes 30 distinct catalog IDs and all
  30 have timing benchmark scripts. This complements `catalog/scripts/validate.py`,
  which already checks that declared benchmark paths exist, by tying the check
  directly to the user-facing AOM/moment inventories.
- Validation:
  - py_compile on `test_aom_moment_facade.py`: PASS.
  - `test_aom_moment_facade.py`: `24 passed`.
  - facade/wrapper/catalog suite:
    `test_aom_moment_facade.py`, `test_moment_model_wrappers.py`,
    `test_catalog_python_bindings.py`: `109 passed`.
  - catalog split/reference/strict-ABI checks: PASS.

AOM PLS superblock route-telemetry accounting follow-up (2026-06-06):

- Fixed `n4m.aom_pls_superblock` telemetry aggregation. Before this patch the
  result only propagated the final selected PLS solve counters, so timing
  artifacts under-reported the PLS work used during the external
  component-selection CV.
- The method now sums PLS route counters for every fold/component solve plus
  the final fit. For the committed CUDA smoke profile (`2` component values,
  `4` folds), the reported PLS work is now `18` CV fits and `9` final fits
  instead of `2` CV fits and `1` final fit.
- Extended `bench_aom_pls_superblock_timing.py` to write the CUDA parallel-fold
  jobs and final-fit host/device counters, and regenerated
  `aom_pls_superblock_timing_cuda_smoke.csv` on one GPU. The first smoke rows
  now report `n_pls_moment_cv_fits=18`,
  `n_pls_moment_host_cv_fits=0`,
  `n_pls_moment_cuda_device_cv_fits=18`,
  `n_pls_moment_cuda_parallel_fold_jobs=18`,
  `n_pls_moment_final_fits=9`,
  `n_pls_moment_host_final_fits=0` and
  `n_pls_moment_cuda_device_final_fits=9`.
- Strengthened wrapper and artifact tests so this accounting cannot regress
  silently.
- Validation:
  - py_compile on touched Python modules, benchmark and tests: PASS.
  - focused PLS-superblock wrapper tests: `2 passed, 72 deselected`.
  - focused diversity CUDA smoke artifact tests: `9 passed, 12 deselected`.
  - wrapper + CUDA-artifact suite:
    `test_moment_model_wrappers.py`,
    `test_aom_moment_cuda_smoke_artifacts.py`: `95 passed`.
  - catalog split/reference/strict-ABI checks: PASS.

AOM Ridge-PLS solve-count telemetry follow-up (2026-06-06):

- Added `n_ridge_pls_fit_calls` to `n4m.aom_ridge_pls_superblock` and
  `n4m.aom_chain_ridge_pls`.
- The counter records deterministic external Ridge-PLS work:
  `n_candidates * cv + 1` native Ridge-PLS solves, i.e. all train-CV candidate
  folds plus the selected final fit. It is CPU solve accounting, not a CUDA
  route counter.
- Exposed the counter through
  `NativeAOMRidgePLSSuperblockRegressor.get_diagnostics()` and
  `NativeAOMChainRidgePLSRegressor.get_diagnostics()`.
- Extended and regenerated the CUDA-build timing smoke CSVs:
  - `aom_ridge_pls_superblock_timing_cuda_smoke.csv` now reports
    `n_ridge_pls_fit_calls=17` for the smoke grid (`4` candidates x `4` folds
    + final fit).
  - `aom_chain_ridge_pls_timing_cuda_smoke.csv` now reports
    `n_ridge_pls_fit_calls=33` (`8` candidates x `4` folds + final fit).
- Validation:
  - py_compile on touched modules/benchmarks/tests: PASS.
  - focused Ridge-PLS wrapper tests: `4 passed, 70 deselected`.
  - focused diversity CUDA artifact tests: `9 passed, 12 deselected`.
  - wrapper + CUDA-artifact suite:
    `test_moment_model_wrappers.py`,
    `test_aom_moment_cuda_smoke_artifacts.py`: `95 passed`.
  - catalog split/reference/strict-ABI checks: PASS.

AOM operator-PLS-stack and ridge-blender cost-route telemetry (2026-06-06):

- Audited `aom_ridge_blender` and `aom_operator_pls_stack`. Neither had cost counters
  in their C++ result structs. No C++ change needed: both methods' costs are fully
  deterministic from existing native scalars.
- Added Python-derived counters:
  - `aom_ridge_blender`: `n_ridge_blender_cv_fits = n_candidates * cv`,
    `n_ridge_blender_final_fits = n_candidates`,
    `n_ridge_blender_fit_calls = n_candidates * (cv + 1)`.
  - `aom_operator_pls_stack`: `n_operator_pls_stack_fit_calls = (n_specs + 1) * cv + 1`,
    `n_operator_pls_stack_pls_fit_calls = n_operator_pls_stack_fit_calls * n_operators`,
    `n_operator_pls_stack_ridge_fit_calls = n_operator_pls_stack_fit_calls`,
    plus the `n_pls_stack_*` / `n_ridge_stack_*` CV/final breakdown counters.
- Propagated to `get_diagnostics()` in both sklearn wrappers, bench CSV rows, smoke CSV
  files, and targeted tests with exact-value assertions.
- Updated `benchmarks/cross_binding/README.md` and the two method pages so the
  documented smoke contract covers replay plus the new fit-count telemetry
  instead of stale fixed ABI/timing values.
- Codex review follow-up regenerated the four timing CSVs from the local scripts:
  dev rows with `N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so`, CUDA smoke
  rows with `CUDA_VISIBLE_DEVICES=0` and `N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so`.
  The smoke rows now report `n_ridge_blender_fit_calls=288` and
  `n_operator_pls_stack_fit_calls=21` / `n_operator_pls_stack_pls_fit_calls=252` /
  `n_operator_pls_stack_ridge_fit_calls=21`.
- Validation:
  - py_compile on all touched Python modules/tests/benchmarks: PASS.
  - focused contract tests: `2 passed, 72 deselected`.
  - diversity CUDA smoke artifact tests: `9 passed, 12 deselected`.
  - wrapper + CUDA-artifact suite: `95 passed`.
  - catalog split/reference/strict-ABI checks: PASS.
  - `git diff --check`: PASS apart from known CRLF warnings on CSV artifacts.

Facade duplicate catalog-role guard follow-up (2026-06-06):

- Added a facade inventory guard requiring duplicate `catalog_id` entries to be
  explicit wrappers, aliases, presets or campaign helpers. At most one entry on
  a facade may claim the primary `catalog_binding` / `direct_native_binding`
  role for a catalog method, and every secondary duplicate must declare
  `wrapper_of`.
- This keeps the AOM/moment reusable product surfaces clear: duplicate catalog
  rows now mean "same native method exposed through a wrapper/preset/helper",
  not two ambiguous primary methods.
- Validation:
  - py_compile on `test_aom_moment_facade.py`: PASS.
  - `test_aom_moment_facade.py`: `26 passed`.
  - facade/wrapper/catalog suite:
    `test_aom_moment_facade.py`, `test_moment_model_wrappers.py`,
    `test_catalog_python_bindings.py`: `111 passed`.
  - catalog split/reference/strict-ABI checks: PASS.

CUDA facade smoke diversity-alias guard follow-up (2026-06-06):

- Strengthened `benchmarks/cross_binding/aom_moment_cuda_facade_smoke.py` so the
  CUDA child process also proves the recent diversity methods are exported
  consistently through the top-level facade:
  `aom_ridge_blender`, `aom_operator_pls_stack`,
  `NativeAOMRidgeBlenderRegressor`, and
  `NativeAOMOperatorPLSStackRegressor`.
- Regenerated `benchmarks/cross_binding/aom_moment_cuda_facade_smoke.json` with
  one visible GPU (`CUDA_VISIBLE_DEVICES=0`,
  `N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so`). The artifact now records
  `aom_diversity_aliases_top_level=true` and
  `aom_diversity_estimators_alias_top_level=true`.
- Added a duplicate-key-safe JSON loader in
  `bindings/python/tests/test_aom_moment_cuda_smoke_artifacts.py` and asserted
  the two new facade proof fields. This catches accidental duplicated report
  keys as well as missing diversity aliases.
- Validation:
  - py_compile on the CUDA facade smoke script and artifact tests: PASS.
  - focused CUDA facade artifact guard: `1 passed, 20 deselected`.
  - full CUDA smoke artifact test file: `21 passed`.

Claude Code release-readiness audit follow-up (2026-06-06):

- Delegated a focused Claude Code audit over the handoff and dirty diff, with
  write permissions available but no long benchmark campaign. Claude did not
  produce a patch before hitting the turn cap.
- The only concrete signal it raised was whether Ridge moment
  `eigen_path/direct_cv` telemetry still needed artifact guards. Codex verified
  this is already covered in `test_aom_moment_cuda_smoke_artifacts.py` and
  `test_aom_benchmark_tools.py`, including the CUDA smoke artifact exact
  assertions for `n_ridge_moment_eigen_path_preparations`,
  `n_ridge_moment_eigen_path_cv_fits`, and
  `n_ridge_moment_direct_cv_fits`.
- Codex also checked that the staged preset estimators
  `NativeAOMSavgolFocusRegressor` and
  `NativeAOMStrictFamilyLiteRegressor` are present in top-level `n4m.__all__`.
  No speculative facade/doc edit was made.
- Validation after the audit:
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

CUDA PLS many-batched route telemetry follow-up (2026-06-06):

- The dedicated
  `benchmarks/cross_binding/moment_sweep_timing_cuda_many_batched_smoke.csv`
  artifact pins the optional exact PLS moment CUDA `many_batched` path with one
  visible GPU, `--cuda-pls-min-device-features 1`, `--cuda-pls-many-batched`,
  and no parallel-fold scheduling.
- The PLS rows in that smoke report host CV fits `0`, CUDA-device CV fits equal
  to total PLS CV fits, `n_pls_moment_cuda_parallel_fold_batches/jobs=0`,
  `n_pls_moment_cuda_many_batched_batches=1`, and
  `n_pls_moment_cuda_many_batched_jobs=n_pls_moment_cv_fits`.
- Regenerated `benchmarks/cross_binding/aom_moment_cuda_facade_smoke.json` so
  its staged/focus/strict facade sections also carry the new many-batched
  fields; they remain `0` in that artifact because it exercises the
  parallel-fold facade smoke route.
- Wrapper fallback coverage now also asserts the new many-batched counters stay
  at zero when CUDA is requested but the device threshold intentionally keeps
  execution on the CPU route.
- Validation:
  - py_compile on the touched smoke/test files: PASS.
  - exact wrapper fallback test: `1 passed`.
  - full CUDA artifact guard file: `22 passed`.
  - full moment wrapper test file: `74 passed`.
  - catalog strict ABI/reference/split checks: PASS.
  - `git diff --check`: PASS.

AOM/moment completion audit follow-up (2026-06-06):

- Re-audited the current porting state against the coverage matrix and the
  active objective. Method/facade/catalog/docs/artifact wiring is internally
  consistent:
  - Claude Code audit reran the targeted AOM/moment suite:
    `test_aom_benchmark_tools.py`, `test_catalog_python_bindings.py`,
    `test_aom_moment_cuda_smoke_artifacts.py`,
    `test_aom_moment_facade.py`, `test_moment_model_wrappers.py`,
    `test_aom_staged_campaign.py`: `169 passed`.
  - Catalog reference, strict-ABI and split-method checks remained green
    (`208` methods, `701/701` exported `n4m_*` symbols covered).
  - Referenced docs, benchmark scripts, CUDA smoke CSV artifacts and coverage
    matrix catalog ids all resolved on disk in the audit.
- Updated `DEFERRALS.md` to reflect the current CUDA state accurately:
  single-fit cuBLAS and bounded exact PLS CV routes (`cuda_pls_parallel_folds`,
  `cuda_pls_many_batched`) ship with smoke coverage, while the full fused
  cartesian/IKPLS-style 200k-chain grinder remains deferred.
- No additional small Python/facade/catalog method gap was found. The remaining
  objective gap is still category-B engine/performance work: fused batched IKPLS
  over many chains/folds/candidates, broader arbitrary-chain moment coverage
  without materialization, grouped CUDA sweep kernels and a larger oracle
  campaign.

Moment SSE BLAS engine slice (2026-06-06):

- Added a bounded CPU-BLAS optimization inside
  `ridge_heldout_sse_from_moments()`: for single-target (`q=1`) held-out moment
  scoring with `p >= 64`, non-CUDA BLAS builds compute `X'X beta` through the
  existing `linalg::gemv` dispatch and finish `beta'X'X beta` by dot product.
- CUDA builds intentionally keep the scalar host path here to avoid introducing
  per-candidate host/device transfers in the SSE loop; the dedicated CUDA PLS
  component routes remain the GPU path.
- This advances the screen-throughput engine work without changing scores,
  candidate ranking or selection policy.
- Validation:
  - dev `n4m_c` build: PASS.
  - BLAS `n4m_c` build: PASS.
  - CUDA `n4m_c` build: PASS.
  - targeted moment wrapper tests on dev and BLAS libs:
    `6 passed, 68 deselected` each.
  - `test_aom_benchmark_tools.py` on dev and BLAS libs: `20 passed` each.
  - dev-vs-BLAS two-process candidate-score comparison on `n=200, p=80`:
    same selected Ridge lambda (`1.0`), max absolute candidate-score delta
    `6.63e-15`.
  - The dev-vs-BLAS candidate-score comparison is now pinned by
    `test_native_sweep_run_blas_sse_scores_match_scalar_build`.
  - `git diff --check`: PASS.

Moment GPU crossover artifact refresh (2026-06-06):

- Added optional Markdown rendering to
  `benchmarks/cross_binding/bench_moment_gpu_crossover.py` via
  `--summary-output`; the summary groups CPU, CUDA default and PLS-only
  CUDA many-batched rows by `(head, n_samples, n_features)` and keeps the
  recommendation source-free (shape/head only, no dataset identity).
- Regenerated `benchmarks/cross_binding/moment_gpu_crossover.csv` and added
  `benchmarks/cross_binding/moment_gpu_crossover.md` from one visible GPU
  (`CUDA_VISIBLE_DEVICES=0`, `--repeats 3`,
  `--cuda-pls-min-device-features 1`,
  `--compare-cuda-pls-many-batched`). The CSV is now ABI `1.21.0`, has 24
  rows and zero errors.
- Current summary: PLS keeps CPU at `260x256`, CUDA default wins at `512x512`
  and `256x1024`; many-batched does not beat CUDA default on the large PLS
  rows in this smoke.
- Validation:
  - `py_compile` on the touched benchmark/test files: PASS.
  - targeted benchmark-tool pytest:
    `1 passed, 20 deselected`; full file: `21 passed`.

PLS exact batch OpenMP aggregation slice (2026-06-06):

- Parallelized the per-chain held-out SSE/result aggregation inside
  `score_pls1_moment_sweeps_score_only()` with `N4M_PARALLEL_FOR_STATIC` after
  the shared PLS prefix fits have been computed.
- Public scores, candidate ordering, ABI and counters stay unchanged. Chain-local
  errors are collected outside the parallel region before touching `Context`.
- Added a subprocess dev-vs-OpenMP guard,
  `test_native_aom_pls_moment_batch_omp_scores_match_scalar_build`, that checks
  candidate score parity and the existing exact PLS batch counters.
- Small synthetic timing smoke (`n=192, p=32`, 31 chains, PLS exact CV,
  components `1/2/4`, five repeats): dev `3.00 ms`, OpenMP with one thread
  `3.10 ms`, OpenMP with four threads `2.45 ms`, same selected score and
  candidate-score shape. This is a modest CPU-path win, not the deferred fused
  CUDA/IKPLS cartesian grinder.
- Validation:
  - dev, OpenMP and CUDA `n4m_c` builds: PASS.
  - full dev `test_moment_model_wrappers.py`: `76 passed`.
  - targeted dev and OpenMP wrapper checks:
    `2 passed, 74 deselected` each.
  - Claude Code Opus/max review found no blocking issue: no data race, no
    ABI/API/counter change, deterministic per-chain accumulation preserved.
    Residual non-blocking note: CI does not appear to build `omp-on`, so the
    new parity guard is mostly a local guard until an OpenMP CI job exists.

CUDA PLS scheduler precedence slice (2026-06-06):

- Fixed `pls1_moment_components_many()` route precedence so the explicit
  `cuda_pls_many_batched=True` / `N4M_CUDA_PLS_MANY_BATCHED=1` tiled scheduler
  is tried before `cuda_pls_parallel_folds=True` when both are set.
- Made `N4M_CUDA_PLS_MANY_LEGACY=1` a real override: it disables the tiled
  many-batched route even when the explicit Python flag or env opt-in is set,
  then falls through to the requested parallel-folds or sequential legacy path.
- Added a live CUDA subprocess guard,
  `test_cuda_pls_many_batched_precedes_parallel_and_legacy_overrides`, that
  checks many-batched counters win with both flags true, legacy env flips the
  route to parallel-fold counters, and candidate scores stay unchanged.
- Updated method docs for `sweep_run` and `aom_chain_sweep_run` with the route
  precedence and legacy override semantics.
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

Staged comparator PLS score-mode slice (2026-06-06):

- Added `pls_score_mode` to `compare_aom_staged_variants.py` config keys and
  labels, matching the staged real-cohort CSV field order emitted by the runner.
  Exact-CV (`cv`) and proxy (`gcv_proxy`) campaign CSV rows now summarize as
  separate variants instead of collapsing behind the same plan/head/budget key.
- Added `test_staged_variant_comparator_distinguishes_pls_score_mode`, which
  also guards the no dataset-name/source routing invariant by proving multiple
  `cv` datasets group together while identity tokens stay out of `config_key`.
- Validation:
  - focused comparator tests:
    `4 passed, 18 deselected`.
  - full benchmark tool tests:
    `22 passed`.
  - py_compile on touched Python files: PASS.
  - `git diff --check`: PASS.

Direct moment heads CPU/CUDA artifact schema slice (2026-06-06):

- Regenerated `benchmarks/cross_binding/direct_moment_heads_timing.csv` and
  `benchmarks/cross_binding/direct_moment_heads_timing_cuda_smoke.csv` with the
  current 9-head schema (`ridge`, `pls`, `pcr`, `cppls`, `weighted_pls`,
  `robust_pls`, `ridge_pls`, `continuum_regression`, `ecr`) over three shapes
  and both native-function/sklearn replay rows.
- Both artifacts now include the current PLS route fields, including
  `n_pls_moment_cuda_many_batched_batches/jobs`. CPU rows prove host routing;
  CUDA PLS rows prove one-GPU device routing with parallel folds enabled and
  many-batched explicitly off.
- Added a CPU artifact guard and strengthened the CUDA artifact guard so stale
  direct-head CSV schemas cannot silently pass release-readiness.
- Validation:
  - focused direct-head artifact tests:
    `2 passed, 21 deselected`.
  - full CUDA/artifact guard file:
    `23 passed`.
  - py_compile on touched benchmark/test files: PASS.
  - `git diff --check`: PASS.

PLS cross-validate reference ABI artifact slice (2026-06-06):

- Added `benchmarks/cross_binding/bench_pls_cross_validate_timing.py` to time
  the reserved `n4m.pls_cross_validate` /
  `n4m_pls_cross_validate` reference surface in both full and score-only modes.
- Regenerated `benchmarks/cross_binding/pls_cross_validate_timing.csv` and
  `benchmarks/cross_binding/pls_cross_validate_timing_cuda_smoke.csv`.
  Each artifact has six rows: three shapes times full/score-only calls.
- The benchmark compares the public PLS CV hook against
  `n4m.sweep_run(heads=("pls",))` on identical folds/component grids. The
  recorded candidate-score, OOF-prediction and prediction max deltas are
  numerical zero.
- CPU rows prove host exact-CV routing on `build/dev-release`; CUDA rows prove
  one-GPU device routing on `build/cuda-on` with
  `CUDA_VISIBLE_DEVICES=0`, `cuda_pls_parallel_folds=True`,
  `cuda_pls_min_device_features=1` and many-batched off.
- Added artifact guards in
  `bindings/python/tests/test_aom_moment_cuda_smoke_artifacts.py` and README
  commands in `benchmarks/cross_binding/README.md`.
- Validation:
  - full CUDA/artifact guard file:
    `25 passed`.
  - py_compile on the new benchmark and touched artifact test: PASS.
  - `git diff --check`: PASS.
- Claude Code Opus/max review was launched for this slice. It verified the
  binding calls, result fields, build/GPU availability and sibling benchmark
  conventions, found no blocker, and was interrupted before making edits
  because the remaining work was limited to log/handoff updates.
- Remaining true gap: this is still only the PLS CV reference ABI/timing hook,
  not the deferred fused/batched IKPLS-style many-chain executor.

Force-moments CPU wide PLS route slice (2026-06-06):

- Fixed `run_aom_chain_sweep` route selection so
  `moment_policy="force_moments"` bypasses the CPU wide Ridge/PLS
  materialization heuristics before trying admissible operator-moment routes.
- The motivating case was PLS exact-CV AOM moment scoring: before this patch,
  identity-only `force_moments` screens at `n=120, p=32`, `cv=4` were rejected
  as `UNSUPPORTED` because `should_materialize_cpu_wide_pls()` fired before the
  moment batch route. They now run as exact PLS moment score-batch jobs with
  zero materialized candidates.
- Added
  `test_aom_pls_force_moments_bypasses_cpu_wide_materialization_heuristic`.
- Validation:
  - rebuilt `build/dev-release`, `build/cuda-on`, `build/omp-on` and
    `build/blas-on` `n4m_c`.
  - targeted force-moments + existing BLAS/OMP parity pytest:
    `3 passed, 78 deselected`.
  - manual dev-release sweep over `p=16,32,48,64,80,96`:
    all passed with `n_materialized_candidates=0` and PLS score-batch jobs.
  - manual dev/OMP/BLAS `p=96` smoke:
    matching selected scores, zero materialized candidates.
  - manual one-GPU CUDA `p=96` smoke:
    zero materialized candidates and four CUDA PLS parallel-fold jobs.
- Remaining true gap is unchanged: this broadens the forced moment screen
  surface, but it is not the fused/batched IKPLS many-chain executor.

Force-moments CPU wide Ridge route slice (2026-06-06):

- Fixed the inner Ridge `score_only` batch path so a route already accepted
  under `moment_policy="force_moments"` is not rejected again by the CPU
  wide-Ridge materialization heuristic.
- Scope is intentionally narrow: non-forced Ridge still keeps the existing CPU
  wide materialization heuristic, and full/refit runs still use the existing
  selected-chain final-fit path.
- The motivating case was Ridge-only AOM moment scoring at `n=40, p=48`,
  `cv=4`, positive lambdas and `score_only=True`; it previously returned
  `UNSUPPORTED` and now returns two Ridge moment candidates, zero materialized
  candidates and one score batch with eight jobs.
- Added
  `test_aom_ridge_force_moments_bypasses_cpu_wide_materialization_heuristic`
  using `n=40, p=64` to cover the banded underdetermined route, and checking
  exact RMSE agreement with an explicit materialized screen.
- Validation:
  - rebuilt `build/dev-release` `n4m_c`.
  - rebuilt `build/cuda-on` `n4m_c`.
  - targeted PLS/Ridge force-moments pytest:
    `2 passed, 80 deselected`.
  - full dev-release `test_moment_model_wrappers.py`:
    `82 passed`.
  - manual one-GPU CUDA smoke with `CUDA_VISIBLE_DEVICES=0`:
    two candidates, zero materialized candidates and one Ridge moment score
    batch with eight jobs.
- Remaining true gap is unchanged: this fixes forced Ridge screen semantics,
  but it is not the fused/batched Ridge/PLS many-chain executor.

Force-moments Ridge banded cap slice (2026-06-06):

- Added a force-only Ridge banded moment feature cap:
  normal routing stays at `p <= 256`, while strict Ridge screens
  under `moment_policy="force_moments"` can now use local banded
  operator-moment scoring up to `p <= 512`.
- CPU `auto` behavior is deliberately unchanged for underdetermined wide Ridge:
  it still selects the exact materialized dual-Ridge scorer when that route is
  cheaper.
- Added
  `test_aom_ridge_force_moments_extends_wide_banded_cap`, covering
  a `finite_difference` chain at `n=40, p=320`, positive Ridge lambdas, zero
  materialized candidates, exact RMSE agreement with materialized scoring and
  unchanged `auto` materialization, plus full/refit output after a moment-only
  candidate screen.
- Validation:
  - rebuilt `build/dev-release` `n4m_c`.
  - rebuilt `build/cuda-on` `n4m_c`.
  - targeted Ridge force-moments pytest:
    `2 passed, 81 deselected`.
  - full dev-release `test_moment_model_wrappers.py`:
    `83 passed`.
  - dev-release `n4m_internal_tests`: PASS.
  - manual one-GPU CUDA smoke with `CUDA_VISIBLE_DEVICES=0` on the same
    `n=40, p=320` finite-difference Ridge screen:
    two candidates, zero materialized candidates and one Ridge moment score
    batch with eight jobs.
- Remaining true gap is unchanged: this broadens strict wide Ridge screening,
  but it is not the fused/batched Ridge/PLS many-chain executor.

Moment facade reusable AOM surface slice (2026-06-06):

- Expanded `n4m.moment` so the moment facade exposes the reusable AOM surfaces
  already available from `n4m` / `n4m.aom`: native AOM preprocess/profile/chain
  sweep functions, `aom_global_select`, `aom_per_component_select`,
  `aom_robust_hpo`, `aom_ridge_blender`, `aom_operator_pls_stack`, and the
  missing reusable sklearn wrappers (`NativeAOMChainSweepRegressor`,
  `NativeAOMScreenRefitRegressor`, `NativeAOMOperatorPLSStackRegressor`,
  `NativeAOMRidgeBlenderRegressor`, `NativeAOMRobustHPORegressor`,
  `NativeAOMPLSRegressor`, `NativeAOMSweepRegressor`, `NativePOPPLSRegressor`).
- Added corresponding `n4m.moment.available_methods()` entries for the
  fit/predict surfaces so the facade advertises native AOM sweeps, generic
  screen-refit, Ridge blender, operator PLS stack, robust HPO, AOM-PLS and
  POP-PLS in addition to the existing moment-specific presets.
- Corrected `pls_cross_validate` inventory metadata to be a secondary helper
  over `sweep_run`, not a second primary `utilities.sweep` catalog binding.
- Validation:
  - `test_aom_moment_facade.py`: `26 passed`.
  - targeted facade/inventory wrapper selection:
    `27 passed, 82 deselected`.
  - full dev-release `test_moment_model_wrappers.py`:
    `83 passed`.
  - full dev-release `test_aom_staged_campaign.py`:
    `16 passed`.
  - Python `py_compile` on touched facade/test files: PASS.
  - CUDA facade smoke on `CUDA_VISIBLE_DEVICES=0`: PASS.
- Remaining true gap is unchanged: this improves reusable method discovery,
  but it is not the fused/batched Ridge/PLS many-chain executor.

Staged real-cohort CSV resume guard slice (2026-06-06):

- Added a header compatibility check in
  `benchmarks/cross_binding/run_aom_staged_real_cohort.py` so `--resume` cannot
  append current telemetry rows into older staged-cohort CSV artifacts whose
  headers lack the latest Ridge/PLS route counters.
- Compatible current-schema CSVs still append normally; stale historical
  outputs now fail fast with an explicit message to use a fresh `--output` or
  migrate the artifact deliberately.
- Validation:
  - targeted staged real-cohort resume tests:
    `2 passed, 22 deselected`.
  - full `test_aom_benchmark_tools.py`:
    `24 passed`.
  - Python `py_compile` on touched benchmark/test files: PASS.
- Remaining true gap is unchanged: this closes a benchmark-resume correctness
  issue, but not the fused/batched Ridge/PLS many-chain executor.

Strict AOM portfolio CPU timing artifact slice (2026-06-06):

- Generated committed CPU/dev-release timing pairs for strict AOM methods that
  previously had committed CUDA smokes only:
  `aom_preprocess_timing.csv`, `aom_ridge_superblock_timing.csv`,
  `aom_ridge_active_superblock_timing.csv`,
  `aom_ridge_mkl_superblock_timing.csv`, `aom_pls_superblock_timing.csv`,
  `aom_ridge_pls_superblock_timing.csv`, `aom_chain_ridge_pls_timing.csv`,
  `aom_ridge_global_timing.csv` and `aom_staged_chain_campaign_timing.csv`.
- Added release guards in
  `bindings/python/tests/test_aom_moment_cuda_smoke_artifacts.py`:
  - `aom_preprocess_timing.csv` covers direct strict-linear operators in hard
    and soft gating modes on `build/dev-release`.
  - strict diversity CPU artifacts cover native + sklearn replay where
    applicable.
  - CPU PLS rows have host PLS counters nonzero and CUDA counters zero.
  - staged CPU timing keeps `selection_uses_test_set=False`.
- Validation:
  - full `test_aom_moment_cuda_smoke_artifacts.py`: `34 passed`.
- Remaining true gap is unchanged: this improves CPU/GPU timing evidence, but
  not the fused/batched Ridge/PLS many-chain executor.

Strict AOM portfolio timing refresh slice (2026-06-06):

- Regenerated the strict AOM portfolio timing artifacts against ABI 1.22.0 on
  both `build/dev-release` and one-GPU `build/cuda-on` where applicable:
  preprocess, AOM selector, Ridge blender, operator PLS stack, robust HPO,
  all strict AOM superblock/chain timing pairs, Ridge global and staged chain
  campaign.
- Strengthened
  `bindings/python/tests/test_aom_moment_cuda_smoke_artifacts.py` so those
  AOM artifacts must report ABI 1.22.0, expected CPU/CUDA library paths,
  native+sklearn replay rows where applicable, CPU host PLS counters on
  dev-release, and CUDA device PLS counters on cuda-on.
- Validation:
  - full `test_aom_moment_cuda_smoke_artifacts.py`: `36 passed`.
- Remaining true gap is unchanged: this refreshes release evidence, but not
  the fused/batched Ridge/PLS many-chain executor.

Sweep/stack/crossover timing refresh slice (2026-06-06):

- Regenerated the remaining stale sweep/stack/crossover timing artifacts
  against ABI 1.22.0:
  `moment_sweep_timing.csv`,
  `moment_sweep_timing_cuda_smoke.csv`,
  `moment_sweep_timing_cuda_many_batched_smoke.csv`,
  `moment_sweep_timing_parallel_flag_smoke.csv`,
  `moment_sweep_timing_min_device_smoke.csv`,
  `moment_stack_timing.csv`,
  `moment_stack_timing_cuda_smoke.csv`,
  `aom_sweep_timing.csv`,
  `aom_sweep_timing_cuda_smoke.csv`,
  `aom_sweep_timing_batch_counter_smoke.csv`,
  `moment_gpu_crossover.csv` and `moment_gpu_crossover.md`.
- Strengthened
  `bindings/python/tests/test_aom_moment_cuda_smoke_artifacts.py` so these
  artifacts must report ABI 1.22.0, expected CPU/CUDA build paths, CPU host vs
  CUDA device PLS counters, AOM exact/proxy batch counters, and the source-free
  CPU/CUDA crossover profile matrix.
- Validation:
  - Python `py_compile` on the artifact test file: PASS.
  - full `test_aom_moment_cuda_smoke_artifacts.py`: `43 passed`.
- Remaining true gap is unchanged: release evidence is now current, but the
  fused/batched Ridge/PLS many-chain executor is still not implemented.

Legacy catalog validator slice (2026-06-06):

- Fixed the legacy `catalog/scripts/validate_catalog.py` gate by aligning
  `catalog/schema/method_v1.json` with the current split method schema:
  Python-backed orchestration methods may use `abi_symbols: []`, and migrated
  `parity.tolerances` blocks are accepted.
- Added a regression test in
  `bindings/python/tests/test_catalog_python_bindings.py` so the legacy
  validator must stay green against the current repo while `methods.yaml`
  remains an auditable legacy source.
- Validation:
  - `catalog/scripts/validate_catalog.py`: PASS.
  - `catalog/scripts/validate.py`: PASS.
  - `catalog/scripts/split_legacy_methods.py --check`: PASS.
  - `test_catalog_python_bindings.py`: `12 passed`.
  - `git diff --check`: PASS.
- Remaining true gap is unchanged: this closes a release/catalog gate mismatch,
  not the fused/batched Ridge/PLS many-chain executor.

AOM/moment inventory objective guard slice (2026-06-06):

- Strengthened `bindings/python/tests/test_aom_moment_facade.py` with explicit
  inventory assertions for the user-facing objective:
  - global configurable screen/refit surfaces:
    `screen_refit_campaign`, `moment_fast_screen_refit_campaign`,
    `staged_chain_campaign`, and
    `NativeAOMStagedChainCampaignRegressor`;
  - winning reusable presets:
    `NativeAOMSavgolFocusRegressor` and
    `NativeAOMStrictFamilyLiteRegressor`;
  - selected-winner reuse:
    `aom_chain_fixed_fit_run` and
    `NativeAOMFixedCandidateRegressor`;
  - direct moment heads, `moment_stack`, and source-free CPU/CUDA backend
    recommendation.
- The guard requires CPU/CUDA capability flags and reuse metadata on those
  inventory rows, so a facade/inventory cleanup cannot silently hide the
  surfaces even if low-level imports still work.
- Validation:
  - `test_aom_moment_facade.py`: `28 passed`.
- Remaining true gap is unchanged: this is public API/readiness evidence, not
  the fused/batched Ridge/PLS many-chain executor.

Screen/refit release-evidence refresh and completion audit slice (2026-06-06):

- Regenerated the remaining stale global screen/refit timing artifacts against
  ABI 1.22.0:
  `aom_screen_refit_scaling.csv`,
  `aom_screen_refit_scaling_cuda_smoke.csv`,
  `aom_mixed_screen_refit_scaling.csv`,
  `aom_mixed_screen_refit_scaling_cuda_smoke.csv`,
  `aom_ridge_refit_scaling.csv`,
  `aom_ridge_refit_scaling_cuda_smoke.csv`,
  `aom_mixed_screen_refit_split_smoke.csv`,
  `aom_screen_refit_parallel_flag_smoke.csv` and
  `aom_screen_refit_min_device_smoke.csv`.
- The six scaling artifacts keep 25 rows each over the same refit-top-k and
  execution-mode grid. PLS/Ridge stay global-only retention, while mixed keeps
  `refit_per_head_top_k=4`; the one-row mixed split smoke uses
  `refit_per_head_top_k=1`.
- Strengthened
  `bindings/python/tests/test_aom_moment_cuda_smoke_artifacts.py` so these
  artifacts now prove ABI 1.22.0, expected dev-release/cuda-on library paths,
  CPU host PLS vs CUDA device PLS counters, Ridge moment refit counters,
  split-head scoring for mixed screens, CUDA PLS flags and min-device-feature
  smoke behavior.
- Validation:
  - `PYTHONPATH=bindings/python/src N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so CUDA_VISIBLE_DEVICES=0 /home/delete/.venv/bin/python -m pytest bindings/python/tests/test_aom_moment_cuda_smoke_artifacts.py -q`:
    `49 passed`.
  - Combined Python readiness slice:
    `test_catalog_python_bindings.py`,
    `test_aom_moment_facade.py`,
    `test_aom_moment_cuda_smoke_artifacts.py`: `89 passed`.
  - `catalog/scripts/validate.py`: PASS.
  - `catalog/scripts/validate_catalog.py`: PASS.
  - `catalog/scripts/validate.py --strict-abi`: PASS,
    `566` method symbols + `136` infra symbols = `702/702`.
  - `catalog/scripts/validate.py --check-references`: PASS,
    `208/208` production methods covered.
  - `catalog/scripts/reconcile_abi.py --check`: PASS.
  - `catalog/scripts/split_legacy_methods.py --check`: PASS.
  - `git diff --check`: PASS.
- Completion audit against the active AOM/moment porting objective:
  - Public method/facade/catalog/docs wiring for AOM and moment surfaces is
    covered by the strengthened facade inventory tests and catalog gates.
  - Winning reusable presets are present and guarded:
    `NativeAOMSavgolFocusRegressor`, `NativeAOMStrictFamilyLiteRegressor`,
    `NativeAOMMomentScreenRefitRegressor`,
    `NativeAOMMomentPLSScreenRefitRegressor`,
    `NativeAOMMomentPLSExactScreenRefitRegressor`,
    `NativeAOMMomentRidgeScreenRefitRegressor`,
    `NativeAOMFixedCandidateRegressor` and the direct moment-head wrappers.
  - Global ultra-configurable test surfaces are present and guarded:
    `aom_chain_screen_refit_campaign`, `aom_moment_screen_refit_campaign`,
    `aom_staged_chain_campaign`, `NativeAOMScreenRefitRegressor`,
    `NativeAOMStagedChainCampaignRegressor`, candidate-pool/refit helpers,
    route summaries, rank diagnostics and preprocessing-impact helpers.
  - CPU and one-GPU CUDA smoke evidence is current for the public timing
    artifacts; PLS exact moment CV device routes are proven by counters on
    cuda-on builds, and dev-release host routes are proven separately.
  - The tests explicitly guard against dataset-name/source/id routing in public
    config options, and production selection remains train-CV/refit based;
    test-rank helpers are offline audit surfaces.
- Remaining true gap is unchanged after this audit: not method wiring, catalog,
  docs, benchmark scripts or committed CPU/GPU release evidence. The open work
  is still category-B engine/performance work: a true fused/batched IKPLS-style
  many-chain/many-fold executor, broader arbitrary-chain moment coverage
  without materialization, and full fused CUDA sweep kernels for the complete
  cartesian.

PLS moment lower-prefix batch recovery for public sweeps (2026-06-06):

- Hardened the public exact PLS moment sweep paths after the method/facade
  readiness audit. `score_pls1_moment_sweep()` and `run_moment_sweep()` now
  keep a shared host/CUDA fold-batch route for lower requested PLS prefixes
  when the maximum requested component is rank-deficient.
- Behavior now matches the many-chain AOM score-only path more closely:
  after a max-prefix failure, the code first calls the fold-batch prefix fitter
  for the next lower requested prefix; only components above the recovered
  prefix are retried fold-by-fold. Lower recovered components keep finite exact
  CV scores, failed higher components remain `inf`, and no materialized fold
  designs are introduced.
- Added a shared PLS prefix-batch counter helper and removed the unused older
  per-job recovery helper.
- Extended the live one-GPU CUDA test so degenerate `sweep_run()` and
  `pls_cross_validate()` calls prove the recovered component-1 prefix uses CUDA
  parallel-fold counters (`2` device jobs in the smoke) and only the failed
  higher component adds one host fallback attempt.
- Validation:
  - `build/dev-release` `n4m_c` and `n4m_internal_tests`: PASS.
  - `build/cuda-on` `n4m_c`: PASS.
  - `./build/dev-release/cpp/tests/n4m_internal_tests`: PASS.
  - full dev-release `test_moment_model_wrappers.py`: `83 passed`.
  - dev-release `test_aom_staged_campaign.py`: `16 passed`.
  - one-GPU CUDA targeted wrapper test:
    `test_cuda_pls_many_batched_precedes_parallel_and_legacy_overrides`:
    `1 passed`.
- Remaining true gap is unchanged: this removes another avoidable fallback in
  public exact PLS moment CV, but it is still not the fused/batched IKPLS
  many-chain/many-fold executor, broader arbitrary-chain moment coverage, or
  full fused CUDA cartesian kernel suite.

AOM score-only campaign facade closure (2026-06-06):

- Exposed `n4m.aom.aom_chain_score_campaign` as the discoverable
  `chain_score_campaign` entry in `n4m.aom.available_methods()`. It is now
  marked as an ultra-configurable score-only strict-linear AOM/moment screen,
  CPU/CUDA capable, tied to `aom_pop.aom_chain_sweep`, and reusing candidate
  tables/checkpointed audit reports rather than final model refits.
- Added exact public config-option coverage for the score campaign from the
  real Python signature, including chain-grid, CV, Ridge/PLS head, scaling,
  moment-policy, PLS route, checkpoint/resume and CPU/CUDA route knobs.
- Strengthened facade tests so this score-only surface is required alongside
  the screen-refit and staged global campaign surfaces, and so `moment_stack`
  remains bounded to strict moment/linear heads:
  `ridge`, `pls`, `pcr`, `continuum`, `ecr`, `cppls`.
- Validation:
  - `PYTHONPATH=bindings/python/src N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so /home/delete/.venv/bin/python -m pytest bindings/python/tests/test_aom_moment_facade.py -q`:
    `28 passed`.
  - `PYTHONPATH=bindings/python/src N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so /home/delete/.venv/bin/python -m pytest bindings/python/tests/test_catalog_python_bindings.py -q`:
    `12 passed`.
  - py_compile on touched Python facade/test files: PASS.
  - `git diff --check`: PASS.
- Remaining true gap is unchanged: method/facade/catalog discoverability is now
  tighter, but the nontrivial remaining work is still the fused/batched IKPLS
  many-chain/many-fold CUDA executor and broader fused cartesian performance
  engine, not public API wiring.

AOM score-campaign CUDA many-batched guard (2026-06-06):

- Strengthened the live CUDA PLS route test so it now covers the actual
  chunked `aom_chain_score_campaign` surface, not only direct
  `sweep_run` / `aom_chain_sweep_run` calls.
- The guard runs a 4-chain, 2-chunk, PLS-only exact-CV campaign with
  `cuda_pls_many_batched=True` and proves that the campaign aggregates
  `2` native score-batch calls, `16` score-batch jobs, `2` CUDA many-batched
  batches and `16` CUDA many-batched jobs.
- The same campaign is rerun with `N4M_CUDA_PLS_MANY_LEGACY=1`; scores and
  candidate identity signatures remain equivalent, while counters move to
  `2` CUDA parallel-fold batches / `16` jobs and `0` many-batched jobs.
- Validation:
  - targeted live CUDA route test:
    `test_cuda_pls_many_batched_precedes_parallel_and_legacy_overrides`:
    `1 passed`.
  - full dev-release `test_moment_model_wrappers.py`: `83 passed`.
  - py_compile on `test_moment_model_wrappers.py`: PASS.
  - `git diff --check`: PASS.
- Remaining true gap is unchanged: this proves the campaign-level route and
  telemetry for the existing many-batched CUDA path, but it is still not the
  fused/batched IKPLS many-chain/many-fold executor or complete fused CUDA
  cartesian kernel suite.

AOM sweep CUDA many-batched timing artifact (2026-06-06):

- Added the committed smoke timing artifact
  `benchmarks/cross_binding/aom_sweep_timing_cuda_many_batched_smoke.csv`,
  generated from `bench_aom_sweep_timing.py` with
  `--cuda-pls-min-device-features 1 --cuda-pls-many-batched` on the one-GPU
  `build/cuda-on` lib.
- The artifact mirrors the existing 49-row AOM sweep timing surface and proves
  that exact-CV PLS rows run on the CUDA many-batched route: PLS rows have
  `n_pls_moment_cuda_parallel_fold_jobs=0`,
  positive `n_pls_moment_cuda_many_batched_batches`, and
  `n_pls_moment_cuda_many_batched_jobs == n_pls_moment_cv_fits`.
- Updated `test_aom_moment_cuda_smoke_artifacts.py` so AOM sweep artifact
  validation distinguishes the CUDA parallel-fold smoke from the CUDA
  many-batched smoke, while keeping CPU artifact expectations unchanged.
- Validation:
  - `test_aom_moment_cuda_smoke_artifacts.py`: `50 passed`.
  - focused AOM sweep artifact tests: `4 passed`.
  - py_compile on touched benchmark/test files: PASS.
  - `git diff --check`: PASS.
- Remaining true gap is unchanged: this adds release evidence for the existing
  AOM many-batched CUDA route; it is not the future fused/batched IKPLS
  cartesian executor.

AOM screen-refit CUDA many-batched timing artifact (2026-06-06):

- Added the committed smoke timing artifact
  `benchmarks/cross_binding/aom_screen_refit_scaling_cuda_many_batched_smoke.csv`,
  generated from `bench_aom_screen_refit_scaling.py` on the one-GPU
  `build/cuda-on` lib with `--cuda-pls-min-device-features 1`,
  `--cuda-pls-many-batched`, and `--backend-min-cuda-product 1`.
- The artifact keeps the full 25-row PLS screen/refit scaling surface
  (`5` refit depths x `5` execution modes). It proves the exact-CV refit stage
  runs on the CUDA many-batched route: `730` PLS moment CV fits,
  `730` CUDA device CV fits, `101` many-batched batches,
  `730` many-batched jobs, and zero CUDA parallel-fold jobs.
- The screen stage in this benchmark is intentionally PLS `gcv_proxy`, so its
  exact-CV CUDA PLS counters stay at zero while proxy batch counters carry the
  screen work. The artifact test now distinguishes this from the exact-CV
  refit route and from the older parallel-fold CUDA smoke.
- Validation:
  - focused screen-refit artifact slice: `7 passed`.
  - full artifact guard file: `51 passed`.
- Remaining true gap is unchanged: this closes another release-evidence hole
  for the current public screen/refit route, but it is still not the fused
  batched IKPLS/cartesian CUDA grinder.

Real-cohort PLS many-batched CUDA smoke artifact (2026-06-06):

- Added the committed one-row held-out real-cohort smoke
  `benchmarks/cross_binding/aom_staged_real_cohort_diesel_pls_many_batched_cuda_smoke.csv`,
  generated with `run_aom_staged_real_cohort.py` on
  `DIESEL/DIESEL_bp50_246_b-a`, PLS-only, `force_moments`, exact-CV PLS,
  `--cuda-pls-min-device-features 1`, `--cuda-pls-many-batched`, and
  `--backend-min-cuda-product 1`.
- The row is ABI `1.22.0`, uses the one-GPU CUDA build, and records
  `selection_uses_test_set=False`. It proves the real train/test benchmark
  runner can now persist the current many-batched route counters:
  screen PLS exact-CV `12/0/12` total/host/device with `2` many-batched
  batches and `12` many-batched jobs; refit PLS exact-CV `6/0/6` with
  `1` many-batched batch and `6` many-batched jobs; zero parallel-fold jobs.
- Added the matching offline oracle join artifacts
  `aom_staged_real_cohort_diesel_pls_many_batched_cuda_smoke_oracle_compare.csv`
  and `.md`. This is a route/benchmark-pipeline smoke only: with four PLS
  chains it is intentionally not competitive with the real AOM/TabPFN oracles.
- Validation:
  - focused real-cohort smoke artifact test: `1 passed`.
  - full artifact guard file: `52 passed`.
- Remaining true gap is unchanged: this proves the current benchmark runner and
  many-batched telemetry on a held-out real split, but it is not the fused
  cartesian/IKPLS CUDA engine.

Coverage-matrix release audit refresh (2026-06-06):

- Updated `docs/architecture/aom_moment_coverage_matrix.md` so the integrated
  campaign rows mention CUDA many-batched batch/job counters and the staged
  real-cohort row points to the ABI `1.22.0` DIESEL PLS many-batched smoke.
- Added a dedicated many-batched artifact note listing the direct moment sweep,
  AOM sweep, screen/refit and real held-out runner smokes. The note explicitly
  scopes them as current route telemetry / exact-CV compatibility evidence,
  not the deferred fused cartesian/IKPLS CUDA engine.
- Validation:
  - facade/catalog inventory tests: `40 passed`.
  - `git diff --check`: PASS.

Stop snapshot for next continuation (2026-06-06):

- User requested: "Met à jour ou créé le handoff, et stop." No new engine work
  was started after that instruction.
- Current branch at stop: `release-readiness-fixes`, pushed to origin before
  this handoff-only update. Latest pushed commit before this note:
  `b213786 docs(aom): refresh many-batched coverage audit`.
- Current objective status:
  - Public AOM/moment method integration is broadly in place: direct moment
    heads, reusable sklearn wrappers, winning/preconfigured strict-linear AOM
    presets, ultra-configurable score/screen/refit/staged campaigns, fixed
    candidate winner reuse, candidate audit/report helpers, docs/catalog
    discoverability and CPU/CUDA smoke artifacts.
  - Current release evidence includes direct moment sweep, AOM sweep,
    screen/refit and real held-out runner CUDA many-batched route smokes, plus
    facade/catalog guards and benchmark artefact guards.
  - The real remaining gap is still engine/performance work: a true fused
    cartesian/IKPLS CUDA many-chain/many-fold executor, broader arbitrary-chain
    moment coverage for unsupported regimes, and broad benchmark campaigns if
    the user wants to quantify final oracle gaps again.
- Recommended next continuation:
  1. Do not spend more time on public API wiring unless an audit finds a
     specific missing export/catalog/doc/test.
  2. If continuing implementation, start from `cpp/src/core/cuda_dispatch.cpp`
     and `cpp/src/core/sweep.cpp` around `pls1_moment_components_many*` /
     `fit_pls1_moment_prefixes_for_folds`. The current many-batched route uses
     strided-batched cuBLAS for the large products but still has per-job scalar
     reductions/host synchronisation, so that is the likely concrete engine
     target.
  3. If not attacking engine work, run a deliberate benchmark campaign rather
     than adding more release-evidence smokes.

PLS many-batched cuBLAS scalar batching (2026-06-06):

- After the stop snapshot, resumed the engine/performance gap rather than API
  wiring. A Claude Code MCP agent was started in `opus`/`max` mode for a review
  of the same hotspot, but it only reached the inspection phase; the session was
  interrupted before edits to avoid concurrent changes.
- Implemented a bounded C++/cuBLAS improvement in
  `cpp/src/core/cuda_dispatch.cpp`:
  - added reusable per-tile device scalar buffers (`dscale`, `dnorm_sq`,
    `dtt`, `dqdot`) to the many-batched PLS workspace;
  - replaced per-job `nrm2` and the two per-job dot-products with
    `cublasDgemmStridedBatched` reductions over the whole tile;
  - replaced per-job copy/scale sequences for normalized weights, PLS
    loadings and C-deflation vectors with `cublasDdgmm` column scaling;
  - kept sign normalization and W/P row-major storage copies per job for now,
    because the repo's CUDA backend is host C++ + cuBLAS only and does not
    compile custom `.cu` kernels.
- Validation completed:
  - `/home/delete/.venv/bin/cmake --build build/cuda-on --target n4m_c -j2`:
    PASS.
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=bindings/python/src
    N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so /home/delete/.venv/bin/python
    -m pytest
    bindings/python/tests/test_moment_model_wrappers.py::test_cuda_pls_many_batched_precedes_parallel_and_legacy_overrides
    -q`: `1 passed`.
  - `/home/delete/.venv/bin/cmake --build build/cuda-on --target
    n4m_internal_tests -j2`: PASS.
  - `CUDA_VISIBLE_DEVICES=0 ./build/cuda-on/cpp/tests/n4m_internal_tests`:
    PASS.
  - Manual 32-chain synthetic PLS exact-CV smoke: many-batched and legacy had
    matching best CV score (`0.3240250845461336` vs
    `0.32402508454613355`); many-batched used `1` many-batched batch /
    `128` jobs, legacy used `1` parallel-fold batch / `128` jobs.
- Remaining true gap is narrower but still real: this reduces scalar cuBLAS
  overhead in the current many-batched path, but it is not a fused
  IKPLS/cartesian CUDA kernel executor and it does not broaden arbitrary-chain
  moment coverage.

PLS many-batched sign gather cleanup (2026-06-06):

- Continued the same `cpp/src/core/cuda_dispatch.cpp` hotspot after the scalar
  batching commit.
- Added a reusable per-tile `dsign` device buffer and host staging vector.
  The many-batched sign convention now gathers each job's dominant-weight value
  into `dsign` with scalar device-device `cublasDcopy_v2` calls, performs one
  device-to-host copy for the tile/component, then applies any negative sign
  flips.
- This removes the previous `copy_d2h_stream_sync` call inside the per-job sign
  loop. Per-job `cublasIdamax_v2` remains because there is no custom CUDA
  kernel in this host-C++ + cuBLAS backend.
- Validation:
  - `/home/delete/.venv/bin/cmake --build build/cuda-on --target n4m_c -j2`:
    PASS.
  - live one-GPU CUDA wrapper route guard:
    `test_cuda_pls_many_batched_precedes_parallel_and_legacy_overrides`:
    `1 passed`.
  - `/home/delete/.venv/bin/cmake --build build/cuda-on --target
    n4m_internal_tests -j2` plus
    `CUDA_VISIBLE_DEVICES=0 ./build/cuda-on/cpp/tests/n4m_internal_tests`:
    PASS.
  - Manual 32-chain synthetic PLS exact-CV smoke: many-batched and legacy had
    matching best CV score (`0.3240250845461336` vs
    `0.32402508454613355`) and expected route counters (`1` many-batched
    batch / `128` jobs vs `1` parallel-fold batch / `128` jobs). The tiny smoke
    was not faster, so do not treat this as a timing claim.

PLS many-batched score-deflation batching (2026-06-06):

- Continued the same many-batched PLS engine cleanup in
  `cpp/src/core/cuda_dispatch.cpp`.
- Replaced the per-job `cublasDaxpy_v2` score-vector deflation
  (`s -= tt*q_load*p_load`) with a tile-level update:
  - compute per-job `-tt*q_load` scales on host;
  - use `cublasDdgmm` to write all scaled `p_load` columns into the reusable
    `dcw` buffer;
  - add that contiguous tile buffer into `ds` with one chunked
    `cublasDaxpy_v2`.
- This removes another component x job cuBLAS call from the current
  host-C++ + cuBLAS path while keeping exact moment PLS scores. Remaining
  non-fused parts include per-job `Idamax` sign choice and W/P strided storage
  copies.
- Validation:
  - `/home/delete/.venv/bin/cmake --build build/cuda-on --target n4m_c -j2`:
    PASS.
  - live one-GPU CUDA wrapper route guard:
    `test_cuda_pls_many_batched_precedes_parallel_and_legacy_overrides`:
    `1 passed`.
  - `/home/delete/.venv/bin/cmake --build build/cuda-on --target
    n4m_internal_tests -j2` plus
    `CUDA_VISIBLE_DEVICES=0 ./build/cuda-on/cpp/tests/n4m_internal_tests`:
    PASS.
  - Manual 32-chain synthetic PLS exact-CV smoke: many-batched and legacy had
    matching best CV score (`0.32402508454613366` vs
    `0.32402508454613355`) with expected route counters. Observed smoke timing
    was `0.249s` many-batched vs `0.313s` legacy, but this is still a tiny
    route smoke rather than the final benchmark.

PLS many-batched W/P tile storage (2026-06-06):

- Continued the same many-batched PLS CUDA engine cleanup in
  `cpp/src/core/cuda_dispatch.cpp`.
- Removed the per-job strided cuBLAS copies that stored `dw` / `dp_load` into
  row-major `W` / `P` device buffers at every component.
- Since `W` and `P` are only needed after the device component loop completes,
  the many-batched workspace now stores them temporarily as component-major
  contiguous tiles. Each component uses one chunked contiguous cuBLAS copy for
  the whole `W` tile and one for the whole `P` tile.
- The existing host output contract is preserved by repacking the copied
  component-major tile into row-major `p x n_components` buffers before the
  prefix-fit code sees it.
- Validation:
  - `/home/delete/.venv/bin/cmake --build build/cuda-on --target n4m_c -j2`:
    PASS.
  - live one-GPU CUDA wrapper route guard:
    `test_cuda_pls_many_batched_precedes_parallel_and_legacy_overrides`:
    `1 passed`.
  - `/home/delete/.venv/bin/cmake --build build/cuda-on --target
    n4m_internal_tests -j2` plus
    `CUDA_VISIBLE_DEVICES=0 ./build/cuda-on/cpp/tests/n4m_internal_tests`:
    PASS.
  - Manual 32-chain synthetic PLS exact-CV smoke: many-batched and legacy had
    matching best CV score (`0.32402508454613366` vs
    `0.32402508454613355`) with expected route counters. Observed smoke timing
    was `0.252s` many-batched vs `0.307s` legacy.
- Remaining engine gap is now mostly the unfused sign path (`Idamax` and
  branchy sign flips) plus the broader missing fused cartesian/IKPLS CUDA
  executor. Solving that cleanly likely requires custom CUDA kernels / CMake
  CUDA-language work, not just host-C++ cuBLAS reshaping.

Final stop handoff update (2026-06-06):

- User requested: "Met à jour ou créé le handoff, et stop." Work stops here;
  no additional code, benchmark or test expansion was started after this
  instruction.
- Repository state at this stop:
  - branch: `release-readiness-fixes`;
  - head: `ccefb1a6c5c7b0a88640a75a1ceee716d55d1cea`
    (`perf(cuda): store PLS W/P as batched tiles`);
  - local branch and `origin/release-readiness-fixes` were identical before
    this handoff-only edit.
- Latest validated implementation state:
  - public AOM/moment method integration is broadly present in
    `nirs4all-methods`: CPU/CUDA facades, reusable sklearn wrappers,
    strict-linear AOM/moment presets, exact and proxy PLS screen/refit variants,
    Ridge/PLS staged campaigns, fixed-candidate reuse, inventories, docs and
    benchmark smoke artifacts;
  - current CUDA PLS many-batched route has been optimized with batched scalar
    reductions/scaling, batched sign-value gather, batched score deflation and
    component-major W/P tile storage;
  - validations after the latest CUDA engine commit passed:
    `cmake --build build/cuda-on --target n4m_c -j2`, the one-GPU pytest guard
    `test_cuda_pls_many_batched_precedes_parallel_and_legacy_overrides`,
    `cmake --build build/cuda-on --target n4m_internal_tests -j2`, and
    `CUDA_VISIBLE_DEVICES=0 ./build/cuda-on/cpp/tests/n4m_internal_tests`;
  - manual 32-chain synthetic exact-CV PLS smoke matched many-batched and
    legacy scores, with expected route counters.
- Important unfinished item from the interrupted next step:
  - after changing internal W/P storage, the existing CUDA guard mostly covers
    score-only route equivalence. A useful next small test is to extend
    `bindings/python/tests/test_moment_model_wrappers.py::test_cuda_pls_many_batched_precedes_parallel_and_legacy_overrides`
    with a non-`score_only` many-batched-vs-legacy comparison of final
    predictions/coefs/candidate scores. This was planned but not implemented.
- Real remaining gap:
  - the current route is still host C++ plus cuBLAS orchestration, not a true
    fused cartesian/IKPLS CUDA executor. The remaining per-job sign path
    (`Idamax` / branchy sign flips) and any true 200k-chain grinder will likely
    require custom CUDA kernels and CMake CUDA-language work.
- Recommended next continuation:
  1. If resuming release-readiness only, add the non-score-only CUDA output
     guard above, run the targeted one-GPU pytest, `py_compile` and
     `git diff --check`, then commit.
  2. If resuming performance work, start in
     `cpp/src/core/cuda_dispatch.cpp` around
     `pls1_moment_components_many_batched_tiled`; attack the remaining sign
     path only if willing to add custom CUDA kernels.
  3. If deciding to stop engineering, run one deliberate benchmark campaign
     from the current branch rather than adding more smoke artifacts.

Continuation update - PLS many-batched full-output guard (2026-06-06):

- Resumed from the handoff and closed the small release-readiness test gap left
  after the W/P tile-storage commit.
- Updated
  `bindings/python/tests/test_moment_model_wrappers.py::test_cuda_pls_many_batched_precedes_parallel_and_legacy_overrides`
  so the same one-GPU subprocess now also runs
  `n4m.sweep_run(... score_only=False)` on:
  - the default CUDA PLS many-batched CV route;
  - the legacy override route forced by `N4M_CUDA_PLS_MANY_LEGACY=1`.
- The new assertions check:
  - default run uses one many-batched CV batch / four jobs and no parallel-fold
    CV batches;
  - legacy override uses one parallel-fold CV batch / four jobs and no
    many-batched CV batches;
  - both routes keep the final full-data PLS fit on CUDA;
  - candidate scores, OOF predictions, final predictions, coefficients,
    intercept and selected CV RMSE match to tight tolerances.
- Validation completed:
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=bindings/python/src
    N4M_LIB_PATH=build/cuda-on/cpp/src/libn4m.so /home/delete/.venv/bin/python
    -m pytest
    bindings/python/tests/test_moment_model_wrappers.py::test_cuda_pls_many_batched_precedes_parallel_and_legacy_overrides
    -q`: `1 passed`;
  - `PYTHONPATH=bindings/python/src /home/delete/.venv/bin/python -m
    py_compile bindings/python/tests/test_moment_model_wrappers.py`: PASS;
  - `git diff --check`: PASS.
- Next useful continuation after this commit:
  1. For release-readiness, run a broader wrapper/artifact subset if desired;
     the specific W/P output-contract guard is now covered.
  2. For performance, the remaining real engine work is still the unfused sign
     path and ultimately a true custom CUDA fused cartesian/IKPLS executor.

Continuation update - PLS many-batched batched sign scaling (2026-06-06):

- Continued the CUDA PLS many-batched engine cleanup in
  `cpp/src/core/cuda_dispatch.cpp`.
- Replaced the remaining per-negative-job `cublasDscal_v2` sign flips with one
  tile-level `cublasDdgmm` when any job in the tile needs a sign flip:
  - `cublasIdamax_v2` remains per job to preserve the deterministic dominant
    loading convention without adding custom CUDA kernels;
  - gathered sign values are converted to a host `+1/-1` vector;
  - the vector is copied once to `dscale`;
  - `cublasDdgmm(CUBLAS_SIDE_RIGHT, ...)` writes the signed weight tile to the
    reusable `douter` buffer;
  - the signed W tile is copied to `dW` before `douter` is reused for the later
    C-deflation vector.
- Validation completed:
  - `/home/delete/.venv/bin/cmake --build build/cuda-on --target n4m_c -j2`:
    PASS;
  - one-GPU CUDA wrapper guard
    `test_cuda_pls_many_batched_precedes_parallel_and_legacy_overrides`:
    `1 passed`;
  - `/home/delete/.venv/bin/cmake --build build/cuda-on --target
    n4m_internal_tests -j2` and
    `CUDA_VISIBLE_DEVICES=0 ./build/cuda-on/cpp/tests/n4m_internal_tests`:
    PASS;
  - manual 32-chain / 128 exact-CV PLS smoke matched many-batched and legacy
    best CV scores with expected route counters (`1` many-batched batch /
    `128` jobs vs `1` parallel-fold batch / `128` jobs).
- Scope note: this removes another component x negative-job cuBLAS loop from
  the current host-C++ + cuBLAS path, but it still is not the true fused
  cartesian/IKPLS CUDA executor.
