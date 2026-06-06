# AOM / Moment Coverage Matrix

Date: 2026-06-06

This file tracks the requested end state: AOM portfolio methods, reusable
winning linear/moment heads, and global preprocessing-selection methods in
`nirs4all-methods`, with CPU and CUDA-build behavior separated.

## Integrated Now

| Surface | User-facing entry | Native ABI | CPU | CUDA build | Status |
|---|---|---:|---:|---:|---|
| AOM logical Python facade | `n4m.aom`, `n4m.aom.available_methods()` | no new ABI | import tested | CUDA-device smoke tested | dedicated AOM import surface re-exporting campaign helpers, screen/refit pool helpers, native sklearn AOM presets and historical AOM portfolio classes while keeping the single `libn4m` runtime and existing top-level imports; inventory metadata identifies global screen/refit, screen/refit pool helpers, selected-winner reuse, chain-grid builders, candidate decode/evaluate helpers, audit/report IO helpers and diversity entries without adding a router, and now includes audit-only `config_options` for chain grids, refit/checkpoint controls, route policy, report serialization and supported CUDA PLS knobs; `benchmarks/cross_binding/aom_moment_cuda_facade_smoke.py` proves the facade aliases the shared runtime and that wide PLS1 AOM chain screens hit the CUDA device-CV route |
| Moment logical Python facade | `n4m.moment`, `n4m.moment.available_methods()` | no new ABI | import tested | CUDA-device smoke tested | dedicated moment import surface re-exporting sufficient-statistics helpers, `sweep_run`, `pls_cross_validate`, direct linear/component heads, direct native sklearn wrappers, `NativeMomentSweepRegressor`, the AOM/moment fast screen-refit function, the mixed/PLS/Ridge `NativeAOMMoment*ScreenRefitRegressor` presets, retained-pool/refit helpers, audit/report helpers and `NativeAOMFixedCandidateRegressor` winner reuse without shadowing the historical `n4m.moments` function; inventory metadata identifies sufficient-statistics, screen, PLS CV reference, direct-head, reusable-estimator, AOM/moment preprocessing-selection, exact-refit, holdout audit, report IO, winner-reuse and backend-planning entries without adding a router, and now includes audit-only `config_options` for row subsets, CV grids, direct-head parameters, AOM chain/refit controls, holdout evaluation, report save/load, fixed-candidate reuse and supported CUDA PLS knobs; `benchmarks/cross_binding/aom_moment_cuda_facade_smoke.py` proves the facade aliases the shared runtime and that wide PLS1 moment screens hit the CUDA device-CV route |
| AOM strict global selector | `n4m.aom_pls` / `n4m.aom_global_select` / `NativeAOMPLSRegressor` (`aom_pop.aom_pls`) | yes | tested | smoke tested | existing catalog method now exposes `input_coefficients` + `intercept` and has ABI-close plus sklearn Python wrappers |
| AOM preprocessing | `n4m.aom_preprocess` / `n4m.aom.aom_preprocess` (`aom_pop.aom_preprocessing`) | yes | tested | existing build path | native operator-bank preprocessing primitive now has ABI-close Python binding, top-level alias, AOM facade inventory entry, catalog binding and CUDA smoke coverage for direct strict-linear identity, degree-1 detrend, Savitzky-Golay smooth/derivative, Norris-Williams, finite difference, Gaussian, Whittaker and FCK hard/soft gating; strict chains and model scoring remain covered by the sweep/campaign operator-moment paths |
| POP-PLS | `n4m.pop_pls` / `n4m.aom_per_component_select` / `NativePOPPLSRegressor` (`aom_pop.pop_pls`) | yes | tested | smoke tested | existing catalog method now exposes `input_coefficients` + `intercept` and has ABI-close plus sklearn Python wrappers |
| AOM robust-HPO strict screen | `n4m.aom_robust_hpo` / `NativeAOMRobustHPORegressor` | yes | tested | smoke tested | compact/wide product screen; compact has 12 chains and wide has 31 strict-linear chains including Gaussian, FCK and Whittaker variants; exports `input_coefficients` and sklearn replay wrapper; `benchmarks/cross_binding/aom_robust_hpo_timing_cuda_smoke.csv` covers both `native_abi` and `native_sklearn` wrapper replay rows on `build/cuda-on` across 3 shapes (6 rows), with `prediction_replay_max_abs_error <= 1e-10` for all `native_sklearn` rows |
| AOM configurable strict sweep | `n4m.aom_sweep_run` | yes | tested | smoke tested | configurable compact/wide chain bank plus Ridge/PLS grids |
| AOM user-defined strict chain sweep | `n4m.aom_chain_sweep_run` | yes | tested | smoke tested | arbitrary caller-provided strict-linear chains plus Ridge/PLS grids |
| AOM fixed selected-chain final fit | `n4m.aom_chain_fixed_fit_run`, `NativeAOMFixedCandidateRegressor(fit_mode="final_only")` (`aom_pop.aom_chain_fixed_fit`) | yes | tested | smoke tested | catalogued individual winner reuse surface; fits one already-verified strict-linear chain/head/parameter on all rows without running CV, folds coefficients back to original input space, and is used by `NativeAOMScreenRefitRegressor` after exact-CV refit so reusable model construction no longer repays CV |
| AOM campaign chain descriptors | `chain_offsets`, `op_kinds`, `param_offsets`, `chain_params`, `n4m.decode_aom_chains`, `n4m.aom_candidate_table` | yes-backed | tested | build path tested | every AOM sweep candidate row can be mapped back to the exact strict-linear preprocessing chain without dataset-name routing |
| AOM candidate route provenance | `candidate_routes`, `score_route_id`, `score_route`, `by_score_route` summaries | yes-backed | tested | smoke tested | every AOM sweep/campaign candidate can be tied to materialized, dense, banded or structured moment scoring without changing the stable candidate score table shape |
| AOM chunked strict-linear campaign helper | `n4m.build_aom_strict_chain_grid`, `n4m.aom_chain_score_campaign`, `n4m.aom_screen_refit_candidate_pool`, `n4m.aom_chain_screen_refit_campaign`, `n4m.aom_refit_candidates`, `n4m.aom_refit_execution_plan`, `n4m.aom_evaluate_candidates`, `n4m.aom_candidate_rank_diagnostics`, `n4m.aom_candidate_route_summary`, `n4m.aom_save_candidate_report`, `n4m.aom_load_candidate_report`, `n4m.aom_candidate_operator_summary` (`aom_pop.aom_chain_screen_refit`) | Python-backed | tested | smoke tested | deterministic compact/wide/lab strict-linear chain grids, including SavGol, Norris-Williams, finite-difference, Gaussian, FCK and Whittaker variants, plus chunked score-only global, per-head and per-score-route top-k aggregation from full chunk score tables, prefix-aware chunk packing via `chain_ordering="prefix"` to increase native moment-prefix cache reuse without changing scores or original chain ids, optional `split_head_scoring="auto"` for mixed Ridge/PLS chunks so the PLS subcall is PLS-only and can hit the native batched exact/proxy PLS path while merged rows keep the same `(chain_id, head, param)` scores, optional `cuda_pls_parallel_folds=True` scheduling for bounded stream-parallel exact PLS1 moment fold jobs, `cuda_pls_many_batched=True` opt-in for the experimental tiled/strided-batched many-job CUDA path with precedence over parallel-fold scheduling unless `N4M_CUDA_PLS_MANY_LEGACY=1`, and `cuda_pls_min_device_features=<positive int>` threshold control on CUDA builds, checkpoint/resume and incremental `max_chunks_per_run` execution for long screens, source-free retained-pool audit via `aom_screen_refit_candidate_pool`, one-call screen -> exact-CV refit campaign reports for retained candidates after proxy/score-only screens, optional `refit_per_head_top_k` inclusion so mixed Ridge/PLS screens can exact-refit per-head winners in addition to the global top rows, grouped exact-CV refit scoring for rows sharing a chain/head, signature-batched exact-CV scoring for retained chains sharing a head/parameter set, explicit union-parameter batching with scored/extra candidate counters, plan-driven `execution_mode="auto"` with bounded extra-score budget, and refit execution planning without touching `X/y`, native batch exact-CV PLS score-only scoring across PLS-only operator-moment chains before the Python campaign top-k aggregation, normalized throughput/route metrics including moment-prefix cache hit fraction, split-head launch counters and CUDA-parallel fold counters, source-free per-head CPU/CUDA backend recommendations from `n_samples`, `n_features`, `head`, CUDA availability and the explicit PLS CUDA threshold, operator/head/route candidate summaries plus explicit route-coverage summaries by head and chain, explicit CV-vs-holdout candidate evaluation/rank-recall diagnostics over `aom_chain_sweep_run`, and JSON/JSONL/CSV candidate report export/reload |
| Staged strict-chain cartesian campaign | `n4m.aom_staged_chain_campaign`, `NativeAOMStagedChainCampaignRegressor`, `NativeAOMSavgolFocusRegressor`, `NativeAOMStrictFamilyLiteRegressor` (`aom_pop.aom_staged_chain_campaign`) | Python-backed over native helpers | tested | CUDA-device smoke tested | catalogued global staged preprocessing-selection method over `aom_pop.aom_chain_screen_refit`; staged compact/wide/lab, focused `savgol_focus` / `strict_family_focus` or custom strict-linear score-only screens over Ridge/PLS/mixed heads; cross-stage deduplication keeps global + per-head retained candidates, exact-CV refits the retained union once, can train-CV-select model config `scale_x_values` such as `[False, True]`, attaches preprocessing-impact and screen-vs-refit rank diagnostics, and can optionally run a held-out audit that is never used for production selection; reports expose `selection_uses_test_set=False`, selected model-config metadata, staged checkpoint/resume counters, `n_remaining_stage_chunks_total`, `n_screen_split_head_chunks`, `n_screen_chunk_score_calls`, Ridge screen counters and PLS CPU/CUDA route counters, all aggregated across model-config grids when present; the real-cohort runner and reusable sklearn estimators default mixed-head campaigns to score-preserving `split_head_scoring="auto"` while the low-level helper keeps `off` for explicit timing comparisons; the generic sklearn estimator omits `X_audit` / `y_audit`, selects by train `refit_cv_rmse`, then fits the chosen row through final-only fixed-candidate reuse with the selected model config; `NativeAOMSavgolFocusRegressor` is the winning fast SavGol-focused reusable preset, while `NativeAOMStrictFamilyLiteRegressor` is the cost-safe strict-family audit preset with a small retained refit budget; `benchmarks/cross_binding/aom_staged_chain_campaign_timing_cuda_smoke.csv` proves the staged PLS path can run through `build/cuda-on` with device exact-CV counters (`n_pls_moment_cuda_device_cv_fits=9`, host CV fits `0`) on one visible GPU, and `benchmarks/cross_binding/aom_moment_cuda_facade_smoke.json` proves the sklearn staged estimator, both staged presets and the PLS exact screen/refit preset alias both facades, reports CUDA device refit counters (`savgol_focus_estimator=8`, `strict_family_lite_estimator=4`, `pls_exact_screen_refit_estimator` screen/refit `8/4`, host CV fits `0`), and includes a mixed Ridge+PLS default-smoke proving split-head auto (`1` split chunk, `2` score calls), PLS screen CUDA routing (`8/0`) and Ridge screen counters (`8/1/8`); per-dataset diagnostics JSON from `--diagnostics-dir` now includes a compact `audit` section (offline only, `audit_only=True`) with the production-selected candidate's test-set rank/score (`selected_cv`), the test-rank oracle (`oracle`) and CV-vs-test Spearman correlation; `benchmarks/cross_binding/summarize_aom_rank_audit.py` reads a directory of such diagnostics and writes a CSV + Markdown comparing selected train-CV rank/score vs audit/test best rank/score; existing compact10 diagnostics artifacts pre-date this feature and lack the `audit` section — rerun campaign to populate |
| Native sklearn sweep/direct estimators | `NativeRidgeRegressor`, `NativePLSRegressor`, `NativePCRRegressor`, `NativeCPPLSRegressor`, `NativeContinuumRegressionRegressor`, `NativeECRRegressor`, `NativeMomentSweepRegressor`, `NativeAOMSweepRegressor`, `NativeAOMChainSweepRegressor`, `NativeAOMScreenRefitRegressor`, `NativeAOMMomentScreenRefitRegressor`, `NativeAOMMomentPLSScreenRefitRegressor`, `NativeAOMMomentPLSExactScreenRefitRegressor`, `NativeAOMMomentRidgeScreenRefitRegressor`, `NativeAOMStagedChainCampaignRegressor`, `NativeAOMSavgolFocusRegressor`, `NativeAOMStrictFamilyLiteRegressor`, `NativeAOMFixedCandidateRegressor` | yes/Python-backed | tested | build path tested | reusable sklearn-style wrappers over native direct heads, sweep MethodResults and campaign helpers; direct wrappers replay native Ridge/PLS/PCR/CPPLS/continuum/ECR predictions from input-space coefficients and reconstructed intercepts; AOM wrappers use exact `input_coefficients` folded into original feature space; preconfigured native AOM wrappers expose `profile_name`, expected compact/wide bank size and selected-head labels where applicable; screen-refit estimators run proxy/score-only campaign plus exact-CV verification before fitting the reusable candidate through final-only fixed fit and expose separate `screen`, `refit`, staged and `final_*` diagnostics; the mixed, PLS GCV, PLS exact-CV, Ridge, staged, SavGol-focused and strict-family-lite staged presets expose the common end-user workflows while keeping custom chain grids, checkpointing and exact-refit budgets configurable where appropriate; fixed candidates can be built from explicit rows or global/per-head/refit campaign winners |
| AOM route policy switch | `moment_policy="auto"`, `"materialized"` or `"force_moments"` on `n4m.aom_sweep_run` / `n4m.aom_chain_sweep_run` | yes | tested | smoke tested | explicit route control; materialized forces the legacy strict-linear chain screen, while force_moments rejects any candidate-screen fallback outside operator moments |
| AOM score-only screen output | `score_only=True` on `n4m.aom_sweep_run` / `n4m.aom_chain_sweep_run` | yes | tested | build path tested | returns candidate scores, route counters, selected ids and fold ids without final selected-chain refit outputs; useful for large preprocessing ranking campaigns |
| AOM PLS1 GCV proxy screen | `pls_score_mode="gcv_proxy"` with `score_only=True`, `n_pls_gcv_proxy_candidates`, `n_pls_gcv_proxy_fits`, `score_metric="pls_gcv_proxy_rmse"` | yes | tested | smoke tested | explicit moment-only first-pass PLS screen that scores all-sample transformed moments with a deterministic PLS1 GCV RMSE proxy; PLS-only operator-moment campaigns batch eligible chains through one internal native proxy-scoring dispatch and skip held-out moment transforms; it is not exact fold CV and must be followed by exact-CV/holdout verification of retained rows |
| PLS/Ridge screen fit-cost audit | `n_ridge_moment_candidates`, `n_ridge_dual_materialized_candidates`, `n_ridge_moment_cv_fits`, `n_ridge_dual_materialized_cv_fits`, `n_ridge_dual_cross_cv_fits`, `n_ridge_moment_score_batch_calls`, `n_ridge_moment_score_batch_jobs`, `n_ridge_moment_final_fits`, `n_ridge_dual_materialized_final_fits`, `n_pls_moment_cv_fits`, `n_pls_moment_host_cv_fits`, `n_pls_moment_cuda_device_cv_fits`, `n_pls_moment_cuda_parallel_fold_batches`, `n_pls_moment_cuda_parallel_fold_jobs`, `n_pls_materialized_cv_fits`, `n_pls_gcv_proxy_fits`, `n_pls_moment_final_fits`, `n_pls_moment_host_final_fits`, `n_pls_moment_cuda_device_final_fits`, `n_pls_materialized_final_fits`, `ridge_cv_fits_per_chain`, `ridge_cv_fits_per_candidate`, `pls_cv_fits_per_chain`, `pls_cv_fits_per_candidate`, `pls_gcv_proxy_fits_per_chain`, `pls_gcv_proxy_fits_per_candidate`, `chains_per_second`, `candidates_per_second`, `projected_200k_chains_seconds`, `n_refit_scored_candidates`, `n_refit_extra_scored_candidates`, `n_refit_global_candidates`, `n_refit_per_head_candidates`, `n_refit_per_head_extra_candidates`, `n_refit_union_candidates`, `bench_aom_screen_refit_scaling.py`, `bench_aom_sweep_timing.py`, `bench_moment_sweep_timing.py` | yes-backed | tested | smoke tested | exposes how many fold-local/final Ridge and PLS fits and explicit proxy fits the current sweep/campaign actually pays, split by moment vs materialized/proxy route, by host vs CUDA-device PLS1 moment execution, and by bounded CUDA stream-parallel exact fold scheduling; score-only moment sweeps avoid full `X/y` copies and Ridge moment score-only uses held-out moments for SSE; exact AOM Ridge-only and PLS-only score-only screens batch all eligible operator-moment chains into one internal native scoring dispatch while preserving exact fold CV scores, and GCV proxy PLS-only screens batch eligible chains while skipping held-out moment transforms; the exact/proxy batch counters are covered by internal tests, Ridge batch scoring parallelizes flattened chain/fold/lambda jobs when `N4M_WITH_OPENMP=ON`, and exact PLS batch scoring parallelizes per-chain held-out SSE/result aggregation in OpenMP builds after shared prefix fits are computed; CUDA PLS1 moment component calls reuse a thread-local device workspace across successive default-path calls to remove repeated allocation churn before a true fused IKPLS grinder exists; `cuda_pls_parallel_folds=True` can run independent exact PLS1 moment jobs in bounded stream/cuBLAS batches on one GPU and reports batches/jobs without changing scores; `cuda_pls_many_batched=True` explicitly selects the existing experimental tiled/strided-batched CUDA many-job route while preserving scores and default-off behavior; `cuda_pls_min_device_features` keeps the default 1024-feature guard but lets benchmark runs explicitly test medium-width PLS device routing; the screen/refit scaling benchmark varies `refit_top_k`, supports PLS, Ridge and mixed Ridge/PLS (`--head pls` / `--head ridge` / `--head mixed`), emits per-head retention counters for `--refit-per-head-top-k`, `--split-head-scoring`, `--cuda-pls-parallel-folds`, `--cuda-pls-many-batched`, `--cuda-pls-min-device-features`, and emits throughput plus 200k-chain projections, `individual` vs `grouped_score` vs `batched_score` vs `union_batched_score` exact-CV refit timings, planned-vs-observed group/scored/extra counters, and reusable final-only fit timing/counters; the main AOM sweep timing CSVs expose `--cuda-pls-parallel-folds`, `--cuda-pls-many-batched` and `--cuda-pls-min-device-features` for one-GPU CUDA device-route smokes and include `native_aom_chain_fixed_fit_pls`, `native_aom_chain_fixed_fit_ridge`, and `native_aom_chain_sweep_pls_exact_score_only` and `native_aom_chain_sweep_ridge_exact_score_only` rows to isolate winner reuse and exact score-only grinder cost, so batched IKPLS/fused CUDA work and Ridge lambda regrouping have measurable baselines |
| AOM Ridge/PLS1 operator-moment scoring | `n4m.aom_sweep_run`, `n4m.aom_chain_sweep_run` with Ridge candidates and compatible single-target PLS1 candidates in dense, banded, or structured strict-linear regimes | no new ABI | tested | tested | exact chain operators applied to raw moments; dense transforms are guarded, identity/SavGol/Norris/finite/Gaussian/FCK use a banded descriptor up to `p <= 256` for Ridge and `p <= 1024` for single-target NIPALS PLS1, `detrend_poly` uses a structured low-rank projection route, and Whittaker uses a structured pentadiagonal solve route; repeated strict-linear prefixes are cached for bounded medium-width operator-moment grids and exposed via cache hit/miss counters; CPU `auto` uses materialized dual Ridge when `p > n_train` and materialized PLS when `min_train < 4p`, while CUDA `auto` keeps the moment route; selected chain materialized once for predictions |
| Live CPU/CUDA moment crossover | `n4m.moment_screen_backend_recommendation`, `bench_moment_gpu_crossover.py`, `moment_gpu_crossover.csv`, `moment_gpu_crossover.md` | benchmark/Python | tested | live timed | launches separate CPU and CUDA Python processes against the rebuilt libs, times score-only Ridge/PLS `sweep_run` on matched synthetic shapes, records route counters plus `speedup_vs_cpu`, `speedup_vs_cuda_default`, `cuda_pls_profile` and `recommended_backend`, and can render a compact Markdown decision table; Python helper exposes the measured crossover as a source-free launch-planning rule using only `n_samples`, `n_features`, `head`, CUDA availability and the explicit PLS CUDA knobs; for PLS it also reports whether the CUDA component loop, exact-CV fold workspace and optional many-batched route are expected to run under the requested threshold; `--compare-cuda-pls-many-batched` runs the default and many-batched CUDA profiles in one CSV for controlled one-GPU comparison; current one-GPU ABI 1.21.0 comparison keeps CPU for PLS at `260x256` and recommends CUDA default for PLS from the measured `512x512` / `256x1024` shapes, with PLS rows using `device_cv=5` when the device threshold is forced low for timing |
| AOM robust-HPO fold-local Python | `AOMRobustHPORegressor`, `AOMRobustHPOCompact`, `AOMRobustHPOWide` | no | tested | Python CPU reference | stateful-chain reference/preset layer |
| AOM portfolio policy/gates | `AOMStructuralPolicy`, `AOMTrueBankEndpointPortfolio`, `AOMMidPEndpointStack`, `AOMEndpointMarginStabilityGate`, `AOMFallbackBlendGate` | no | tested | Python CPU reference | source-free reference layer |
| AOM-Ridge blender | `n4m.aom_ridge_blender`, `NativeAOMRidgeBlenderRegressor`, `AOMRidgeBlender` | yes for native function | tested | smoke tested for native function and native sklearn wrapper; flexible Python estimator is CPU reference | native compact/wide strict-linear Ridge OOF simplex blend now exports replayable `input_coefficients`; wide has 31 strict-linear chains including Gaussian, FCK and Whittaker variants; `benchmarks/cross_binding/aom_ridge_blender_timing_cuda_smoke.csv` now includes `native_aom_ridge_blender_sklearn` rows with `prediction_replay_max_abs_error <= 1e-10`; Python estimator remains flexible explicit-candidate reference |
| AOM-Ridge global | `n4m.aom_ridge_global`, `NativeAOMRidgeGlobalRegressor` (`aom_pop.ridge_global`) | Python-backed over native `aom_chain_sweep_run` | tested | CUDA-build smoke tested, not fused GPU yet | catalogued donor-style strict-linear AOM Ridge global selector; converts strict single operators into one-op chains, selects one operator plus one positive Ridge alpha by native CV, and returns folded `input_coefficients` plus `intercept`; intentionally excludes donor branch-global reference-dependent, MKL/kernel and nonlinear modes; `benchmarks/cross_binding/aom_ridge_global_timing_cuda_smoke.csv` includes function + sklearn rows on `build/cuda-on` with `ridge_backend=native_aom_chain_sweep` and replay error at numerical-noise level |
| AOM-Ridge superblock | `n4m.aom_ridge_superblock`, `NativeAOMRidgeSuperblockRegressor` (`aom_pop.ridge_superblock`) | Python-backed over native `aom_preprocess` + native `ridge` | tested | CUDA-build smoke tested, not fused GPU yet | catalogued donor-style strict-linear AOM Ridge superblock reference over the moment-compatible single-operator bank; concatenates strict operator views, selects/fits Ridge alpha by fold-local CV or fixed alpha through the native Ridge binding, applies train-fold centering/block RMS scaling to validation folds, and folds the final superblock coefficients back into `input_coefficients` plus `intercept` for exact replay; intentionally excludes branch/global reference-dependent, MKL/kernel and nonlinear AOM Ridge modes; `benchmarks/cross_binding/aom_ridge_superblock_timing_cuda_smoke.csv` includes function + sklearn rows on `build/cuda-on` with `ridge_backend=native` and replay error at numerical-noise level |
| AOM-Ridge active superblock | `n4m.aom_ridge_active_superblock`, `NativeAOMRidgeActiveSuperblockRegressor` (`aom_pop.ridge_active_superblock`) | Python-backed over native `aom_preprocess` + native `ridge` | tested | CUDA-build smoke tested, not fused GPU yet | catalogued donor-style strict-linear AOM Ridge active-superblock reference; screens a train-only active operator subset inside every alpha-CV fold using response signatures from the actual `aom_preprocess` outputs, screens once on full calibration rows for the final fit, then folds the active superblock coefficients back into `input_coefficients` plus `intercept`; intentionally excludes branch/global reference-dependent, MKL/kernel and nonlinear AOM Ridge modes; `benchmarks/cross_binding/aom_ridge_active_superblock_timing_cuda_smoke.csv` includes function + sklearn rows on `build/cuda-on` with `ridge_backend=native` and replay error at numerical-noise level |
| AOM-Ridge MKL-light superblock | `n4m.aom_ridge_mkl_superblock`, `NativeAOMRidgeMKLSuperblockRegressor` (`aom_pop.ridge_mkl_superblock`) | Python-backed over native `aom_preprocess` + native `ridge` | tested | CUDA-build smoke tested, not fused GPU yet | catalogued donor-style strict-linear AOM Ridge MKL-light reference; learns non-negative train-only KTA weights over strict operator blocks inside every alpha-CV fold, refits weights on full calibration rows, fits native Ridge on the equivalent weighted superblock, and folds final coefficients back into `input_coefficients` plus `intercept`; intentionally excludes branch/global reference-dependent preprocessing, row-reference-dependent preprocessing, nonlinear kernels and nonlinear AOM Ridge modes; `benchmarks/cross_binding/aom_ridge_mkl_superblock_timing_cuda_smoke.csv` includes function + sklearn rows on `build/cuda-on` with `mkl_mode=alignment`, `ridge_backend=native` and replay error at numerical-noise level |
| AOM-PLS superblock | `n4m.aom_pls_superblock`, `NativeAOMPLSSuperblockRegressor` (`aom_pop.aom_pls_superblock`) | Python-backed over native `aom_preprocess` + native `pls` | tested | CUDA-device smoke tested, not fused GPU yet | catalogued donor-style strict-linear AOM-PLS superblock reference; concatenates strict operator views, selects the PLS component count by train CV, fits through the native PLS binding and folds final superblock coefficients back into `input_coefficients` plus `intercept`; intentionally excludes row-reference-dependent preprocessing, MKL/kernel, nonlinear lifts and dataset/source routing; `benchmarks/cross_binding/aom_pls_superblock_timing_cuda_smoke.csv` includes function + sklearn rows on `build/cuda-on` with `pls_backend=native`, CUDA device PLS final-route counters and replay error at numerical-noise level |
| AOM Ridge-PLS superblock | `n4m.aom_ridge_pls_superblock`, `NativeAOMRidgePLSSuperblockRegressor` (`aom_pop.aom_ridge_pls_superblock`) | Python-backed over native `aom_preprocess` + native `ridge_pls` | tested | CUDA-build smoke tested, not fused GPU yet | catalogued donor-style strict-linear AOM Ridge-PLS superblock reference; concatenates strict operator views, selects the PLS component count and Ridge-PLS penalty by train CV, fits through the native `ridge_pls` binding, and folds final superblock coefficients back into `input_coefficients` plus `intercept`; intentionally excludes row-reference-dependent preprocessing, MKL/kernel, nonlinear lifts and dataset/source routing; `benchmarks/cross_binding/aom_ridge_pls_superblock_timing_cuda_smoke.csv` includes function + sklearn rows on `build/cuda-on` with `ridge_pls_backend=native` and replay error at numerical-noise level |
| AOM chain Ridge-PLS | `n4m.aom_chain_ridge_pls`, `NativeAOMChainRidgePLSRegressor` (`aom_pop.aom_chain_ridge_pls`) | Python-backed over native strict AOM operators + native `ridge_pls` | tested | CUDA-build smoke tested, not fused GPU yet | catalogued donor-style strict/raw-base single-chain Ridge-PLS selector; applies each strict-linear AOM chain sequentially, selects chain, PLS component count and Ridge-PLS penalty by train CV, fits through native `ridge_pls`, and folds the selected chain coefficients back into `input_coefficients` plus `intercept`; intentionally excludes SNV, MSC, EMSC, OSC, row-reference-dependent preprocessing, nonlinear lifts, kernels and dataset/source routing; `benchmarks/cross_binding/aom_chain_ridge_pls_timing_cuda_smoke.csv` includes function + sklearn rows on `build/cuda-on` with `selection_mode=chain_ridge_pls`, `ridge_pls_backend=native` and replay error at numerical-noise level |
| AOM operator PLS stack | `n4m.aom_operator_pls_stack`, `NativeAOMOperatorPLSStackRegressor`, `AOMOperatorPLSStack` | yes for native function | tested | smoke tested for native function and native sklearn wrapper; flexible Python estimator is CPU reference | native compact/wide strict-operator PLS1 score stack with Ridge head now exports folded input-space coefficients; wide has 31 strict-linear operators including Gaussian, FCK and Whittaker variants; `benchmarks/cross_binding/aom_operator_pls_stack_timing_cuda_smoke.csv` now includes `native_aom_operator_pls_stack_sklearn` rows with `prediction_replay_max_abs_error <= 1e-10`; Python estimator remains flexible custom-operator/gated reference |
| Row-additive moment substrate | `n4m.moments`, `n4m.moments_train_from_heldout` | yes | tested | tested | raw/centered `X'X`, `X'Y`, `Y'Y` plus fold subtraction |
| Moment Ridge/PLS sweep | `n4m.sweep_run` | yes | tested | tested | exact Ridge CV; `p <= n_train` via moments, `p > n_train` via reused dual kernels with optional cross-kernel scoring; wide dual Gram/cross/prediction/coefficient products now use `linalg::gemm`, which routes to cuBLAS in CUDA builds; compatible single-target NIPALS/regression PLS1 grids score from train/held-out moments, with CPU/BLAS linalg kernels, a scalar host loop in CUDA builds for medium widths, and a device-resident cuBLAS component loop plus reused fold workspace for very wide `p >= 1024` PLS1 moment CV by default; `cuda_pls_min_device_features` can lower or raise that threshold for controlled timing campaigns; `cuda_pls_parallel_folds=True` optionally schedules independent exact PLS1 moment jobs in bounded CUDA stream batches and reports batches/jobs; `cuda_pls_many_batched=True` selects the experimental tiled/strided-batched many-design CUDA path before parallel-fold scheduling unless `N4M_CUDA_PLS_MANY_LEGACY=1` forces the non-batched route; materialized prefix fallback remains for other PLS regimes |
| Direct Ridge head | `n4m.ridge`, `NativeRidgeRegressor` | yes | smoke tested | smoke tested | reusable linear/moment head with direct function and sklearn wrapper; `bench_direct_moment_heads_timing.py` records native vs wrapper fit+predict timing and replay error |
| Direct PLS head | `n4m.pls`, `NativePLSRegressor` | Python-backed over native sweep | smoke tested | CUDA-device smoke tested | reusable moment PLS head over `n4m_sweep_run`; fixed `n_components` or train-CV `pls_components` grid, replayable input-space coefficients/intercept, route diagnostics and CUDA PLS knobs shared with the sweep path; `direct_moment_heads_timing_cuda_smoke.csv` forces `cuda_pls_min_device_features=1` and `cuda_pls_parallel_folds=True` and covers all 9 direct heads with function + sklearn replay rows across 3 shapes (54 rows total); PLS rows report device CV fits equal to total PLS moment CV fits and host CV fits at zero |
| Direct PCR head | `n4m.pcr`, `NativePCRRegressor` | yes | smoke tested | smoke tested | reusable PCR/SVD head with direct `n4m_pcr_fit` MethodResult ABI, original-input coefficients, predictions, centering/scaling metadata and sklearn replay wrapper |
| CPPLS head | `n4m.cppls`, `NativeCPPLSRegressor` | yes | smoke tested | smoke tested | reusable linear/moment head with direct function and sklearn wrapper; `bench_direct_moment_heads_timing.py` records native vs wrapper fit+predict timing and replay error |
| Continuum regression head | `n4m.continuum_regression`, `NativeContinuumRegressionRegressor` | yes | smoke tested | smoke tested | reusable linear/moment head with direct function and sklearn wrapper; `bench_direct_moment_heads_timing.py` records native vs wrapper fit+predict timing and replay error |
| ECR head | `n4m.ecr`, `NativeECRRegressor` | yes | smoke tested | smoke tested | reusable linear/moment head with direct function and sklearn wrapper; `bench_direct_moment_heads_timing.py` records native vs wrapper fit+predict timing and replay error |
| Weighted PLS head | `n4m.weighted_pls`, `NativeWeightedPLSRegressor` | yes | smoke tested | smoke tested | reusable direct head; sqrt(w)-prescaled weighted PLS route; exports replayable `coefficients`, `predictions`, `x_mean` and `y_mean`; sklearn wrapper reconstructs the intercept from the native means; `bench_direct_moment_heads_timing.py` records native and wrapper timing across three shapes |
| Robust PLS head | `n4m.robust_pls`, `NativeRobustPLSRegressor` | yes | smoke tested | smoke tested | reusable direct head; Huber IRLS over weighted PLS fits; Python helper default `max_irls_iter=5`; exports replayable `coefficients`, `predictions`, `x_mean` and `y_mean`; sklearn wrapper reconstructs the intercept from the native means; `bench_direct_moment_heads_timing.py` records native and wrapper timing across three shapes |
| Ridge-augmented PLS head | `n4m.ridge_pls`, `NativeRidgePLSRegressor` | yes | smoke tested | smoke tested | reusable direct head; ridge-regularised augmented SIMPLS route; exports replayable `coefficients`, `predictions`, `x_mean` and `y_mean`; sklearn wrapper reconstructs the intercept from the native means; `bench_direct_moment_heads_timing.py` records native and wrapper timing across three shapes |
| Moment-model OOF stack | `NativeMomentStackRegressor` (`models.ensembles.moment_stack`) | Python-backed over native heads | smoke tested | CUDA-device smoke tested for PLS base | train-only OOF Ridge meta-model over native Ridge/PLS/PCR/continuum/ECR/CPPLS predictions; the PLS base now reuses the direct `NativePLSRegressor` rather than a separate generic sweep wrapper; no transformed-spectrum stacking, nonlinear lift or dataset-name routing; fitted objects expose `base_oof_diagnostics_`, `base_final_diagnostics_` and aggregate PLS route counters for OOF and final base fits, so CPU/GPU routing can be audited without retaining validation data; `benchmarks/cross_binding/moment_stack_timing_cuda_smoke.csv` proves a PLS-only stack on `build/cuda-on` used CUDA for both OOF and final PLS base sweeps (`n_base_oof_pls_moment_cuda_device_cv_fits=16`, host `0`; final device CV fits `4`, host `0`) |

## Not Yet Integrated

Grid note: `iter_aom_strict_chain_grid` now exposes the same deterministic
compact/wide/lab strict-linear bank as `build_aom_strict_chain_grid`, but with
stable chain ids, `start`/`stop` slicing and optional chunks for incremental
campaign launchers. It is an inventory/runtime helper only and does not change
candidate scores.

Benchmark note: campaign timing scripts now expose
`--backend-min-cuda-product` alongside the CUDA PLS knobs, so the source-free
CPU/CUDA launch recommendation used in reports can be reproduced or overridden
from scripted runs without changing candidate scores.

| Surface | Required shape before completion | Reason it is still open |
|---|---|---|
| Fused/batched IKPLS mode for `n4m_sweep_run` | native batch/sweep ABI for device/host-fused free component CV over many PLS variants | ABI 1.22.0 exposes `n4m_pls_cross_validate` as an exact PLS-only reference surface and tests equivalence to the PLS branch of `n4m_sweep_run`; `bench_pls_cross_validate_timing.py` now pins CPU/CUDA smoke artifacts and route counters for that reference hook; current ABI v1 already scores compatible single-target PLS1 grids from moment cross-products, i.e. an IKPLS-style sufficient-statistics route, and falls back to one max-component materialized fit per fold for other PLS regimes, but it is still not fused across many chains/folds/candidates |
| full arbitrary-chain AOM moment screen | strict-linear operator descriptors applying `A*S` and `A*C*A'` across Ridge/PLS and wide regimes without materializing transformed `X` | Banded descriptors now cover local linear operators, structured low-rank projection covers `detrend_poly`, and structured pentadiagonal solves cover Whittaker; multi-target/non-NIPALS PLS, very wide Ridge solves and fused batched GPU execution still fall back or remain open |
| CUDA fused sweep | grouped GEMM/operator kernels and backend parity vs CPU fp64 | CUDA build works; fused moment/sweep kernels are not shipped |
| WASM/WebGPU sweep | optional lite backends | design-only |

## Invariants

- `models.pls.kernel` / `kernel_pls` (non-linear kernel PLS) is **intentionally
  excluded** from the AOM/moment porting scope. The moment substrate operates on
  linear operators applied to the raw feature matrix; non-linear kernel
  functions (`K = XX'`) do not fit the strict-linear AOM/operator-moment
  architecture and would require a separate kernel-moment substrate. Kernel PLS
  remains in the catalog but outside this porting track.
- Deployable selection must not route by dataset name, source name, database
  name, dataset id, or equivalent identity fields.
- Python portfolio metadata is audit-only; tests assert metadata changes do not
  change selected route or predictions.
- `n4m.aom_ridge_blender` uses outer-fold OOF predictions to solve a
  non-negative simplex blend and is now catalogued as `aom_pop.ridge_blender`.
  The native surface is restricted to compact/wide strict-linear AOM chains and
  positive Ridge lambdas. The native result folds the weighted candidate Ridge
  coefficients back into `input_coefficients` plus `intercept`, and
  `NativeAOMRidgeBlenderRegressor` predicts from that state. `AOMRidgeBlender`
  remains the Python reference layer for explicit custom estimators. Neither
  surface is a fused GPU blender yet.
- `n4m.aom_ridge_superblock` is deliberately narrower than donor
  `AOMRidgeRegressor`: it ports the strict-linear superblock idea only. It does
  not include `branch_global`, MKL/kernel routing, row-reference-dependent
  preprocessing, nonlinear lifts or any dataset/source-name routing. It is a
  Python orchestration over native `aom_preprocess` and native `ridge`; it is
  catalogued and timed, but still not a fused GPU superblock grinder.
- `n4m.aom_ridge_active_superblock` ports the donor strict active-superblock
  idea without donor-private covariance helpers: active scores are defined on
  native `aom_preprocess` outputs, screened fold-locally during alpha CV and
  then re-screened on full calibration data for the final fit. It is an
  active-pruning model, not a screen-recall guarantee for arbitrary 200k-chain
  preprocessing campaigns.
- `n4m.aom_ridge_global` ports the donor strict global Ridge idea: select one
  strict operator and one Ridge alpha by train CV, then expose a reusable final
  model. It is intentionally narrower than the existing broad
  `NativeAOMMomentRidgeScreenRefitRegressor` campaign preset.
- `n4m.aom_pls_superblock` ports the donor strict AOM-PLS superblock idea:
  concatenate fixed strict-linear operator views and fit a PLS head whose
  coefficients are folded back to raw input space. It is not `soft` operator
  weighting, active PLS pruning, MKL/kernel weighting, or a fused GPU
  many-operator grinder.
- `n4m.aom_chain_ridge_pls` ports the donor strict/raw-base single-chain
  Ridge-PLS idea: each candidate is a sequential strict-linear chain, one PLS
  component count and one Ridge-PLS penalty scored by train CV. It deliberately
  excludes SNV, MSC, EMSC, OSC, row-reference-dependent preprocessing,
  nonlinear lifts, kernels and any dataset/source routing.
- `n4m.aom_operator_pls_stack` is catalogued as
  `aom_pop.operator_pls_stack`. It is single-target PLS1, uses fixed
  strict-linear compact/wide operator banks, and exposes the final score stack
  plus Ridge head. It also folds the selected strict-linear stack into
  `input_coefficients` plus `input_intercept`, so
  `NativeAOMOperatorPLSStackRegressor` can predict on new spectra without
  rebuilding stack features. `AOMOperatorPLSStack` remains the Python reference
  layer for custom operator matrices and optional baseline admission gates.
- The native `aom_robust_hpo` method is a compact/wide product screen. It now
  folds the selected strict-linear chain coefficients back into
  `input_coefficients`, so `X @ input_coefficients + intercept` exactly
  replays the native predictions and `NativeAOMRobustHPORegressor` can predict
  on new spectra. This is not a claim of 200k-chain fused GPU acceleration.
- `n4m.aom_sweep_run` is the configurable native strict-linear AOM sweep over
  fixed compact/wide banks. It allows custom Ridge/PLS grids and explicit folds,
  while `n4m.aom_chain_sweep_run` accepts arbitrary caller-provided
  strict-linear chains through a flat descriptor. Both expose
  `moment_policy="materialized"` to force the legacy materialized route for
  timing comparisons or production route guarding, and
  `moment_policy="force_moments"` / `"moments_only"` to fail fast whenever a
  candidate screen would need a materialized fallback. The selected chain may
  still be materialized once after scoring to populate public predictions and
  folded coefficients. Both now export `input_coefficients`, folded into the
  original feature space, so the native sklearn wrappers can predict on new
  `X` without Python-side chain replay.
- Both AOM sweep surfaces expose `score_only=True` for large ranking passes.
  It keeps the candidate table, selected ids, route counters and fold ids, but
  skips selected-model output buffers and final refits. The result also keeps
  the flat chain descriptor (`chain_offsets`, `op_kinds`, `param_offsets`,
  `chain_params`), so broad campaigns can decode the exact preprocessing chain
  behind each `chain_id` through `n4m.decode_aom_chains` or
  `n4m.aom_candidate_table`. This is an output/cost control, not a new scoring
  rule; materialized candidate-screen routes still pay their fold-local
  scoring fits because they are not batched IKPLS yet.
- `n4m.build_aom_strict_chain_grid` and `n4m.aom_chain_score_campaign` are
  Python campaign helpers over the same native strict-linear ABI. They provide
  deterministic compact/wide/lab grids, custom cartesian family templates,
  chunked `score_only=True` execution, global top-k aggregation, per-head
  top-k aggregation for mixed Ridge/PLS audits, per-score-route top-k
  aggregation for materialized/dense/banded/structured route audits, and
  optional checkpoint/resume. `pls_score_mode="gcv_proxy"` is fingerprinted
  separately from exact-CV campaigns and keeps proxy rows labelled through
  `score_metric`.
  Checkpoints are guarded by a fingerprint of the chain grid, folds,
  hyperparameters and `X/y` contents, so an interrupted long screen can resume
  without mixing incompatible candidate scores. Loaded top-k rows are filtered
  to the chunks actually present in a checkpoint before new chunks are
  appended, which keeps manual partial checkpoints coherent for global,
  per-head and per-route top-k lists. The optional
  `max_chunks_per_run` budget lets schedulers advance a campaign in bounded
  increments; reports expose `complete`, `n_remaining_chunks` and
  `processed_chunks_this_run`. They do not add a new selection heuristic and
  do not route by dataset identity.
- `split_head_scoring` controls how mixed Ridge/PLS chunks are scored. A single
  mixed native call uses *none* of the batched head-homogeneous fast paths, so
  `"off"` keeps `n_ridge_moment_score_batch_calls`,
  `n_pls_moment_score_batch_calls` and `n_pls_gcv_proxy_batch_calls` at `0`.
  `"auto"` scores each mixed chunk as a Ridge-only call plus a PLS-only call and
  merges the rows, which turns on the Ridge moment batch (`C*L*K` jobs for `C`
  chains, `L` ridge lambdas, `K` folds) and the PLS exact moment batch
  (`pls_score_mode="cv"`, `C*K` jobs) or GCV-proxy batch
  (`pls_score_mode="gcv_proxy"`, `C` jobs). The merge is score-preserving: the
  retained `(chain_id, head, param)` candidate scores and the selected winner
  are identical to `"off"`, so it changes launch shape and route counters only,
  never scores, ranking or predictions. The low-level campaign helpers
  (`aom_chain_score_campaign`, `aom_chain_screen_refit_campaign`) default to
  `"off"`; the sklearn screen/refit estimators (`NativeAOMScreenRefitRegressor`
  and the `NativeAOMMomentScreenRefitRegressor` mixed preset) default to
  `"auto"`. For single-head screens `"auto"` is inert and `n_split_head_chunks`
  stays `0`.
- `n4m.aom_moment_screen_refit_campaign` is the functional fast-profile
  counterpart to `NativeAOMMomentScreenRefitRegressor` and is exposed from both
  `n4m.aom` and `n4m.moment`. It wraps `aom_chain_screen_refit_campaign` with
  strict moment routes, prefix chunk ordering, split-head mixed scoring, PLS
  GCV-proxy screening, exact-CV refit and `refit_execution="auto"`. It adds
  `campaign_preset="moment_fast_screen_refit"` to the combined report but does
  not change the legacy defaults of the lower-level campaign helpers.
- Campaign and chunk reports expose normalized performance and route metrics
  (`candidates_per_second`, `ms_per_candidate`, route fractions, etc.) derived
  from native counters. These fields are for CPU/GPU audit and run planning;
  they do not affect candidate scores or ranking.
- `NativeAOMFixedCandidateRegressor.from_candidate(row)` is the reusable
  bridge from a decoded campaign row to a fitted model. `from_campaign(report,
  head=None|"ridge"|"pls", rank=0)` selects the global or per-head campaign
  winner directly before delegating to the same fixed-candidate path. It refits
  exactly that chain/head/parameter through `aom_chain_sweep_run` and predicts
  from folded input-space coefficients.
- `n4m.aom_evaluate_candidates` refits decoded top-k rows on an explicit train
  split and scores them on a caller-provided eval split. It reports
  `cv_rank`, `eval_rank` and `rank_delta` for CV-vs-holdout analysis, but it
  does not change the native fit or route selection.
- `n4m.aom_candidate_rank_diagnostics` summarizes screen-vs-holdout rank
  agreement from evaluated rows or reloaded reports. It reports rank
  correlation, rank drift and top-k overlap/recall. It is an audit tool for
  screen recall, not a selection rule.
- `n4m.aom_candidate_report_records` and `n4m.aom_save_candidate_report`
  serialize campaign/eval candidate rows for offline comparison and replay.
  They preserve decoded chains, add a CSV-friendly `chain_json`, and drop
  prediction arrays by default. They do not introduce a ranking heuristic or
  any dataset-identity route.
- `n4m.aom_load_candidate_report` reads JSON/JSONL/CSV candidate reports back
  into refittable rows. CSV rows restore `chain` from `chain_json`, so saved
  campaign winners can be passed directly to
  `NativeAOMFixedCandidateRegressor.from_candidate`.
- `n4m.aom_candidate_operator_summary` groups scored candidate rows by head,
  preprocessing operator, operator/head pair and chain length. It is an
  inspection aid for pruning or expanding future grids; it does not alter
  scores, top-k ordering or route selection.
- `n4m.aom_candidate_preprocessing_impact` adds stage/option/position impact
  summaries over the same scored rows, including improvement versus an identity
  baseline when such a row is present. It is an analysis surface only and does
  not introduce dataset-identity routing or a new selection rule.
- AOM sweep route counters are split by scoring route and model head:
  `n_ridge_operator_moment_candidates`, `n_pls_operator_moment_candidates`,
  `n_ridge_materialized_candidates`, and `n_pls_materialized_candidates`.
  This makes broad campaigns auditable without inferring route provenance from
  candidate table rows.
- AOM sweep result rows also include `candidate_routes`, with route codes
  `0=materialized`, `1=dense_operator_moment`,
  `2=banded_operator_moment`, and `3=structured_operator_moment`. Python
  decoded candidate rows expose `score_route_id` and `score_route`, and
  operator summaries include `by_score_route`. These fields are diagnostics
  only and do not change candidate scores or rank order.
- PLS sweep fit counters are split by scoring route and by CV/final phase:
  `n_pls_moment_cv_fits`, `n_pls_materialized_cv_fits`,
  `n_pls_moment_final_fits`, and `n_pls_materialized_final_fits`. Campaign
  reports also expose `pls_cv_fits_per_chain` and
  `pls_cv_fits_per_candidate`. These fields are diagnostics only; they do not
  affect scores, ranking, route policy or dataset selection.
- `n4m_moments_*` is now a reusable raw/centered sufficient-statistics layer;
  `n4m_sweep_run` uses moment fold subtraction where it is efficient,
  precomputed dual Ridge and optional held-out/train cross-kernel scoring when
  `p > n_train`, and train/held-out moment scoring for compatible
  single-target NIPALS/regression PLS1 component grids. The wide dual Ridge
  train Gram, held-out cross-kernel, held-out prediction and coefficient
  reconstruction products use the shared `linalg::gemm` dispatch, so CUDA
  builds execute those products through cuBLAS. Compatible PLS1 moment-prefix
  dense products use `linalg` in CPU/BLAS builds, but keep the scalar host-side
  path in CUDA builds because the current cuBLAS dispatch copies each
  micro-kernel host/device and benchmarked slower; this is an explicit guard
  until a device-resident batched IKPLS workspace exists. Other PLS regimes
  keep materialized prefix scoring with coefficient-prefix reuse across
  component counts.
- The CUDA PLS many-job dispatcher has an experimental opt-in tiled
  strided-batched cuBLAS path (`N4M_CUDA_PLS_MANY_BATCHED=1`) that preserves
  exact scores and falls back to the legacy sequential-many workspace path, but
  smoke timing did not justify making it the default. The production gap is
  still device-resident scalar reductions/glue kernels or a fused IKPLS-style
  screen.
- AOM Ridge candidate rows with `p <= n_train` transform strict-linear chain
  operators into moment space and score held-out SSE from moments, including
  inside mixed Ridge+PLS sweeps. Dense medium-wide Ridge grids use this exact
  path when all lambdas are strictly positive and `p <= 48`; local linear
  banded operators extend the route to `p <= 256` for Ridge and `p <= 1024`
  for compatible single-target NIPALS PLS1. `detrend_poly` chains use an exact
  structured low-rank projection transform and Whittaker chains use the exact
  structured pentadiagonal solve for `(I + lambda D2'D2)^-1`; both can compose
  with the local banded operators. CPU `auto` routes Ridge `p > n_train` rows
  through the exact materialized dual-Ridge scorer and PLS rows with
  `min_train < 4p` through the exact materialized PLS prefix scorer; CUDA
  `auto` keeps the operator-moment route in those cells. Feature caps and
  backend route choices are compute-route guards, not scoring heuristics.
- `moment_policy="force_moments"` is the strict native screen mode: it accepts
  only chains/heads whose candidate scores are computed through operator
  moments and returns `UNSUPPORTED` instead of silently leaving the moment
  substrate. This does not change the post-selection materialization used to
  expose predictions and `input_coefficients`.
- The full moment-engine goal remains open until batched IKPLS, operator
  descriptors for the remaining regimes, and fused CUDA sweep/parity evidence
  exist.
