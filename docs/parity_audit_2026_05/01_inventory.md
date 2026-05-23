# Parity audit — Step 1: implemented-method inventory

Date: 2026-05-16
Scope: every numerical method implemented in the C++ core that should be
externally parity-checked. Built from `cpp/include/pls4all/p4a.h`,
`cpp/src/core/*.{cpp,hpp}`, `bindings/python/src/pls4all/_methods.py`,
and `cpp/tests/abi/test_*`.

Legend for **Reach** column:
- **public-fit**: dedicated `n4m_<name>_fit/_run/_compute` ABI symbol,
  exposed in Python via `pls4all._methods`.
- **dispatch**: routed through `n4m_model_fit` by setting
  `Algorithm × Solver × Deflation`. Reachable from Python via
  `pls4all.Model.fit(ctx, cfg, X, Y)`.
- **pipeline**: routed through `n4m_pipeline_fit / _transform` with an
  `OperatorKind` enum.
- **aom-public**: routed through `n4m_aom_global_select` /
  `n4m_aom_per_component_select`.
- **internal-only**: implemented in C++ but not exposed via the C ABI;
  currently only reachable from `cpp/tests/abi/*`. Needs ABI exposure
  before it can be parity-gated by the Python runner.

Legend for **Registry** column:
- **YES**: already a `MethodSpec` in `benchmarks/parity_timing/registry.py`
- **MISSING**: not in registry.

---

## 1. Core PLS regression algorithms (N4M_ALGO_PLS_REGRESSION)

| # | Method | Solver enum | Deflation | Reach | Registry | Notes |
|---|--------|-------------|-----------|-------|----------|-------|
| 1 | NIPALS PLS1/PLS2 | `NIPALS` | `REGRESSION` | dispatch | MISSING | Reference algorithm. |
| 2 | Orthogonal-scores PLS | `ORTHOGONAL_SCORES` | `REGRESSION` | dispatch | MISSING | Martens orthogonal scores. |
| 3 | SIMPLS | `SIMPLS` | `REGRESSION` | dispatch | MISSING | de Jong 1993. |
| 4 | Kernel-algorithm PLS (linear) | `KERNEL_ALGORITHM` | `REGRESSION` | dispatch | MISSING | Lindgren/Wold linear kernel-PLS. |
| 5 | Wide-kernel PLS | `WIDE_KERNEL` | `REGRESSION` | dispatch | MISSING | Rännar 1994 widekernel. |
| 6 | SVD-based PLS | `SVD` | `REGRESSION` | dispatch | MISSING | Direct SVD on covariance. |
| 7 | Power-iteration PLS | `POWER` | `REGRESSION` | dispatch | MISSING | Power method on covariance. |
| 8 | Randomized SVD PLS | `RANDOMIZED_SVD` | `REGRESSION` | dispatch | MISSING | Halko/Martinsson 2011. |

## 2. Other core algorithms

| # | Method | Algorithm enum | Solver | Deflation | Reach | Registry | Notes |
|---|--------|----------------|--------|-----------|-------|----------|-------|
| 9  | PLS-Canonical (NIPALS) | `PLS_CANONICAL` | `NIPALS` | `CANONICAL` | dispatch | MISSING | Mode A symmetric. |
| 10 | PLS-Canonical (SVD)    | `PLS_CANONICAL` | `SVD`    | `CANONICAL` | dispatch | MISSING | SVD variant. |
| 11 | PLS-SVD                | `PLS_SVD`       | `SVD`    | `CANONICAL` | dispatch | MISSING | Cross-cov SVD scores only. |
| 12 | PLS-DA                 | `PLS_DA`        | any-PLS  | `REGRESSION`| dispatch | MISSING | Dummy-coded Y. |
| 13 | OPLS                   | `OPLS`          | `NIPALS` | `ORTHOGONAL`| dispatch | MISSING | Trygg & Wold 2002. |
| 14 | OPLS-DA                | `OPLS_DA`       | `NIPALS` | `ORTHOGONAL`| dispatch | MISSING | One predictive + ortho. |
| 15 | OPLS-DA (multiclass)   | `OPLS_DA`       | `NIPALS` | `ORTHOGONAL`| dispatch | MISSING | Phase 4m. |
| 16 | PCR                    | `PCR`           | `SVD`    | `REGRESSION`| dispatch | MISSING | PCA + OLS. |
| 17 | MB-PLS                 | `MB_PLS` enum exists, NOT dispatched | n/a | n/a | **internal-only** | MISSING | `cpp/src/core/mb_pls.{cpp,hpp}` exists; `n4m_model_fit` rejects `N4M_ALGO_MB_PLS` and there is no `n4m_mb_pls_fit` symbol. |
| 18 | LW-PLS                 | `LW_PLS` enum exists, NOT dispatched | n/a | n/a | **internal-only** | MISSING | Same situation: `cpp/src/core/lw_pls.{cpp,hpp}` exists; no public ABI wrapper. |
| 19 | PLS-LDA                | derived from PLS_DA | n/a | n/a | **internal-only** | MISSING | Phase 4p; needs ABI exposure. |
| 20 | PLS-Logistic           | derived from PLS_DA | n/a | n/a | **internal-only** | MISSING | Phase 4q; needs ABI exposure. |

> **NOTE (codex review)**: `n4m_model_fit` actually only dispatches `PLS_REGRESSION`, `PLS_CANONICAL`, `PLS_SVD`, `PLS_DA`, `OPLS`, `OPLS_DA`, `PCR`. It rejects `MB_PLS`, `LW_PLS`, `AOM_PLS`, `SPARSE_PLS` even though those enums exist. So MB-PLS and LW-PLS are **not** reachable from any public ABI today; their C++ implementations are exercised only by `cpp/tests/abi/test_mb_pls.cpp` and `cpp/tests/abi/test_lw_pls.cpp`. They join PLS-LDA / PLS-Logistic in the "needs ABI shim" bucket.

## 3. Existing extra-PLS public-fit entries

These ALREADY have a `MethodSpec` in the registry. Listed for completeness.

| # | Method | ABI symbol | Registry name |
|---|--------|------------|---------------|
| 21 | sparse SIMPLS | `n4m_sparse_simpls_fit` | `sparse_simpls` |
| 22 | DI-PLS | `n4m_di_pls_fit` | `di_pls` |
| 23 | Recursive PLS | `n4m_recursive_pls_run` | `recursive_pls` |
| 24 | CPPLS | `n4m_cppls_fit` | `cppls` |
| 25 | Weighted PLS | `n4m_weighted_pls_fit` | `weighted_pls` |
| 26 | Robust PLS | `n4m_robust_pls_fit` | `robust_pls` |
| 27 | Ridge PLS | `n4m_ridge_pls_fit` | `ridge_pls` |
| 28 | Continuum regression | `n4m_continuum_regression_fit` | `continuum_regression` |
| 29 | N-PLS | `n4m_n_pls_fit` | `n_pls` |
| 30 | Kernel PLS (RBF) | `n4m_kernel_pls_fit` | `kernel_pls_rbf` |
| 31 | O2PLS | `n4m_o2pls_fit` | `o2pls` |
| 32 | Approximate-PRESS | `n4m_approximate_press_compute` | `approximate_press` |
| 33 | Sparse PLS-DA | `n4m_sparse_pls_da_fit` | `sparse_pls_da` |
| 34 | Group sparse PLS | `n4m_group_sparse_pls_fit` | `group_sparse_pls` |
| 35 | Fused sparse PLS | `n4m_fused_sparse_pls_fit` | `fused_sparse_pls` |
| 36 | PLS monitoring | `n4m_pls_monitoring_run` | `pls_monitoring` |
| 37 | One-SE rule | `n4m_one_se_rule_compute` | `one_se_rule` |
| 38 | SO-PLS | `n4m_so_pls_fit` | `so_pls` |
| 39 | OnPLS | `n4m_on_pls_fit` | `on_pls` |
| 40 | ROSA | `n4m_rosa_fit` | `rosa` |
| 41 | Bagging PLS | `n4m_bagging_pls_fit` | `bagging_pls` |
| 42 | Boosting PLS | `n4m_boosting_pls_fit` | `boosting_pls` |
| 43 | Random-subspace PLS | `n4m_random_subspace_pls_fit` | `random_subspace_pls` |
| 44 | PLS-GLM | `n4m_pls_glm_fit` | `pls_glm` |
| 45 | PLS-QDA | `n4m_pls_qda_fit` | `pls_qda` |
| 46 | PLS-Cox | `n4m_pls_cox_fit` | `pls_cox` |
| 47 | PDS | `n4m_pds_fit` | `pds` |
| 48 | DS  | `n4m_ds_fit`  | `ds` |
| 49 | MIR-PLS | `n4m_mir_pls_fit` | `mir_pls` |
| 50 | Missing-aware NIPALS | `n4m_missing_aware_nipals_fit` | `missing_aware_nipals` |
| 51 | PLS Hotelling T² | `n4m_pls_diagnostics_compute` | `pls_diagnostic_t2` |
| 52 | PLS Q residuals  | `n4m_pls_diagnostics_compute` | `pls_diagnostic_q` |
| 53 | PLS DModX | `n4m_pls_diagnostics_compute` | `pls_diagnostic_dmodx` |

## 4. AOM / POP public ABI

| # | Method | ABI symbol | Registry | Notes |
|---|--------|------------|----------|-------|
| 54 | Global AOM-SIMPLS selection | `n4m_aom_global_select` | MISSING | Phase 6f. Bench oracle: `nirs4all/bench/AOM_v0/aompls`. |
| 55 | Per-component POP-PLS selection | `n4m_aom_per_component_select` | MISSING | Phase 6f. Same bench oracle. |
| 56 | AOM preprocessing fit/transform | **not on public ABI** | MISSING | Phase 6a. `aom_preprocessing.{cpp,hpp}` is reachable only via the AOM selectors (`aom_global_select` / `aom_per_component_select`) which use it internally. There is no `n4m_aom_preprocess_fit` ABI symbol. |

## 5. Preprocessing operators (N4M_OP_*)

Reach: pipeline (via `n4m_pipeline_fit/_transform`).

| # | Operator | Enum | Registry | Notes |
|---|----------|------|----------|-------|
| 57 | Identity | `IDENTITY` | MISSING | Trivial — round-trip check. |
| 58 | Center | `CENTER` | MISSING | Column-mean removal. |
| 59 | Autoscale | `AUTOSCALE` | MISSING | Z-score. |
| 60 | Pareto scaling | `PARETO_SCALE` | MISSING | sqrt-σ scaling. |
| 61 | SNV | `SNV` | MISSING | Standard normal variate. |
| 62 | MSC | `MSC` | MISSING | Multiplicative scatter correction. |
| 63 | EMSC | `EMSC` | MISSING | Extended MSC. |
| 64 | Detrend (poly) | `DETREND_POLY` | MISSING | Polynomial detrending. |
| 65 | Savitzky-Golay smoothing | `SAVGOL_SMOOTH` | MISSING | SG smoother. |
| 66 | Savitzky-Golay derivative | `SAVGOL_DERIVATIVE` | MISSING | SG derivative. |
| 67 | Norris-Williams | `NORRIS_WILLIAMS` | MISSING | Gap derivative. |
| 68 | AsLS baseline | `ASLS_BASELINE` | MISSING | Eilers asymmetric LS. |
| 69 | OSC | `OSC` | MISSING | Wold OSC. |
| 70 | EPO | `EPO` | MISSING | External-parameter orthogonalization. |
| 71 | Wavelet denoise | `WAVELET_DENOISE` | MISSING | Multi-level. Pipeline reach. |
| 72 | Finite difference | `FINITE_DIFFERENCE` | MISSING | **AOM-only**: not in `pipeline_fit/transform` switch; reachable only as an `OperatorBank` entry inside `aom_global_select` / `aom_per_component_select`. |
| 73 | Whittaker | `WHITTAKER` | MISSING | **AOM-only**, same as §72. |
| 74 | FCK | `FCK` | MISSING | **AOM-only**, same as §72. Fractional convolution kernel. |

## 6. Variable selection methods (Phase 5*)

All have a C++ implementation under `cpp/src/core/<name>_selection.{cpp,hpp}`
and a regression C++ test, but **none are exposed via the C ABI**. They are
NOT reachable from the Python parity runner today. Needed: either ABI
exposure (preferred — would let Python adapters run them directly) or a
cpp-test-driven harness that reads expected outputs from a fixture file
and writes a "selector parity" report. Adding ABI exposure is non-trivial
(23 selectors × output-struct accessor surface) and is its own roadmap
item. For Step 1 we record them all; Step 4 will decide which path to
take per selector.

| # | Method | Phase | Source file | Tested in cpp/tests/abi |
|---|--------|-------|-------------|-------------------------|
| 75 | VIP ranker | 5a | `variable_importance.cpp` | `test_variable_importance.cpp` |
| 76 | Coefficient-magnitude ranker | 5a | `variable_selection.cpp` | `test_variable_selection.cpp` |
| 77 | Selectivity-ratio ranker | 5a | `variable_importance.cpp` | `test_variable_importance.cpp` |
| 78 | Interval / iPLS / mwPLS scan | 5b | `interval_selection.cpp` | `test_interval_selection.cpp` |
| 79 | MCUVE / stability | 5c | `stability_selection.cpp` | `test_stability_selection.cpp` |
| 80 | UVE | 5d | `uve_selection.cpp` | `test_uve_selection.cpp` |
| 81 | SPA | 5e | `spa_selection.cpp` | `test_spa_selection.cpp` |
| 82 | CARS | 5f | `cars_selection.cpp` | `test_cars_selection.cpp` |
| 83 | Random Frog | 5g | `random_frog_selection.cpp` | `test_random_frog_selection.cpp` |
| 84 | SCARS | 5h | `scars_selection.cpp` | `test_scars_selection.cpp` |
| 85 | GA-PLS | 5i | `ga_selection.cpp` | `test_ga_selection.cpp` |
| 86 | Shaving | 5j | `shaving_selection.cpp` | `test_shaving_selection.cpp` |
| 87 | BVE-PLS | 5k | `bve_selection.cpp` | `test_bve_selection.cpp` |
| 88 | T2-PLS | 5l | `t2_selection.cpp` | `test_t2_selection.cpp` |
| 89 | WVC-PLS | 5m | `wvc_selection.cpp` | `test_wvc_selection.cpp` |
| 90 | EMCUVE | 5n | `emcuve_selection.cpp` | `test_emcuve_selection.cpp` |
| 91 | Randomization-test | 5o | `randomization_selection.cpp` | `test_randomization_selection.cpp` |
| 92 | biPLS | 5p | `bipls_selection.cpp` | `test_bipls_selection.cpp` |
| 93 | siPLS | 5q | `sipls_selection.cpp` | `test_sipls_selection.cpp` |
| 94 | WVC-threshold | 5r | `wvc_selection.cpp` | `test_wvc_selection.cpp` |
| 95 | REP-PLS | 5s | `rep_selection.cpp` | `test_rep_selection.cpp` |
| 96 | IPW-PLS | 5t | `ipw_selection.cpp` | `test_ipw_selection.cpp` |
| 97 | ST-PLS | 5u | `st_selection.cpp` | `test_st_selection.cpp` |

## 7. Validation / metrics (utility kernels — explicitly excluded)

These are orthogonal to "models". Already covered by the Phase 3 test
suite against NumPy / scikit-learn for metrics, fold plans, etc. We
include them only for completeness — they are **not** new parity-gate
candidates and the runner deliberately ignores them.

| # | Group | Notes |
|---|-------|-------|
| 98 | Regression metrics (RMSE, MAE, R², bias, RPD, RPIQ) | Already parity-checked vs sklearn/numpy in cpp tests. |
| 99 | Classification metrics (binary + multiclass + AUC + calibration) | Ditto. |
| 100 | Cross-validation engine + split plans (k-fold, repeated, MC, KS, SPXY, leave-one-out, holdout) | Ditto. |
| – | `component_coefficients_compute` | Internal helper that materialises `B(a)` for component prefixes; not a model. |
| – | `cross_validate_component_prefixes` (`model_selection.cpp`) | Internal CV scoring used by AOM/POP. Tested against fixtures. |
| – | `aom_operators` (FCK / Whittaker / FD math) | Already covered by §5 entries 72-74 (AOM-only reach). |

---

## Summary

| Category | Implemented | Reachable from Python today | In registry | Action for Step 4 |
|----------|-------------|------------------------------|-------------|-------------------|
| Core PLS regression solvers (§1, items 1-8) | 8 | 8 (via `Model.fit` + Algorithm/Solver) | 0 | wire 8 new MethodSpecs |
| Other core algos (§2 items 9-16) — PLSCanonical, PLSSVD, PLS-DA, OPLS, OPLS-DA, multiclass OPLS-DA, PCR | 8 | 8 (via `Model.fit`) | 0 | wire 8 new MethodSpecs |
| Other core algos (§2 items 17-20) — MB-PLS, LW-PLS, PLS-LDA, PLS-Logistic | 4 | **0 — ABI gap** | 0 | needs 4 ABI shims (`n4m_mb_pls_fit`, `n4m_lw_pls_fit`, `n4m_pls_lda_fit`, `n4m_pls_logistic_fit`) |
| Extra-PLS public-fit (§3) | 33 | 33 | 33 | upgrade refs (drop paper-only where possible) |
| AOM / POP public ABI (§4 items 54-55) | 2 | 2 | 0 | wire 2 new MethodSpecs (bench oracle) |
| AOM preprocessing public (§4 item 56) | 1 | **0 — ABI gap** | 0 | needs `n4m_aom_preprocess_fit` shim |
| Pipeline preprocessing operators (§5 items 57-71) | 15 | 15 (via `pipeline_fit/transform`) | 0 | wire 15 new MethodSpecs |
| AOM-only operators (§5 items 72-74) — FCK, Whittaker, finite-diff | 3 | reachable only via the `OperatorBank` inside AOM selectors | 0 | wire via bench oracle |
| Variable selectors (§6 items 75-97) | 23 | **0 — ABI gap** | 0 | needs ABI shims OR cpp-test harness (Step 4 decision) |
| **TOTAL** | **97 numerical methods** | **66 reachable + 3 AOM-only** | **33 in registry** | — |

So 33 of 97 implemented numerical methods are parity-gated today —
**~34% coverage**. **66 of 97 are reachable from Python without any
ABI changes**; the other 28 (4 + 1 + 23) require ABI work or a
non-Python harness, and 3 AOM-only operators need bench-oracle plumbing.

ABI exposure work that gates the registry-coverage ceiling, in priority
order:

1. **§4 item 56** — `n4m_aom_preprocess_fit` (small; parity already
   defined against bench oracle).
2. **§2 items 17-20** — MB-PLS, LW-PLS, PLS-LDA, PLS-Logistic
   public-fit symbols (4 new ABI entries).
3. **§6 items 75-97** — 23 selector public symbols (large; defer to
   follow-up phase, OR use a cpp-test fixture-driven harness as an
   intermediate measure).

Step 4 wires the **66 directly reachable methods + 3 AOM-only via
bench oracle = 69 MethodSpecs**; the remaining 28 (4+1+23) are tracked
as ABI-blocked and will land in a follow-up phase.

---

## Audit-end update (2026-05-16)

After Phases 47-53 the actual gate covers **68 methods, 132 reference
rows, 64 `ok` methods, 4 `paper_only`** (`fused_sparse_pls`, `mir_pls`,
`scars_select`, `sipls_select`). All 23 selector ABI shims listed above
were delivered (Option A). Methods *added* during the audit that did
not appear in the original 97-method inventory:

| Phase | Method | Source |
|------:|--------|--------|
| 47 | `gpr_pls` | new pls4all kernel (GPR-on-PLS) |
| 48 | `pso_select` | new pls4all kernel (Binary PSO) |
| 49 | `vissa_select` | new pls4all kernel (Deng VISSA) |
| 50 | `ecr` | new pls4all kernel ported from libPLS `ecr.m` |
| 51 | `iriv_select` | new pls4all kernel ported from libPLS `iriv.m` |
| 52 | `irf_select` | new pls4all kernel ported from libPLS `irf.m` |
| 53 | `vip_spa_select` | new pls4all kernel mirroring `auswahl.VIP_SPA` |

See [`03_closeout.md`](03_closeout.md) for the full progression.
