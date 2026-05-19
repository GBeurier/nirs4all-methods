# Methods catalogue

Every algorithm in `benchmarks.parity_timing.registry.METHODS` documented with its parameters, bibliographic source, math principle, every binding's signature, and its parity + timing rows.

_Total methods_: **71**. Grouped by family below.

```{toctree}
:hidden:
:glob:
:maxdepth: 1

*
```

## Core PLS

| Method | Description | Tolerance | Refs |
|--------|-------------|-----------|------|
| [`cppls`](cppls.md) | CPPLS (column-power-rescaled SIMPLS) | `10.0` | R |
| [`opls`](opls.md) | Orthogonal PLS (Trygg & Wold 2002) | `0.001` | R |
| [`pcr`](pcr.md) | Principal Components Regression | `1e-06` | Py, R |
| [`pls`](pls.md) | SIMPLS PLS regression baseline | `0.1` | Py, R, ikpls, mixOmics |
| [`recursive_pls`](recursive_pls.md) | Recursive (moving-window) PLS | `0.1` | Py, R |

## Sparse

| Method | Description | Tolerance | Refs |
|--------|-------------|-----------|------|
| [`fused_sparse_pls`](fused_sparse_pls.md) | Fused sparse PLS (§7) | `0.05` | — |
| [`group_sparse_pls`](group_sparse_pls.md) | Group sparse PLS (§7) | `10.0` | R |
| [`sparse_pls_da`](sparse_pls_da.md) | Sparse PLS-DA (§7) | `2.0` | R |
| [`sparse_simpls`](sparse_simpls.md) | Sparse SIMPLS with soft-threshold lambda | `1.0` | R |

## Ensemble

| Method | Description | Tolerance | Refs |
|--------|-------------|-----------|------|
| [`bagging_pls`](bagging_pls.md) | Bagging PLS (§20) | `2.0` | Py |
| [`boosting_pls`](boosting_pls.md) | Boosting PLS (§20) | `10.0` | R |
| [`random_subspace_pls`](random_subspace_pls.md) | Random-subspace PLS (§20) | `2.0` | Py |

## Robust / weighted

| Method | Description | Tolerance | Refs |
|--------|-------------|-----------|------|
| [`robust_pls`](robust_pls.md) | Robust PLS (Huber IRLS over weighted SIMPLS) | `2.0` | R |
| [`weighted_pls`](weighted_pls.md) | Sample-weighted PLS (sqrt(w)-prescaled SIMPLS) | `0.1` | Py |

## Nonlinear / local

| Method | Description | Tolerance | Refs |
|--------|-------------|-----------|------|
| [`continuum_regression`](continuum_regression.md) | Continuum regression (interpolates PLS / OLS) | `0.2` | R |
| [`gpr_pls`](gpr_pls.md) | GPR-on-PLS — RBF Gaussian Process on PLS scores (§47) | `1e-08` | Py |
| [`kernel_pls_rbf`](kernel_pls_rbf.md) | Non-linear kernel PLS (RBF kernel) | `2.0` | R |
| [`lw_pls`](lw_pls.md) | LW-PLS — Locally-weighted PLS (§17 Phase 4) | `5.0` | Py |

## Multi-block / cross-modal

| Method | Description | Tolerance | Refs |
|--------|-------------|-----------|------|
| [`mb_pls`](mb_pls.md) | MB-PLS — Multi-block PLS (§17 Phase 4) | `2.0` | Py |
| [`mir_pls`](mir_pls.md) | MIR-PLS — Inverse-regression PLS (§13) | `0.05` | — |
| [`n_pls`](n_pls.md) | N-PLS — 3-way tensor PLS (Bro 1996) | `10.0` | Py |
| [`o2pls`](o2pls.md) | O2PLS — bi-directional OPLS (Trygg & Wold 2003) | `1.0` | R |
| [`on_pls`](on_pls.md) | OnPLS — Orthogonal multi-block decomposition (§18) | `10.0` | Py |
| [`rosa`](rosa.md) | ROSA — Response-Oriented Sequential Alternation (§19) | `1.0` | R |
| [`so_pls`](so_pls.md) | SO-PLS — Sequential & Orthogonalized multi-block PLS (§17) | `1.0` | R |

## Calibration transfer

| Method | Description | Tolerance | Refs |
|--------|-------------|-----------|------|
| [`di_pls`](di_pls.md) | Domain-invariant PLS | `0.02` | Py |
| [`ds`](ds.md) | DS — Direct Standardization (§13) | `0.5` | R |
| [`ecr`](ecr.md) | Elastic Component Regression (Phase 50) | `0.001` | Py |
| [`pds`](pds.md) | PDS — Piecewise Direct Standardization (§13) | `0.5` | R |

## Classification & GLM

| Method | Description | Tolerance | Refs |
|--------|-------------|-----------|------|
| [`pls_cox`](pls_cox.md) | PLS-Cox (§5) — Cox PH on PLS scores | `5.0` | R |
| [`pls_glm`](pls_glm.md) | PLS-GLM (§5) — softmax/Poisson IRLS on PLS scores | `0.5` | R |
| [`pls_lda`](pls_lda.md) | PLS-LDA — LDA on PLS scores (§17 Phase 4) | `5.0` | Py |
| [`pls_logistic`](pls_logistic.md) | PLS-Logistic — Logistic regression on PLS scores | `5.0` | Py |
| [`pls_qda`](pls_qda.md) | PLS-QDA (§5) — quadratic discriminant on PLS scores | `10.0` | Py |

## Missing data

| Method | Description | Tolerance | Refs |
|--------|-------------|-----------|------|
| [`missing_aware_nipals`](missing_aware_nipals.md) | Missing-aware NIPALS PLS (§13) | `10.0` | R |

## Regularised

| Method | Description | Tolerance | Refs |
|--------|-------------|-----------|------|
| [`ridge_pls`](ridge_pls.md) | Ridge-augmented PLS | `0.1` | Py |

## Diagnostic

| Method | Description | Tolerance | Refs |
|--------|-------------|-----------|------|
| [`aom_preprocess`](aom_preprocess.md) | AOM preprocessing pipeline (§17 Phase 4) | `5.0` | Py |
| [`approximate_press`](approximate_press.md) | Approximate-PRESS component selection (§29) | `10.0` | R |
| [`one_se_rule`](one_se_rule.md) | One-SE component selection rule (§10) | `10.0` | R |
| [`pls_diagnostic_dmodx`](pls_diagnostic_dmodx.md) | PLS Distance-to-Model X (§9) | `5.0` | R |
| [`pls_diagnostic_q`](pls_diagnostic_q.md) | PLS Q residuals / SPE (§9) | `5.0` | R |
| [`pls_diagnostic_t2`](pls_diagnostic_t2.md) | PLS Hotelling T² (§9) | `10.0` | R |
| [`pls_monitoring`](pls_monitoring.md) | PLS process monitoring (T²/Q thresholds + alarms) (§19) | `10.0` | R |

## Variable selector

| Method | Description | Tolerance | Refs |
|--------|-------------|-----------|------|
| [`bipls_select`](bipls_select.md) | biPLS backward interval elimination (§18 Phase 5p) | `0.7` | R |
| [`bve_select`](bve_select.md) | Backward Variable Elimination (§18 Phase 5k) | `1.4` | R |
| [`cars_select`](cars_select.md) | CARS competitive adaptive reweighted sampling | `1.4` | R |
| [`emcuve_select`](emcuve_select.md) | EMCUVE ensemble MC-UVE (§18 Phase 5n) | `1.6` | R |
| [`ga_select`](ga_select.md) | GA-PLS genetic algorithm selection | `1.3` | R |
| [`interval_select`](interval_select.md) | Interval/iPLS forward selection (§18 Phase 5b) | `1.0` | R |
| [`ipw_select`](ipw_select.md) | IPW-PLS iterative predictor weighting (§18 Phase 5t) | `1.0` | R |
| [`irf_select`](irf_select.md) | Interval Random Frog (Phase 52) | `1.3` | Py |
| [`iriv_select`](iriv_select.md) | Iteratively Retains Informative Variables (Phase 51) | `1.0` | Py |
| [`pso_select`](pso_select.md) | PSO-PLS — Binary Particle Swarm variable selection (§48) | `1.4` | Py |
| [`random_frog_select`](random_frog_select.md) | Random Frog selection (§18 Phase 5g) | `1.35` | Py |
| [`randomization_select`](randomization_select.md) | Randomization-test selector (§18 Phase 5o) | `1.3` | R |
| [`rep_select`](rep_select.md) | REP-PLS repeated VIP selection (§18 Phase 5s) | `1.8` | R |
| [`scars_select`](scars_select.md) | SCARS stability + CARS (§18 Phase 5h) | `1.0` | — |
| [`shaving_select`](shaving_select.md) | Shaving iterative variable trimming | `0.8` | R |
| [`sipls_select`](sipls_select.md) | siPLS synergistic interval selection (§18 Phase 5q) | `0.7` | — |
| [`spa_select`](spa_select.md) | SPA Successive Projections (§18 Phase 5e) | `1.2` | R |
| [`st_select`](st_select.md) | ST-PLS soft-thresholded sparse PLS (§18 Phase 5u) | `2.1` | R |
| [`stability_select`](stability_select.md) | Stability/MCUVE selection (§18 Phase 5c) | `1.35` | R |
| [`t2_select`](t2_select.md) | T²-PLS loading-weight selection (§18 Phase 5l) | `1.2` | R |
| [`uve_select`](uve_select.md) | UVE noise-thresholded selection (§18 Phase 5d) | `0.72` | R |
| [`variable_select_coef`](variable_select_coef.md) | \|Coef\| top-k selection (§18 Phase 5a, method=1) | `1.1` | R |
| [`variable_select_sr`](variable_select_sr.md) | Selectivity-Ratio top-k (§18 Phase 5a, method=2) | `1.3` | R |
| [`variable_select_vip`](variable_select_vip.md) | VIP top-k variable selection (§18 Phase 5a, method=0) | `0.8` | R |
| [`vip_spa_select`](vip_spa_select.md) | VIP_SPA — VIP-mask then SPA greedy (Phase 53) | `1.3` | Py |
| [`vissa_select`](vissa_select.md) | VISSA-PLS — Variable Iterative Space Shrinkage (§49) | `2.5` | Py |
| [`wvc_select`](wvc_select.md) | WVC weighted-variable-component top-k | `0.7` | R |
| [`wvc_threshold_select`](wvc_threshold_select.md) | WVC threshold-based selection (§18 Phase 5r) | `0.7` | R |

---

See the [benchmark overview](../benchmarks/overview.md) for how parity and timing are measured, and the [GitHub Pages dashboard](../landing/dashboard.md) for an interactive cross-method comparison.