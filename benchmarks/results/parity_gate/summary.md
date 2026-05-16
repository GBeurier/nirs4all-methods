# Parity gate report

Host: `Linux-6.6.114.1-microsoft-standard-WSL2-x86_64-with-glibc2.35`
pls4all: `0.97.0+abi.1.14.0`
Python: `3.11.15`
NumPy: `2.4.5`

## Pass quality breakdown

| Quality | Count | Definition |
|---------|------:|------------|
| tight    | 34 | `rmse_rel < 30% of tolerance` — high-confidence parity. |
| moderate | 10 | 30-60% of tolerance — real agreement. |
| loose    | 4 | 60-90% of tolerance — algorithmic divergence likely; passes with margin. |
| **weak** | **12** | **>= 90% of tolerance** — passes barely; tolerance widened to accept stochastic-RNG or algorithmic-convention divergence. **Treat as smoke check, not bit parity.** |

### Weak-parity passes (read with caution)

These pass the gate but rely on widened tolerances. Their external reference and the pls4all implementation use the same algorithm family but differ in RNG, convention or parameter mapping; do not read a green check as bit-exact agreement.

| Method | Reference | rmse_rel | tolerance | margin |
|--------|-----------|---------:|----------:|-------:|
| `variable_select_vip` | `plsVarSel` | 6.325e-01 | 7.00e-01 | 9.6% |
| `variable_select_coef` | `pls` | 1.000e+00 | 1.10e+00 | 9.1% |
| `variable_select_sr` | `plsVarSel` | 1.183e+00 | 1.30e+00 | 9.0% |
| `interval_select` | `mdatools` | 9.129e-01 | 1.00e+00 | 8.7% |
| `stability_select` | `plsVarSel` | 1.265e+00 | 1.35e+00 | 6.3% |
| `cars_select` | `enpls` | 1.291e+00 | 1.40e+00 | 7.8% |
| `random_frog_select` | `libPLS` | 1.265e+00 | 1.35e+00 | 6.3% |
| `ga_select` | `plsVarSel` | 1.195e+00 | 1.30e+00 | 8.1% |
| `bve_select` | `plsVarSel` | 1.291e+00 | 1.40e+00 | 7.8% |
| `t2_select` | `plsVarSel` | 1.155e+00 | 1.20e+00 | 3.8% |
| `rep_select` | `plsVarSel` | 1.700e+00 | 1.80e+00 | 5.6% |
| `st_select` | `plsVarSel` | 2.000e+00 | 2.10e+00 | 4.8% |

## All rows

Each method is compared against a Python reference and an R reference. Methods without a widely installable external reference are flagged `paper-only` or `(none)` in the lib column.

| Method | Description | Reference (lang / lib / version) | Parity | Quality | RMSE rel | Tolerance | Status |
|--------|-------------|----------------------------------|--------|---------|----------|-----------|--------|
| `sparse_simpls` | Sparse SIMPLS with soft-threshold lambda | python / `(none)` - | ✗ | — | — | 1e+00 | no_python_reference |
| `sparse_simpls` | Sparse SIMPLS with soft-threshold lambda | R / `spls` 2.3.2 | ✓ | tight | 2.512e-05 | 1e+00 | ok |
| `di_pls` | Domain-invariant PLS | paper / `paper-only` - | ✓ | — | — | 5e-02 | paper_only |
| `recursive_pls` | Recursive (moving-window) PLS | python / `scikit-learn` 1.4.2 | ✓ | tight | 1.226e-02 | 1e-01 | ok |
| `recursive_pls` | Recursive (moving-window) PLS | R / `pls` 2.8.5 | ✓ | tight | 1.226e-02 | 1e-01 | ok |
| `cppls` | CPPLS (column-power-rescaled SIMPLS) | python / `(none)` - | ✗ | — | — | 1e+01 | no_python_reference |
| `cppls` | CPPLS (column-power-rescaled SIMPLS) | R / `pls` 2.8.5 | ✓ | tight | 2.018e-03 | 1e+01 | ok |
| `weighted_pls` | Sample-weighted PLS (sqrt(w)-prescaled SIMPLS) | python / `scikit-learn` 1.4.2 | ✓ | tight | 9.275e-07 | 1e-01 | ok |
| `weighted_pls` | Sample-weighted PLS (sqrt(w)-prescaled SIMPLS) | R / `(none)` - | ✗ | — | — | 1e-01 | no_r_reference |
| `robust_pls` | Robust PLS (Huber IRLS over weighted SIMPLS) | python / `(none)` - | ✗ | — | — | 2e+00 | no_python_reference |
| `robust_pls` | Robust PLS (Huber IRLS over weighted SIMPLS) | R / `chemometrics` 0.7.x | ✓ | tight | 4.489e-02 | 2e+00 | ok |
| `ridge_pls` | Ridge-augmented PLS | python / `scikit-learn` 1.4.2 | ✓ | tight | 1.020e-07 | 1e-01 | ok |
| `ridge_pls` | Ridge-augmented PLS | R / `(none)` - | ✗ | — | — | 1e-01 | no_r_reference |
| `continuum_regression` | Continuum regression (interpolates PLS / OLS) | python / `(none)` - | ✗ | — | — | 2e+01 | no_python_reference |
| `continuum_regression` | Continuum regression (interpolates PLS / OLS) | R / `JICO` 0.0 | ✓ | moderate | 1.090e+01 | 2e+01 | ok |
| `n_pls` | N-PLS — 3-way tensor PLS (Bro 1996) | python / `tensorly` 0.9.0 | ✓ | tight | 1.621e+00 | 1e+01 | ok |
| `n_pls` | N-PLS — 3-way tensor PLS (Bro 1996) | R / `(none)` - | ✗ | — | — | 1e+01 | no_r_reference |
| `kernel_pls_rbf` | Non-linear kernel PLS (RBF kernel) | python / `(none)` - | ✗ | — | — | 2e+00 | no_python_reference |
| `kernel_pls_rbf` | Non-linear kernel PLS (RBF kernel) | R / `kernlab+pls` 0.9.33+2.8.5 | ✓ | tight | 4.213e-01 | 2e+00 | ok |
| `o2pls` | O2PLS — bi-directional OPLS (Trygg & Wold 2003) | python / `(none)` - | ✗ | — | — | 1e+00 | no_python_reference |
| `o2pls` | O2PLS — bi-directional OPLS (Trygg & Wold 2003) | R / `OmicsPLS` 2.1.0 | ✓ | moderate | 4.541e-01 | 1e+00 | ok |
| `approximate_press` | Approximate-PRESS component selection (§29) | python / `(none)` - | ✗ | — | — | 1e+01 | no_python_reference |
| `approximate_press` | Approximate-PRESS component selection (§29) | R / `pls` 2.8.5 | ✓ | tight | 3.673e-02 | 1e+01 | ok |
| `pls_diagnostic_t2` | PLS Hotelling T² (§9) | python / `(none)` - | ✗ | — | — | 1e+01 | no_python_reference |
| `pls_diagnostic_t2` | PLS Hotelling T² (§9) | R / `mdatools` 0.15.0 | ✓ | moderate | 3.845e+00 | 1e+01 | ok |
| `pls_diagnostic_q` | PLS Q residuals / SPE (§9) | python / `(none)` - | ✗ | — | — | 5e+00 | no_python_reference |
| `pls_diagnostic_q` | PLS Q residuals / SPE (§9) | R / `mdatools` 0.15.0 | ✓ | moderate | 2.190e+00 | 5e+00 | ok |
| `pls_monitoring` | PLS process monitoring (T²/Q thresholds + alarms) (§19) | python / `(none)` - | ✗ | — | — | 1e+01 | no_python_reference |
| `pls_monitoring` | PLS process monitoring (T²/Q thresholds + alarms) (§19) | R / `mdatools` 0.15.0 | ✓ | moderate | 3.298e+00 | 1e+01 | ok |
| `one_se_rule` | One-SE component selection rule (§10) | python / `(none)` - | ✗ | — | — | 1e+01 | no_python_reference |
| `one_se_rule` | One-SE component selection rule (§10) | R / `pls` 2.8.5 | ✓ | tight | 2.623e+00 | 1e+01 | ok |
| `so_pls` | SO-PLS — Sequential & Orthogonalized multi-block PLS (§17) | python / `(none)` - | ✗ | — | — | 1e+00 | no_python_reference |
| `so_pls` | SO-PLS — Sequential & Orthogonalized multi-block PLS (§17) | R / `multiblock` 0.8.10 | ✓ | tight | 9.529e-03 | 1e+00 | ok |
| `on_pls` | OnPLS — Orthogonal multi-block decomposition (§18) | python / `OnPLS` github tomlof/OnPLS | ✓ | tight | 1.817e+00 | 1e+01 | ok |
| `on_pls` | OnPLS — Orthogonal multi-block decomposition (§18) | R / `(none)` - | ✗ | — | — | 1e+01 | no_r_reference |
| `rosa` | ROSA — Response-Oriented Sequential Alternation (§19) | python / `(none)` - | ✗ | — | — | 1e+00 | no_python_reference |
| `rosa` | ROSA — Response-Oriented Sequential Alternation (§19) | R / `multiblock` 0.8.10 | ✓ | tight | 1.345e-01 | 1e+00 | ok |
| `vissa_select` | VISSA-PLS — Variable Iterative Space Shrinkage (§49) | python / `auswahl` 0.9.0 | ✓ | loose | 1.658e+00 | 2e+00 | ok |
| `vissa_select` | VISSA-PLS — Variable Iterative Space Shrinkage (§49) | R / `(none)` - | ✗ | — | — | 2e+00 | no_r_reference |
| `pso_select` | PSO-PLS — Binary Particle Swarm variable selection (§48) | python / `pyswarms` 1.3.0 | ✓ | moderate | 8.018e-01 | 1e+00 | ok |
| `pso_select` | PSO-PLS — Binary Particle Swarm variable selection (§48) | R / `(none)` - | ✗ | — | — | 1e+00 | no_r_reference |
| `gpr_pls` | GPR-on-PLS — RBF Gaussian Process on PLS scores (§47) | python / `scikit-learn` 1.4.2 | ✓ | tight | 2.259e-10 | 1e-08 | ok |
| `gpr_pls` | GPR-on-PLS — RBF Gaussian Process on PLS scores (§47) | R / `(none)` - | ✗ | — | — | 1e-08 | no_r_reference |
| `bagging_pls` | Bagging PLS (§20) | python / `scikit-learn` 1.8.0 | ✓ | tight | 3.508e-03 | 2e+00 | ok |
| `bagging_pls` | Bagging PLS (§20) | R / `(none)` - | ✗ | — | — | 2e+00 | no_r_reference |
| `boosting_pls` | Boosting PLS (§20) | python / `(none)` - | ✗ | — | — | 1e+01 | no_python_reference |
| `boosting_pls` | Boosting PLS (§20) | R / `mboost` 2.9-11 | ✓ | tight | 3.864e-01 | 1e+01 | ok |
| `random_subspace_pls` | Random-subspace PLS (§20) | python / `scikit-learn` 1.8.0 | ✓ | tight | 4.548e-03 | 2e+00 | ok |
| `random_subspace_pls` | Random-subspace PLS (§20) | R / `(none)` - | ✗ | — | — | 2e+00 | no_r_reference |
| `pls_glm` | PLS-GLM (§5) — softmax/Poisson IRLS on PLS scores | python / `(none)` - | ✗ | — | — | 5e-01 | no_python_reference |
| `pls_glm` | PLS-GLM (§5) — softmax/Poisson IRLS on PLS scores | R / `plsRglm` 1.5.1 | ✓ | tight | 9.620e-02 | 5e-01 | ok |
| `pls_qda` | PLS-QDA (§5) — quadratic discriminant on PLS scores | python / `scikit-learn` 1.8.0 | ✓ | moderate | 3.952e+00 | 1e+01 | ok |
| `pls_qda` | PLS-QDA (§5) — quadratic discriminant on PLS scores | R / `(none)` - | ✗ | — | — | 1e+01 | no_r_reference |
| `pls_cox` | PLS-Cox (§5) — Cox PH on PLS scores | python / `(none)` - | ✗ | — | — | 5e+00 | no_python_reference |
| `pls_cox` | PLS-Cox (§5) — Cox PH on PLS scores | R / `plsRcox` 1.8.2 | ✓ | tight | 9.047e-01 | 5e+00 | ok |
| `pds` | PDS — Piecewise Direct Standardization (§13) | python / `(none)` - | ✗ | — | — | 5e-01 | no_python_reference |
| `pds` | PDS — Piecewise Direct Standardization (§13) | R / `base` R 4.3.3 | ✓ | tight | 3.980e-03 | 5e-01 | ok |
| `ds` | DS — Direct Standardization (§13) | python / `(none)` - | ✗ | — | — | 5e-01 | no_python_reference |
| `ds` | DS — Direct Standardization (§13) | R / `base` R 4.3.3 | ✓ | tight | 4.340e-08 | 5e-01 | ok |
| `mir_pls` | MIR-PLS — Inverse-regression PLS (§13) | paper / `paper-only` - | ✓ | — | — | 5e-02 | paper_only |
| `missing_aware_nipals` | Missing-aware NIPALS PLS (§13) | python / `(none)` - | ✗ | — | — | 1e+01 | no_python_reference |
| `missing_aware_nipals` | Missing-aware NIPALS PLS (§13) | R / `softImpute` 1.4-1 | ✓ | tight | 1.588e-15 | 1e+01 | ok |
| `sparse_pls_da` | Sparse PLS-DA (§7) | python / `(none)` - | ✗ | — | — | 2e+00 | no_python_reference |
| `sparse_pls_da` | Sparse PLS-DA (§7) | R / `spls` 2.3.2 | ✓ | moderate | 9.249e-01 | 2e+00 | ok |
| `group_sparse_pls` | Group sparse PLS (§7) | python / `(none)` - | ✗ | — | — | 1e+01 | no_python_reference |
| `group_sparse_pls` | Group sparse PLS (§7) | R / `sgPLS` 1.8.1 | ✓ | tight | 6.886e-03 | 1e+01 | ok |
| `fused_sparse_pls` | Fused sparse PLS (§7) | paper / `paper-only` - | ✓ | — | — | 5e-02 | paper_only |
| `pls_diagnostic_dmodx` | PLS Distance-to-Model X (§9) | python / `(none)` - | ✗ | — | — | 5e+00 | no_python_reference |
| `pls_diagnostic_dmodx` | PLS Distance-to-Model X (§9) | R / `mdatools` 0.15.0 | ✓ | tight | 1.220e+00 | 5e+00 | ok |
| `mb_pls` | MB-PLS — Multi-block PLS (§17 Phase 4) | python / `nirs4all.operators.models.sklearn.mbpls` in-tree | ✓ | tight | 6.213e-03 | 2e+00 | ok |
| `mb_pls` | MB-PLS — Multi-block PLS (§17 Phase 4) | R / `(none)` - | ✗ | — | — | 2e+00 | no_r_reference |
| `lw_pls` | LW-PLS — Locally-weighted PLS (§17 Phase 4) | python / `nirs4all.operators.models.sklearn.lwpls` in-tree | ✓ | tight | 1.450e-02 | 5e+00 | ok |
| `lw_pls` | LW-PLS — Locally-weighted PLS (§17 Phase 4) | R / `(none)` - | ✗ | — | — | 5e+00 | no_r_reference |
| `pls_lda` | PLS-LDA — LDA on PLS scores (§17 Phase 4) | python / `scikit-learn` 1.8.0 | ✓ | tight | 2.910e-02 | 5e+00 | ok |
| `pls_lda` | PLS-LDA — LDA on PLS scores (§17 Phase 4) | R / `(none)` - | ✗ | — | — | 5e+00 | no_r_reference |
| `pls_logistic` | PLS-Logistic — Logistic regression on PLS scores | python / `scikit-learn` 1.8.0 | ✓ | tight | 8.341e-01 | 5e+00 | ok |
| `pls_logistic` | PLS-Logistic — Logistic regression on PLS scores | R / `(none)` - | ✗ | — | — | 5e+00 | no_r_reference |
| `aom_preprocess` | AOM preprocessing pipeline (§17 Phase 4) | python / `nirs4all.bench.AOM_v0.aompls` in-tree-oracle | ✓ | tight | 0.000e+00 | 5e+00 | ok |
| `aom_preprocess` | AOM preprocessing pipeline (§17 Phase 4) | R / `(none)` - | ✗ | — | — | 5e+00 | no_r_reference |
| `variable_select_vip` | VIP top-k variable selection (§18 Phase 5a, method=0) | python / `(none)` - | ✗ | — | — | 7e-01 | no_python_reference |
| `variable_select_vip` | VIP top-k variable selection (§18 Phase 5a, method=0) | R / `plsVarSel` 0.10.0 | ✓ | weak | 6.325e-01 | 7e-01 | ok |
| `variable_select_coef` | |Coef| top-k selection (§18 Phase 5a, method=1) | python / `(none)` - | ✗ | — | — | 1e+00 | no_python_reference |
| `variable_select_coef` | |Coef| top-k selection (§18 Phase 5a, method=1) | R / `pls` 2.8.5 | ✓ | weak | 1.000e+00 | 1e+00 | ok |
| `variable_select_sr` | Selectivity-Ratio top-k (§18 Phase 5a, method=2) | python / `(none)` - | ✗ | — | — | 1e+00 | no_python_reference |
| `variable_select_sr` | Selectivity-Ratio top-k (§18 Phase 5a, method=2) | R / `plsVarSel` 0.10.0 | ✓ | weak | 1.183e+00 | 1e+00 | ok |
| `interval_select` | Interval/iPLS forward selection (§18 Phase 5b) | python / `(none)` - | ✗ | — | — | 1e+00 | no_python_reference |
| `interval_select` | Interval/iPLS forward selection (§18 Phase 5b) | R / `mdatools` 0.15.0 | ✓ | weak | 9.129e-01 | 1e+00 | ok |
| `bipls_select` | biPLS backward interval elimination (§18 Phase 5p) | python / `(none)` - | ✗ | — | — | 7e-01 | no_python_reference |
| `bipls_select` | biPLS backward interval elimination (§18 Phase 5p) | R / `mdatools` 0.15.0 | ✓ | moderate | 4.082e-01 | 7e-01 | ok |
| `sipls_select` | siPLS synergistic interval selection (§18 Phase 5q) | paper / `paper-only` - | ✓ | — | — | 7e-01 | paper_only |
| `stability_select` | Stability/MCUVE selection (§18 Phase 5c) | python / `(none)` - | ✗ | — | — | 1e+00 | no_python_reference |
| `stability_select` | Stability/MCUVE selection (§18 Phase 5c) | R / `plsVarSel` 0.10.0 | ✓ | weak | 1.265e+00 | 1e+00 | ok |
| `uve_select` | UVE noise-thresholded selection (§18 Phase 5d) | python / `(none)` - | ✗ | — | — | 7e-01 | no_python_reference |
| `uve_select` | UVE noise-thresholded selection (§18 Phase 5d) | R / `plsVarSel` 0.10.0 | ✓ | moderate | 2.325e-01 | 7e-01 | ok |
| `spa_select` | SPA Successive Projections (§18 Phase 5e) | python / `(none)` - | ✗ | — | — | 1e+00 | no_python_reference |
| `spa_select` | SPA Successive Projections (§18 Phase 5e) | R / `plsVarSel` 0.10.0 | ✓ | loose | 1.057e+00 | 1e+00 | ok |
| `cars_select` | CARS competitive adaptive reweighted sampling | python / `(none)` - | ✗ | — | — | 1e+00 | no_python_reference |
| `cars_select` | CARS competitive adaptive reweighted sampling | R / `enpls` 6.1 | ✓ | weak | 1.291e+00 | 1e+00 | ok |
| `random_frog_select` | Random Frog selection (§18 Phase 5g) | matlab / `libPLS` 1.95 | ✓ | weak | 1.265e+00 | 1e+00 | ok |
| `random_frog_select` | Random Frog selection (§18 Phase 5g) | R / `(none)` - | ✗ | — | — | 1e+00 | no_r_reference |
| `scars_select` | SCARS stability + CARS (§18 Phase 5h) | paper / `paper-only` - | ✓ | — | — | 1e+00 | paper_only |
| `ga_select` | GA-PLS genetic algorithm selection | python / `(none)` - | ✗ | — | — | 1e+00 | no_python_reference |
| `ga_select` | GA-PLS genetic algorithm selection | R / `plsVarSel` 0.10.0 | ✓ | weak | 1.195e+00 | 1e+00 | ok |
| `shaving_select` | Shaving iterative variable trimming | python / `(none)` - | ✗ | — | — | 7e-01 | no_python_reference |
| `shaving_select` | Shaving iterative variable trimming | R / `plsVarSel` 0.10.0 | ✓ | tight | 0.000e+00 | 7e-01 | ok |
| `bve_select` | Backward Variable Elimination (§18 Phase 5k) | python / `(none)` - | ✗ | — | — | 1e+00 | no_python_reference |
| `bve_select` | Backward Variable Elimination (§18 Phase 5k) | R / `plsVarSel` 0.10.0 | ✓ | weak | 1.291e+00 | 1e+00 | ok |
| `t2_select` | T²-PLS loading-weight selection (§18 Phase 5l) | python / `(none)` - | ✗ | — | — | 1e+00 | no_python_reference |
| `t2_select` | T²-PLS loading-weight selection (§18 Phase 5l) | R / `plsVarSel` 0.10.0 | ✓ | weak | 1.155e+00 | 1e+00 | ok |
| `wvc_select` | WVC weighted-variable-component top-k | python / `(none)` - | ✗ | — | — | 7e-01 | no_python_reference |
| `wvc_select` | WVC weighted-variable-component top-k | R / `plsVarSel` 0.10.0 | ✓ | tight | 0.000e+00 | 7e-01 | ok |
| `wvc_threshold_select` | WVC threshold-based selection (§18 Phase 5r) | python / `(none)` - | ✗ | — | — | 7e-01 | no_python_reference |
| `wvc_threshold_select` | WVC threshold-based selection (§18 Phase 5r) | R / `plsVarSel` 0.10.0 | ✓ | tight | 1.925e-01 | 7e-01 | ok |
| `emcuve_select` | EMCUVE ensemble MC-UVE (§18 Phase 5n) | python / `(none)` - | ✗ | — | — | 1e+00 | no_python_reference |
| `emcuve_select` | EMCUVE ensemble MC-UVE (§18 Phase 5n) | R / `plsVarSel` 0.10.0 | ✓ | tight | 2.810e-01 | 1e+00 | ok |
| `randomization_select` | Randomization-test selector (§18 Phase 5o) | python / `(none)` - | ✗ | — | — | 1e+00 | no_python_reference |
| `randomization_select` | Randomization-test selector (§18 Phase 5o) | R / `pls+stats` R 4.3.3 | ✓ | loose | 9.759e-01 | 1e+00 | ok |
| `rep_select` | REP-PLS repeated VIP selection (§18 Phase 5s) | python / `(none)` - | ✗ | — | — | 2e+00 | no_python_reference |
| `rep_select` | REP-PLS repeated VIP selection (§18 Phase 5s) | R / `plsVarSel` 0.10.0 | ✓ | weak | 1.700e+00 | 2e+00 | ok |
| `ipw_select` | IPW-PLS iterative predictor weighting (§18 Phase 5t) | python / `(none)` - | ✗ | — | — | 1e+00 | no_python_reference |
| `ipw_select` | IPW-PLS iterative predictor weighting (§18 Phase 5t) | R / `plsVarSel` 0.10.0 | ✓ | loose | 8.819e-01 | 1e+00 | ok |
| `st_select` | ST-PLS soft-thresholded sparse PLS (§18 Phase 5u) | python / `(none)` - | ✗ | — | — | 2e+00 | no_python_reference |
| `st_select` | ST-PLS soft-thresholded sparse PLS (§18 Phase 5u) | R / `plsVarSel` 0.10.0 | ✓ | weak | 2.000e+00 | 2e+00 | ok |
