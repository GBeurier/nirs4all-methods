# Parity tolerances

This table records the documented per-pair tolerances between every two reference PLS implementations `pls4all` cross-checks against. Numbers in **bold** are the relaxations applied after Codex review of the Phase 0 roadmap.

For cross-algorithm rows the comparator gates on `predictions`, `coefficients`, `intercept`, `x_mean`, `x_scale`, `y_mean`, `y_scale` only — raw latent matrices (`weights_W`, `loadings_P`, `y_loadings_Q`, `rotations_R`, `scores_T`, `y_scores_U`) are diagnostic when algorithms differ.

| Row key | A | B | Variant | Algorithm | abs_tol | rel_tol | Comparison set | Notes |
|---|---|---|---|---|---|---|---|---|
| `sklearn/PLSRegression/self_roundtrip`     | sklearn 1.4 `PLSRegression`    | sklearn (round-trip)               | regression | NIPALS              | 1e-15  | 1e-15  | all                                              | sanity |
| `sklearn-vs-pls-kernelpls`                 | sklearn 1.4 `PLSRegression`    | R pls 2.8 `plsr(method='kernel')`  | regression | NIPALS vs kernel    | 5e-7   | 1e-6   | predictions, coefficients, intercept, x/y mean+scale | cross-algorithm |
| `sklearn-vs-pls-simpls`                    | sklearn 1.4 `PLSRegression`    | R pls 2.8 `plsr(method='simpls')`  | regression | NIPALS vs SIMPLS    | 1e-5   | 1e-4   | predictions, coefficients, intercept, x/y mean+scale | cross-algorithm |
| `sklearn-vs-mixomics-pls`                  | sklearn 1.4 `PLSRegression`    | mixOmics 6.28 `pls`                | regression | NIPALS              | **1e-7** | **1e-7** | predictions, coefficients, intercept, x/y mean+scale | mixOmics scales differently — predictions agree |
| `sklearn-vs-nirs4all-simpls`               | sklearn 1.4 `PLSRegression`    | nirs4all 0.8 `SIMPLS`              | regression | SIMPLS              | 1e-6   | 1e-5   | predictions, coefficients, intercept, x/y mean+scale | cross-algorithm |
| `pls4all-numpy-simpls`                     | pls4all NumPy SIMPLS fixture   | pls4all C++ `P4A_SOLVER_SIMPLS`    | regression | SIMPLS              | 1e-9   | 1e-9   | predictions, coefficients, intercept, x/y mean+scale, latent arrays | solver parity |
| `sklearn-canonical-vs-mixomics-canonical`  | sklearn 1.4 `PLSCanonical`     | mixOmics 6.28 `pls(mode='canonical')` | canonical | NIPALS           | 1e-7   | 1e-7   | predictions, coefficients, intercept, x/y mean+scale | cross-impl |
| `ropls-opls-vs-nirs4all-opls`              | ropls 1.34 `opls`              | nirs4all 0.8 `OPLS`                | orthogonal | OPLS-NIPALS         | 1e-5   | 1e-4   | predictions only                                  | OPLS sign + ordering vary |
| `nirs4all-aom-vs-pls4all-aom`              | nirs4all 0.8 `AOM_PLS`         | pls4all (Phase 6)                  | AOM        | SIMPLS + op-bank    | **TBD**| **TBD**| predictions                                       | tolerance frozen once the pls4all reference is frozen |
| `pls4all-refcpu-vs-pls4all-blas`           | pls4all REFERENCE_CPU          | pls4all BLAS                       | any        | any                 | **1e-10** | **1e-9** | all                                            | same algorithm, different SGEMM path; fixed reduction order, no FMA |
| `pls4all-refcpu-vs-pls4all-cuda`           | pls4all REFERENCE_CPU          | pls4all CUDA                       | any        | any                 | 1e-9   | 1e-8   | predictions                                       | fp64 GPU drift envelope |
| `pls4all-linux-vs-pls4all-macos-arm64`     | pls4all REFERENCE_CPU (Linux x86_64) | pls4all REFERENCE_CPU (macOS arm64) | any   | any                 | **1e-11** | **1e-11** | all                                          | cross-platform sanity |

The `tolerance_table_row` key in each fixture's `comparison_policy` block points to a row by name. The comparator resolves missing per-field tolerances against this row.
