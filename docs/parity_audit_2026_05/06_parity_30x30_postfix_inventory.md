# Inventaire post-fixes parite stricte 30x30 - 2026-05-20

Source CSV: `benchmarks/cross_binding/results/parity_30x30_strict.csv`

Politique: toute methode avec reference executable doit avoir
`reference_parity_rmse_rel <= 1e-3` sur le dataset random 30x30. Les seules
exceptions valides restent les methodes `paper_only`.

## Etat global apres corrections

- Lignes CSV: 3123.
- Echecs d'execution: 0.
- Lignes avec metrique reference finie: 2671.
- Lignes reference `> 1e-3`: 1558.
- Lignes binding `> 1e-3`: 18.
- Methodes strictement OK: 26.
- Methodes `paper_only` exemptes: 4.
- Methodes avec drift binding restant: 3.
- Methodes avec drift reference restant: 43.

## Corrections validees dans ce round

| famille | resultat post-fix |
|---|---:|
| `recursive_pls` 30x30 window cap + Julia | `1.49e-14`, binding `4.0e-15` |
| `aom_pls` | `4.69e-16`, binding `1.78e-15` |
| `pop_pls` | `6.99e-16`, binding `0` |
| `mb_pls` | `4.24e-16`, binding `1.78e-15` |
| `lw_pls` | `7.83e-16`, binding `2.66e-15` |
| `so_pls` | `5.12e-06`, binding `0` |
| `on_pls` | `2.68e-05`, binding `0` |
| `aom_preprocess` | `0`, binding `0` |
| `shaving_select` | `0`, binding `0` |
| `variable_select_coef` | `0`, binding `0` |
| JS runtime failures (`pls`, `pcr`, `opls`, diagnostics, `gpr_pls`) | tous OK execution |
| R selector binding drift (`bve`, `uve`, `emcuve`, `shaving`) | binding `0` |

Diagnostic actualise: `mb_pls` et `lw_pls` ont bien `python_nirs4all` comme
reference sanctionnee et passent maintenant strictement. Ils ne sont pas des
exceptions. `so_pls` et `on_pls` passent aussi la gate stricte; `rosa` reste un
vrai drift reference face a `multiblock::rosa` en mode canonique R.

## Methodes strictement OK

| methode | reference | max rmse_rel | max binding diff |
|---|---:|---:|---:|
| `on_pls` | `ref_python_onpls` | 2.6765e-05 | 0 |
| `so_pls` | `ref_r_multiblock` | 5.1208e-06 | 0 |
| `ecr` | `ref_matlab_libpls` | 1.5974e-06 | 1.78e-15 |
| `ds` | `ref_r_base` | 1.6973e-07 | 8.88e-16 |
| `pds` | `ref_r_base` | 3.5726e-08 | 4.44e-16 |
| `gpr_pls` | `ref_python_scikit_learn` | 1.5435e-09 | 5.82e-14 |
| `pcr` | `ref_python_scikit_learn` | 7.6542e-13 | 3.74e-12 |
| `pls_diagnostic_t2` | `ref_r_mdatools` | 1.8617e-14 | 3.91e-14 |
| `missing_aware_nipals` | `ref_r_softimpute` | 1.6175e-14 | 1.78e-15 |
| `recursive_pls` | `ref_python_scikit_learn` | 1.4928e-14 | 4.00e-15 |
| `pls_monitoring` | `ref_r_mdatools` | 7.1725e-15 | 6.48e-14 |
| `opls` | `ref_r_ropls` | 6.8983e-15 | 3.53e-14 |
| `pls_diagnostic_q` | `ref_r_mdatools` | 1.6701e-15 | 4.62e-14 |
| `pls_diagnostic_dmodx` | `ref_r_mdatools` | 1.2442e-15 | 8.88e-16 |
| `lw_pls` | `ref_python_nirs4all` | 7.8324e-16 | 2.66e-15 |
| `pop_pls` | `ref_python_nirs4all` | 6.9917e-16 | 0 |
| `aom_pls` | `ref_python_nirs4all` | 4.6941e-16 | 1.78e-15 |
| `mb_pls` | `ref_python_nirs4all` | 4.2368e-16 | 1.78e-15 |
| `aom_preprocess` | `ref_python_nirs4all` | 0 | 0 |
| `bipls_select` | `ref_r_mdatools` | 0 | 0 |
| `iriv_select` | `ref_matlab_libpls` | 0 | 0 |
| `randomization_select` | `ref_r_pls_stats` | 0 | 0 |
| `shaving_select` | `ref_r_plsvarsel` | 0 | 0 |
| `t2_select` | `ref_r_plsvarsel` | 0 | 0 |
| `variable_select_coef` | `ref_r_pls` | 0 | 0 |
| `wvc_select` | `ref_r_plsvarsel` | 0 | 0 |

## Exceptions paper_only valides

- `mir_pls`
- `fused_sparse_pls`
- `scars_select`
- `sipls_select`

## Drifts binding restants

Ces lignes indiquent encore une divergence entre bindings/backends internes
avant meme de regarder la reference externe.

| methode | reference | max rmse_rel | max binding diff | diagnostic |
|---|---:|---:|---:|---|
| `bagging_pls` | `ref_python_scikit_learn` | 0.211826 | 0.556503 | JS/WASM RNG/subsampling divergent |
| `random_subspace_pls` | `ref_python_scikit_learn` | 0.0924876 | 0.744132 | JS/WASM RNG/subspace divergent |
| `di_pls` | `ref_python_diplslib` | 0.017799 | 0.0709344 | R binding still differs from C++ path |

## Drifts reference restants

| methode | reference | max rmse_rel | binding diff |
|---|---:|---:|---:|
| `pls_qda` | `ref_python_scikit_learn` | 2.85617 | 0 |
| `st_select` | `ref_r_plsvarsel` | 1.61245 | 0 |
| `stability_select` | `ref_r_plsvarsel` | 1.34164 | 0 |
| `irf_select` | `ref_matlab_libpls` | 1.31426 | 0 |
| `spa_select` | `ref_r_plsvarsel` | 1.18322 | 0 |
| `pso_select` | `ref_python_pyswarms` | 1.04881 | 0 |
| `ga_select` | `ref_r_plsvarsel` | 1 | 0 |
| `rep_select` | `ref_r_plsvarsel` | 1 | 0 |
| `variable_select_sr` | `ref_r_plsvarsel` | 1 | 0 |
| `vissa_select` | `ref_python_auswahl` | 1 | 0 |
| `pls_cox` | `ref_r_plsrcox` | 0.95641 | 3.33e-16 |
| `vip_spa_select` | `ref_python_auswahl` | 0.912871 | 0 |
| `pls_lda` | `ref_python_scikit_learn` | 0.906674 | 0 |
| `random_frog_select` | `ref_matlab_libpls` | 0.894427 | 0 |
| `uve_select` | `ref_r_plsvarsel` | 0.894427 | 0 |
| `bve_select` | `ref_r_plsvarsel` | 0.866025 | 0 |
| `ipw_select` | `ref_r_plsvarsel` | 0.845154 | 0 |
| `approximate_press` | `ref_r_pls` | 0.82606 | 0 |
| `n_pls` | `ref_python_tensorly` | 0.825789 | 1.78e-15 |
| `cars_select` | `ref_r_enpls` | 0.816497 | 0 |
| `emcuve_select` | `ref_r_plsvarsel` | 0.816497 | 0 |
| `pls_logistic` | `ref_python_scikit_learn` | 0.796761 | 0 |
| `variable_select_vip` | `ref_r_plsvarsel` | 0.774597 | 0 |
| `interval_select` | `ref_r_mdatools` | 0.707107 | 0 |
| `sparse_pls_da` | `ref_r_spls` | 0.494747 | 8.88e-16 |
| `one_se_rule` | `ref_r_pls` | 0.49223 | 0 |
| `boosting_pls` | `ref_r_mboost` | 0.45708 | 8.88e-16 |
| `wvc_threshold_select` | `ref_r_plsvarsel` | 0.417029 | 0 |
| `bagging_pls` | `ref_python_scikit_learn` | 0.211826 | 0.556503 |
| `kernel_pls_rbf` | `ref_r_kernlab_pls` | 0.185322 | 0 |
| `robust_pls` | `ref_r_chemometrics` | 0.176278 | 2.66e-15 |
| `o2pls` | `ref_r_omicspls` | 0.140165 | 8.88e-16 |
| `continuum_regression` | `ref_r_jico` | 0.0944624 | 1.78e-15 |
| `random_subspace_pls` | `ref_python_scikit_learn` | 0.0924876 | 0.744132 |
| `group_sparse_pls` | `ref_r_sgpls` | 0.0884986 | 0 |
| `pls_glm` | `ref_r_plsrglm` | 0.0880831 | 2.36e-15 |
| `pls` | `ref_python_scikit_learn` | 0.0532762 | 4e-15 |
| `sparse_simpls` | `ref_r_spls` | 0.0532504 | 5.33e-15 |
| `cppls` | `ref_r_pls` | 0.0345951 | 1.78e-15 |
| `rosa` | `ref_r_multiblock` | 0.0296069 | 0 |
| `di_pls` | `ref_python_diplslib` | 0.017799 | 0.0709344 |
| `weighted_pls` | `ref_python_scikit_learn` | 0.00416263 | 1.78e-15 |
| `ridge_pls` | `ref_python_scikit_learn` | 0.00164031 | 1.78e-15 |

## Priorite de fixes suivante

1. Corriger les 3 drifts binding restants: `bagging_pls`,
   `random_subspace_pls`, `di_pls`.
2. Corriger les gros drifts classification/score: `pls_qda`, `pls_lda`,
   `pls_logistic`, `pls_cox`, `sparse_pls_da`, puis `pls_glm`.
3. Corriger les selectors `plsVarSel`/`auswahl`/`libPLS` restants en
   separant les vrais bugs methode des references stochastic/score-scale.
4. Repasser la matrice 30x30 complete apres chaque groupe de corrections.
