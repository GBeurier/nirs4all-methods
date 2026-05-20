# Inventaire post-correction dashboard et PLS-QDA - 2026-05-20

Sources:

- Gate stricte: `benchmarks/cross_binding/results/parity_30x30_strict.csv`
- Dashboard local: `docs/_static/bench-data.json`
- Dataset gate: random `30x30`, `threads=1`, `seed_base=1234567890`
- Politique: `reference_parity_rmse_rel <= 1e-3` pour toute methode avec
  reference executable. Exceptions uniquement `paper_only`.

## Ce qui etait faux dans le dashboard

Le dashboard local melangeait la gate stricte `parity_30x30_strict.csv` avec
des fichiers `dashboard_refresh_*.csv` plus anciens. Il pouvait donc afficher
des divergences issues de runs `100x50`, `1000x500` ou de probes anciennes.

Correction appliquee: quand `parity_30x30_strict.csv` existe, le builder du
dashboard lit uniquement cette gate stricte. Le dashboard local reconstruit ne
contient maintenant qu'une taille: `30x30`.

## Ce qui etait un vrai bug methode

`pls_qda` etait reellement faux cote sortie C ABI:

- les scores latents ne suivaient pas exactement la projection
  `sklearn.cross_decomposition.PLSRegression`;
- la prediction exposee etait un proxy de distance, pas le
  `QuadraticDiscriminantAnalysis.predict_proba`;
- le comparateur rejetait aussi certaines sorties equivalentes en contenu mais
  sauvegardees en `90` au lieu de `30x3`.

Correction appliquee:

- projection PLS-QDA alignee sur `sklearn` pour cette gate;
- sortie `predictions` alignee sur `QDA(reg_param=0.01).predict_proba`;
- reshape comparateur autorise seulement quand un cote est vraiment 1D et que
  le nombre total d'elements correspond.

Resultat `pls_qda`: `2.85617` avant correction, `2.5169e-16` apres correction
sur le probe strict 30x30, puis lignes remplacees dans la gate stricte.

## Etat global actuel

CSV strict:

- lignes CSV: `3123`
- lignes avec metrique reference finie: `2695`
- lignes reference `> 1`: `205`
- methodes reference `> 1`: `5`
- lignes reference `> 1e-3`: `1537`
- methodes reference `> 1e-3`: `42`
- lignes binding `> 1e-3`: `18`
- methodes binding `> 1e-3`: `3`
- methodes strictement OK: `27`
- exceptions `paper_only`: `4`

Dashboard reconstruit:

- lignes dashboard: `73`
- tailles dashboard: `30x30` uniquement
- cellules dashboard: `1110`
- cellules exactes/OK: `849`
- cellules reference `> 1`: `70`, reparties sur `5` methodes
- cellules reference `> 1e-3`: `535`, reparties sur `42` methodes

## Methodes encore au-dessus de 1

Ces divergences ne sont plus un bug d'affichage. Les bindings internes sont
coherents, mais l'algorithme ou l'adaptateur ne reproduit pas la reference.

| methode | reference | max ref rmse_rel | lignes CSV >1e-3 | diagnostic |
|---|---:|---:|---:|---|
| `st_select` | `ref_r_plsvarsel` | 1.61245 | 41 | implementation hard-threshold vs chemin `plsVarSel::stpls` |
| `stability_select` | `ref_r_plsvarsel` | 1.34164 | 41 | plan de folds/top-k different de `plsVarSel::mcuve_pls` |
| `irf_select` | `ref_matlab_libpls` | 1.31426 | 41 | selection top-K interne vs strategie finale libPLS |
| `spa_select` | `ref_r_plsvarsel` | 1.18322 | 41 | heuristique greedy SPA differente de `plsVarSel::spa_pls` |
| `pso_select` | `ref_python_pyswarms` | 1.04881 | 41 | PSO interne/RNG/repair differents de `pyswarms.discrete.BinaryPSO` |

## Toutes les methodes encore au-dessus de 1e-3

| methode | reference | max ref rmse_rel | max binding diff | lignes >1e-3 |
|---|---:|---:|---:|---:|
| `st_select` | `ref_r_plsvarsel` | 1.61245 | 0 | 41 |
| `stability_select` | `ref_r_plsvarsel` | 1.34164 | 0 | 41 |
| `irf_select` | `ref_matlab_libpls` | 1.31426 | 0 | 41 |
| `spa_select` | `ref_r_plsvarsel` | 1.18322 | 0 | 41 |
| `pso_select` | `ref_python_pyswarms` | 1.04881 | 0 | 41 |
| `vissa_select` | `ref_python_auswahl` | 1 | 0 | 41 |
| `variable_select_sr` | `ref_r_plsvarsel` | 1 | 0 | 41 |
| `ga_select` | `ref_r_plsvarsel` | 1 | 0 | 41 |
| `rep_select` | `ref_r_plsvarsel` | 1 | 0 | 41 |
| `pls_cox` | `ref_r_plsrcox` | 0.95641 | 3.33067e-16 | 45 |
| `vip_spa_select` | `ref_python_auswahl` | 0.912871 | 0 | 41 |
| `pls_lda` | `ref_python_scikit_learn` | 0.906674 | 0 | 21 |
| `uve_select` | `ref_r_plsvarsel` | 0.894427 | 0 | 41 |
| `random_frog_select` | `ref_matlab_libpls` | 0.894427 | 0 | 41 |
| `bve_select` | `ref_r_plsvarsel` | 0.866025 | 0 | 41 |
| `ipw_select` | `ref_r_plsvarsel` | 0.845154 | 0 | 41 |
| `approximate_press` | `ref_r_pls` | 0.82606 | 0 | 41 |
| `n_pls` | `ref_python_tensorly` | 0.825789 | 1.77636e-15 | 41 |
| `cars_select` | `ref_r_enpls` | 0.816497 | 0 | 41 |
| `emcuve_select` | `ref_r_plsvarsel` | 0.816497 | 0 | 41 |
| `pls_logistic` | `ref_python_scikit_learn` | 0.796761 | 0 | 21 |
| `variable_select_vip` | `ref_r_plsvarsel` | 0.774597 | 0 | 41 |
| `interval_select` | `ref_r_mdatools` | 0.707107 | 0 | 41 |
| `sparse_pls_da` | `ref_r_spls` | 0.494747 | 8.88178e-16 | 21 |
| `one_se_rule` | `ref_r_pls` | 0.49223 | 0 | 41 |
| `boosting_pls` | `ref_r_mboost` | 0.45708 | 8.88178e-16 | 41 |
| `wvc_threshold_select` | `ref_r_plsvarsel` | 0.417029 | 0 | 41 |
| `bagging_pls` | `ref_python_scikit_learn` | 0.211826 | 0.556503 | 41 |
| `kernel_pls_rbf` | `ref_r_kernlab_pls` | 0.185322 | 0 | 41 |
| `robust_pls` | `ref_r_chemometrics` | 0.176278 | 2.66454e-15 | 41 |
| `o2pls` | `ref_r_omicspls` | 0.140165 | 8.88178e-16 | 17 |
| `continuum_regression` | `ref_r_jico` | 0.0944624 | 1.77636e-15 | 41 |
| `random_subspace_pls` | `ref_python_scikit_learn` | 0.0924876 | 0.744132 | 41 |
| `group_sparse_pls` | `ref_r_sgpls` | 0.0884986 | 0 | 41 |
| `pls_glm` | `ref_r_plsrglm` | 0.0880831 | 2.35922e-15 | 21 |
| `pls` | `ref_python_scikit_learn` | 0.0532762 | 3.9968e-15 | 1 |
| `sparse_simpls` | `ref_r_spls` | 0.0532504 | 5.32907e-15 | 41 |
| `cppls` | `ref_r_pls` | 0.0345951 | 1.77636e-15 | 41 |
| `rosa` | `ref_r_multiblock` | 0.0296069 | 0 | 21 |
| `di_pls` | `ref_python_diplslib` | 0.017799 | 0.0709344 | 16 |
| `weighted_pls` | `ref_python_scikit_learn` | 0.00416263 | 1.77636e-15 | 41 |
| `ridge_pls` | `ref_python_scikit_learn` | 0.00164031 | 1.77636e-15 | 41 |

## Drifts binding restants

Ces lignes indiquent encore une divergence entre bindings/backends internes,
independamment de la reference externe.

| methode | reference | max ref rmse_rel | max binding diff | backend implique |
|---|---:|---:|---:|---|
| `random_subspace_pls` | `ref_python_scikit_learn` | 0.0924876 | 0.744132 | `js_wasm` |
| `bagging_pls` | `ref_python_scikit_learn` | 0.211826 | 0.556503 | `js_wasm` |
| `di_pls` | `ref_python_diplslib` | 0.017799 | 0.0709344 | R bindings/compat |

## Methodes strictement OK apres correction

`aom_pls`, `aom_preprocess`, `bipls_select`, `ds`, `ecr`, `gpr_pls`,
`iriv_select`, `lw_pls`, `mb_pls`, `missing_aware_nipals`, `on_pls`, `opls`,
`pcr`, `pds`, `pls_diagnostic_dmodx`, `pls_diagnostic_q`,
`pls_diagnostic_t2`, `pls_monitoring`, `pls_qda`, `pop_pls`,
`randomization_select`, `recursive_pls`, `shaving_select`, `so_pls`,
`t2_select`, `variable_select_coef`, `wvc_select`.

## Exceptions paper_only valides

`fused_sparse_pls`, `mir_pls`, `scars_select`, `sipls_select`.

## Priorites de correction

1. Corriger les 5 methodes `> 1`: `st_select`, `stability_select`,
   `irf_select`, `spa_select`, `pso_select`.
2. Corriger les drifts binding: `random_subspace_pls` JS/WASM,
   `bagging_pls` JS/WASM, `di_pls` R.
3. Corriger les gros drifts de surface de prediction/classification:
   `pls_cox`, `pls_lda`, `pls_logistic`, `sparse_pls_da`.
4. Repasser la gate complete apres chaque groupe et ne reconstruire le
   dashboard qu'a partir de `parity_30x30_strict.csv`.
