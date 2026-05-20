# Inventaire apres recalcul des scores bindings - 2026-05-20

Source: `benchmarks/cross_binding/results/parity_30x30_strict.csv`

## Probleme corrige

Plusieurs bindings affichaient une divergence reference sans score numerique,
alors que les lignes C++ equivalentes avaient un score. La cause n'etait pas
un run absent: les predictions existaient, mais la gate stricte avait ete
calculee avant que le comparateur accepte les sorties sauvegardees comme
vecteur plat par une partie des bindings R/MATLAB.

Exemples de shapes concernees:

- `(30, 30)` vs `(900,)`
- `(30, 4)` vs `(120,)`
- `(30, 3)` vs `(90,)`
- `(30, 2)` vs `(60,)`
- `(10, 2)` vs `(20,)`

Correction appliquee:

- recalcul de `reference_parity_*` et `binding_parity_*` depuis les fichiers
  de predictions existants;
- regeneration de `benchmarks/cross_binding/results/parity_30x30_strict.csv`;
- regeneration de `docs/_static/bench-data.json`;
- ajout d'une regression test qui interdit les lignes successful/non-paper
  sans score de reference quand C++ a un score.

Resultat: `264` lignes successful/non-paper sans score numerique avant
recalcul, `0` apres recalcul.

## Etat dashboard

- lignes dashboard: `73`
- cellules dashboard: `1110`
- cellules exactes/OK: `903`
- cellules reference `> 1`: `74`, reparties sur `6` methodes
- cellules reference sans score numerique alors que C++ en a un: `0`

## Etat CSV strict

- lignes CSV: `3123`
- lignes reference `> 1`: `221`
- methodes reference `> 1`: `6`
- lignes reference `> 1e-3`: `1713`
- methodes reference `> 1e-3`: `44`
- lignes binding `> 1e-3`: `66`
- methodes binding `> 1e-3`: `6`

## Methodes encore au-dessus de 1

| methode | reference | max ref rmse_rel | lignes CSV >1e-3 | diagnostic |
|---|---:|---:|---:|---|
| `st_select` | `ref_r_plsvarsel` | 1.61245 | 41 | drift methode selector |
| `stability_select` | `ref_r_plsvarsel` | 1.34164 | 41 | drift methode selector |
| `irf_select` | `ref_matlab_libpls` | 1.31426 | 41 | drift methode selector |
| `spa_select` | `ref_r_plsvarsel` | 1.18322 | 41 | drift methode selector |
| `pso_select` | `ref_python_pyswarms` | 1.04881 | 41 | drift methode/RNG |
| `on_pls` | `ref_python_onpls` | 1.00264 | 16 | drift R bindings visible apres recalcul des scores |

## Drifts binding internes maintenant visibles

| methode | max binding diff | backends impliques |
|---|---:|---|
| `so_pls` | 3.88218 | R bindings/compat |
| `on_pls` | 0.834084 | R bindings/compat |
| `random_subspace_pls` | 0.744132 | JS/WASM |
| `bagging_pls` | 0.556503 | JS/WASM |
| `pls_glm` | 0.231014 | R bindings/compat |
| `di_pls` | 0.0709344 | R bindings/compat |

## Toutes les methodes encore au-dessus de 1e-3

| methode | reference | max ref rmse_rel | max binding diff | lignes >1e-3 |
|---|---:|---:|---:|---:|
| `st_select` | `ref_r_plsvarsel` | 1.61245 | 0 | 41 |
| `stability_select` | `ref_r_plsvarsel` | 1.34164 | 0 | 41 |
| `irf_select` | `ref_matlab_libpls` | 1.31426 | 0 | 41 |
| `spa_select` | `ref_r_plsvarsel` | 1.18322 | 0 | 41 |
| `pso_select` | `ref_python_pyswarms` | 1.04881 | 0 | 41 |
| `on_pls` | `ref_python_onpls` | 1.00264 | 0.834084 | 16 |
| `vissa_select` | `ref_python_auswahl` | 1 | 0 | 41 |
| `variable_select_sr` | `ref_r_plsvarsel` | 1 | 0 | 41 |
| `ga_select` | `ref_r_plsvarsel` | 1 | 0 | 41 |
| `rep_select` | `ref_r_plsvarsel` | 1 | 0 | 41 |
| `pls_cox` | `ref_r_plsrcox` | 0.95641 | 3.33067e-16 | 45 |
| `vip_spa_select` | `ref_python_auswahl` | 0.912871 | 0 | 41 |
| `pls_lda` | `ref_python_scikit_learn` | 0.906674 | 0 | 45 |
| `uve_select` | `ref_r_plsvarsel` | 0.894427 | 0 | 41 |
| `random_frog_select` | `ref_matlab_libpls` | 0.894427 | 0 | 41 |
| `bve_select` | `ref_r_plsvarsel` | 0.866025 | 0 | 41 |
| `ipw_select` | `ref_r_plsvarsel` | 0.845154 | 0 | 41 |
| `approximate_press` | `ref_r_pls` | 0.82606 | 0 | 41 |
| `n_pls` | `ref_python_tensorly` | 0.825789 | 1.77636e-15 | 41 |
| `cars_select` | `ref_r_enpls` | 0.816497 | 0 | 41 |
| `emcuve_select` | `ref_r_plsvarsel` | 0.816497 | 0 | 41 |
| `pls_logistic` | `ref_python_scikit_learn` | 0.796761 | 0 | 45 |
| `variable_select_vip` | `ref_r_plsvarsel` | 0.774597 | 0 | 41 |
| `interval_select` | `ref_r_mdatools` | 0.707107 | 0 | 41 |
| `so_pls` | `ref_r_multiblock` | 0.631386 | 3.88218 | 16 |
| `sparse_pls_da` | `ref_r_spls` | 0.494747 | 2.16011e-08 | 45 |
| `one_se_rule` | `ref_r_pls` | 0.49223 | 0 | 41 |
| `boosting_pls` | `ref_r_mboost` | 0.45708 | 8.88178e-16 | 41 |
| `wvc_threshold_select` | `ref_r_plsvarsel` | 0.417029 | 0 | 41 |
| `bagging_pls` | `ref_python_scikit_learn` | 0.211826 | 0.556503 | 41 |
| `kernel_pls_rbf` | `ref_r_kernlab_pls` | 0.185322 | 0 | 41 |
| `robust_pls` | `ref_r_chemometrics` | 0.176278 | 2.66454e-15 | 41 |
| `o2pls` | `ref_r_omicspls` | 0.140165 | 8.88178e-16 | 41 |
| `pls_glm` | `ref_r_plsrglm` | 0.136561 | 0.231014 | 45 |
| `continuum_regression` | `ref_r_jico` | 0.0944624 | 1.77636e-15 | 41 |
| `random_subspace_pls` | `ref_python_scikit_learn` | 0.0924876 | 0.744132 | 41 |
| `group_sparse_pls` | `ref_r_sgpls` | 0.0884986 | 0 | 41 |
| `pls` | `ref_python_scikit_learn` | 0.0532762 | 3.9968e-15 | 1 |
| `sparse_simpls` | `ref_r_spls` | 0.0532504 | 5.32907e-15 | 41 |
| `cppls` | `ref_r_pls` | 0.0345951 | 1.77636e-15 | 41 |
| `rosa` | `ref_r_multiblock` | 0.0296069 | 0 | 45 |
| `di_pls` | `ref_python_diplslib` | 0.017799 | 0.0709344 | 16 |
| `weighted_pls` | `ref_python_scikit_learn` | 0.00416263 | 1.77636e-15 | 41 |
| `ridge_pls` | `ref_python_scikit_learn` | 0.00164031 | 1.77636e-15 | 41 |

## Priorite immediate apres ce recalcul

1. Corriger les drifts binding internes maintenant visibles:
   `so_pls`, `on_pls`, `pls_glm`, `di_pls`, puis JS/WASM
   `random_subspace_pls` et `bagging_pls`.
2. Corriger ensuite les six methodes `> 1` cote reference.
3. Recalculer la gate stricte apres chaque groupe de corrections pour eviter
   de reintroduire des cellules sans score numerique.
