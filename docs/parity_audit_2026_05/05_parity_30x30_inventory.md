# Inventaire parite stricte 30x30 - 2026-05-20

Source CSV: `benchmarks/cross_binding/results/parity_30x30_strict.csv`

Commande de generation principale:

```bash
PYTHONPATH=. python benchmarks/cross_binding/orchestrator.py \
  --algorithms all --sizes 30x30 --threads 1 \
  --libp4a-build all-cpu --n-runs 1 \
  --reference-backends registry \
  --out-csv benchmarks/cross_binding/results/parity_30x30_strict.csv \
  --force --flush-each-cell --timeout 1200
```

Rerun cible effectue apres correction de l'adaptateur de parametres:

```bash
PYTHONPATH=. python benchmarks/cross_binding/orchestrator.py \
  --algorithms recursive_pls --sizes 30x30 --threads 1 \
  --libp4a-build all-cpu --n-runs 1 \
  --reference-backends registry \
  --out-csv benchmarks/cross_binding/results/parity_30x30_strict.csv \
  --resume-existing --force --flush-each-cell --timeout 1200
```

Politique appliquee: toute methode avec reference executable doit etre
`reference_parity_rmse_rel <= 1e-3` sur 30x30. Les seules exemptions sont les
methodes reellement `paper_only`.

## Resume global

- Lignes CSV: 3071.
- Lignes OK execution: 3060. Echecs execution: 11.
- Lignes avec metrique de reference finie: 2608.
- Lignes reference `> 1e-3`: 1745.
- Lignes de binding divergence `> 1e-3`: 89.
- Methodes strictement OK: 10.
- Methodes `paper_only` exemptes: 4.
- Methodes avec echecs d'execution: 10.
- Methodes avec drift binding/backend: 13.
- Methodes avec drift reference uniforme: 36.

Les repetitions par ligne sont attendues: une meme divergence methode est
repliquee sur plusieurs bindings/builds. L'analyse par methode ci-dessous est
donc le niveau de pilotage pour les fixes.

## Exceptions paper_only valides

Ces methodes n'ont pas de reference executable dans le registre courant et sont
les seules exceptions a la gate `1e-3`:

- `mir_pls`
- `fused_sparse_pls`
- `sipls_select`
- `scars_select`

Toutes les autres methodes doivent converger vers `<= 1e-3`, y compris les
references sanctionnees `python_nirs4all`.

## Diagnostic important BLAS/OMP

Sur le dataset 30x30, `so_pls`, `rosa`, `mb_pls`, `lw_pls`, `aom_pls` et
`pop_pls` ne montrent pas un drift specifique BLAS+OMP: la divergence a la
reference est uniforme sur `dev-release`, `blas-on`, `omp-on`, `blas-omp` et
les bindings internes ont une divergence binding nulle ou numeriquement
negligeable.

Conclusion: c'est un vrai drift par rapport a la reference executable, ou un
adaptateur/oracle de reference incorrect. Ce n'est pas un probleme kernel
BLAS+OMP localise.

Focus methodes demandees:

| methode | reference | rmse_rel reference | binding max_diff | diagnostic |
|---|---:|---:|---:|---|
| `so_pls` | `r_multiblock` | 0.631386 | 0 | drift reference uniforme |
| `rosa` | `r_multiblock` | 0.0296069 | 0 | drift reference uniforme |
| `mb_pls` | `python_nirs4all` | 0.0845981 | <= 6.3e-15 | drift reference uniforme |
| `lw_pls` | `python_nirs4all` | 0.0542661 | <= 1.8e-15 | drift reference uniforme |
| `aom_pls` | `python_nirs4all` | 0.392498 | <= 1.8e-15 | drift reference uniforme |
| `pop_pls` | `python_nirs4all` | 2.77371 | 0 | drift reference uniforme |
| `aom_preprocess` | `python_nirs4all` | 0 | 0 | OK strict |

## Correction deja faite dans ce passage

`recursive_pls` echouait en 30x30 parce que le registre gardait
`window_size=60`, invalide pour `n=30`. L'adaptateur
`bench_registry_common.adapted_params()` borne maintenant `window_size` par la
taille du dataset. Apres rerun:

- `recursive_pls` passe la reference Python et R a `~4.15e-15`.
- C++/Python/R/MATLAB/JS passent la binding gate.
- Deux lignes Julia (`dev-release`, `blas-on`) echouent encore a l'execution:
  probleme binding/env separe, pas le bug `window_size`.

## Echecs d'execution restants

| methode | backend | lignes | commentaire |
|---|---:|---:|---|
| `recursive_pls` | `julia` | 2 | `dev-release` et `blas-on` echouent dans le script Julia; `omp-on` et `blas-omp` passent |
| `pls` | `js_wasm` | 1 | Node imprime `Node.js v22.21.1`, script non-zero |
| `pcr` | `js_wasm` | 1 | Node imprime `Node.js v22.21.1`, script non-zero |
| `opls` | `js_wasm` | 1 | Node imprime `Node.js v22.21.1`, script non-zero |
| `pls_diagnostic_t2` | `js_wasm` | 1 | Node imprime `Node.js v22.21.1`, script non-zero |
| `pls_diagnostic_q` | `js_wasm` | 1 | Node imprime `Node.js v22.21.1`, script non-zero |
| `pls_monitoring` | `js_wasm` | 1 | Node imprime `Node.js v22.21.1`, script non-zero |
| `vissa_select` | `js_wasm` | 1 | Node imprime `Node.js v22.21.1`, script non-zero |
| `pso_select` | `js_wasm` | 1 | Node imprime `Node.js v22.21.1`, script non-zero |
| `gpr_pls` | `js_wasm` | 1 | Node imprime `Node.js v22.21.1`, script non-zero |

## Methodes strictement OK

| methode | reference | max rmse_rel |
|---|---:|---:|
| `ecr` | `matlab_libpls` | 1.59738e-06 |
| `ds` | `r_base` | 1.69733e-07 |
| `pds` | `r_base` | 3.57257e-08 |
| `missing_aware_nipals` | `r_softimpute` | 1.61753e-14 |
| `aom_preprocess` | `python_nirs4all` | 0 |
| `bipls_select` | `r_mdatools` | 0 |
| `iriv_select` | `matlab_libpls` | 0 |
| `randomization_select` | `r_pls_stats` | 0 |
| `t2_select` | `r_plsvarsel` | 0 |
| `wvc_select` | `r_plsvarsel` | 0 |

## Methodes avec drift binding/backend

Ces methodes ont au moins une ligne interne dont le binding differe d'un autre
binding/build de plus de `1e-3`. Le drift reference peut aussi exister.

| methode | reference | max rmse_rel | lignes binding >1e-3 | backends impliques |
|---|---:|---:|---:|---|
| `pls_glm` | `r_plsrglm` | 5.85684e+06 | 1 | `js_wasm` |
| `variable_select_sr` | `r_plsvarsel` | 1.09545 | 1 | `js_wasm` |
| `uve_select` | `r_plsvarsel` | 1.09545 | 16 | `r_tier1`, `r_tier2`, `r_pls_compat`, `r_mdatools_compat` |
| `pls_cox` | `r_plsrcox` | 1.01159 | 1 | `js_wasm` |
| `emcuve_select` | `r_plsvarsel` | 1 | 16 | `r_tier1`, `r_tier2`, `r_pls_compat`, `r_mdatools_compat` |
| `variable_select_vip` | `r_plsvarsel` | 0.894427 | 1 | `js_wasm` |
| `bve_select` | `r_plsvarsel` | 0.866025 | 16 | `r_tier1`, `r_tier2`, `r_pls_compat`, `r_mdatools_compat` |
| `variable_select_coef` | `r_pls` | 0.632456 | 1 | `js_wasm` |
| `shaving_select` | `r_plsvarsel` | 0.632456 | 16 | `r_tier1`, `r_tier2`, `r_pls_compat`, `r_mdatools_compat` |
| `bagging_pls` | `python_scikit_learn` | 0.211826 | 1 | `js_wasm` |
| `random_subspace_pls` | `python_scikit_learn` | 0.0924876 | 1 | `js_wasm` |
| `pls_diagnostic_dmodx` | `r_mdatools` | 0.0835848 | 1 | `js_wasm` |
| `di_pls` | `python_diplslib` | 0.017799 | 17 | `js_wasm`, R bindings/compat |

## Methodes avec drift reference uniforme

Ces methodes ont des bindings/builds internes coherents entre eux, mais
divergent de la reference executable. C'est la liste prioritaire des fixes
methode ou adaptateur-reference.

| methode | reference | max rmse_rel | lignes >1e-3 |
|---|---:|---:|---:|
| `pls_qda` | `python_scikit_learn` | 2.85617 | 17 |
| `pop_pls` | `python_nirs4all` | 2.77371 | 41 |
| `st_select` | `r_plsvarsel` | 1.61245 | 41 |
| `sparse_pls_da` | `r_spls` | 1.36404 | 17 |
| `stability_select` | `r_plsvarsel` | 1.34164 | 41 |
| `irf_select` | `matlab_libpls` | 1.31426 | 41 |
| `spa_select` | `r_plsvarsel` | 1.18322 | 41 |
| `on_pls` | `python_onpls` | 1.00264 | 17 |
| `ga_select` | `r_plsvarsel` | 1 | 41 |
| `rep_select` | `r_plsvarsel` | 1 | 41 |
| `vip_spa_select` | `python_auswahl` | 0.912871 | 41 |
| `pls_lda` | `python_scikit_learn` | 0.906674 | 17 |
| `random_frog_select` | `matlab_libpls` | 0.894427 | 41 |
| `ipw_select` | `r_plsvarsel` | 0.845154 | 41 |
| `approximate_press` | `r_pls` | 0.82606 | 41 |
| `n_pls` | `python_tensorly` | 0.825789 | 41 |
| `cars_select` | `r_enpls` | 0.816497 | 41 |
| `pls_logistic` | `python_scikit_learn` | 0.796761 | 17 |
| `interval_select` | `r_mdatools` | 0.707107 | 41 |
| `so_pls` | `r_multiblock` | 0.631386 | 17 |
| `one_se_rule` | `r_pls` | 0.49223 | 41 |
| `boosting_pls` | `r_mboost` | 0.45708 | 41 |
| `wvc_threshold_select` | `r_plsvarsel` | 0.417029 | 41 |
| `aom_pls` | `python_nirs4all` | 0.392498 | 41 |
| `kernel_pls_rbf` | `r_kernlab_pls` | 0.185322 | 41 |
| `robust_pls` | `r_chemometrics` | 0.176278 | 41 |
| `o2pls` | `r_omicspls` | 0.140165 | 17 |
| `continuum_regression` | `r_jico` | 0.0944624 | 41 |
| `group_sparse_pls` | `r_sgpls` | 0.0884986 | 41 |
| `mb_pls` | `python_nirs4all` | 0.0845981 | 41 |
| `lw_pls` | `python_nirs4all` | 0.0542661 | 41 |
| `sparse_simpls` | `r_spls` | 0.0532504 | 41 |
| `cppls` | `r_pls` | 0.0345951 | 41 |
| `rosa` | `r_multiblock` | 0.0296069 | 17 |
| `weighted_pls` | `python_scikit_learn` | 0.00416263 | 41 |
| `ridge_pls` | `python_scikit_learn` | 0.00164031 | 41 |

## Ordre de fixes propose

1. Stabiliser le harness et les bindings qui faussent la lecture:
   `recursive_pls` est corrige cote parametres, restent les echecs `julia`
   et les echecs `js_wasm`.
2. Corriger les references sanctionnees `python_nirs4all` non conformes:
   `pop_pls`, `aom_pls`, `mb_pls`, `lw_pls`. Ces methodes ne sont pas
   exemptes.
3. Corriger les drift multiblocs externes: `so_pls`, `rosa`, puis `on_pls`.
4. Corriger les gros drift prediction/classification:
   `pls_glm`, `pls_qda`, `sparse_pls_da`, `pls_lda`, `pls_logistic`,
   `pls_cox`.
5. Corriger les families selection/mask contre `plsVarSel`, `libPLS`,
   `mdatools`, `auswahl`: commencer par les cas avec binding drift
   (`bve`, `uve`, `emcuve`, `shaving`), puis les drifts uniformes.
6. Repasser la matrice 30x30 jusqu'a obtenir zero methode avec reference
   executable au-dessus de `1e-3`.
