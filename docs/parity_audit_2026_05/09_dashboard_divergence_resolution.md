# Resolution des divergences dashboard > 1e-3 - 2026-05-20

Source stricte: `benchmarks/cross_binding/results/parity_30x30_strict.csv`,
regenere apres correction puis relu par `docs/_static/bench-data.json`.

## Contrat retenu

La resolution ne cache pas les gates divergentes.

- Gate 1, bindings pls4all -> C++: `binding_parity_max_diff <= 1e-6`.
- Gate 2, reference externe: `reference_parity_rmse_rel <= 1e-3`.
- `rmse_rel <= 1e-6` reste affiche comme exact.
- `MethodSpec.rmse_rel_tol` reste un budget diagnostic/variant. Il ne peut
  pas transformer une divergence stricte en parite de release.
- Quand une methode C++ est utile mais ne suit pas le contrat exact du
  package externe, la ligne reste divergente jusqu'a ce qu'une variante
  package-compatible soit exposee et testee.

## Etat mesure apres corrections

- CSV strict: `3155` records, dont `2856` comparaisons de reference finies.
- Gate 1 bindings: `0` methode et `0` ligne au-dessus de `1e-6`.
- Gate 2 references: `1509` lignes restent au-dessus de `1e-3`.
- Dashboard regenere: `513` cellules restent affichees comme `divergent`
  cote reference stricte, pas comme `qualitative`.

## Divergences resolues

| methode | type de fix | resultat strict |
|---|---|---|
| `di_pls` | R embarquait un `model.cpp` stale. Le vendored `bindings/r/.../model.cpp` a ete resynchronise avec `cpp/src/core/model.cpp`. | binding max `3.33e-15`; reference max `3.14e-14`; Gate 1 et Gate 2 passent. |
| `wvc_select` | R embarquait un `wvc_selection.cpp` stale. Le vendored R a ete resynchronise. | binding max `0`; reference max `0`; Gate 1 et Gate 2 passent. |
| `wvc_threshold_select` | Meme drift vendored R que `wvc_select`. | binding max `0`; reference max `0`; Gate 1 et Gate 2 passent. |
| `on_pls` | Divergence de dashboard due a des lignes stale. Rerun strict avec les artefacts courants. | binding max `0`; reference max `2.68e-05`; Gate 1 et Gate 2 passent. |
| `so_pls` | Divergence de dashboard due a des lignes stale. Rerun strict avec les artefacts courants. | binding max `0`; reference max `5.12e-06`; Gate 1 et Gate 2 passent. |
| `bagging_pls` | Le C++ utilisait `std::uniform_int_distribution` sur `mt19937_64`; la sequence n'est pas portable entre libstdc++ et Emscripten/libc++. Remplace par SplitMix64 + tirage borne portable, puis sync R/WASM. | Gate 1 passe partout, binding max `2.22e-15`. Gate 2 sklearn reste divergent: ref max `0.234567`. |
| `random_subspace_pls` | Meme probleme de RNG portable, plus `std::shuffle` non portable. Remplace par Fisher-Yates SplitMix64, puis sync R/WASM. | Gate 1 passe partout, binding max `0`. Gate 2 sklearn reste divergent: ref max `0.111212`. |

## Divergences restantes et contrat

Ces methodes ne sont pas marquees comme passees. Elles restent des echecs
stricts Gate 2 dans le dashboard. La resolution correcte est une variante
package-compatible ou une reimplementation exacte du package externe; elargir
la tolerance serait une dissimulation.

| methode | max rmse_rel | explication du non-fix strict |
|---|---:|---|
| `st_select` | `1.61245` | plsVarSel et C++ ne suivent pas le meme chemin de shrink/selection. Variante plsVarSel necessaire. |
| `stability_select` | `1.34164` | Selection stochastic/RNG plsVarSel non reproduite. Variante package necessaire. |
| `irf_select` | `1.31426` | Chaine stochastic libPLS differente. Une compat libPLS exacte doit fixer seed, folds et trajectoire. |
| `spa_select` | `1.18322` | Heuristique de depart SPA differente de plsVarSel. Variante plsVarSel necessaire. |
| `pso_select` | `1.04881` | Dynamique BinaryPSO/RNG pyswarms differente. Variante pyswarms necessaire. |
| `vissa_select` | `1` | Trajectoire VISSA/seed CV auswahl differente. Variante auswahl necessaire. |
| `variable_select_sr` | `1` | Formule SR package vs energie de reconstruction C++. Variante plsVarSel necessaire. |
| `ga_select` | `1` | GA stochastic plsVarSel non reproduit. Variante package necessaire. |
| `rep_select` | `1` | Vote/repetition C++ different du meilleur candidat CV plsVarSel. Variante package necessaire. |
| `pls_cox` | `0.95641` | plsRcox utilise le pipeline Cox/Bastien; C++ est une approximation PLS-then-Cox. |
| `vip_spa_select` | `0.912871` | auswahl fait une recherche SPA multi-departs/CV differente. |
| `pls_lda` | `0.906674` | sklearn PLS+LDA ne renvoie pas le meme contrat de scores que C++. |
| `uve_select` | `0.894427` | UVE stochastic plsVarSel et survivor set C++ divergent. |
| `random_frog_select` | `0.894427` | Random Frog libPLS/RNG non reproduit exactement. |
| `bve_select` | `0.866025` | Arret step-count/min-features plsVarSel different du VIP-cut C++. |
| `ipw_select` | `0.845154` | Survivor set plsVarSel different du top-k C++. |
| `approximate_press` | `0.82606` | R `pls` compare un PRESS/LOO exact; C++ expose une approximation. |
| `n_pls` | `0.825789` | tensorly/PARAFAC proxy et N-PLS C++ ne sont pas le meme algorithme. |
| `cars_select` | `0.816497` | enpls MC/CARS analogue mais pas trajectoire CARS C++ exacte. |
| `emcuve_select` | `0.816497` | Ensemble MCUVE stochastic plsVarSel non reproduit. |
| `pls_logistic` | `0.796761` | sklearn logistic sur scores vs IRLS C++: contrat different. |
| `variable_select_vip` | `0.774597` | Top-k VIP comparable, conventions package differentes. |
| `interval_select` | `0.707107` | Protocole CV/greedy mdatools different. |
| `sparse_pls_da` | `0.494747` | `splsda` renvoie des classes dures; C++ renvoie des scores/probas. |
| `one_se_rule` | `0.49223` | R `pls` execute un vrai CV; C++ consomme une matrice fold-RMSE. |
| `boosting_pls` | `0.45708` | mboost booste des weak learners lineaires; C++ booste des weak learners PLS. |
| `bagging_pls` | `0.234567` | Binding fixe. Pour Gate 2, sklearn `BaggingRegressor` impose son bootstrap/RNG/base-estimator. Variante sklearn-compatible necessaire. |
| `kernel_pls_rbf` | `0.185322` | Deflation/centrage kernel differents de kernlab. |
| `robust_pls` | `0.176278` | PRM chemometrics vs Huber/IRLS C++. |
| `o2pls` | `0.140165` | OmicsPLS iteratif vs peel-then-PLS C++. |
| `random_subspace_pls` | `0.111212` | Binding fixe. Pour Gate 2, sklearn `BaggingRegressor(max_features=...)` impose sa selection de sous-espaces. Variante sklearn-compatible necessaire. |
| `continuum_regression` | `0.0944624` | Parametrisation JICO differente. |
| `group_sparse_pls` | `0.0884986` | Penalisation groupe sgPLS differente. |
| `pls_glm` | `0.0880831` | plsRglm/Bastien IRLS exact different du PLS+IRLS C++. Variante `pls_glm_plsrglm` necessaire. |
| `pls` | `0.0532762` | Une ligne externe non-canonique diverge encore; les bindings pls4all stricts restent verts. |
| `cppls` | `0.0345951` | CPPLS R Liland vs Powered-PLS Indahl C++: variantes differentes. |
| `rosa` | `0.0296069` | Strategie greedy multiblock R `multiblock` differente. |

## Validation executee

- rebuild C++: `build/dev-release`, `build/blas-on`, `build/omp-on`,
  `build/blas-omp`;
- reinstall R: `R CMD INSTALL bindings/r/pls4all`;
- rebuild JS/WASM: `npm run build:wasm && npm run build:ts &&
  npm run copy:wasm`;
- rerun strict cible sur `di_pls`, `pls_glm`, `on_pls`, `so_pls`,
  `wvc_select`, `wvc_threshold_select`, `bagging_pls`,
  `random_subspace_pls`;
- recomputation dashboard depuis le CSV strict;
- `pytest -q benchmarks/cross_binding/tests/test_orchestrator_parity.py
  benchmarks/cross_binding/tests/test_landing_dashboard.py
  benchmarks/cross_binding/tests/test_registry_common.py`;
- `python -m benchmarks.parity_timing.lockfile --check`;
- `ctest --test-dir build/blas-omp --output-on-failure`;
- `npm run test:methods` dans `bindings/js`.
