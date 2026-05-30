# Tier 0 — stochastic-method inventory (authoritative)

Derived from source: the set of C-kernel files that contain a file-local
`splitmix64` generator (`grep -rln splitmix64 cpp/src/core/`) is the *true*
stochastic surface. Everything else is deterministic or a Python composite that
bypasses the C kernel.

## The 14 splitmix64-using core files

**12 stochastic selectors** (RNG-parity scope):
`cars_selection`, `emcuve_selection`, `ga_selection`, `iriv_selection`,
`irf_selection`, `pso_selection`, `random_frog_selection`,
`randomization_selection`, `scars_selection`, `stability_selection`,
`uve_selection`, `vissa_selection`.

**2 infrastructure** (also RNG, but not selector parity):
`model.cpp` (randomized-SVD component sampling — context-seed + component
offset), `validation.cpp` (CV fold assignment).

## Dashboard parity policy (2026-05-30)

The dashboard now separates three selector outcomes instead of collapsing all
feature-mask differences into a red failure:

- `selection_exact` → exact selector parity (`Jaccard = 1.00`).
- `selection_divergence_allowed` → `cross_check` with the recorded reason
  (for example `t2_select`, where the statistic matches but the loading-weight
  convention differs).
- documented `selection_mismatch` methods → `cross_check` with an amber
  `BD J` badge and a tooltip explaining why exact feature-mask parity is not
  expected.

Unmapped `selection_mismatch` rows still render as real `divergent` failures.
This keeps the matrix strict by default while making the known RNG/noise/model
exceptions explicit.

## Post-investigation selector verdicts (2026-05-30)

| method | canonical default | reference + its RNG | verdict |
|---|---|---|---|
| `uve_select` | R `plsVarSel::mcuve_pls` | R-MT | canonical dashboard path exact; legacy C++ path remains a documented `BD J` cross-check because its noise count, RNG, and folds differ |
| `emcuve_select` | R `mcuve_pls` ensemble | R-MT | exact in the current matrix |
| `stability_select` | R `mcuve_pls` stability analogue | R-MT | `BD J` cross-check — R mcuve stability and n4m legacy stability use different subsampling/std/RNG-noise conventions |
| `cars_select` | R `enpls.fs` analog | R-MT | `BD J` cross-check — executable reference is an enpls importance analog, not n4m CARS proper |
| `spa_select` | R `plsVarSel::spa_pls` | R-MT | `BD J` cross-check — R uses random subsampling + Wilcoxon/permutation logic; n4m legacy SPA is deterministic |
| `ga_select` | R GA selector | R-MT | cross_check — GA is highly RNG/impl-sensitive |
| `iriv_select` | NumPy IRIV port | numpy PCG64/default_rng | `BD J` cross-check — NumPy/scipy rank tests and n4m legacy candidate generation are not the same algorithmic path |
| `random_frog_select` | auswahl `RandomFrog` | numpy RandomState-MT | `BD J` cross-check — different MCMC proposal chain and random stream |
| `vissa_select` | auswahl `VISSA` | numpy RandomState-MT | `BD J` cross-check — different submodel sampling/aggregation path |
| `irf_select` | auswahl `IntervalRandomFrog` | numpy RandomState-MT | `BD J` cross-check — different interval proposal chain |
| `randomization_select` | R randomization (`pvals<alpha`) | R-MT | exact in the current dashboard through the canonical base-R adapter; native R-MT C++ path remains an optional future parity hardening |
| `rep_select` | R `plsVarSel::rep` | R-MT | `BD J` cross-check — repeated VIP Monte-Carlo selection vs bounded backward-elimination steps |
| `shaving_select` | R shaving | R-MT | `BD J` cross-check — CV-error survivor path vs explicit step-count trajectory |
| `bve_select` | R `plsVarSel::bve` | R-MT | `BD J` cross-check — R sampling/VIP elimination vs deterministic greedy CV-RMSE elimination |
| `ipw_select` | R `plsVarSel::ipw` | R-MT | `BD J` cross-check — R RC filter/package thresholds vs n4m top-k iterative scores |
| `st_select` | R `plsVarSel::st` | R-MT | `BD J` cross-check — relative shrink ladder vs absolute coefficient thresholds |
| `pso_select` | Python `pyswarms` BinaryPSO | NumPy/Python PSO RNG | `BD J` cross-check — different swarm updates, velocity handling, and RNG streams |
| `scars_select` | NumPy SCARS port | NumPy RNG | `BD J` cross-check — different subsampling, shrinkage, and random stream details |
| `wvc_select` | R WVC | deterministic/current R path | exact in the current matrix |
| `wvc_threshold_select` | R WVC threshold | deterministic/current R path | `BD J` cross-check — median-scaled R threshold vs n4m min-selected threshold convention |
| `t2_select` | R `plsVarSel::T2_pls` | loading-weight convention, not RNG | documented `selection_divergence_allowed` cross-check |

Notes:
- The **numpy/sklearn composites** (`bagging_pls`, `boosting_pls`,
  `random_subspace_pls`, `n_pls`, `pls_qda`, `pls_glm`, `pls_cox`,
  `pls_logistic`, `group_sparse_pls`) BYPASS the C kernel — `not_available` for
  C-kernel bindings, never an RNG fix.
- `iriv_select` uses **PCG64/default_rng** in the NumPy reference path.
- The auswahl trio (`random_frog`, `vissa`, `irf`) uses **numpy RandomState
  MT19937** through sklearn `check_random_state`.

## RNG engines required (and status)

| engine | reproduces | status |
|---|---|---|
| splitmix64 | n4m's own legacy streams | exists; legacy selector streams are intentionally preserved |
| PCG64 | `numpy.random.default_rng` | **DONE** (`rng_pcg64`) — covers IRIV-style NumPy references |
| R-MT + Inversion | base R `set.seed`/`runif`/`rnorm` | **DONE** (`rng_mt_r`, bit-exact) — covers the R selectors |
| numpy RandomState-MT | legacy `numpy.random.RandomState` | **DONE** (`rng_numpy_mt`, bit-exact) — covers auswahl-style references |

## Completed rollout

1. **Tier 0 inventory** — done.
2. **RNG engines** — splitmix64 preservation plus PCG64, R-MT, and
   numpy-RandomState-MT engines are available behind the additive
   `n4m_rng_kind_t` ABI.
3. **UVE pilot** — R-exact MCUVE path is available through `rng_kind=MT_R`
   while the legacy splitmix path remains byte-compatible.
4. **Dashboard contract** — known RNG/noise/model selector mismatches are
   rendered as documented `cross_check`/`BD J` cells; unmapped mismatches remain
   red failures.

Further hardening would mean porting entire external selector algorithms into
C++ (for example R `bve_pls`, R `spa_pls`, auswahl VISSA/RandomFrog). That is a
method implementation project, not an RNG-engine swap.
