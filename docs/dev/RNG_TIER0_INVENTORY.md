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

## Per-method decision (canonical path × reference RNG × verdict)

| method | canonical default | reference + its RNG | verdict |
|---|---|---|---|
| `uve_select` | R `plsVarSel::mcuve_pls` | R-MT | **compat (Tier C pilot)** — reference IS the target algorithm |
| `emcuve_select` | R `mcuve_pls` ensemble | R-MT | compat — ensemble of mcuve |
| `stability_select` | R `mcuve_pls` (stability) | R-MT | compat |
| `cars_select` | R `enpls.fs` (closest analog) | R-MT | **cross_check** — analog, not CARS proper (registry says so) |
| `spa_select` | R (deterministic SPA) | — (R but deterministic) | canonical/compat — SPA is deterministic; verify no RNG |
| `ga_select` | R GA selector | R-MT | cross_check — GA is highly RNG/impl-sensitive |
| `iriv_select` | **NumPy port** (`default_rng`) | numpy PCG64 | compat via **PCG64 (exists)** — not R |
| `random_frog_select` | auswahl `RandomFrog` | numpy **RandomState-MT** | compat via **Tier A** RNG |
| `vissa_select` | auswahl `VISSA` | numpy RandomState-MT | compat via Tier A |
| `irf_select` | auswahl `IntervalRandomFrog` | numpy RandomState-MT | compat via Tier A |
| `randomization_select` | R randomization (`pvals<alpha`) | R-MT | compat — NOTE C kernel uses `<=` vs R `<` (off-by-boundary, Codex) |
| `rep_select` | R `plsVarSel::rep` | R-MT | compat / cross_check |
| `shaving_select` | R shaving | R-MT | cross_check |
| `bve_select` | R `plsVarSel::bve` | R-MT | compat / cross_check |
| `ipw_select` | R `plsVarSel::ipw` | R-MT | compat / cross_check |
| `st_select` | R `plsVarSel::st` | R-MT | compat / cross_check |
| `pso_select` | (C kernel; PSO) | splitmix | review — is there an external ref? |
| `scars_select` | (C kernel) | splitmix | review |
| `wvc_select` / `wvc_threshold_select` | R | R-MT/deterministic | review (largely deterministic) |
| `t2_select` | R `plsVarSel::T2_pls` | (loading-weight convention, not RNG) | cross_check (already documented, #13) |

Notes:
- The **numpy/sklearn composites** (`bagging_pls`, `boosting_pls`,
  `random_subspace_pls`, `n_pls`, `pls_qda`, `pls_glm`, `pls_cox`,
  `pls_logistic`, `group_sparse_pls`) BYPASS the C kernel — `not_available` for
  C-kernel bindings, never an RNG fix.
- `iriv_select` needs **PCG64** (already shipped), not R-MT — cheap Tier-A-style
  win.
- The auswahl trio (`random_frog`, `vissa`, `irf`) needs **numpy RandomState
  MT19937** (Tier A engine), because auswahl uses sklearn `check_random_state`.

## RNG engines required (and status)

| engine | reproduces | status |
|---|---|---|
| splitmix64 | n4m's own current streams | exists (file-local ×14) — **freeze with golden tests** |
| PCG64 | `numpy.random.default_rng` | **exists** (`rng_pcg64`) — covers iriv |
| R-MT + Inversion | base R `set.seed`/`runif`/`rnorm` | **DONE** (`rng_mt_r`, bit-exact) — covers the R selectors |
| numpy RandomState-MT | legacy `numpy.random.RandomState` | **Tier A — to build** — covers the auswahl trio |

## Order of work (Codex-aligned)
1. **Tier 0** (this doc) — done.
2. **Tier B golden tests** — freeze each of the 14 splitmix streams before any
   refactor (default `rng_kind` must stay byte-identical).
3. **Tier B abstraction** — `n4m_rng_kind_t` enum + `n4m_config_set_rng_kind`
   (additive ABI) + a tagged `n4m_rng` dispatch; wire the pilot (uve) first.
4. **Tier A** — numpy RandomState-MT engine + bit-exact test vs numpy.
5. **Tier C** — UVE pilot: with the abstraction + R-MT, port mcuve_pls's noise
   model + draw order + fold semantics → Jaccard 1.0; the other R selectors
   follow the same pattern (compat) or stay documented cross_check (analogs).
