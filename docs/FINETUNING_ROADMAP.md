# Finetuning implementation roadmap — native `libn4m` HPO, all languages

**Status:** implementation roadmap (companion to [`NATIVE_FINETUNING.md`](NATIVE_FINETUNING.md), which holds the *why* and the architecture). Revised twice: after an internal adversarial pass (corrections marked **[review]**) and after a **Codex read-only review** (corrections marked **[codex]**, transcript in [`reviews/finetuning-roadmap/`](reviews/finetuning-roadmap/)).
**Goal:** a **native finetuner in `nirs4all-methods`** — every HPO algorithm from the strategy implemented once in C++/C-ABI, reachable **from a nirs4all pipeline in every language** (Python, R, MATLAB/Octave, WASM), with **parity, docs and bibliography like every other method**, while **Optuna stays in Python as an optional sampler**.
**End-user acceptance:** in a pipeline, a model step's `finetune_params` resolves to a *native* sampler (`sampler:"random"|"sobol"|"cmaes"|"tpe"|…`, optionally `pruner:"asha"|"hyperband"|"median"`) and runs identically in Python/R/MATLAB/WASM; `sampler:"optuna"` keeps the Python engine. Same seed → bit-identical trial sequence across bindings.

---

## 0. How to read this roadmap

- **Unit of delivery.** The finetuner is a **new C-ABI category** (`optimization`, header `cpp/include/n4m/optimization.h`, included by `n4m.h`) with **one handle-based ask/tell surface** and **many samplers/pruners behind two enums**. The *ABI surface* lands **once** (Phase F0); each later sampler is a `Sampler`/`Pruner` subclass + a parity fixture + a doc/bib page.
- **Sampler ⟂ pruner is a hard split. [review]** A sampler proposes; a pruner/scheduler decides early-stopping over intermediate scores. They **compose** (`sampler=tpe` × `pruner=median`), like Optuna's `sampler=`/`pruner=`. `n4m_optimizer_create` takes both via an options struct (§4-F0). Never flatten pruners into the sampler enum.
- **Enum values ARE an ABI-minor change. [codex]** Adding a value to `n4m_sampler_kind_t`/`n4m_pruner_kind_t` exports no new *symbol*, but the repo's own rule classes a new enum value as **ABI MINOR** (`cpp/include/n4m/n4m_version.h:11`). Therefore F0 **reserves the full numeric range of every enum now**; unimplemented values return `N4M_ERR_UNSUPPORTED` until their phase lands — so no later phase touches the ABI at all. (Bindings still add their language-side enum mirror + docs when a value is activated; that is not a `libn4m` ABI change.)
- **Definition of done per sampler/pruner** — the 6-surface method-add checklist (`.github/PULL_REQUEST_TEMPLATE/method-add.md`) applies, minus new symbols after F0:
  1. **C++ core** — a `Sampler`/`Pruner` subclass under `cpp/src/core/optimization/<name>.{hpp,cpp}`. **[review]** New core `.cpp` are **not** auto-globbed — list them explicitly in `add_library(n4m_core OBJECT …)` in `cpp/src/CMakeLists.txt` (only `cpp/src/c_api/*.cpp` is `CONFIGURE_DEPENDS`-globbed via `n4m_targets.cmake`, so `c_api_optimization.cpp` auto-compiles).
  2. **ABI** — no new symbol; activate a reserved enum value. Enums are 4-byte (`N4M_STATIC_ASSERT(sizeof(...)==4, …)`).
  3. **Tests** — cases in `cpp/tests/test_optimization.cpp`, compiled into the single `n4m_tests` binary by the **hand-rolled zero-dependency harness** (`cpp/tests/harness.hpp` + `main.cpp` — *not* doctest, despite older CLAUDE.md wording); register the file in `cpp/tests/CMakeLists.txt`. Cover the ask/tell contract, determinism-given-seed, a converging non-trivial objective, and a correct prune.
  4. **Reference & parity — needs a NEW schema, the prediction-oriented one does not fit. [codex]** `benchmarks/parity_timing/registry.py`'s `MethodSpec` (`:8668`) and `parity/scripts/per_method_parity.py` compare **prediction arrays** (`:179`, `:265`) — they cannot hold a pruner decision fixture, a TPE/CMA state fixture, or an ask/tell trace. Add an **`HpoSpec`** registry schema + dedicated comparators (`parity/comparator/{pruner_decisions,sampler_state,study_trace}.py`) + fixtures (`parity/fixtures/opt_<name>_v1.json`) + an **explicit CI job** (the cross-binding runner is currently *manual*, not a gate — `.github/workflows/parity-gate.yml:157`). This machinery is itself an F0/F2 deliverable, not a reuse.
  5. **Catalog** — `catalog/methods/optimization.<name>.yaml` (regenerated from `catalog/methods.yaml` via `split_legacy_methods.py`, which maps an **arbitrary dotted `method_id`** to a file — two- or three-segment both exist, e.g. `catalog/methods.yaml:675`). Specify `namespace`/`leaf`/`c_surface` + reference coverage; run `catalog/scripts/validate.py --strict-abi` **and** `--check-references` (a separate gate; the donor-category allow-list at `validate.py:43` must gain `optimization`).
  6. **Bindings + docs** — thin wrapper where in scope; `docs/methods/<name>.md` with **bibliography**; `CHANGELOG.md`; dashboard timing (`benchmarks/cross_binding/donor_ops.py`) if benchmarked.
- **Parity model reused, adapted for stochasticity** (§3): cross-binding stays the tight **1e-12** gate (bindings vs C++ native, same seed → same study); reference parity is per-sampler (bit-exact only for deterministic math, *state-level* for TPE/CMA-ES, *decision-level* for pruners).
- **Optuna coexistence is a hard invariant.** Nothing removes `nirs4all.optimization.optuna`. The native engine is added *alongside* and selected by a string; in R/MATLAB/WASM it is the only engine.

---

## 1. Target end-state (what "done" looks like)

```
pipeline step:  { "model": PLSRegression(),
                  "finetune_params": { "n_trials": 60, "sampler": "tpe", "pruner": "asha",
                                       "metric": "rmse", "eval_mode": "mean",
                                       "model_params": { "n_components": ("int", 1, 30) } } }

Python  → OptunaManager (default)  OR  native optimizer (routed by string)          ← both available
R       → n4m_tune(spec, X, y, sampler="tpe", pruner="asha", seed=…)                 ← native only
MATLAB  → n4m.finetune(X, Y, space, struct('Sampler','tpe','Pruner','asha', …))      ← native only
WASM    → await n4m.finetune({ searchSpace, sampler:"tpe", pruner:"asha", … })       ← native only
```

Every route: (a) accepts the **same search-space DSL** (or loudly rejects the unsupported subset, §3-D6); (b) produces a **`TrialResult` stream** with per-trial `status ∈ {completed, pruned, failed}`, `score`, `duration`, `error`; (c) is **seedable, timeout-bounded, resumable, warm-startable**; (d) at a fixed seed yields **bit-identical** trials across bindings (native samplers only). Each sampler/pruner has a **doc page + citations** and a **parity verdict** in CI.

---

## 2. The sampler & pruner set (single source of truth for scope)

### 2a. Samplers — `n4m_sampler_kind_t`

| # | id (`optimization.<id>`) | Algorithm | Space kinds | Parity tier (§3) | Phase | Build |
|---|---|---|---|---|---|---|
| 1 | `random` | Uniform random search | all | Tier C | F1 | reuse RNG |
| 2 | `sobol` | Sobol low-discrepancy | int/float/log | **A unscrambled; B scrambled** | F1 | low-new |
| 3 | `lhs` | Latin Hypercube | int/float/log | Tier C | F1 | low-new |
| 4 | `ternary` | Ternary unimodal-int search (ports `BinarySearchSampler`) | single unimodal int | **Tier A** | F1 | low-new |
| 5 | `ga` | GA over typed candidates (generalize `ga_select`) | discrete/categorical/perm | Tier B-state | F3 | reuse loop |
| 6 | `pso` | PSO over typed candidates (generalize `pso_select`) | binary/categorical | Tier B-state | F3 | reuse loop |
| 7 | `cmaes` | CMA-ES (+ **mandatory independent-random fallback** for categorical/conditional dims) | continuous/ordinal | Tier B-state | F4 | med-new |
| 8 | `tpe` | Tree-structured Parzen Estimator (**conditional/tree handling is core**) | mixed + conditional | Tier B-state | F4 | high-new |
| 9 | `gp_ei` | GP + Expected Improvement | small int/float | Tier B — **optional / cuttable** | F4 | high-new |

**`grid` is NOT a libn4m sampler [review].** Exhaustive enumeration is dag-ml's `enumerate_variants` (the parity-locked single source of truth); a small finite native space is covered by `random`/`sobol` over categoricals.

**Categorical values & string lifetime [codex].** Optuna categories may be `bool|int|float|str`. The ABI takes **typed categorical payloads** (a tagged value list), not only `const char*`; returned labels are **core-owned UTF-8 copies valid until the search space is destroyed** (never host-freed, per rule 3), and the **integer index is the canonical identity** for cross-binding parity. String-only would silently break bool/int categories round-tripping through Optuna's DSL.

**Warm-start / enqueue.** Every sampler accepts host-injected candidates via `n4m_optimizer_enqueue(...)` (Optuna's `enqueue_trial`/`force_params`): evaluate the published/current-best preprocessing chain first; seed from a prior study for `nirs4all-papers` reproducibility. **Reserved in F0.**

### 2b. Pruners / schedulers — `n4m_pruner_kind_t` [review]

| id | Algorithm | Fidelity axis | Parity tier | Phase |
|---|---|---|---|---|
| `none` | no early stopping | — | — | F0 |
| `median` | median/percentile stopping (Vizier) | any intermediate-score stream | Tier B-decision | F2 |
| `asha` | asynchronous successive halving | see §2c | Tier B-decision | F2 |
| `hyperband` | bracket scheduler over ASHA (owns (n,r) allocation, pairs with a base sampler) | see §2c | Tier B-decision | F2 |
| `racing` | Hoeffding/Bernstein racing (fold-safe early-stop, §2c) | CV folds | Tier B-decision | F2 |

Pruner verdicts are pure functions of the intermediate-score history + schedule (no RNG) — the one cleanly decision-testable case.

### 2c. Fidelity axis — the load-bearing correction [review + codex]

Successive-halving/Hyperband/ASHA assume **rank-preservation across fidelities**. `fidelity = fraction of CV folds` violates it: folds are exchangeable, so a fold-subset is a *higher-variance estimator of the identical target*, not a lower-fidelity proxy — with "smaller k" it systematically prefers smaller `n_components` (fewer training samples ⇒ fewer stable latent variables), so ASHA can **prune the config that wins at full data** (rank inversion). Fold count also gives ≤5× dynamic range, far short of what Hyperband's guarantees assume.

**The native fidelity axes, in priority:**
1. **PLS `n_components` learning curve (primary) — with a HARD scope caveat [codex].** The claim "CV-RMSE at 1..K is a free by-product of one fit" is true **only for the exact PLS1 / NIPALS / regression-deflation sweep path** (`cpp/src/core/sweep.cpp:2052`, `sweep.hpp:3`, `q == 1`), and even there it is **one max-K prefix fit *per CV fold*** (`cpp/tests/test_sweep.cpp:525`), not one global fit. The generic component-CV path loops `k=1..K` calling CV each time (`cpp/src/core/model_selection.cpp:69`) — no free prefix. **F2 must therefore either (a) scope the learning-curve fidelity to the eligible PLS1/NIPALS route, or (b) first build a real prefix-CV engine for SIMPLS/multi-target.** State which per method.
2. **Subsample fraction (general).** Increasing sample fraction; honest low-fidelity, caveated for very small n.
3. **Epochs (DL, host-side, Flavor C).** Host reports per-epoch validation as rungs.
4. **CV folds → `racing`, not ASHA.** The correct fold-based early-stop is statistical racing (Hoeffding/Bernstein; Maron & Moore). Document the rank-preservation assumption in `parity/tolerances.md`.

`tell_intermediate(trial, step, value)` is axis-agnostic; `step` is the rung (a `n_components` count / subsample rung / epoch).

### 2d. `auto` policy

Finite categorical → dag-ml grid. Single unimodal int → `ternary`. Combinatorial → `ga`/`pso`. Continuous, budget≥~30 → `cmaes` (independent fallback for non-continuous dims), else `gp_ei` (if built). Mixed/conditional budget≥~20 → `tpe`. First ~10 trials `sobol`/`random`. Pruner defaults to `asha` on the `n_components` axis where the learning curve is eligible, else `median`, else `none`.

---

## 3. Decisions that MUST be locked in F0 (they change the frozen ABI)

The F0 ABI is a one-shot freeze, so the entire *shape* — options, results, status, batch, metric, constraints — must be settled now or a later phase forces a second ABI-minor break.

- **D1. Sampler ⟂ pruner + grid-is-dag-ml + Hyperband owns bracket scheduling.** (§0, §2)
- **D2. Fidelity axis** = `n_components` learning-curve (scoped to eligible PLS1/NIPALS), subsample, epochs; `racing` for folds; the `tell_intermediate` step semantics. (§2c)
- **D3. Async pruning** (ASHA/median) so `tell_intermediate → out_should_prune` is sound per-tell.
- **D4. Result/status model [codex, Blocker].** `tell(score)` alone cannot represent a failed trial, a pruned terminal state, a timeout/cancellation, an error string, or wall-clock. Freeze: `n4m_trial_status_t {RUNNING, COMPLETED, PRUNED, FAILED}`; a terminal `n4m_optimizer_tell_result(opt, trial_id, status, score, const char* error)` (with `tell(score)` = `tell_result(COMPLETED, score, NULL)`); per-trial `duration`; and the `TrialResult` stream carrying all of it (matches Optuna's `TrialSummary`/`FinetuneResult` — `nirs4all/optimization/optuna.py:281,445`).
- **D5. Batch / parallel ask [codex, Major].** Studio is async/streamable; a scalar `ask` with no imputation makes parallel asks propose duplicate candidates. Freeze either `n4m_optimizer_ask_batch(opt, n, out[])` or an **active-trial-aware** scalar `ask` with an explicit **constant-liar** strategy set in the options struct. Reproducibility is guaranteed for sequential tell-order; parallel documents the liar policy.
- **D6. Metric / task / direction ABI [codex, Major].** A bare `int32_t metric` is regression-only. nirs4all resolves direction from metric **and task type**, covering regression **and** classification (`nirs4all/core/metrics.py:8,119`; `optuna.py:578`). Freeze an objective/metric surface with: a metric enum spanning both task types, a direction (auto-from-metric or explicit), multi-target aggregation, and **multi-objective explicitly reserved/unsupported** (a reserved vector-objective shape, returning `N4M_ERR_UNSUPPORTED` until built).
- **D7. Native-routable DSL subset [codex, Major].** The Optuna DSL also has `sorted_tuple` and static scalar pass-through (`optuna.py:1146,1299`). F0 either implements these param kinds or the native path **loudly rejects** any `finetune_params` key it can't honor (`sorted_tuple`, `fold_strategy:"individual"` beyond host-looping, `eval_mode` if not native, unknown keys) — never silently drops — so `sampler:"tpe"` and `sampler:"optuna"` optimize the **same** space and objective.
- **D8. SearchSpace constraint richness [codex, Major].** dag-ml's generation constraints are `mutex/requires/exclude` **plus `pick/arrange/count`** and multi-parent conditions (`nirs4all/pipeline/dagml/detect.py:263`). A single-value `set_condition(child, parent, parent_equals)` is too weak → Flavor A (grid) and Flavor B (native) would evaluate **different spaces**. Freeze a **generic constraint ABI** (`n4m_search_space_add_constraint(kind, refs[], n)` with `kind ∈ {mutex_group, requires, exclude, condition_in, condition_not_in}` + multi-parent), co-designed JSON-identical with the dag-ml `SearchSpace` contract, with explicit `N4M_ERR_UNSUPPORTED` for anything it can't express.
- **D9. Forward-compatible options struct [codex].** `n4m_optimizer_create(ctx, space, const n4m_optimizer_options_t* opts, out)` where `n4m_optimizer_options_t { size_t struct_size; sampler_kind; pruner_kind; direction; eval_mode; metric; liar_kind; uint64_t seed; double timeout_seconds; int32_t n_startup_trials; /* reserved[] */ }`. The `struct_size` prologue lets fields be added later without an ABI break.
- **D10. Resume mechanism (resolved by MT6).** N4MOPT v1 is the versioned
  little-endian `n4m_optimizer_save/load` blob. The F0-reserved signatures were
  activated without new ABI surface; see `docs/architecture/serialization.md`.
- **D11. `n4m_finetune_estimator` is a thin driver, not a second control loop [codex].** It reuses the same ask/tell primitives, the **identical `TrialResult` + best-param schema**, and a **single `validation_plan` (no nested CV, no selection, no leakage enforcement — those belong to dag-ml)**. Documented as single-level convenience so it cannot diverge from the dag-ml Tuner path; alternatively deferred until the dag-ml contract is frozen.

**Parity tiers (unchanged intent, restated):** Tier A = deterministic math bit-exact (`ternary`; **unscrambled** Sobol only, vs `scipy.stats.qmc.Sobol(scramble=False, bits=30)` with pinned direction file + index convention — the useful *scrambled* Sobol is Tier-B behavioral). Tier B-decision = pruners on canned history (RNG-free verdicts, clean). Tier B-state = TPE/CMA-ES assert **state** (split index / distribution update), never the RNG-entangled sample. Tier C = self-consistency + regret (`random`, `lhs`). Cross-binding is Tier-A-tight (1e-12) for every native sampler.

---

## 4. Phase plan

Each phase ships as `roadmap/phase-Fn-*.md` checkpoints in the existing idiom (Status / Methods / Parity-gate / ABI-delta / Local-gate), tagged `x.y.z+abi.M.m.p`.

### Phase F0 — ABI foundations + decisions (the ONE ABI-blocker PR)

The only ABI-surface-changing PR. Lands the *complete* surface so later samplers/pruners are reserved enum values. **Keep it a pure additive-symbol PR — RNG-consolidation is F3.**

- [ ] **Lock decisions D1–D11 (§3)** in `NATIVE_FINETUNING.md` + `docs/abi/changes_log.md`.
- [ ] **Search-space contract** — `n4m_search_space_t` + `create/destroy/add_int/add_float/add_categorical(typed values)/add_ordinal` + the **generic constraint API** (D8). `model.`/`train.` name-prefix preserves the split; `__` nested names verbatim; **`sorted_tuple` + static-scalar pass-through** implemented or explicitly rejected (D7). Categorical labels core-owned, index canonical (D-categoricals).
- [x] **Optimizer handle + ask/tell** — `n4m_optimizer_{create(options),ask,ask_batch,tell,tell_result,tell_intermediate,best,get_trials(since_id),enqueue,destroy,save,load}`; `n4m_trial_get_{int,float,category,id,rung,duration}` + `n4m_trial_is_active`; enums `n4m_param_kind_t`, `n4m_sampler_kind_t`, `n4m_pruner_kind_t`, `n4m_opt_direction_t`, `n4m_eval_mode_t`, `n4m_trial_status_t`, `n4m_metric_t`, `n4m_liar_kind_t`, `n4m_constraint_kind_t` (all 4-byte, static-asserted).
- [ ] **Metric/objective surface (D6)** — regression + classification metrics, direction (auto/explicit), multi-target aggregation, reserved multi-objective returning `N4M_ERR_UNSUPPORTED`.
- [x] **Pure-native entry (D11 / MT8)** — `n4m_finetune_estimator(ctx, estimator, X, Y, plan, space, options, n_trials, out_result)`: thin selection-only driver over the same ask/tell and rich-trace machinery, with one globally preflighted `validation_plan`. Its closed regression registry exposes `PLS_REGRESSION`, `PLS_CANONICAL`, `PLS_SVD`, `OPLS`, `SPARSE_PLS` and `PCR`; five routes tune `n_components`, while sparse PLS tunes any non-empty subset of `n_components` / `sparsity_lambda` with native defaults for omitted axes. Unsupported routes and malformed schemas fail before the study; timeout after a completion returns an explicit partial result. It performs no final refit, nested CV or DAG callback.
- [ ] **HPO parity machinery (D4/§0-DoD-4)** — `HpoSpec` registry schema + `parity/comparator/{pruner_decisions,sampler_state,study_trace}.py` + fixture format + an **explicit CI job** in `parity-gate.yml`.
- [ ] **Category scaffolding** — `cpp/src/core/optimization/` (**add each .cpp to `cpp/src/CMakeLists.txt`**), `cpp/src/c_api/c_api_optimization.cpp` (auto-globbed), `cpp/include/n4m/optimization.h` (`#include` in `n4m.h`), `cpp/tests/test_optimization.cpp` (register in `cpp/tests/CMakeLists.txt`), catalog `optimization` category + donor policy in `validate.py:43`, `docs/methods/_finetuning_bibliography.bib`.
- [ ] **ABI gate** — regenerate `cpp/abi/expected_symbols_{linux,macos,windows}.txt` (all three), `docs/abi/changes_log.md`, ABI **minor bump 2.0 → 2.1** (`cpp/include/n4m/n4m_version.h`).
- **Gate:** `ctest --preset dev-release`; ABI diff clean; `scripts/bump_version.sh --check`; SONAME/linkage audit; a `random`+`none` path exercises ask/tell/tell_result end-to-end.

### Phase F1 — deterministic & baseline samplers (Tier A/C)

Samplers `random`, `sobol` (scrambled + unscrambled), `lhs`, `ternary`; all consume the shared `n4m_rng` (`cpp/src/core/common/rng_engine.h`) directly — **no GA/PSO involvement yet**.

- [ ] **Sobol parity** — unscrambled Tier-A vs `scipy.stats.qmc.Sobol(scramble=False, bits=30)` (pinned direction file + index convention); scrambled Tier-B + cross-binding. `ternary` deterministic fixture; `random`/`lhs` self-consistency + regret via the new `study_trace` comparator.
- [ ] Tests; catalog; `docs/methods/{random,sobol,lhs,ternary}.md` + bib; CHANGELOG.
- **Gate:** C++-native determinism + Tier-A reference parity (bindings land F6); `roadmap/phase-F1-deterministic-samplers.md`.

### Phase F2 — pruners & multi-fidelity (highest-value native win — on the *right, scoped* axis)

Pruners `median`, `asha`, `hyperband`, `racing`; the `tell_intermediate` rung machinery on the **eligible `n_components` learning curve** (D2/§2c) and subsample.

- [ ] Implement the eligible-PLS1 learning-curve fidelity (**one max-K prefix fit per fold**, sweep.cpp route) and the subsample fidelity; **explicitly reject / fall back** for SIMPLS/multi-target unless the prefix-CV engine is built (D2 caveat). `tell_intermediate → should_prune` async semantics.
- [ ] `racing` (Hoeffding/Bernstein) as the fold-safe early-stop; document the rank-preservation assumption + why fold-fraction violates it in `parity/tolerances.md`.
- [ ] **Tier B-decision parity** vs Optuna `MedianPruner`/`SuccessiveHalvingPruner`/`HyperbandPruner` on canned histories (`pruner_decisions.py`); assert Hyperband (n,r) bracket accounting.
- [ ] Tests (a config that should be pruned is; a rank-inverting fold-fraction case is rejected by design); catalog; `docs/methods/{median_pruner,asha,hyperband,racing}.md` + bib; CHANGELOG.
- **Gate:** decision-level parity verdicts; `roadmap/phase-F2-pruners-multifidelity.md`.

### Phase F3 — population samplers + RNG consolidation

- [ ] **RNG consolidation [review, HERE not F0]** — migrate `ga_selection.cpp`/`pso_selection.cpp` off their file-local `splitmix64` onto the shared `n4m_rng`, guarded by their existing parity fixtures; re-freeze intentionally if numbers move (the R-parity RNG work shows this is delicate — hence out of the ABI-blocker PR).
- [ ] Generalize the GA/PSO loops from `vector<uint8_t>` masks to the **typed candidate** (`Population<Candidate>`); the feature-selection entry points become adapters — re-verify their fixtures.
- [ ] Expose `ga`/`pso` (reserved enum values); Tier B-state parity (pyswarms/genalg/nirs4all donors).
- [ ] Tests; catalog; `docs/methods/{ga_search,pso_search}.md` (cross-link selectors) + bib; CHANGELOG.
- **Gate:** GA/PSO selector fixtures still bit-exact (or re-frozen); `roadmap/phase-F3-population.md`.

### Phase F4 — model-based samplers

- [ ] `cmaes` (continuous/ordinal; PSO population pattern + `n4m_rng_next_normal`; covariance via the from-scratch linalg) **+ mandatory independent-random fallback for categorical/conditional dims** (mirrors Optuna's `CmaEsSampler`, which cannot handle them and warns). Tier B-state vs `pycma` distribution update.
- [ ] `tpe` — **tree/conditional handling is core** (per-branch densities, Parzen good/bad split, `l/g` argmax over drawn candidates). Tier B-state (split index + fixed-candidate `l/g` argmax) vs Optuna; `optuna-compat` flag; regret benchmark.
- [ ] `gp_ei` — **optional / cuttable**; only if mixed/conditional-kernel + marginal-likelihood work is scoped; reuse core Cholesky; Tier B vs skopt. May be deferred indefinitely — TPE+CMA-ES+ASHA is the committed Optuna-replacement surface.
- [ ] Tests; catalog; `docs/methods/{cmaes,tpe,gp_ei}.md` + bib; CHANGELOG.
- **Gate:** TPE + CMA-ES state-level parity; `roadmap/phase-F4-model-based.md`.

### Phase F5 — cross-repo orchestration (reachable *from the pipeline*) — SPLIT into 3 checkpoints [codex]

Acceptance spans **three repos** and cannot be proven in `nirs4all-methods` alone; today `dag-ml` rejects `finetune_params` (`nirs4all/pipeline/dagml/run_backend.py:624`) and forces them to Python (`detect.py:170`).

- **F5a — `dag-ml-data` `SearchSpace` contract.** One JSON-identical contract (validated by both repos' `validate_contracts.py`): the DSL + the **full D8 constraint richness** (co-frozen with F0's constraint ABI) + study controls (`n_trials 5–500 | timeout | direction | random_state | sampler | pruner | eval_mode | metric`). Gate: contract validation byte-identical.
- **F5b — `dag-ml` Tuner execution.** Make `NodeKind::Tuner` run the ask/tell loop inside `NestedCvSpec` inner folds (`fit inner-train → validation CandidateScore → tell`), `stable_json_fingerprint`-keyed; reuse `select_candidate` + `resolve_refit_variant`; Flavor B (native estimator) and Flavor C (host model, incl. DL epoch-fidelity with host-side partial-model caching keyed by trial id). Thin `dagml_tuner_ask_json`/`dagml_tell_json` FFI/WASM + `TrialResult` stream. Gate: dag-ml `cargo fmt/clippy/test`; a native pipeline finetune runs through dag-ml.
- **F5c — `nirs4all` controller dispatch + best-param normalization.** In the controller's `_execute_finetune` (`nirs4all/controllers/models/base_model.py:718`, which today unconditionally builds `OptunaManager` at `:851`), add a **real dispatch/factory** (parse `sampler`/`pruner`; `optuna`/`auto` → `OptunaManager`, else native). A **best-param normalization adapter** maps the native flat `model.`/`train.`-prefixed trial into the exact nirs4all nested `model_params`/`train_params` dict the refit `config_extractor` expects — so refit is identical whichever engine won. Stop demoting native-model `finetune_params` in `detect._FORCE_PYTHON_STEP_KEYS`. Gate: pipeline acceptance test (native sampler drives a real pipeline; best-params refit correctly); Optuna path unchanged.

### Phase F6 — bindings + cross-binding gate [codex: paths corrected]

- [ ] **Python** — `N4MSearchCV` in **`bindings/python/src/n4m/model_selection/`** (the `n4m.sklearn` namespace was removed in the ABI-2 migration; idiomatic classes live in role packages re-exported from `n4m._impl`); the raw ABI-close tier is `n4m._impl.native` (+ argtypes in `bindings/python/src/n4m/_ffi_decls.py`) — **not** a `n4m.python`/`python.py` module. (The controller dispatch itself is F5c.)
- [ ] **R** — S3 `parsnip`/`tune` engine + `n4m_tune()` (`.Call` into `bindings/r/n4m/`, col-major/1-based) driving ask/tell in R.
- [ ] **MATLAB/Octave** — `n4m.finetune(X,Y,space,opts)` as a `bindings/matlab/mex/n4m_*_mex.c` shim + `+n4m/` package; **no `bayesopt` dependency** (breaks Octave); ask/tell driven from `.m`.
- [ ] **WASM** — `async finetune(config)` (`bindings/js/`) over one synchronous native call; progress via **batched re-entry**, no Asyncify, no per-trial JS callback.
- [ ] **Cross-binding parity gate** — extend `benchmarks/cross_binding/orchestrator.py` with the ask/tell loop and wire it as a **real CI gate** (currently manual); assert Python≡R≡WASM≡MATLAB studies at **1e-12** for every deterministic-given-seed sampler; add native finetune ops to `donor_ops.py` for timing.
- **Gate:** cross-binding parity green; **Optuna-coexistence test** (same pipeline, `sampler:"optuna"` vs `sampler:"tpe"` both run, both reported, best-params normalized identically); `nirs4all-core` re-exports the finetuner; `roadmap/phase-F6-bindings.md`.

### Phase F7 — Studio, persistence, streaming, dashboard

- [x] **Native persistence/resume substrate (MT6)** — N4MOPT v1 activates the
  reserved `n4m_optimizer_save/load` symbols and Python `Optimizer.save/load`,
  preserving all nine sampler/five pruner sequential trajectories. Host storage,
  `study_name`, Studio streaming and graphical resume remain F7 integration work.
- [ ] **Streaming** — `n4m_optimizer_get_trials(..., since_id)` feeding Studio's `AutoMLStatus`/`TrialResult` (progress, running-best, trials_completed/total, elapsed, pruned/failed reported, timeout honoured).
- [ ] **Docs & dashboard** — Sphinx pages; native finetune timing in `docs/_static/bench-data.json`; a worked pipeline example per language; `parity/tolerances.md` finetuning section.
- **Gate:** Studio drives a native study with live progress + resume + warm-start; `roadmap/phase-F7-studio-persistence.md`.

---

## 5. Bibliography (per sampler — canonical citations for the doc pages)

Collect in `docs/methods/_finetuning_bibliography.bib`. **[review] `sobol` venue corrected (SIAM J. Sci. Comput., not ACM TOMS); keep both Joe & Kuo papers distinct.**

| Sampler / pruner | Primary reference(s) | Reference impl (donor) |
|---|---|---|
| `random` | Bergstra & Bengio, *Random Search for Hyper-Parameter Optimization*, JMLR 13 (2012), 281–305 | self / nirs4all |
| `sobol` | Sobol′, USSR Comput. Math. & Math. Phys. 7 (1967), 86–112; Joe & Kuo, *Constructing Sobol Sequences with Better Two-Dimensional Projections*, **SIAM J. Sci. Comput. 30 (2008), 2635–2654** (scipy's direction numbers); *cf.* Joe & Kuo, *Remark on Algorithm 659*, ACM TOMS 29 (2003), 49–57 | `scipy.stats.qmc.Sobol` |
| `lhs` | McKay, Beckman & Conover, Technometrics 21 (1979), 239–245 | `scipy.stats.qmc.LatinHypercube` |
| `ternary` | ports nirs4all `BinarySearchSampler` (unimodal ternary search) | nirs4all (self-consistency) |
| `median` | Golovin et al., *Google Vizier*, KDD (2017) — median stopping rule | Optuna `MedianPruner` |
| `asha` | Jamieson & Talwalkar, *Non-stochastic Best Arm Identification and HPO*, AISTATS (2016); Karnin, Koren & Somekh, ICML (2013); Li et al. (ASHA), MLSys (2020) | Optuna `SuccessiveHalvingPruner` |
| `hyperband` | Li, Jamieson, DeSalvo, Rostamizadeh & Talwalkar, *Hyperband*, JMLR 18 (2018) / ICLR (2017) | Optuna `HyperbandPruner` |
| `racing` | Maron & Moore, *Hoeffding Races*, NeurIPS (1993); Mnih, Szepesvári & Audibert, *Empirical Bernstein Stopping*, ICML (2008) | nirs4all (self-consistency) |
| `cmaes` | Hansen & Ostermeier, Evol. Comput. 9 (2001), 159–195; Hansen, *CMA-ES Tutorial* (2016), arXiv:1604.00772 | `pycma` |
| `tpe` | Bergstra, Bardenet, Bengio & Kégl, NeurIPS (2011); Bergstra, Yamins & Cox, ICML (2013) | Optuna `TPESampler` |
| `gp_ei` | Jones, Schonlau & Welch, J. Glob. Optim. 13 (1998), 455–492; Snoek, Larochelle & Adams, NeurIPS (2012) | `scikit-optimize` |
| `ga` | Leardi & Lupiáñez González, Chemom. Intell. Lab. Syst. 41 (1998), 195–207 | R `genalg` / nirs4all (`ga_select` donor) |
| `pso` | Kennedy & Eberhart, *A discrete binary version of the particle swarm algorithm*, IEEE SMC (1997), 4104–4108 | `pyswarms` BinaryPSO (`pso_select` donor) |

---

## 6. Two-phase execution & agent parallelism

The work is split into **two execution phases** to (a) let the finetuner stabilize inside `nirs4all-methods` before anything downstream depends on it, and (b) **not disturb a concurrent agent working in `dag-ml` / `nirs4all-core`**. The F-phases above (F0–F7) map onto these two execution phases.

### 6.1 Is the two-phase constraint well-founded? — Yes, and it is the natural architecture

- **Phase 1 (methods) is 100% self-contained.** F0–F4 + the methods-side bindings + the HPO parity machinery all live in `nirs4all-methods`. **Zero edits to `dag-ml`, `dag-ml-data`, or `nirs4all-core`.**
- **Cross-language finetuning of *native* models already works at the end of Phase 1 — without dag-ml.** The methods bindings (R/WASM/MATLAB/Python) expose the optimizer directly, so a binding user can `finetune` a native PLS/estimator in any language before Phase 2 starts. dag-ml is needed only for **pipeline-orchestrated** finetuning (native OOF/leakage/selection/lineage), **cross-language *pipeline* reachability**, and **host-model (Flavor C)** coordination.
- **The one coupling is neutralized.** F0's SearchSpace constraint ABI (D8) is designed against the **already-implemented, parity-locked dag-ml generation vocabulary** (`generation.rs`, `detect.py:263`) — not against unwritten F5a work — so Phase 1 freezes it **unilaterally**, with no coordination with the dag-ml agent. Phase 2's F5a then maps its JSON contract onto the already-frozen ABI.
- **Phase 2 depends on Phase 1, never the reverse** — exactly the requested ordering (Phase 1 stable & validated → Phase 2).

So the constraint is not a compromise; it is how the layering already wants to run.

### 6.2 The phase boundary

| Execution phase | Repos touched | F-phases | Deliverable |
|---|---|---|---|
| **Phase 1 — engine** | `nirs4all-methods` only | **F0, F1, F2, F3, F4** + methods bindings + HPO parity/CI | A stable, validated, parity-gated native finetuner in `libn4m`, idiomatic and bit-reproducible in every binding, callable directly (incl. `n4m_finetune_estimator` from R/WASM/MATLAB/Python). |
| **Phase 1-bridge — Python consumer** *(optional, dag-ml-free, safe to run any time after Phase 1's Python binding)* | `nirs4all` (Python) — **not** dag-ml/core | the dag-ml-free half of **F5c** + **F6-Python** | A `NativeOptimizerManager` behind the controller `_execute_finetune` dispatch + `N4MSearchCV` + best-param normalization, reusing nirs4all's **Python-engine** fold/eval/refit machinery (the same path `OptunaManager` uses today). Result: `sampler:"tpe"` works in a nirs4all **Python** pipeline. **Touches neither dag-ml nor nirs4all-core**, so it does not disturb the other agent. |
| **Phase 2 — orchestration** *(after Phase 1 is stable; coordinate with / hand to the dag-ml agent)* | `dag-ml`, `dag-ml-data`, `nirs4all-core` (+ the dag-ml-backend half of nirs4all) | **F5a, F5b**, core re-export, dag-ml-backend routing, Flavor C, **F7** | Native finetuning **through the dag-ml engine**: cross-language *pipeline* reachability, native OOF/leakage/selection/lineage, host-model tuning, Studio persistence/streaming. |

> **Note on F5c.** Codex framed F5c as cross-repo because *routing finetune through the dag-ml backend* needs the dag-ml Tuner. But the nirs4all **Python engine** can drive the native optimizer directly (host-drives-the-loop), bypassing dag-ml — that half of F5c is **Phase 1-bridge** and dag-ml-free. The dag-ml-backend half of F5c stays in Phase 2.

### 6.3 Phase 1 dependency DAG — maximum agent parallelism

`F0` is the **only** serialization point (it freezes the ABI everything compiles against — see [`FINETUNING_F0_PR.md`](FINETUNING_F0_PR.md)). Once it merges, Phase 1 forks into independent tracks against the frozen `optimization.h`:

```
                          ┌──────────────────────────── F0 (ONE PR — ABI freeze + random/none slice) ───────────────────────────┐
                          │  serial gate: header + enums + options + ask/tell + finetune_estimator + ABI snapshots + parity CI  │
                          └───────────────────────────────────────────────┬───────────────────────────────────────────────────┘
   after F0 merges, all tracks run CONCURRENTLY (arrows = intra-track order only):
   ── Track S (samplers) ······  sobol  ∥  lhs  ∥  ternary                         [3 agents, fully independent]
   ── Track P (pruners) ·······  [fidelity-engine agent] → median ∥ asha ∥ hyperband ∥ racing   [1 → 4 agents]
   ── Track G (population) ·····  [RNG-consolidation agent] → ga ∥ pso             [1 → 2 agents; isolated re-freeze]
   ── Track M (model-based) ····  cmaes  ∥  tpe  ∥  (gp_ei?)                        [2–3 agents, independent]
   ── Track Q (parity/CI) ·····  HpoSpec + comparators + cross-binding CI gate     [1 agent, START FIRST — every DoD gates on it]
   ── Track B (bindings) ······  python → ( R ∥ MATLAB/Octave ∥ WASM ) → cross-binding parity gate   [1 → 3 agents]
```

**Concurrency peaks at ~10–14 agents.** Immediately after F0: `sobol`, `lhs`, `ternary`, the fidelity-engine, the RNG-consolidation, `cmaes`, `tpe`, the parity machinery, and the Python binding can all start at once (~9). As the two sub-dependencies clear (fidelity engine → 4 pruners; RNG consolidation → `ga`/`pso`; Python binding → R/MATLAB/WASM), the fan-out grows.

**Intra-Phase-1 ordering rules (the only real constraints):**
1. **Everything waits on F0** (the ABI freeze). Nothing else in Phase 1 touches the header/version/snapshots.
2. **Track Q starts first.** Each sampler/pruner's *merge gate* needs the `HpoSpec` + comparators + CI job. Implementation can proceed against local tests in parallel, but land Q early.
3. **Pruners wait on the fidelity engine** (the eligible-PLS1 `n_components` learning-curve producer, §2c-1) — one shared sub-task, then median/asha/hyperband/racing parallelize.
4. **`ga`/`pso` wait on the RNG-consolidation** sub-task (isolated because it may re-freeze GA/PSO fixtures — keep it off the critical path of the other tracks).
5. **R/MATLAB/WASM bindings wait on the Python binding** proving the wrapper pattern (Python is also the cross-binding parity reference); they then parallelize and can wrap samplers incrementally as each lands.
6. **The cross-binding parity gate** (tail of Track B) needs Track Q + the bindings + the samplers under test.

Phase 1 is "done, stable & validated" when: all Phase-1 samplers/pruners pass their tier gate, cross-binding parity is a green CI gate at 1e-12, and `n4m_finetune_estimator` is validated end-to-end. **Only then does Phase 2 begin** — handed to (or coordinated with) the dag-ml/core agent, against the now-frozen ABI.

### 6.4 Phase 2 sequencing (for later; owned by the dag-ml/core agent)

```
Phase-1 frozen ABI ──► F5a (dag-ml-data SearchSpace + constraints, maps onto frozen ABI)
                         └─► F5b (dag-ml Tuner exec: ask/tell in nested CV, Flavor B/C)
                               └─► nirs4all-core re-export  ∥  dag-ml-backend routing in nirs4all
                                     └─► F7 (Studio persistence/streaming; persistence ABI already shipped in F0)
```
F5a/F5b are independently reviewable; core re-export and dag-ml-backend routing parallelize once F5b lands.

---

## 7. Per-phase acceptance gates (summary)

- **Always:** `cmake --build --preset dev-release` + `ctest --preset dev-release`; `clang-format`/`clang-tidy`; `git diff --check`.
- **F0 (ABI blocker):** ABI symbol diff clean vs regenerated `expected_symbols_{linux,macos,windows}.txt`; `docs/abi/changes_log.md`; `bump_version.sh --check`; SONAME/linkage; **complete** surface reserved (options/result/status/batch/metric/constraints/enqueue/since_id/save-load-if-chosen) so no second blocker PR; the HPO parity CI job exists (even if only `random` is wired).
- **Per sampler/pruner:** `n4m_tests` case incl. non-trivial convergence / correct prune; parity fixture frozen; verdict at its tier (A ≤1e-12; B-decision verdict match; B-state distribution/split match; C self-consistency + regret); `validate.py --strict-abi --check-references` green; `docs/methods/<name>.md` + bib; `CHANGELOG.md`.
- **F5a/b/c:** `validate_contracts.py` SearchSpace + constraints byte-identical; dag-ml `cargo fmt/clippy/test`; nirs4all pipeline acceptance (native drives a real pipeline; best-params refit identical; Optuna path unchanged).
- **F6:** cross-binding parity at 1e-12 as a **CI gate**; per-binding smoke; Optuna-coexistence test.
- **F7:** Studio live-progress + resume + warm-start; dashboard timing refreshed.

---

## 8. Risks & mitigations

1. **PLS learning-curve fidelity over-scoped [codex].** Free prefix-CV exists only for the PLS1/NIPALS sweep route and is per-fold, not global; generic CV re-fits per k. Mitigation: scope F2 to the eligible route or build a prefix-CV engine for SIMPLS/multi-target first. (§2c-1)
2. **Fold-fraction fidelity is ML-unsound.** Mitigation: `n_components` learning-curve / subsample / `racing`; document in `tolerances.md`. (§2c)
3. **F0 result/status/metric/constraint under-freeze → a second ABI-minor break [codex].** Mitigation: freeze the full options/result/status/batch/metric/constraint surface + reserve all enum values now (D4–D9); `struct_size` prologue for forward-compat.
4. **Two control loops diverge (`n4m_finetune_estimator` vs dag-ml Tuner) [codex].** Mitigation: the native entry is a thin single-level driver over the same primitives + schema; nested CV / selection / leakage stay in dag-ml (D11).
5. **HPO parity has nowhere to live [codex].** Mitigation: `HpoSpec` + dedicated comparators + a real CI job — designed in F0, not retrofitted onto the prediction-only `MethodSpec`.
6. **Cross-repo sequencing hidden by a single "F5" [codex].** Mitigation: F5a/F5b/F5c each with cross-repo gates; acceptance spans methods + dag-ml + nirs4all.
7. **Sampler/pruner conflation / DSL divergence.** Mitigation: split enums (D1); native-routable subset with loud rejects (D7); constraint richness (D8).
8. **Parallel ask duplicates trials.** Mitigation: `ask_batch` / active-trial-aware ask + constant-liar (D5).
9. **Parity over-claims (Sobol/TPE).** Mitigation: unscrambled-only Sobol Tier-A; TPE/CMA-ES assert state; pruners decision-level. (§3)
10. **RNG consolidation moving GA/PSO numbers.** Mitigation: in F3, not the ABI PR; re-freeze with a note.
11. **CMA-ES on mixed spaces / GP-EI scope creep.** Mitigation: mandatory independent fallback; GP-EI explicitly optional.
12. **Categorical string lifetime / typed values [codex].** Mitigation: core-owned label copies, canonical index, typed categorical payloads.

---

## 9. What this roadmap does NOT do (scope guard)

- Does not remove or reduce Optuna in Python — Optuna stays a first-class default-in-Python sampler.
- Does not put DL objective evaluation in `libn4m` — torch/TF training stays host-side (Flavor C: native `ask`, host evaluation with epoch-fidelity, native `tell`).
- Does not chase bit-parity with Optuna's stochastic *samples* — cross-binding bit-parity (native RNG) is the guarantee; reference parity is per-tier.
- Does not re-implement grid enumeration in `libn4m` — that is dag-ml's parity-locked oracle.
- Does not support multi-objective in v1 — reserved as `N4M_ERR_UNSUPPORTED` (D6).
- Does not add a mandatory dependency — every sampler is against the from-scratch core.

> **Companion reconciliation.** `NATIVE_FINETUNING.md` §4.2/§4.4/§5/§6 have been annotated/corrected to match this roadmap (sampler ⟂ pruner, grid-is-dag-ml, `n_components`-not-fold fidelity, `pruner`/`eval_mode` controls, `n4m.model_selection` not `n4m.sklearn`). This roadmap is authoritative where they still differ.
