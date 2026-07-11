# Native cross-language finetuning for nirs4all

**Status:** strategy / design proposal (not yet implemented)
**Scope:** ecosystem-wide — `nirs4all` (Python), `dag-ml`, `dag-ml-data`, `nirs4all-methods` (`libn4m`), `nirs4all-core`, the R/WASM/MATLAB/Octave bindings, `nirs4all-studio`
**Author:** drafted from a full-ecosystem survey (see the *Evidence map* appendix for file:line anchors)
**Question it answers:** how do we get *clean, native, reproducible* hyperparameter finetuning into nirs4all that works identically in Python, R, MATLAB/Octave and WASM — not just Python/Optuna?

---

## 0. TL;DR — the recommended path

Today, all *adaptive* finetuning is Python-only: it lives entirely in `nirs4all.optimization.optuna.OptunaManager`. `dag-ml` already owns the *deterministic* half (variant generation + selection) natively in Rust, but has **no sampler and no candidate-evaluation loop**. `nirs4all-methods` already ships native black-box optimizers (GA/PSO/CARS/…) but their objective is hard-wired to PLS.

The clean, ML-logical, code-honest path is a **three-seam architecture**:

1. **The search *algorithm* lives once, natively, in `libn4m`** as a dependency-free **ask/tell optimizer** (typed search space + `n4m_optimizer_t` with `ask()`/`tell()`), reusing the RNG, CV/validation-plan and result-bag substrate methods already has. This is the *only* way to get bit-identical finetuning across R/MATLAB/Octave/WASM.
2. **`dag-ml` owns the *control loop* and every OOF/leakage/nested-CV/selection/lineage guarantee** — it drives `ask → schedule candidate → score → tell → select → refit`. Its `Tuner` node, `GenerationSpec`, `select_candidate` and nested-CV plumbing already exist; it gains a sampler seam and a per-trial score-return path, not a new subsystem.
3. **The *evaluator* is pluggable.** For a methods model (PLS-family) the objective stays fully native (no boundary crossing). For a host model (torch/TF/sklearn) **the host drives the loop** (`ask` → host fits & scores → `tell`), so **no C→host objective callback is needed** in the common case. Optuna becomes an *optional sampler*, not the engine.

**Direct answers to the four questions:**

| # | Question | Answer |
|---|---|---|
| Q1 | Is finetuning customisable in dag-ml, or Python-specific? | Today the **adaptive search is Python-specific** (Optuna). dag-ml *already* owns generation + selection + a `Tuner` node natively, but not the sampler. Making finetuning customisable-from-any-language = add the ask/tell sampler seam + a declarative `SearchSpace` contract. Then it is a **dag-ml capability**, chosen by a string (`sampler:"tpe"`), not a Python feature. |
| Q2 | What is needed to use finetuning *from* dag-ml? | (a) a native `libn4m` ask/tell optimizer; (b) a JSON-identical `SearchSpace` contract in `dag-ml-data`; (c) wire the `Tuner` node to run the loop inside nested CV and return per-candidate scores via the existing controller vtable / PredictionStore. No new project. |
| Q3 | Could R/WASM/MATLAB provide idiomatic finetuning for their nirs4all-core ports? | **Yes** — each binding wraps the native optimizer thinly (R `tune`/`mlr3`-style, MATLAB `bayesopt`-like struct, WASM promise-based, Python `SearchCV`). For **WASM and Octave** the native optimizer is effectively the only path that is *simultaneously reproducible-across-bindings, toolbox-free and single-source* (MATLAB's `bayesopt` and JS `hpjs` exist but are proprietary / niche / non-portable). |
| Q4 | Which finetuning methods to implement in nirs4all-methods? | In priority order: **uniform random, Sobol/LHS, grid (delegate to dag-ml), ASHA/median-pruned successive-halving & Hyperband, a ternary sampler for unimodal ints (`n_components`), reuse GA/PSO for combinatorial spaces, CMA-ES for continuous, TPE for mixed spaces**; GP-EI is optional/cuttable. Default `auto` policy dispatches on space shape + budget (Optuna-analogous). |

---

## 1. Three meanings of "finetuning" (disambiguation first)

The word "finetuning" collapses three distinct capabilities in nirs4all. Keeping them separate is essential — they have different owners and different maturity.

1. **Enumeration / grid search** — deterministically enumerate a *finite* set of candidate pipelines (the `_or_` / `_grid_` / `_cartesian_` / `_zip_` generator DSL, with `mutex`/`requires`/`exclude` constraints) and pick the best by CV. This is **variant generation + selection**. In real NIRS workloads this is the *dominant* case: the AOM Talanta recipe is a **600-combination preprocessing grid** (5 norms × 10 smoothers × 3 baselines × 4 OSC), all categorical, crossed with one model HP.
2. **Adaptive / sequential search** — propose the next candidate *from the scores of previous ones* (TPE, CMA-ES, Bayesian/GP, successive-halving pruning). This is the Optuna intelligence. It matters for the continuous / mixed-space tail: PLS `n_components`, Ridge `alpha` (log-uniform), DL learning rates, and epoch-budgeted DL training.
3. **Combinatorial variable / preprocessing-chain selection** — search over feature subsets or operator chains (GA-PLS, PSO-PLS, CARS, VISSA, IRIV…). methods already implements these natively, but as *self-contained* selectors, not as a generic optimizer.

A "finetune" in a nirs4all pipeline today mixes (1) and (2): a model step carries `finetune_params` (adaptive, Optuna) and the surrounding pipeline may carry `_or_`/`_grid_` generators (enumeration). The end-state must cover all three coherently.

---

## 2. État des lieux — current finetuning, per package

| Package | What it does for finetuning today | Native? Portable? |
|---|---|---|
| **nirs4all (Python)** | The *entire* adaptive engine: `OptunaManager` — TPE/CMA-ES/Grid/Random + a custom ternary `BinarySearchSampler`; median/successive-halving/Hyperband pruners; single/grouped/individual fold strategies; `best/mean/robust_best` aggregation; multi-phase studies; the full search-space DSL; `stack_params()`. Wired per model step via `finetune_params`, dispatched by the model controllers, best params flow into refit. | **Python-only.** None of it crosses the C ABI. |
| **dag-ml** | Owns the **deterministic** half natively in Rust: variant `GENERATION` (`enumerate_variants`, Cartesian/Zip + `mutex/requires/exclude` — parity-locked to the nirs4all oracle) and `SELECTION` (`select_candidate`, Min/Max objective + tie-break). Has a first-class `Tuner` **node type**, `search_space_fingerprint`, and nested inner-CV plumbing (`NestedCvSpec`). | Generation + selection are native. **But no sampler, no HPO loop, no host-objective callback.** The `Tuner` node *compiles* but is dispatched to an external host controller — it is not natively executed. |
| **nirs4all-methods (`libn4m`)** | Ships native metaheuristic optimizers over **feature-subset** spaces: GA-PLS, PSO-PLS, CARS, SCARS, VISSA, IRIV, IRF, random-frog. Each takes `(ctx, cfg, X, Y, validation_plan, hyperparams…, seed)` and computes **CV-RMSE of PLS internally**. Reusable substrate: a seeded RNG engine (`n4m_rng`: splitmix64/PCG64/R-MT/NumPy-MT), the `n4m_validation_plan_t` fold contract (9 generators), `cross_validate_regression`, the `n4m_method_result_t` result bag. | Native + portable — but the **objective is PLS-locked**, there is **no generic `evaluate(candidate)→score` seam**, and **no function-pointer/callback anywhere** in the public ABI. |
| **dag-ml-data** | Data contracts (schemas, axes, `FoldSet`, envelope). | No search-space contract yet. |
| **Bindings (R / WASM / MATLAB / Octave / Python)** | Every binding is a pure data-in/data-out marshaller. **No binding passes a host function pointer into `libn4m`.** | The callback question is *moot* for the recommended design (host drives the loop; only R→C / JS→WASM repeated calls, which already work). |
| **nirs4all-core** | Re-exports formats/io/datasets/methods/dag_ml/dag_ml_data. Executes today only a tiny operator subset (KS split, SNV, SG, PLS). | **Finetuning is explicitly out of core's current scope** — it enters only after the engine lands upstream (methods + dag-ml) and clears the operator gate. |
| **nirs4all-studio** | Never reimplements the library — it *serializes* the editor state to/from the exact `finetune_params` dict; the AutoML API needs live per-trial streaming (`TrialResult{params,score,status∈{completed,failed,pruned}}`), running-best, `n_trials` 5–500, `timeout`, seed, and an `include_preprocessing_search` toggle. | Consumer. Sets hard requirements: **async, streamable, prunable, resumable**. |
| **nirs4all-lab / -aom** | Real HPO recipes (AOM 600-combo preprocessing grid + model-HP tail; TabPFN search spaces; a `study_pp_selection_pls` preprocessing-selection study). Python-only prototypes. | Show the *actual* search shapes: **preprocessing-heavy categorical with a small continuous/int model tail.** |

**The single load-bearing fact:** the deterministic half (enumerate + select) is native and (nearly) cross-language; the *adaptive* half (sample + prune + the propose→score→propose loop) exists **only in Python/Optuna**. That is the whole gap.

### 2.1 Two honest corrections from adversarial verification

These sharpen the plan and correct over-optimistic framings:

- **methods substrate is *partial* reuse, not "assembly."** The CV/validation-plan is genuinely factored and the RNG *engine* exists — but the GA/PSO loops carry **file-local copies** of splitmix64 (they don't consume the shared engine), the "PLS CV-RMSE fitness" is a **copy-pasted `subset_rmse` per selector shaped around a feature mask** (no generic candidate/objective seam), and the population loops are bespoke over `vector<uint8_t>` masks with no `Population<T>` abstraction. A native optimizer reuses the *primitives* (folds, CV, RNG kinds, result bag, calling convention) and the *proven patterns*, but is **real new C++** — a generic objective seam, a typed candidate representation, a generic loop, and a new ask/tell ABI. Budget it as a build, not glue.
- **dag-ml's generation is only *transitively* exposed, and even grid eval is host-side.** `select_candidate` is directly exposed over the C-ABI **and** WASM. `enumerate_variants` has **no dedicated FFI symbol** — a host reaches the enumerated grid only bundled inside `build_execution_plan`'s `ExecutionPlan.variants`. And dag-ml **never evaluates a candidate**: it hands each fully-specified variant to a host controller to fit/score, then picks the winner. So "grid finetuning already works cross-language" is true for *generation + selection*, but the *evaluation* is always host-side — which is exactly why the native-evaluator path (flavor B below) is the real portability win.

---

## 3. The recommended architecture

### 3.1 One rule, two seams

> **`dag-ml` owns the search *control loop* and all ML invariants (OOF, leakage, nested CV, selection, lineage, fingerprints). The *sampler* and the *evaluator* are the two pluggable seams. The sampler lives natively in `libn4m`. The evaluator is native for methods models, host-side for host models.**

This maps cleanly onto what already exists: `dag-ml`'s `Tuner` node is already a scheduled OOF-producing node with a node-local `inner_cv`; it just needs to (a) pull candidates from a sampler instead of enumerating them all up front, and (b) receive a per-trial score.

### 3.2 Three flavors, three loops

**Flavor A — enumeration / grid (native *today*, minus evaluation).**
```
compile → enumerate_variants (all N, constraint-pruned) → [host fits/scores each] → select_candidate → refit winner
```
No sampler. Already cross-language for generation + selection. Covers the 600-combo preprocessing grids that dominate real workloads.

**Flavor B — adaptive over a NATIVE methods model (fully native, no host in the loop).**
```
loop t:
  cand  = n4m_optimizer_ask(space)          # sampler in libn4m
  score = fit_cv(cand, inner-train folds)    # native PLS/estimator, CV-RMSE
  n4m_optimizer_tell(cand, score)
until budget → best → select_candidate → refit
```
**This is the portability payoff.** No boundary crossing per trial ⇒ bit-identical in Python/R/MATLAB/Octave/WASM. It is the only design that gives portable *adaptive* finetuning.

**Flavor C — adaptive over a HOST model (torch/TF/sklearn) — sampler native, evaluation delegated.**
```
loop t:
  cand   = n4m_optimizer_ask(space)          # sampler still native
  result = host_controller.fit(cand, inner-fold-train ids)   # Python/etc. scores it
  n4m_optimizer_tell(cand, result.score)
until budget → best → select_candidate → refit (host controller)
```
The host **drives** the loop and evaluates; the sampler stays native. dag-ml owns the coordination; Python keeps only what only Python can do (train a torch model in-process).

### 3.3 Why "host drives the loop", not a C→host objective callback

The host-driven ask/tell pattern is what Optuna itself (`study.ask`/`study.tell`), scikit-optimize, Nevergrad and Ax all converged on. It is strictly better than a C→host objective callback for this codebase:

- **No new function-pointer ABI** in the common path ⇒ it can't violate "never throw across the boundary", can't clobber the thread-local error buffer by re-entering `n4m_*`, and doesn't force "callback on the calling thread only, no OpenMP".
- **R's evaluator is single-threaded / non-reentrant and errors via `longjmp`** — a C→R objective callback threaded through a native inner loop is exactly the `PROTECT`/unwinding hazard. Ask/tell inverts control: R calls `ask`, evaluates in R, calls `tell`. Hazard gone.
- **WASM is synchronous, single-thread, no Asyncify, `addFunction` unexported** — a per-trial JS callback needs Asyncify (large cost, off). Ask/tell keeps the whole search inside one synchronous WASM call (flavor B), with progress via *batched re-entry*, never a per-trial callback.
- **Python `ctypes.CDLL` releases the GIL** — a re-entrant callback re-acquires it and can deadlock under OMP; ask/tell returns to Python which evaluates on its own thread.

An **optional** callback (`n4m_optimizer_run(opt, obj_fn, user, n_trials)`) can still be offered as a Python/R convenience for the core to own the iteration/pruning bookkeeping — status-returning, single-threaded, non-reentrant by contract. It is never the portable contract.

---

## 4. What to build in `nirs4all-methods`

### 4.1 Search-space contract (typed, C-ABI, 1:1 with the existing DSL)

An opaque `n4m_search_space_t` built with typed adders; kinds are 4-byte enums (guard-railed like every n4m enum):

```c
typedef enum n4m_param_kind_t {
  N4M_PARAM_INT=0, N4M_PARAM_FLOAT=1, N4M_PARAM_LOG_INT=2,
  N4M_PARAM_LOG_FLOAT=3, N4M_PARAM_CATEGORICAL=4, N4M_PARAM_ORDINAL=5
} n4m_param_kind_t;

N4M_API n4m_status_t n4m_search_space_create(n4m_search_space_t** out);
N4M_API n4m_status_t n4m_search_space_add_int(n4m_search_space_t*, const char* name,
                      int64_t low, int64_t high, int64_t step, int32_t log);
N4M_API n4m_status_t n4m_search_space_add_float(n4m_search_space_t*, const char* name,
                      double low, double high, double step, int32_t log); /* step<=0 => continuous */
N4M_API n4m_status_t n4m_search_space_add_categorical(n4m_search_space_t*, const char* name,
                      const char* const* choices, int32_t n_choices);
N4M_API n4m_status_t n4m_search_space_add_ordinal(n4m_search_space_t*, const char* name,
                      const double* values, int32_t n_values);
N4M_API n4m_status_t n4m_search_space_set_condition(n4m_search_space_t*, const char* child,
                      const char* parent, const char* parent_equals);
N4M_API void         n4m_search_space_destroy(n4m_search_space_t*);
```

**DSL parity is mechanical** with the Studio/Optuna vocabulary (`{int, float, log_float, categorical}`, list `[token, low, high, step]` / `[token, choices]` or dict `{type, low, high, step, choices, log}`): `["int",lo,hi,step]`→`add_int`; `["log_float",lo,hi]`→`add_float(...,log=1)`; `["categorical",cs]`→`add_categorical`. **Nested params** keep the `__` flattened name verbatim (the string is opaque to the core). The `model_params` vs `train_params` split is preserved by a `model.` / `train.` name prefix, so Studio's serializer round-trips unchanged. This requirement is non-negotiable — break the DSL vocabulary and Studio breaks.

### 4.2 Optimizer handle + ask/tell

> **Superseded by [`FINETUNING_ROADMAP.md`](FINETUNING_ROADMAP.md) §2–3 (adversarial review).** The enum below is the original sketch. The implementation splits **sampler ⟂ pruner** into two enums (`n4m_sampler_kind_t` × `n4m_pruner_kind_t`, composed in `n4m_optimizer_create`), drops `GRID` (exhaustive enumeration is dag-ml's parity-locked oracle, not a libn4m sampler), and moves `HALVING`/Hyperband/median into the pruner enum. The multi-fidelity axis is the **PLS `n_components` learning curve** / subsample / racing — **not** CV-fold fraction (which breaks the rank-preservation successive-halving assumes). See the roadmap for the authoritative surface.

```c
typedef enum n4m_sampler_kind_t {
  N4M_SAMPLER_RANDOM=0, N4M_SAMPLER_GRID=1, N4M_SAMPLER_SOBOL=2, N4M_SAMPLER_LHS=3,
  N4M_SAMPLER_HALVING=4, N4M_SAMPLER_CMAES=5, N4M_SAMPLER_TPE=6, N4M_SAMPLER_GP_EI=7,
  N4M_SAMPLER_GA=8, N4M_SAMPLER_PSO=9, N4M_SAMPLER_TERNARY=10
} n4m_sampler_kind_t;

N4M_API n4m_status_t n4m_optimizer_create(n4m_context_t*, const n4m_search_space_t*,
                      n4m_sampler_kind_t, n4m_opt_direction_t, uint64_t seed,
                      n4m_optimizer_t** out);
N4M_API n4m_status_t n4m_optimizer_ask(n4m_optimizer_t*, n4m_trial_t** out_trial); /* BORROWED */
N4M_API n4m_status_t n4m_trial_get_int(const n4m_trial_t*, const char* name, int64_t* out);
N4M_API n4m_status_t n4m_trial_get_float(const n4m_trial_t*, const char* name, double* out);
N4M_API n4m_status_t n4m_trial_get_category(const n4m_trial_t*, const char* name,
                      int32_t* out_index, const char** out_label);
N4M_API n4m_status_t n4m_optimizer_tell(n4m_optimizer_t*, int64_t trial_id, double score);
N4M_API n4m_status_t n4m_optimizer_tell_intermediate(n4m_optimizer_t*, int64_t trial_id,
                      int32_t step, double score, int32_t* out_should_prune);
N4M_API n4m_status_t n4m_optimizer_best(const n4m_optimizer_t*, n4m_trial_t** out_best, double* out_score);
N4M_API n4m_status_t n4m_optimizer_get_trials(const n4m_optimizer_t*, n4m_method_result_t** out);
N4M_API void         n4m_optimizer_destroy(n4m_optimizer_t*);
```

The handle owns a seeded `n4m_rng` and the full trial table (it is **stateful** — that is what makes pruning possible). It is one-context-per-thread like every n4m handle. `out_trial`/`out_score` are core-owned scratch, only *borrowed* by the host; the sole host-freed object is the `n4m_method_result_t` trace (the sanctioned core-allocated + `n4m_method_result_destroy` pattern). `n4m_optimizer_get_trials` packs params + scores + status so Studio can stream `TrialResult`s.

### 4.3 Pure-native finetune (zero host code, the portability win)

```c
N4M_API n4m_status_t n4m_finetune_estimator(
    n4m_context_t*, n4m_algorithm_t estimator,
    const n4m_matrix_view_t* X, const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan, const n4m_search_space_t* space,
    n4m_sampler_kind_t, n4m_opt_direction_t, int32_t metric,
    int32_t n_trials, uint64_t seed, n4m_method_result_t** out_result);
```
Internally: the ask/tell loop, feeding the sampled config into `n4m_config_set_*` then `cross_validate_regression` over `plan`, reading `cv.metrics.rmse` back into `tell`. This one entry point gives R/MATLAB/Octave/WASM adaptive finetuning of a native model with a single synchronous call.

### 4.4 Algorithm set + priorities

> **Superseded by [`FINETUNING_ROADMAP.md`](FINETUNING_ROADMAP.md) §2–3 (Codex + adversarial review).** The table below is the original sketch and is retained for the ML rationale only. The authoritative facts: (a) `grid` is **not** a libn4m sampler (dag-ml owns enumeration); (b) successive-halving/Hyperband/median are **pruners** (a separate `n4m_pruner_kind_t`), not samplers; (c) the multi-fidelity axis is the **PLS `n_components` learning curve** (one max-K fit per CV fold, for eligible PLS1/NIPALS paths — *not* free for SIMPLS/multi-target), subsample, or DL epochs — **not** CV-fold fraction (which breaks the rank-preservation successive-halving assumes).

| # | Algorithm | ML rationale | Build cost | Kind |
|---|---|---|---|---|
| 1 | **Uniform random** | strong baseline; startup seeding for all others | reuse RNG + result bag | sampler |
| 2 | **Sobol / LHS** | better coverage than i.i.d. at small budgets; ideal warm-start for the 600-combo spaces | low-new (quasi-random kernels) | sampler |
| 3 | ~~Grid~~ → **dag-ml** | exhaustive over finite categorical spaces | **delegate to dag-ml** `enumerate_variants` + `select_candidate` | *not a libn4m sampler* |
| 4 | **Successive-halving / Hyperband / median** | prune weak configs early on the `n_components`/subsample/epoch fidelity axis | med-new (rung bookkeeping + `tell_intermediate`) | **pruner** |
| 5 | **Ternary int search** | `n_components` is unimodal in CV-RMSE → O(log n) vs a linear sweep; ports `BinarySearchSampler` | low-new | sampler |
| 6 | **GA / PSO (reuse)** | combinatorial preprocessing-chains / variable selection | reuse loops, generalize genotype | sampler |
| 7 | **CMA-ES** | continuous params + model-HP tail (with independent-random fallback for categoricals) | med-new (population pattern + `next_normal`) | sampler |
| 8 | **TPE** | Optuna default; mixed + **conditional/tree** spaces | high-new (Parzen split, clean-room feasible) | sampler |
| 9 | **GP-EI Bayesian** | small budgets, smooth continuous objectives | high-new — **optional / cuttable** | sampler |

**Default `auto` policy** (Optuna-analogous): pure finite categorical → grid (dag-ml); single unimodal int → ternary; combinatorial → GA/PSO; continuous-only budget ≥ ~30 → CMA-ES, else GP-EI (if built); mixed budget ≥ ~20 → TPE; first ~10 trials always Sobol/random seeding; whenever an intermediate-score axis exists (`n_components`/subsample/epochs) → wrap in an ASHA/median pruner (racing for fold-based stopping).

### 4.5 ABI implications

All additive symbols ⇒ **ABI minor bump (2.0 → 2.1)**. Design the *entire* symbol surface in Phase 1 so later samplers are enum values, not new symbols. Regenerate `cpp/abi/expected_symbols_{linux,macos,windows}.txt` and add a `docs/abi/changes_log.md` entry in the same PR (the diff fails closed).

---

## 5. What to build in `dag-ml` / `dag-ml-data`

- **`dag-ml-data`: a `SearchSpace` descriptor** — one new *declarative* contract, JSON-identical across `dag-ml` ↔ `dag-ml-data` (validated by each repo's `validate_contracts.py`). It encodes the DSL exactly (`int|float|log_float|categorical` + the `sorted_tuple`/static-scalar pass-through the Optuna DSL supports, list/dict form, `model_params` vs `train_params`, and the **full `mutex/requires/exclude` + `pick/arrange/count` constraint richness** so it lowers straight into `GenerationSpec` for flavor A — single-value `set_condition` is *not* enough; see roadmap §3-decision-7) plus study controls (`n_trials 5–500 | timeout | direction | random_state | sampler | pruner | eval_mode`). Because it is declarative, any binding can build it — no callback needed to *define* a search.
- **`dag-ml`: wire the `Tuner` node to the loop.** Reuse `NestedCvSpec`/`build_inner_fold_set` so the whole search runs inside the inner folds of the current outer-train partition (leakage refused structurally). Per trial: `ask` → fit on inner-train only → validation `CandidateScore` → `tell`; each candidate gets a `stable_json_fingerprint` keyed on params+fold+seed (dedup/cache/lineage fall out for free). After the budget, the *same* `select_candidate` the grid path uses picks the winner, which refits on full outer-train. The only structural change: the candidate stream is *sampler-generated incrementally* instead of `enumerate_variants` emitting all up front.
- **`dag-ml`: a score-return path, not a new callback family.** Reuse the existing `DagMlControllerVTable{fit,predict,invoke}`: the `Tuner` issues a `NodeTask` per candidate and the host returns a `NodeResult` carrying the validation score (the metric field / PredictionStore row becomes the defined channel). Add only thin `dagml_tuner_ask_json` / `dagml_tell_json` wrappers over the `libn4m` optimizer handle, plus a `TrialResult` stream for Studio. The raw sampler callback, if any, lives in `libn4m`, never in dag-ml.

**Migration from the Python `run_backend` shim** (staged, each stage shippable):
- *Stage 0 (done):* `_or_`/grid generation already lowers natively — flavor A needs no Python.
- *Stage 1:* stop demoting `_grid_`/dict-form/multi-model generators (`detect._FORCE_PYTHON_STEP_KEYS`, `_generation_kind`) — route them through `GenerationSpec`. Enumeration finetuning becomes fully native.
- *Stage 2:* land the `libn4m` optimizer + `SearchSpace`; wire flavor B; drop `finetune_params` from the force-Python set for native-model steps.
- *Stage 3:* wire flavor C via the vtable score-return; Optuna retained only as a *selectable sampler*. End state = ADR-17 (`DEFAULT_ENGINE="dag-ml"`).

---

## 6. Per-language idiomatic finetuning

| Language | Idiomatic API | Native ecosystem | Decision |
|---|---|---|---|
| **Python** | `n4m.model_selection.N4MSearchCV(...)` (the `n4m.sklearn` namespace was removed in the ABI-2 migration), keep `finetune_params` | rich (Optuna, Ray Tune, skopt) | **both** — keep Optuna for power users; `sampler="native"` for cross-language parity |
| **R** | `parsnip`/`tune` engine + `n4m_tune(...)` | rich (`tune`, `mlr3tuning`, `ParBayesianOptimization`) | **both** — `tune_bayes` drives R→C per trial (already works); `n4m_tune` uses the native sampler for reproducibility |
| **MATLAB / Octave** | `n4m.finetune(X,Y,space,opts)` struct API | `bayesopt` (MATLAB only, **proprietary toolbox, absent in Octave**) | **native only** — wrapping `bayesopt` breaks the shared Octave target |
| **JS / WASM** | `await n4m.finetune({searchSpace, sampler, ...})` | essentially none (`hpjs` is niche JS, not WASM) | **native only** — one synchronous native call; progress via batched re-entry, no Asyncify |

```python
# Python
search = N4MSearchCV(pipeline, search_space={"pls__n_components": ("int", 1, 30)},
                     n_trials=50, sampler="tpe", seed=42, cv=5)   # sampler="native" ⇒ cross-lang parity
```
```r
# R — native sampler, bit-identical across bindings
res <- n4m_tune(pls_n4m(n_components = tune()), X, y, sampler = "tpe", n_trials = 50, seed = 42)
```
```matlab
% MATLAB/Octave — no bayesopt dependency
res = n4m.finetune(X, Y, struct('n_components', n4m.intspace(1,30)), ...
                   struct('Sampler','tpe','NumTrials',50,'Seed',42,'CV',5));
```

**The reproducibility argument (why native, not wrap).** Only a native sampler over a native RNG gives **bit-identical HPO across all four languages**. Wrapping each language's own optimizer means Optuna-TPE (Python) vs `ParBayesianOptimization` (R) vs `bayesopt` (MATLAB) — different samplers, different RNGs, so even at the same seed the trial trajectories diverge and the selected "best" config differs by language. That breaks the ecosystem's "same binary everywhere" promise exactly where it is most valuable: Studio round-trips, `nirs4all-papers` reproducibility kits, `nirs4all-benchmarks` scoring. **Wrapping is an idiomatic convenience (Python/R); the native optimizer is the portable contract, and the only contract for MATLAB/Octave/WASM.**

> **Framing correction (from verification):** do **not** claim "native is the only way HPO is *possible* in these languages" — MATLAB `bayesopt` and JS `hpjs` exist. Claim the precise thing: native is the only path that is *simultaneously* reproducible-across-all-bindings, toolbox-free and single-source. For PLS the objective is *already* a native kernel, so native is the *natural* home, not a fallback forced by host limitations.

---

## 7. Hard engineering decisions to settle *before* building

These come out of adversarial review; the design is sound but underspecified here, and each is a real fork:

1. **Pruning semantics must be asynchronous.** A per-`tell` verdict (`tell_intermediate → out_should_prune`) is only sound for **ASHA / median / percentile** pruners, *not* classic synchronous successive-halving (which compares a whole rung cohort). Commit explicitly to ASHA/median semantics, or `N4M_SAMPLER_HALVING` silently implements a different algorithm than the Hyperband literature it cites. Add trial `budget`/`rung` accessors and — for **epoch-fidelity** DL — an `ask`-side "resume a suspended trial" path plus host-side partial-model caching keyed by trial id. Fold-fidelity is survivable without this; epoch-fidelity is not.
2. **Batch-ask, determinism and concurrency.** "identical seed → identical ask sequence" holds only for **sequential tell-order**. Under parallel evaluation, completion order is nondeterministic and adaptive samplers (TPE, CMA-ES mid-generation) diverge. Decide: either scope reproducibility to sequential mode, or make within-batch tells order-invariant (generation-synchronous samplers buffer tells and update per batch). State handle thread-safety (Python GIL serializes cheaply; R/WASM have no threads). Acknowledge the **structural limit**: an in-process handle confines parallelism to one process, unlike Optuna's storage-mediated multi-process studies — fine for Studio's single backend, a gap for HPC scale-out.
3. **Persistence/resume is new ABI surface.** There is *no* handle-serialization precedent in `libn4m`. Choose now: a versioned little-endian `n4m_optimizer_save/load` blob, or **replay-based resume** (`create(seed)` + replay the recorded ask/tell log — matches dag-ml's lineage philosophy, but requires the deterministic tell-order from decision 2). Studio streaming itself is fine via `n4m_optimizer_get_trials` (add a `since_id` incremental form).
4. **Parity gate for stochastic samplers ≠ the PLS gate.** Do **not** chase bit-parity with Optuna's TPE (its behavior is dominated by undocumented, version-drifting constants — `prior_weight`, `n_ei_candidates`, bandwidth clips, multivariate mode — plus NumPy RNG order). Use a **statistical/behavioral** gate: match sampling distributions and regret curves within tolerance on fixed benchmark spaces, and assert **decision-level** parity on canned `(params, score)` histories (isolates deterministic logic from RNG). Keep the existing 1e-12/1e-8 numeric gates for the deterministic PLS objective. Cross-binding parity (Python≡R≡WASM≡MATLAB at the same seed) *is* the tight ~1e-12 gate, via the `benchmarks/cross_binding` harness.
5. **TPE yes, GP-EI maybe never.** TPE-lite from scratch is a few hundred dependency-free lines and genuinely feasible. GP-EI needs marginal-likelihood hyperparameter optimization, PSD-jitter robustness and mixed/conditional kernels — demote to "optional, possibly cut". **TPE + CMA-ES + ASHA covers the practical Optuna-replacement surface.**

---

## 8. Non-goals — what stays Python-only

The optimizer owns **SEARCH**; the host owns **EVALUATION**. Explicitly *not* in `libn4m`:

- **DL objective evaluation** — fitting a torch/TF model to score a trial needs the framework in-process; `libn4m` never links torch. The native sampler proposes via `ask`; the host trains and returns the score via `tell` (flavor C).
- **SHAP / explain** — stays in `nirs4all` Python; not a search concern.
- **Optuna storage backends / dashboards / RDB resume UI** — the native study exposes a serializable state (or replay log); persistence and visualization are host/Studio responsibilities.
- **Framework-specific param nesting** (`compile_`/`fit_`/`optimizer_` bucketing, `set_params` merge) — stays in Python controllers; the ABI sees only a flat typed param vector.

---

## 9. Phased roadmap

- **Phase 1 — native optimizer skeleton + pure-native objective.** Ship the *complete* `n4m_optimizer_t` + search-space builders + `ask`/`tell`/`tell_intermediate` + enums, with samplers `random`, `sobol/lhs`, `ASHA-halving`, `ternary`; objective = internal CV-RMSE over native PLS (`n4m_finetune_estimator`). No host callback. Minor ABI bump; regenerate all three symbol snapshots. Settle decisions 1–4 here.
- **Phase 2 — dag-ml Tuner drives it cross-language + host-objective loop.** `SearchSpace` contract in dag-ml-data; `Tuner` node runs `ask → controller.fit → tell` inside nested CV; flavor B fully native, flavor C via the existing vtable. Extend `parity-gate.yml` + the cross-binding orchestrator with the ask/tell loop.
- **Phase 3 — CMA-ES / TPE (+ optional GP) + idiomatic wrappers.** Samplers as enum values; per-language wrappers (Python `SearchCV`, R `tune`/`mlr3`, MATLAB struct, WASM promise). Optuna demoted to a selectable sampler.

---

## 10. Bottom line

- The **deterministic** half of finetuning is already native in dag-ml (generation + selection); the **adaptive** half is Python-only (Optuna). That asymmetry is the entire problem.
- The clean fix is **not** "port Optuna to Rust/C++ wholesale" and **not** "let each language wrap its own HPO". It is a **native ask/tell optimizer in `libn4m`** (one algorithm, every binding), **orchestrated by dag-ml's Tuner node** (one set of OOF/leakage/selection guarantees), with **the host driving evaluation** (native for PLS, Python for DL). This is the ML-standard decomposition (ask/tell = Optuna/skopt/Nevergrad/Ax) and it respects every n4m ABI rule.
- It makes finetuning a **dag-ml capability chosen by a string**, portable and bit-reproducible in R/WASM/MATLAB/Octave, with Optuna reduced to an optional sampler and DL evaluation the *only* thing that stays Python — exactly the North Star boundary ("portable kernels incl. finetune move to methods; dag-ml covers coordination cross-language").
- Build honestly: methods gives you strong *primitives* and *proven patterns* but the optimizer is real new C++; settle the pruning/determinism/persistence/parity decisions in §7 before writing samplers; and treat GP-EI as optional. TPE + CMA-ES + ASHA + the reused GA/PSO/ternary is a full, defensible Optuna-replacement surface.

---

## Appendix — evidence map (file:line)

**nirs4all (Python HPO):** `nirs4all/optimization/optuna.py` (`OptunaManager`, `BinarySearchSampler`, sampler/pruner/fold-strategy logic); `controllers/models/base_model.py:682,711,742,752,851` (dispatch → finetune → OptunaManager → refit); `pipeline/steps/parser.py:52` (`finetune_params` reserved); `_generator/…` (`_or_`/`_grid_`/`_mutex_` tokens); `pipeline/execution/refit/{config_extractor.py:625,662, executor.py:347,390}` (best-params → refit); `pipeline/dagml/{run_backend.py:117, detect.py:173,182}` (finetune forced to Python).

**dag-ml:** `crates/dag-ml-core/src/generation.rs:521,570,652` (`enumerate_variants`, constraints, seed derivation), `:1000-1217` (parity tests); `selection.rs:337,461` (`select_candidate`); `graph.rs:33` (`NodeKind::Tuner`); `plan.rs:200,950` (nested CV, enumerate-in-plan); `fold.rs:472,519,544` (`NestedCvSpec`); `crates/dag-ml-capi/src/lib.rs:386,1646,1688` (controller vtable; select FFI); `crates/dag-ml-wasm/src/lib.rs:147,197` (select + plan-build over WASM); `crates/dag-ml-py/src/in_process.rs:242,341` (per-variant fit closure, refit); `dsl/tests.rs:3486` (`compiles_tuner_as_external_prediction_node`); `docs/{CAPABILITY_MATRIX.md:24-26, COORDINATOR_SPEC.md:286, TARGET_RESPONSIBILITY_SPLIT.md}`.

**nirs4all-methods:** `cpp/include/n4m/feature_selection.h:108-228` (GA/PSO/CARS/VISSA/… native optimizers); `cpp/include/n4m/n4m.h:403-411,838-905` (`n4m_validation_plan_t`, result bag); `cpp/src/core/{ga_selection.cpp:236,259,385-655, pso_selection.cpp:28,191,334-385}` (bespoke loops, copy-pasted `subset_rmse`/splitmix64 — the "partial reuse" finding); `cpp/src/core/common/rng_engine.h:44-76` (shared RNG engine, unused by GA/PSO); `cpp/src/core/{validation.hpp,cross_validation.cpp:177}` (folds + CV); `cpp/src/c_api/c_api_method_result.cpp` (result packing). No function-pointer typedef anywhere in `cpp/include/n4m/`.

**Consumers:** `nirs4all-studio/api/{pipeline_canonical_finetune.py:8-81, pipeline_canonical.py:802-1519, automl.py:85-128, execution_driver.py:253-290}` (canonical DSL + streaming/pruning/resume requirements); `nirs4all-core/docs/{CAPABILITIES.md, OPERATORS.md:9-23}` (operator gate; finetune out of current scope); `nirs4all-aom/paper/repro/{hpo_recipe_frequency.py, run_linear_hpo_paper_aom.py:134-166}` (600-combo preprocessing grid + model-HP tail).

**Bindings:** `nirs4all-methods/bindings/{python/src/n4m/_ffi.py:124, r/pls4all/src/r_dispatch.c:45,558, js/{src/ffi.ts, CMakeLists.txt:68-78}, matlab/mex/n4m_method_fit_mex.c}` (pure marshallers; WASM single-thread/no-Asyncify/`addFunction` unexported).
