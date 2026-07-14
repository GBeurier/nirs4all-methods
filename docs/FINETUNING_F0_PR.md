# PR F0 — native HPO optimizer ABI surface (`optimization` category)

**Companion to** [`FINETUNING_ROADMAP.md`](FINETUNING_ROADMAP.md) (phases, §3 decisions) and [`NATIVE_FINETUNING.md`](NATIVE_FINETUNING.md) (architecture).
**Scope of this PR:** land the **complete, frozen** C-ABI surface for the native finetuner + a minimal working `random`/`none` slice + all scaffolding + ABI snapshots. Every later sampler/pruner/binding is then **additive** (a `Sampler`/`Pruner` subclass + a reserved enum value activated), never an ABI change.
**Why it is one PR:** F0 is the single serialization point of the whole effort. Once it merges, Phase-1 work forks into ~10 parallel agent tracks (see the roadmap §6). Getting the surface complete here is what avoids a second ABI-minor break.
**Constraint respected:** this PR touches **only `nirs4all-methods`**. It does not touch `dag-ml`, `dag-ml-data`, `nirs4all-core`, or `nirs4all` (Python). The constraint-safe design of the SearchSpace constraint ABI is in §2-D8 below.

---

## 1. Conventions confirmed against the current tree (so the PR is buildable)

- Status enum already has everything needed: `N4M_ERR_NOT_IMPLEMENTED = 11` ("deferred public surface", `n4m.h:85`) for a **reserved-but-unactivated** sampler/pruner; `N4M_ERR_CANCELLED = 17` (`n4m.h:91`) for timeout; `N4M_ERR_UNSUPPORTED = 10` for a param/constraint kind the space can't express. **No new status value is needed.**
- `n4m_algorithm_t` (`n4m.h:424`, `N4M_ALGO_PLS_REGRESSION=0…`) is reused by `n4m_finetune_estimator`. `n4m_array_t` (`n4m.h:422`) + `n4m_array_free` are the core-allocated/`free` blob type for `save`. `n4m_method_result_t` is the result bag. `n4m_validation_plan_t` is the fold contract.
- There is **no** existing `n4m_metric_t` enum (`metrics.h` is functions only) — F0 introduces it.
- CMake: core `.cpp` are listed by name in `cpp/src/CMakeLists.txt`; only `cpp/src/c_api/*.cpp` is `CONFIGURE_DEPENDS`-globbed (`n4m_targets.cmake`). Tests are the hand-rolled `harness.hpp` compiled into one `n4m_tests` binary, files registered in `cpp/tests/CMakeLists.txt`.
- 4-byte enum discipline: every enum below gets `N4M_STATIC_ASSERT(sizeof(...)==4, …)` next to the existing block (`n4m.h:966+`).

---

## 2. Decisions resolved (concrete answers to roadmap §3 D1–D11)

| # | Decision | Resolution baked into F0 |
|---|---|---|
| D1 | sampler ⟂ pruner; grid | Two enums (`n4m_sampler_kind_t` ⟂ `n4m_pruner_kind_t`), composed in the options struct. **No `grid` sampler** (dag-ml owns enumeration). |
| D2 | fidelity axis | `tell_intermediate(trial, step, value)` is axis-agnostic; the *engine* that produces the stream (n_components learning-curve etc.) is **F2**, not F0. F0 only freezes the ABI. |
| D3 | async pruning | `tell_intermediate → out_should_prune` returns a per-tell verdict (ASHA/median semantics). |
| D4 | result/status | `n4m_trial_status_t` + `n4m_optimizer_tell_result(status, score, error)` + trial `duration`/`status` accessors. `tell(score)` = `tell_result(COMPLETED, score, NULL)`. |
| D5 | batch/parallel | `n4m_optimizer_ask_batch` + active-trial-aware scalar `ask`; `n4m_liar_kind_t` in the options (default `NONE` ⇒ sequential-only reproducibility). |
| D6 | metric/task | `n4m_metric_t` spans regression **and** classification; direction auto-derived unless `direction` set; single scalar objective in v1, **multi-objective returns `N4M_ERR_NOT_IMPLEMENTED`** (reserved). |
| D7 | DSL subset | `sorted_tuple` + typed categoricals implemented; the binding rejects any `finetune_params` key the native space can't hold (enforced Phase-1-bridge, but the ABI is complete). |
| D8 | constraint richness | Generic `n4m_search_space_add_constraint(kind, refs…)` covering `mutex_group/requires/exclude/condition_in/condition_not_in`. **Designed against the already-implemented, parity-locked dag-ml generation vocabulary** (`dag-ml/crates/dag-ml-core/src/generation.rs`, `nirs4all/pipeline/dagml/detect.py:263`) — so F0 freezes it unilaterally, no coordination with the dag-ml agent. `pick/arrange/count` are dag-ml **variant-generation** operators (they expand the space, they are not per-sample constraints) → they stay dag-ml-side and are out of the native constraint ABI by design. |
| D9 | options struct | `n4m_optimizer_options_t { size_t struct_size; …; uint8_t reserved[64]; }` + `n4m_optimizer_options_init()`; forward-compatible via the `struct_size` prologue. |
| D10 | resume | `n4m_optimizer_save/load` were **declared and reserved** by F0. MT6/F7 has now activated them with portable N4MOPT v1 without changing the reserved signatures, ABI version or SONAME. |
| D11 | `n4m_finetune_estimator` | A **thin single-level driver** over the same ask/tell primitives + the identical result schema; one `validation_plan`, **no nested CV / selection / leakage** (those are dag-ml's). |

---

## 3. The frozen header — `cpp/include/n4m/optimization.h` (sketch)

```c
/* SPDX-License-Identifier: CECILL-2.1 */
#ifndef N4M_OPTIMIZATION_H
#define N4M_OPTIMIZATION_H
#include "n4m/n4m.h"          /* status, matrix view, context, config, algorithm,
                                 array, validation_plan, method_result, N4M_API */
#ifdef __cplusplus
extern "C" {
#endif

/* ---- enums (all 4-byte, static-asserted in n4m.h) ------------------------ */

typedef enum n4m_param_kind_t {
  N4M_PARAM_INT = 0, N4M_PARAM_FLOAT = 1, N4M_PARAM_LOG_INT = 2, N4M_PARAM_LOG_FLOAT = 3,
  N4M_PARAM_CATEGORICAL = 4, N4M_PARAM_ORDINAL = 5, N4M_PARAM_SORTED_TUPLE = 6
} n4m_param_kind_t;

typedef enum n4m_cat_type_t {           /* typed categorical payload (D7) */
  N4M_CAT_STR = 0, N4M_CAT_INT = 1, N4M_CAT_FLOAT = 2, N4M_CAT_BOOL = 3
} n4m_cat_type_t;

typedef enum n4m_constraint_kind_t {    /* D8 — covers the dag-ml generation vocabulary */
  N4M_CONSTRAINT_MUTEX_GROUP = 0, N4M_CONSTRAINT_REQUIRES = 1, N4M_CONSTRAINT_EXCLUDE = 2,
  N4M_CONSTRAINT_CONDITION_IN = 3, N4M_CONSTRAINT_CONDITION_NOT_IN = 4
} n4m_constraint_kind_t;

typedef enum n4m_sampler_kind_t {       /* full range reserved now; unimplemented → NOT_IMPLEMENTED */
  N4M_SAMPLER_RANDOM = 0, N4M_SAMPLER_SOBOL = 1, N4M_SAMPLER_LHS = 2, N4M_SAMPLER_TERNARY = 3,
  N4M_SAMPLER_GA = 4, N4M_SAMPLER_PSO = 5, N4M_SAMPLER_CMAES = 6, N4M_SAMPLER_TPE = 7,
  N4M_SAMPLER_GP_EI = 8
} n4m_sampler_kind_t;

typedef enum n4m_pruner_kind_t {
  N4M_PRUNER_NONE = 0, N4M_PRUNER_MEDIAN = 1, N4M_PRUNER_ASHA = 2,
  N4M_PRUNER_HYPERBAND = 3, N4M_PRUNER_RACING = 4
} n4m_pruner_kind_t;

typedef enum n4m_opt_direction_t {
  N4M_OPT_AUTO = 0, N4M_OPT_MINIMIZE = 1, N4M_OPT_MAXIMIZE = 2   /* AUTO ⇒ derive from metric */
} n4m_opt_direction_t;

typedef enum n4m_eval_mode_t {          /* fold-score aggregation (nirs4all parity) */
  N4M_EVAL_BEST = 0, N4M_EVAL_MEAN = 1, N4M_EVAL_ROBUST_BEST = 2
} n4m_eval_mode_t;

typedef enum n4m_metric_t {             /* D6 — regression + classification */
  N4M_METRIC_RMSE = 0, N4M_METRIC_MSE = 1, N4M_METRIC_MAE = 2, N4M_METRIC_R2 = 3,
  N4M_METRIC_ACCURACY = 16, N4M_METRIC_BALANCED_ACCURACY = 17, N4M_METRIC_F1 = 18,
  N4M_METRIC_LOGLOSS = 19
  /* gaps left for growth; multi-objective is a reserved future vector shape */
} n4m_metric_t;

typedef enum n4m_liar_kind_t {          /* D5 — constant-liar for batch/parallel ask */
  N4M_LIAR_NONE = 0, N4M_LIAR_MIN = 1, N4M_LIAR_MEAN = 2, N4M_LIAR_MAX = 3
} n4m_liar_kind_t;

typedef enum n4m_trial_status_t {       /* D4 */
  N4M_TRIAL_RUNNING = 0, N4M_TRIAL_COMPLETED = 1, N4M_TRIAL_PRUNED = 2, N4M_TRIAL_FAILED = 3
} n4m_trial_status_t;

/* ---- forward-compatible options struct (D9) ------------------------------ */

typedef struct n4m_optimizer_options_t {
  size_t              struct_size;        /* MUST be sizeof(n4m_optimizer_options_t) */
  n4m_sampler_kind_t  sampler;
  n4m_pruner_kind_t   pruner;
  n4m_opt_direction_t direction;
  n4m_eval_mode_t     eval_mode;
  n4m_metric_t        metric;
  n4m_liar_kind_t     liar;
  uint64_t            seed;
  double              timeout_seconds;    /* 0 = none; on expiry ask returns N4M_ERR_CANCELLED */
  int32_t             n_startup_trials;   /* random/sobol seeding before adaptive kicks in */
  uint8_t             reserved[64];       /* forward-compat; must be zeroed */
} n4m_optimizer_options_t;

N4M_API void n4m_optimizer_options_init(n4m_optimizer_options_t* opts);  /* sets struct_size + defaults */

/* ---- search space -------------------------------------------------------- */

typedef struct n4m_search_space_s n4m_search_space_t;

N4M_API n4m_status_t n4m_search_space_create(n4m_search_space_t** out);
N4M_API void         n4m_search_space_destroy(n4m_search_space_t*);
N4M_API n4m_status_t n4m_search_space_add_int   (n4m_search_space_t*, const char* name,
                        int64_t lo, int64_t hi, int64_t step, int32_t log);
N4M_API n4m_status_t n4m_search_space_add_float (n4m_search_space_t*, const char* name,
                        double lo, double hi, double step, int32_t log);   /* step<=0 ⇒ continuous */
N4M_API n4m_status_t n4m_search_space_add_categorical(n4m_search_space_t*, const char* name,
                        n4m_cat_type_t type, const void* values, int32_t n_values);
N4M_API n4m_status_t n4m_search_space_add_ordinal(n4m_search_space_t*, const char* name,
                        const double* values, int32_t n_values);
N4M_API n4m_status_t n4m_search_space_add_sorted_tuple(n4m_search_space_t*, const char* name,
                        int32_t length, double lo, double hi, int32_t element_is_int);
/* generic constraint (D8): refs are parallel arrays of param names + optional labels */
N4M_API n4m_status_t n4m_search_space_add_constraint(n4m_search_space_t*, n4m_constraint_kind_t kind,
                        const char* const* param_refs, const char* const* label_refs, int32_t n_refs);

/* ---- optimizer handle + ask/tell ---------------------------------------- */

typedef struct n4m_optimizer_s n4m_optimizer_t;
typedef struct n4m_trial_s     n4m_trial_t;   /* BORROWED — core-owned, valid until its terminal tell */

N4M_API n4m_status_t n4m_optimizer_create(n4m_context_t*, const n4m_search_space_t*,
                        const n4m_optimizer_options_t*, n4m_optimizer_t** out);
N4M_API void         n4m_optimizer_destroy(n4m_optimizer_t*);
N4M_API n4m_status_t n4m_optimizer_enqueue(n4m_optimizer_t*, const n4m_trial_t* warm_start); /* D-warmstart */
N4M_API n4m_status_t n4m_optimizer_ask(n4m_optimizer_t*, n4m_trial_t** out_trial);
N4M_API n4m_status_t n4m_optimizer_ask_batch(n4m_optimizer_t*, int32_t n,
                        n4m_trial_t** out_trials, int32_t* out_count);          /* D5 */
N4M_API n4m_status_t n4m_optimizer_tell(n4m_optimizer_t*, int64_t trial_id, double score);
N4M_API n4m_status_t n4m_optimizer_tell_result(n4m_optimizer_t*, int64_t trial_id,
                        n4m_trial_status_t status, double score, const char* error); /* D4 */
N4M_API n4m_status_t n4m_optimizer_tell_intermediate(n4m_optimizer_t*, int64_t trial_id,
                        int32_t step, double score, int32_t* out_should_prune);  /* D3 */
N4M_API n4m_status_t n4m_optimizer_best(const n4m_optimizer_t*, n4m_trial_t** out_best, double* out_score);
N4M_API n4m_status_t n4m_optimizer_get_trials(const n4m_optimizer_t*, int64_t since_id,
                        n4m_method_result_t** out);                             /* streaming, D-stream */

/* F0 reserved persistence (D10); activated by N4MOPT v1 in MT6/F7 */
N4M_API n4m_status_t n4m_optimizer_save(const n4m_optimizer_t*, n4m_array_t** out_blob);
N4M_API n4m_status_t n4m_optimizer_load(n4m_context_t*, const uint8_t* blob, uint64_t n,
                        n4m_optimizer_t** out);

/* ---- trial accessors ----------------------------------------------------- */

N4M_API n4m_status_t n4m_trial_get_id      (const n4m_trial_t*, int64_t* out);
N4M_API n4m_status_t n4m_trial_get_int     (const n4m_trial_t*, const char* name, int64_t* out);
N4M_API n4m_status_t n4m_trial_get_float   (const n4m_trial_t*, const char* name, double* out);
N4M_API n4m_status_t n4m_trial_get_category(const n4m_trial_t*, const char* name,
                        int32_t* out_index, const char** out_label);  /* label core-owned, UTF-8 */
N4M_API n4m_status_t n4m_trial_is_active   (const n4m_trial_t*, const char* name, int32_t* out);
N4M_API n4m_status_t n4m_trial_get_rung    (const n4m_trial_t*, int32_t* out);
N4M_API n4m_status_t n4m_trial_get_status  (const n4m_trial_t*, n4m_trial_status_t* out);
N4M_API n4m_status_t n4m_trial_get_duration(const n4m_trial_t*, double* out_seconds);

/* ---- pure-native single-level driver (D11) ------------------------------- */

N4M_API n4m_status_t n4m_finetune_estimator(n4m_context_t*, n4m_algorithm_t estimator,
                        const n4m_matrix_view_t* X, const n4m_matrix_view_t* Y,
                        const n4m_validation_plan_t* plan, const n4m_search_space_t*,
                        const n4m_optimizer_options_t*, int32_t n_trials,
                        n4m_method_result_t** out_result);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* N4M_OPTIMIZATION_H */
```

**ABI-rule compliance:** `extern "C"` only, no STL/Eigen/exceptions across the boundary; the C++ core throws and `c_api_optimization.cpp` catches → `n4m_status_t` (as `ga_selection.cpp:675` already does). `n4m_trial_t`/`n4m_optimizer_t`/`n4m_search_space_t` are **opaque, core-owned**; the only host-freed objects are the `n4m_method_result_t` (via `n4m_method_result_destroy`) and the `n4m_array_t` blob (via `n4m_array_free`). Category labels and trial pointers are **borrowed**, valid until their owner is destroyed / the trial reaches a terminal tell. Matrix views stay stride-aware. No function pointer is added (host drives the loop), so no callback hazard.

---

## 4. File-by-file change list (the 6 method-add surfaces + scaffolding)

| # | Surface | Files | What lands in F0 |
|---|---|---|---|
| 1 | **C++ core** | `cpp/src/core/optimization/` (NEW): `search_space.{hpp,cpp}`, `optimizer.{hpp,cpp}`, `samplers/random.{hpp,cpp}`, `pruners/none.{hpp,cpp}`, `finetune_estimator.cpp`; **register each `.cpp` in `cpp/src/CMakeLists.txt`** | The handle types, the typed search space + generic constraints, the ask/tell state machine + trial table, the `random` sampler, the `none` pruner, and `n4m_finetune_estimator` wired to `cross_validate_regression` over one `validation_plan`. Every reserved sampler/pruner returns `N4M_ERR_NOT_IMPLEMENTED` from a dispatch stub. |
| 2 | **C ABI** | `cpp/include/n4m/optimization.h` (NEW); `cpp/include/n4m/n4m.h` (add `#include` + the `N4M_STATIC_ASSERT(sizeof(enum)==4)` block); `cpp/src/c_api/c_api_optimization.cpp` (NEW, auto-globbed) | The full frozen surface of §3. `extern "C"` wrappers catch-and-translate. |
| 3 | **ABI snapshots** | `cpp/abi/expected_symbols_{linux,macos,windows}.txt`; `docs/abi/changes_log.md`; `cpp/include/n4m/n4m_version.h` | Regenerate all three platforms (Linux locally via `nm -D … | diff`, macOS/Windows via CI artifacts); one changelog line per new symbol; **ABI minor bump 2.0 → 2.1** (`N4M_ABI_VERSION_MINOR`). |
| 4 | **Tests** | `cpp/tests/test_optimization.cpp` (NEW; register in `cpp/tests/CMakeLists.txt`) | Harness cases: options-init/struct_size guard; search-space build incl. a constraint + a sorted_tuple + a typed categorical; `create → ask → tell_result(COMPLETED) → best` on a synthetic quadratic converges; `ask_batch` returns N distinct active trials; a reserved sampler returns `N4M_ERR_NOT_IMPLEMENTED`; `n4m_finetune_estimator(random)` over a real `validation_plan` beats a fixed baseline; determinism-given-seed (same seed → identical `random` sequence). |
| 5 | **Catalog** | `catalog/methods.yaml` (+ regen `catalog/methods/optimization.{random,none}.yaml` via `split_legacy_methods.py`); `catalog/scripts/validate.py:43` (add `optimization` to the donor-category allow-list) | The `optimization` category + the two seed entries with `abi_symbols`/`tu`/`headers`/`parity`/`publications`; `validate.py --strict-abi --check-references` green. |
| 6 | **HPO parity machinery** | `benchmarks/parity_timing/registry.py` (`HpoSpec` dataclass); `parity/comparator/{study_trace,pruner_decisions,sampler_state}.py` (NEW skeletons); `.github/workflows/parity-gate.yml` (HPO job skeleton) | The registry schema + comparator skeletons + a CI job that runs `random` self-consistency. This is the gate every later sampler plugs into. |
| — | **Docs** | `docs/methods/_finetuning_bibliography.bib` (NEW); `docs/methods/random.md` (NEW); `CHANGELOG.md`; `docs/methods/optimization.md` (category overview) | The bibliography seed + the first method page + a changelog entry. |

---

## 5. F0 acceptance / green gate

- [ ] `cmake --build --preset dev-release` + `ctest --preset dev-release --output-on-failure` green; single suite `./build/dev-release/cpp/tests/n4m_tests --test-case="*optimization*"`.
- [ ] `clang-format`/`clang-tidy` clean.
- [ ] **ABI diff clean** vs regenerated `cpp/abi/expected_symbols_{linux,macos,windows}.txt`; `docs/abi/changes_log.md` updated; `scripts/bump_version.sh --check` green; SONAME/linkage audit shows only libc/libstdc++/libgcc/libm.
- [ ] `catalog/scripts/validate.py --strict-abi --check-references` green.
- [ ] The HPO parity CI job runs (even if only `random`).
- [ ] `n4m_cli --abi-info` prints the new triple; `--selfcheck` still passes.
- [ ] **Reviewer sign-off (Codex + human):** the surface is *complete* — options/result/status/batch/metric/constraints/enqueue/since_id/save-load are all present, so no Phase-1 sampler/pruner will need an ABI change. This is the one thing F0 must get right.

---

## 6. Why F0 is the only serialization point

After F0 merges, **nothing else in Phase 1 touches the ABI header, the version, or the symbol snapshots.** Adding `sobol` = a new `Sampler` subclass + activating `N4M_SAMPLER_SOBOL` in the core dispatch + a fixture + a doc page. That is why the ~10 Phase-1 tracks in the roadmap §6 can run as independent parallel agents against the frozen `optimization.h`: they share a compile-time contract that never moves again.
