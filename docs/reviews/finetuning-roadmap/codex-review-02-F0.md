# Codex review — F0 (native HPO ABI + random sampler)

**Date:** 2026-07-10 · **Reviewer:** `codex exec` (codex-cli 0.144.1, reasoning_effort=medium, sandbox=read-only) over the F0 worktree.
**Target:** cpp/include/n4m/optimization.h, cpp/src/core/optimization/optimizer.{hpp,cpp}, cpp/src/c_api/c_api_optimization.cpp, cpp/tests/test_optimization.cpp.
**Disposition:** all 14 findings (4 Blocker, 9 Major, 1 Minor) ACCEPTED and applied ("Codex wins"). See the F0 fixups commit; regression tests + standalone-include guards added.

---

**Blocker**

1. `optimization.h` is not self-contained. `cpp/include/n4m/optimization.h:8` includes `n4m.h`, while `n4m.h:953` includes `optimization.h` and then static-asserts HPO enums at `n4m.h:1017`. Direct `#include "n4m/optimization.h"` fails because the include guard skips the enum definitions.
Fix: move HPO enum static asserts into `optimization.h` after the enum definitions, or split base ABI types into a non-recursive header. Add C and C++ include-only CI tests for every public role header.

2. Some exported status-returning trial accessors can throw across the C ABI. The name-based accessors call `Trial::find(const std::string&)` (`optimizer.cpp:29`), so `const char* name` conversion may allocate/throw, but `n4m_trial_get_int/float/category/is_active` have no try/catch (`c_api_optimization.cpp:477`, `485`, `493`, `503`).
Fix: wrap every exported accessor in try/catch and map `bad_alloc` to `N4M_ERR_OUT_OF_MEMORY`; better, make `Trial::find` take `std::string_view`.

3. `struct_size` handling is unsafe and does not preserve defaults. `create` and `finetune` zero-init options, then do `memcpy(..., copy > 0 ? copy : sizeof(o))` (`c_api_optimization.cpp:292-296`, `546-550`). If `struct_size == 0`, it full-copies anyway; if an older partial struct omits fields, absent fields become zero rather than `options_init` defaults (`eval_mode` becomes BEST, `n_startup_trials` becomes 0).
Fix: initialize with `n4m_optimizer_options_init(&o)`, reject `struct_size < sizeof(uint64_t)`, copy exactly `min(struct_size, sizeof(o))`, never substitute full size for zero, and validate enum ranges/reserved bytes.

4. Conditional constraints do not implement the frozen ABI. The header documents condition labels as `{refs[1..]}` (`optimization.h:38`), but `add_constraint` only consumes `ref[1]`/`label[1]` (`c_api_optimization.cpp:253-264`). The core stores only one `cond_parent` (`optimizer.hpp:45-48`), silently dropping different parents.
Fix: represent condition constraints as constraints, not a single field on `ParamSpec`; consume all refs, implement multi-label/multi-parent semantics, or reject unsupported shapes with `N4M_ERR_UNSUPPORTED`.

**Major**

5. Constraint rejection can silently return an invalid trial. `sample()` retries 200 times, then keeps the last invalid sample (`optimizer.cpp:86-147`), and `ask()` returns `N4M_OK` (`optimizer.cpp:220-253`).
Fix: make sampling return status/bool; after retry exhaustion return `N4M_ERR_INVALID_ARGUMENT` or validate constraint satisfiability at optimizer creation.

6. Enqueue/warm-start can violate constraints and breaks deterministic streams. `ask()` samples first, consuming RNG, then overrides queued values (`optimizer.cpp:224-249`), does not re-run `constraints_ok`, and rounds/clamps categorical indices (`optimizer.cpp:235-245`).
Fix: validate queued names/ranges/categories up front, sample only unforced dimensions, re-run conditions and constraints, and reject invalid forced candidates instead of clamping silently.

7. Log/numeric ranges are under-validated. `add_float` accepts NaN/Inf and log ranges with `low <= 0` (`c_api_optimization.cpp:130-144`); `sample_numeric` silently falls back to linear sampling when log bounds are invalid (`optimizer.cpp:66-76`).
Fix: reject non-finite bounds/steps and require positive bounds for log params at search-space construction.

8. `MUTEX_GROUP` semantics are wrong for a mutex group. It fails only when all refs are present (`optimizer.cpp:183-190`), allowing two of three mutually exclusive refs.
Fix: count present refs and fail when count > 1, or rename/document this as an all-of forbidden-set constraint and add a real mutex.

9. `n4m_finetune_estimator` silently ignores almost the entire search space. It only reads `"n_components"` (`c_api_optimization.cpp:567-574`); all other sampled params affect the trial trace but not the estimator, and missing `n_components` falls back to `Config`’s default 2 (`config.hpp:36`).
Fix: for F0, require exactly supported estimator params or reject unsupported names/kinds with `N4M_ERR_UNSUPPORTED`.

10. `n4m_finetune_estimator` does not return the documented trial trace. The header promises best params plus trial trace (`optimization.h:171-176`), but the result only stores `trial_scores`, `n_trials`, and `metric` (`c_api_optimization.cpp:601-610`), unlike `get_trials` (`c_api_optimization.cpp:427-444`).
Fix: share a trial-trace packing helper and include ids, scores, status, rung, duration, and params consistently.

11. All CV failures still return `N4M_OK`. Failed CV trials are marked failed and skipped (`c_api_optimization.cpp:576-582`); if no trial completes, `best == nullptr` and the function still returns a result with no `best_score` (`c_api_optimization.cpp:591-612`).
Fix: track completed trials and return `N4M_ERR_NOT_FITTED` or the last CV status when none complete.

12. Metric direction defaults are easy to get wrong. `options_init` sets `direction = N4M_OPT_MINIMIZE` (`c_api_optimization.cpp:81`), while auto-derivation only happens for `N4M_OPT_AUTO` (`optimizer.cpp:54-55`). Setting metric to `R2` alone will minimize R2.
Fix: make the default direction `N4M_OPT_AUTO`, or require/validate explicit direction whenever metric changes.

13. Timeout and duration are frozen but nonfunctional. `timeout_seconds` promises `ask` returns `N4M_ERR_CANCELLED` (`optimization.h:91`), but `ask()` never checks time (`optimizer.cpp:220-253`). `duration_seconds` exists (`optimizer.hpp:81`) but `tell_result` never sets it (`optimizer.cpp:261-269`).
Fix: store trial start time, enforce optimizer timeout in `ask`, and set duration on terminal tell.

**Minor**

14. Void destroy wrappers do not follow the header’s catch-and-swallow rule. `n4m.h:13-16` says destroy/free wrappers catch; `n4m_search_space_destroy` and `n4m_optimizer_destroy` are raw deletes (`c_api_optimization.cpp:105-106`, `317-318`). Current destructors should not throw, but the ABI rule should be uniform.
Fix: wrap both in `try { delete ...; } catch (...) {}`.

**Top 3 Must-Fix Before F1**

1. Fix public header self-containment and add include-only ABI tests.
2. Harden the ABI boundary: trial accessor try/catch plus safe `struct_size` copying/defaulting.
3. Make constraints authoritative: implement condition semantics, reject unsatisfied constraints, and validate enqueue candidates.