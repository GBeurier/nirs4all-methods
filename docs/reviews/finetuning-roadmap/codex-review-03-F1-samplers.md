# Codex review — F1 samplers (ternary + lhs)

**Date:** 2026-07-10 · **Reviewer:** `codex exec` (codex-cli 0.144.1, reasoning_effort=medium, read-only).
**Target:** ternary.cpp, lhs.cpp, optimizer.{hpp,cpp} (override_numeric/numeric_from_unit), test_optimization.cpp.
**Disposition:** 0 Blocker, 5 Major, 2 Minor — all applied. Ternary reworked into grid-index space (step-aware, reserves running trials, skips inactive/off-domain history, bounded arithmetic); base enqueue validates numeric ranges + categorical indices; LHS seed domain-separated + size_t index compare. +2 regression tests.

---

**Blocker**
None found.

**Major**
1. Ternary repeats candidates for parallel/batch asks. `next_ternary_value()` ignores RUNNING trials and only records completed scores, so repeated `ask()` calls before `tell()` all see the same history and can all return `low_` or the same next candidate. See `ternary.cpp:45-52`, `optimizer.cpp:285-290`, `c_api_optimization.cpp:401-406`.  
Fix: include running target values in `tested` as reservations, but only use completed trials in scored `hist`.

2. Ternary bypasses integer `step`. Base numeric sampling snaps/clamps stepped values in `numeric_from_unit()`, but ternary returns raw `low/high/mid/bisect` values directly. See `optimizer.cpp:85-90`, `optimizer.cpp:180-181`, `ternary.cpp:55-58`, `ternary.cpp:82-87`.  
Fix: run ternary over grid indices, then map back with `low + idx * step` and clamp to `high`.

3. Forced numeric values can poison ternary state. `enqueue()` validates names only, numeric forced values are stored raw, and ternary trusts every completed target value. An out-of-range forced best can make the spiral search never reach the legal range and return the bad value again. See `optimizer.cpp:294-298`, `optimizer.cpp:108-110`, `ternary.cpp:47-51`, `ternary.cpp:91-100`.  
Fix: validate forced numeric values against range/step at enqueue/sample time, and have ternary ignore out-of-domain history defensively.

4. Ternary counts inactive conditional values as evidence. Conditions are applied after sampling, but ternary history only checks that the param exists, not that it was active. See `optimizer.cpp:187`, `optimizer.cpp:193-209`, `ternary.cpp:47-50`.  
Fix: skip `tp == nullptr || !tp->active`; consider not auto-selecting conditional integer axes as the ternary target.

5. Ternary has signed-overflow/precision hazards for legal `int64_t` ranges. The API accepts `int64_t` bounds, but ternary converts through `double`/`llround()` and computes `high_ - low_`, `best_v ± d` in signed integers. See `c_api_optimization.cpp:135-150`, `ternary.cpp:26-27`, `ternary.cpp:57`, `ternary.cpp:96-98`.  
Fix: store exact integer bounds internally or reject too-wide ternary ranges; compute in checked unsigned/index space.

**Minor**
6. LHS uses the same seed stream as the base RNG. `Optimizer` and `LhsSampler` both seed SplitMix64 with `opts.seed`, creating avoidable correlation between LHS design and categorical/fallback random draws. See `optimizer.cpp:58`, `lhs.cpp:35-36`.  
Fix: domain-separate the LHS seed with a sampler constant or split seed.

7. LHS index comparison can wrap at huge trial counts. `trials().size()` is cast to `int32_t` before comparing with `n_startup_`, then the original `size_t` indexes `unit_`. See `lhs.cpp:57-65`.  
Fix: compare as `if (i >= static_cast<std::size_t>(n_startup_))`.

Top must-fix before the next sampler: ternary pending-trial reservations, stepped-grid handling, forced-value validation, and inactive conditional history. Existing `build/dev-debug/cpp/tests/n4m_tests` passed: 367 passed, 0 failed.