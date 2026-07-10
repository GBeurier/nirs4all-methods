No Blockers found.

- **Major** - [sobol.cpp](/home/delete/nirs4all/_worktrees/native-hpo-phase1/cpp/src/core/optimization/sobol.cpp:42): Sobol can read past `kSobolSv[d][c]` after `2^30` successful asks. At ask index `1,073,741,824`, `next_index_ == 2^30 - 1`, so `c == 30`, but [sobol_direction.hpp](/home/delete/nirs4all/_worktrees/native-hpo-phase1/cpp/src/core/optimization/sobol_direction.hpp:13) only has columns `0..29`. Minimal fix: guard `i >= (1LL << kSobolBits)` / `c >= kSobolBits` and either return an ask error or fall back to base random.

- **Major** - [gp.cpp](/home/delete/nirs4all/_worktrees/native-hpo-phase1/cpp/src/core/optimization/gp.cpp:20): `is_continuous()` includes `INT` and `LOG_INT`, and stepped floats are also treated as GP axes. But [gp.cpp](/home/delete/nirs4all/_worktrees/native-hpo-phase1/cpp/src/core/optimization/gp.cpp:190) stores raw unit coords before `decode_candidate()` snaps int/step values. Concrete failure: `add_int("k", 1, 4, 1, 0)` has no continuous axis, but GP still fits on raw `u`; many distinct `u` values decode to the same `k`, so the surrogate models fake distinct observations and does not use the intended independent fallback. Minimal fix: build `cont_axes_` from `ParamSpec`, not kind alone; include only non-int float/log-float axes with no snapping, e.g. `!p.is_int && p.step <= 0.0`.

- **Major** - [gp.cpp](/home/delete/nirs4all/_worktrees/native-hpo-phase1/cpp/src/core/optimization/gp.cpp:38): `M_PI` is non-standard C++17. This can fail to compile on conforming libc++/MSVC-style environments. Minimal fix: use a local `constexpr double kPi = 3.141592653589793238462643383279502884;`.

- **Minor** - [test_optimization.cpp](/home/delete/nirs4all/_worktrees/native-hpo-phase1/cpp/tests/test_optimization.cpp:697): `test_gp_ei_converges` is a weak GP regression guard. A broken GP path that always falls back to random acquisition can still satisfy `bs < 0.5` on favorable seeds, and the test does not cover maximize EI sign, duplicate/all-equal observations, zero/single continuous axes, or discrete fallback. Minimal fix: add targeted edge-case tests plus a deterministic maximize test; consider comparing GP-EI against a fixed random baseline or tightening the threshold across several fixed seeds.

Correct/OK in the reviewed scope: Sobol Gray-code order and cache behavior are correct for normal sequential asks, batch asks, and same-index constraint retry. `dim >= 52` falls back without table OOB. EI sign for minimize/maximize is correct. GP duplicate/all-equal continuous observations should not crash with the current jitter path. The Sobol C++ and Python tests compare ordered matrices, so they do pin Gray-code order for covered dimensions.
---

## Resolution (all findings applied)

- **Major — Sobol OOB after 2^30 asks** (sobol.cpp): added an `exhausted_` flag. When
  the Gray-code advance would need column `c >= kSobolBits` (i.e. past 2^30 points),
  the sampler stops and `override_numeric`/`override_categorical` return `false`, so
  the base uniform sampler takes over. No OOB read; the sequence degrades gracefully.

- **Major — GP fake-distinct observations from decode snapping** (gp.cpp): the GP now
  stores the **decoded** continuous-axis coordinates (via `unit_from_numeric` on the
  trial's post-`decode_candidate` value), not the raw pre-snap proposal. Two asks that
  decode to the same integer now share one GP coordinate (duplicate rows, kept PD by
  the diagonal jitter), so the surrogate models real observations. For plain float
  axes the decode round-trips, so this is identical to the raw coordinate there.
  *Deviation from the literal suggestion (excluding int axes):* integer axes are kept
  in the GP because `n_components` — the canonical NIRS hyperparameter — is an int, and
  continuous relaxation of integers is standard (and consistent with the already-approved
  CMA-ES). Storing decoded coords eliminates the exact defect Codex described (fake
  distinct observations) without crippling integer optimisation.

- **Major — `M_PI` non-standard in C++17** (gp.cpp): replaced with a local
  `constexpr double kPi`.

- **Minor — weak GP test**: added `test_gp_ei_maximize` (deterministic MAXIMIZE peak —
  a flipped EI sign would flee the optimum) and `test_gp_ei_edge_cases` (pure-categorical
  → random fallback returns the right best; constant objective → all-equal `y` + duplicate
  decoded coords must not destabilise the Cholesky or throw across the ABI).

384 C++ tests pass; Sobol Tier-A parity and the Python smoke still green. ABI unchanged.
