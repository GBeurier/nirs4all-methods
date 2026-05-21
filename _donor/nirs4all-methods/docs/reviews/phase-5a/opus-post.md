# Phase 5a — Opus independent post-review transcript

**Date**: 2026-05-18
**Reviewer**: Claude Opus 4.7 (via `feature-dev:code-reviewer` agent)
**Scope**: Phase 5a — baseline correction core (Detrend, AsLS, AirPLS, ArPLS) + Phase 5 opener cleanup (linalg + fixture parser + pybaselines vendoring + Derivate doc fix)
**Verdict**: **REVISE** → 3 high-confidence fixes applied below. Medium / low items tracked in DEFERRALS.

## Summary

Reviewer cross-checked the iterative solvers against the frozen NumPy reference and the pybaselines source (Eilers 2005 / Zhang 2010 / Baek 2015 papers). Algorithm fidelity is good. Three blocking items needed pre-commit:

## High-confidence issues — all FIXED

### #1. Missing algorithm docs and review subdirectory
**Fix**: Created `docs/algorithms/{detrend,asls,airpls,arpls}.md` + `docs/reviews/phase-5a/` (this file).

### #2. AirPLS test tolerance globally loosened
The test was applying `1e-6 / 1e-7` to all 4 AirPLS cases, but only the `high_lam` (lam=1e7) case actually needs the loosened bar.
**Fix**: Per-case dispatch in `cpp/tests/test_preprocessing_baselines.cpp:verify_airpls_parity`: `high_lam` keeps `1e-6 / 1e-7`; all other cases enforce `1e-7 / 1e-8` per the brief contract.

### #3. Frozen pybaselines reference not reproducibly validated
The `n4m_parity_pybaselines_ref/__init__.py` docstring claimed "Validated once against pybaselines==1.1.4" but no committed script proved it.
**Fix**: Added `parity/python_generator/scripts/validate_frozen_pybaselines_ref.py` — imports both `n4m_parity_pybaselines_ref` and the actually-installed `pybaselines` package, runs each baseline operator on a deterministic synthetic spectrum (PCG64 seed 20260518), asserts max-abs-error < 1e-10. Manual attestation: run after any pin bump or frozen-ref change.

## Medium-confidence items — deferred

- **#4** ArPLS extreme-arg clip diverges from `scipy.special.expit` semantics on values < -700 (practical effect: zero). Will revisit if numerical issues surface — current clip is conservative and consistent.
- **#5** `relative_l2_diff` uses naive `sqrt(sum(d²))` instead of NumPy's scaled `dnrm2`. ULP-level difference, no parity impact. Comment refinement deferred.
- **#6** AirPLS `y_l1 == 0` divergence from Python (only exits via `neg_count < 2` rather than `conv_metric < tol`). Practical effect: identical convergence point. Pathological case not exercised. Defer.
- **#7** Smoke tests only check `isfinite`. Strengthen with bounded-value assertions in Phase 5b cleanup.
- **#8** `require_per_case_output_shape=true` unnecessary for baseline ops (shape preserved). Cosmetic.

## Low-priority deferrals (Phase 5b)

- **#9** AirPLS LDLT vs spsolve compounding divergence at lam=1e7 (~1e-7 abs after 50 iter). Not a bug — structural difference between banded LDLT and SuperLU pivot strategies. Document with a residual-check in debug builds.
- **#10** Iterative ops allocate L/D buffers inside the iteration loop (max_iter mallocs/frees). Lift to row-scope: `n4m_banded5_factor_into(...)`. ~5–10× call-overhead win on max_iter=50.
- **#11** Duplicated `build_penalty_diagonals` / `relative_l2_diff` across asls.c / airpls.c / arpls.c. Extract to `core/common/banded_solver.{c,h}` as `n4m_second_diff_penalty_pent5` + a shared utils header.

## Things done well

- **Linalg extraction is clean**: `n4m_householder_qr`, `n4m_apply_qt`, `n4m_back_solve_R` with well-documented contracts. 3 callers (EMSC + SG main + SG edge) use them consistently. Phases 0-4 byte-exact preserved.
- **`fixture_parser.hpp` API design is clean**: zero-deps, structured `Fixture` / `Case` types, templated `params_get_*` helpers. 1087 LOC eliminated from test files.
- **Banded LDLT is mathematically sound**: standard Cholesky-style recurrence, `(n, 2)` L storage, zero-pivot returns `N4M_ERR_NUMERICAL_FAILURE`. Validated independently on random pentadiagonal systems (residual ~9e-16).
- **Iteration loops faithfully mirror Python ref**: iteration counts, convergence-check position, exit_early semantics, exp clip bounds all match.
- **AsLS / AirPLS / ArPLS implementations are scientifically correct**: papers cited, weight update formulae correct, convergence semantics consistent with pybaselines.
- **§9 Derivate banner correctly updated** to honestly state input-shape memoization (addresses prior Codex M2).
- **§11 banner is honest and informative**: clear `out = X - baseline` output convention.
- **ABI: 106 symbols, version-script clean, 0 leak**.
- **CMake integration**: linalg.c + banded_solver.c + 4 baselines added to `n4m_core`. ABI version bumped to 1.5.0.

## Verification (post-fix)

```
cmake --build --preset dev-debug                           → 51/51 targets clean
./build/dev-debug/cpp/cli/n4m_cli --selfcheck → OK
./build/dev-debug/cpp/tests/n4m_tests         → 69/69 passing (61 prior + 8 new)
nm -D --defined-only build/dev-debug/cpp/src/libn4m.so.1.5.0 | awk '$2=="T" {print $3}' | wc -l → 106
wc -l cpp/include/n4m/n4m.h                   → 913
```
