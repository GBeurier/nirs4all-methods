# Phases 1+2 combined — Codex code + architecture review transcript

**Date**: 2026-05-18
**Reviewer**: Codex (via `general-purpose` agent)
**Scope**: Phases 1 (PCG64 RNG) + 2 (7 stateless preprocessings). Per the user's cadence: every two phases an additional Codex review checks code quality + architectural choices.
**Verdict**: ACCEPT with Phase 3 prerequisites → applied below.

## Summary

The reviewer confirmed the two-layer C engine + C ABI veneer is sound, naming is consistent, visibility hygiene is correct, and parity strategy is rigorous. Three architectural decisions to make before Phase 3 (which introduces stateful ops):

## Major architectural recommendations — addressed

### M1. Standardize C ABI naming
**Decision**: Stay with `c4a_<category>_<op>_*`:
- `c4a_pp_*` — preprocessing (Phases 2-10)
- `c4a_split_*` — splitters (Phase 11)
- `c4a_filter_*` — filters (Phases 12-14)
- `c4a_aug_*` — augmentations (Phases 15-18)
- `c4a_util_*` — utilities (Phases 19-21)
- `c4a_rng_*` — RNG (Phase 1)
- `c4a_metric_*` — NIRS / transfer metrics (Phases 19-20)

Locked in `c4a.h` header comment for Phases 3-21.

### M2. Keep distinct opaque types per operator
**Decision**: No collapsing. Each operator gets its own `c4a_pp_<name>_handle_t`. Python/R/MATLAB tier-2 wrappers hand-roll one class per operator regardless. Per-operator boilerplate (~21 lines × 7 ops) is small and acceptable.

### M3. No per-operator `_to_bytes / _from_bytes`
**Decision**: Adopt pls4all's pipeline-level serialization in Phase 11+. Bindings serialize constructor parameters, not engine state. Saves 84+ ABI symbols.

### M4. `c4a_array_t` orphans
**Decision**: **Delete orphan declarations in Phase 3 opener.** Caller-allocates pattern (current Phase 2 default) is the right convention. Re-add `c4a_array_t` only when Phase 9 (CARS/MCUVE) needs variable-shape output.

### M5. LogTransform stateless miscategorisation
**Decision**: Document the per-call recomputation in c4a.h (done). Proper `_fit/_transform` split lands in Phase 3 alongside MSC/EMSC.

### M6. ABI contract for `_fit/_transform`
**Decision**: To be declared in c4a.h §-Phase 3 placeholder before Phase 3 implementation:
- `_fit` before `_transform` → currently no contract; Phase 3 will use `C4A_ERR_NOT_FITTED` (already declared).
- Calling `_fit` twice is allowed (overwrites).
- `_fit` resets fitted state.

## Code quality recommendations — applied

| # | Recommendation | Status |
|---|---|---|
| R1 | Rename test Runner from `phase-0-smoke` to `chemometrics4all` | ✅ Done — `cpp/tests/main.cpp:73` |
| R2 | Add `parity/python_generator/pyproject.toml` with numpy 1.26.4 pin | ✅ Done — pins numpy 1.26.4 + scipy 1.11.4 + scikit-learn 1.4.2 + PyWavelets 1.6.0 + pybaselines 1.1.4 |
| R3 | Trim orphan pls4all declarations in c4a.h (~1900 lines) | ⏳ Deferred to Phase 3 opener (big surgery; needs its own commit) |
| R4 | Delete `c4a_array_t` orphan declarations | ⏳ Deferred to Phase 3 opener (paired with R3) |
| R5 | Add empty `c4a_linux.map` / `chemometrics4all.def` skeletons | ✅ Done — `cpp/src/c_api/c4a_linux.map` with `CHEMOMETRICS4ALL_1 { global: c4a_*; local: *; }`. CMake auto-picks it up. Confirmed: all 57 exported symbols are `c4a_*`. |

## Code quality observations — deferred

- **O1** Fixture compression for Phase 6+ — defer.
- **O2** nirs4all source loading via importlib + absolute path → git submodule in Phase 3.
- **O3** Shared fixture cache for tests — defer to Phase 6.
- **O4** `dev-asan_ubsan` preset — defer to Phase 5.
- **O5** Workflow re-activation schedule documented (parity-gate → end of Phase 2; abi-check → Phase 3; docs → Phase 4; sanitizers/coverage → Phase 5; version-sync + release-python → Phase 22).
- **O6** Phase 2 commit blocked on this review — unblocking now.

## Process recommendations — partially applied

- **P1** Pre-implementation Codex reviews → adopt from Phase 3 onward (every phase brief gets reviewed before code).
- **P2** Aggregate verdicts in `docs/reviews/PHASES.md` index — to add at Phase 3.
- **P3** Track deferrals in `docs/reviews/DEFERRALS.md` — to add at Phase 3.
- **P4** Commit briefs before implementation — adopt from Phase 3.

## Things done well

- Two-layer C engine + C ABI veneer correctly applied across 7 operators.
- Symbol export hygiene: hidden visibility + version script (Phase 2 fix) → 0 leakage.
- NumPy parity rigor (true division vs reciprocal-multiplication documented).
- Big-endian hex JSON encoding for double fixtures.
- Consistent internal contract (`_state_new/_state_free/_apply`) per operator.
- Dual-versioning discipline (project 0.1.0 + ABI 1.2.0).
