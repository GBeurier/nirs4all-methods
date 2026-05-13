# Codex re-review · Phase 0 roadmap · revision 2

- **Date**: 2026-05-14 (same day as rev 1 review)
- **Codex CLI**: v0.125.0
- **Model**: gpt-5.5 (reasoning_effort=medium)
- **Subject of review**: `roadmap/phase-0.md` revision 2 (the post-rev-1-correction draft).
- **Companion reads**: `Direction_Technique.md`, the previous review transcript `0001-roadmap-review.md`.
- **Outcome**: 16 actionable items (10 partial fixes from rev 1 + 6 new issues introduced by rev 2). All corrections accepted; revision 3 incorporates each of them.

## Prompt sent to Codex

```
You previously reviewed revision 1 of pls4all/roadmap/phase-0.md and listed 10 critical, 13 important, 6 minor corrections. Revision 2 (the current file at /home/delete/nirs4all/pls4all/roadmap/phase-0.md) is supposed to address every one of them. Read:

- roadmap/phase-0.md (revision 2 under verification)
- Direction_Technique.md (canonical spec)
- docs/reviews/phase-0/0001-roadmap-review.md (your previous review + the disposition committed)

Verify that revision 2 actually fixes the issues you raised. For each item from your previous review, mark it as one of:

- resolved — the fix is correct.
- partial — addressed but with new problems or gaps (specify them).
- regressed — the original concern is now made worse.
- unaddressed — the issue is not fixed.

Then list any NEW issues introduced by revision 2 (e.g. an enum field accidentally renamed inconsistently, a new function with wrong signature, a contradiction between sections).

Be terse — bullet points. Reference the section numbers from revision 2 (the headings may have shifted by one). Do not write any files. Do not edit anything. Output your full report on stdout.

If revision 2 is clean (everything resolved, no new issues), say so explicitly at the top of the report: a single line "## Verdict: approved for implementation."
```

## Codex response

## Verdict: not approved.

Note: the prior review file contains **10 critical, 15 important, 6 minor** actionable bullets, not 10/13/6. I verified all 31.

**Critical**
- resolved — Phase 1 model ABI surface added: §4.7, §4.10.
- partial — serialization functions added, but §4.8 still lacks an ABI-version field/constant in the serialized format.
- partial — SIMPLS moved to solver, but §4.5 keeps `P4A_ALGO_KERNEL_PLS` and references nonexistent `KERNEL_RBF`/future kernel setters.
- resolved — AOM/POP composable bank/gating surface added: §4.5, §4.6.
- resolved — preprocessing pipeline fit/transform ABI added: §4.3, §4.5, §4.6.
- resolved — blanket `noexcept`/catch rule added: §4.0, §7.1.
- resolved — error buffer pinned to 4096 bytes: §4.11.
- resolved — ILP32/`armeabi-v7a` excluded and documented: §3.5, §10.1.
- partial — CI matrix aligned in §1/§5.1/§5.8, but §3.5 preset names do not match the 10+universal2 labels.
- resolved — Python and R both generate all 3 fixtures: §1, §5.5, §6.3.

**Important**
- resolved — strided matrix-view init/tests added; negative strides rejected: §4.2, §7.1.
- resolved — Linux/macOS undefined-symbol link checks added: §3.3.
- partial — dead-code stripping added, but §3.3 applies link flags unconditionally and uses `CMAKE_BUILD_TYPE`; multi-config/debug behavior is wrong, and C API sources do not get `-ffunction-sections -fdata-sections`.
- resolved — universal2 CI cell added: §5.1.
- partial — dependency audit added, but §5.3 omits `embind`, `nlohmann-json`, `yaml-cpp`; §1 and §5.3 forbidden lists disagree.
- partial — `parity-gate.yml` is required, but §5.8 says "after PR 15" while §9 creates it in PR 17.
- resolved — fuzzing targets added; serialization fuzzer explicitly deferred to Phase 1: §7.3.
- resolved — TSAN + multi-context stress + same-context policy added: §5.2, §7.2, §8.
- partial — fork test added, but async-signal policy contradicts itself: §7.2 says `last_error` is safe-ish; §7.5 says nothing in `p4a_*` is safe.
- partial — NaN/Inf policy still not explicit for algorithm inputs; §7.5 defers data behavior to Phase 1 and says "NaN/Inf inputs to matrix-view setters", which is not meaningful for integer shape/stride setters.
- partial — §6.4 paragraph is correct, but table rows omit intercept/means/scales and one cross-algorithm row still mentions W/P/Q/R sign-invariance.
- resolved — tolerances relaxed and AOM marked TBD: §6.4.
- resolved — public `_COUNT` sentinels removed/documented: §4.2, §7.5.
- resolved — exact patch pins / lockfiles added: §6.2, §6.3.
- resolved — double-destroy marked UB; only `destroy(NULL)` tested: §7.1.

**Minor**
- resolved — `PLS4ALL_BUILD_SHARED` now guards shared target/install: §3.1, §3.3.
- resolved — CLI install guarded: §3.1.
- resolved — `OPLS` names fixed: §4.5.
- resolved — `P4A_SOLVER_KERNEL_ALGORITHM` rename done: §4.5.
- resolved — operations notes moved out of technical risks: §10, §11.
- partial — doctest has tag + SHA-256, but §11 still does not name the source URL explicitly.

**New Issues Introduced**
- §3.1/§3.3: generated `p4a_export.h` is used from `${CMAKE_BINARY_DIR}/generated`, but install only copies `cpp/include/pls4all`; installed headers will miss `p4a_export.h`.
- §3.3: `pls4all_c_static` lacks the generated include directory, unlike `pls4all_c`; static build can fail to include generated `p4a_export.h`.
- §4.5: "Getters mirror setters identically; signatures elided" leaves public ABI symbols unspecified despite §4 saying Phase 0 declares the surface.
- §4.5: "every PLS knob ... declared NOW" conflicts with "kernel ML solvers ... declared by ... setters in Phase 4".
- §4.2/§7.5/§8: `_COUNT` is removed, but docs still reference `_COUNT` stability notes; confusing and stale.
- §5.1 vs §3.5: `cmake --preset ci-<label>` cannot work as written unless presets are named after the table labels.

## Disposition

All 16 items addressed in revision 3. Per the project's review rule, no items were rejected.

| # | Concern | Resolution in rev 3 |
|---|---------|---------------------|
| 1 | §4.8 missing ABI version in serialized format | Added `p4a_serialization_inspect` function + ABI-version-in-buffer commentary. |
| 2 | §4.5 `P4A_ALGO_KERNEL_PLS` references nonexistent future setters | Removed `KERNEL_PLS` from `p4a_algorithm_t`; deferred kernel-type surface to Phase 4 explicitly. |
| 3 | §3.5 preset names don't match §5.1 labels | Renamed all CI presets to match table labels exactly (`ci-linux-gcc12-release`, etc.). |
| 4 | §3.3 `CMAKE_BUILD_TYPE` breaks multi-config | Switched to generator expressions (`$<$<NOT:$<CONFIG:Debug>>:…>`) for both compile and link options. |
| 5 | §3.3 C API sources missed `-ffunction-sections -fdata-sections` | Extracted `pls4all_apply_c_target` helper that applies the flags to both shared and static C API targets. |
| 6 | §5.3 forbidden list incomplete | Added `embind`, `nlohmann-json`, `yaml-cpp` + cross-referenced to §1 acceptance criterion. |
| 7 | §5.8 vs §9 parity-gate PR number disagreement | Aligned to PR 17 in both places. |
| 8 | §7.2 vs §7.5 async-signal contradiction | Unified policy: no `p4a_*` is async-signal-safe, full stop. |
| 9 | §7.5 NaN/Inf claim about "matrix-view setters" misleading | Reworded: NaN/Inf rejected by scalar `set_tol`; matrix-view inits take only integers; data NaN/Inf handled per-algorithm in Phase 1. |
| 10 | §6.4 cross-algo row still mentions sign-invariance | Updated comparison-set column to list explicit fields (`predictions, coefficients, intercept, x/y means and scales`); removed all sign-invariance mentions for cross-algorithm rows. |
| 11 | §11 doctest missing source URL | Added <https://github.com/doctest/doctest> and <https://github.com/nlohmann/json>. |
| 12 | Generated `p4a_export.h` not installed | Added explicit `install(FILES ${CMAKE_BINARY_DIR}/generated/pls4all/p4a_export.h DESTINATION include/pls4all)`. |
| 13 | `pls4all_c_static` missing generated include | Both targets now go through the shared `pls4all_apply_c_target()` helper that sets include dirs identically. |
| 14 | §4.5 "getters elided" | Fully listed every getter signature paired with each setter. |
| 15 | §4.5 "every knob declared NOW" vs "kernel solvers in Phase 4" | Removed the contradicting comment; the only forward-deferred piece is the **nonlinear** kernel-type enum (not a knob, a new enum). |
| 16 | §4.2/§7.5/§8 stale `_COUNT` mentions | Removed the leftover "P4A_BACKEND__COUNT is not part of public ABI" comment; clarified in §7.5 that `_COUNT` lives only inside internal C++ via `static_assert`. |

Next step: revision 3 is committed alongside both review transcripts. Phase 0 implementation begins immediately. Per-PR Codex reviews catch any further drift.
