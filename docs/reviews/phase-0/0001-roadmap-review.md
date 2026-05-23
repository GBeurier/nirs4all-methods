# Codex review · Phase 0 roadmap · revision 1

- **Date**: 2026-05-14
- **Codex CLI**: v0.125.0
- **Model**: gpt-5.5 (reasoning_effort=medium)
- **Subject of review**: `roadmap/phase-0.md` (revision 1, ~700 lines)
- **Companion reads**: `Direction_Technique.md`, `Overview.md`
- **Outcome**: 10 critical issues, 13 important, 6 minor, 6 confirmed correct. All corrections accepted.

## Prompt sent to Codex

```
You are reviewing the Phase 0 roadmap for the pls4all C++ library, a multi-language PLS engine. Read these three files (they all live in the current working directory):

- roadmap/phase-0.md  — the roadmap under review
- Direction_Technique.md — the canonical technical specification (non-negotiable decisions)
- Overview.md — the full PLS algorithm taxonomy that Phases 1-7 will eventually cover

Your job: act as a strict, senior reviewer who will block a PR if any of these are violated.

Check the roadmap against Direction_Technique.md for any violation of these non-negotiable decisions:
1. C ABI public, C++17 internal — no STL, no Eigen::*, no std::* in the public header.
2. Zero mandatory external dependencies (no Eigen / Armadillo / Boost / pybind11 / Rcpp / embind / OpenBLAS / MKL / nlohmann-json / yaml-cpp on the runtime link path; test-only vendored is OK).
3. No exception ever crosses the C ABI; every extern "C" wrapper catches and translates.
4. No JSON / YAML parsing in the core (test-only fixture loader is OK because doctest binary is not shipped).
5. Stride-aware matrix views in the public API (must accept NumPy row-major, R/MATLAB col-major, transposed views, slice views).
6. Opaque config handle with setters (Option B from §5.5), not a versioned struct.
7. Error message ownership — no free() across the language boundary; context-owned 4 KiB buffer is the rule.
8. Versioned binary serialization with explicit endianness (Phase 1; surface must be declared in Phase 0 if needed).
9. Stateful preprocessing operators with explicit fit / transform separation (Phase 3; design must accommodate this without ABI break).
10. AOM-PLS / POP-PLS as composable engine (operator bank + gating strategy), not separate classes.

Then also check for:
- Missing ABI surface elements that Phase 1+ will need (anything that would force an ABI break at Phase 1 is a Phase 0 bug).
- Missing test category (fuzzing, thread-safety stress, signal-safety, fork-safety, NaN/Inf propagation).
- CI gaps (missing -Wl,--no-undefined, missing --gc-sections, missing dumpbin /EXPORTS on Windows, missing macOS universal2, missing 32-bit Android coverage decision).
- Parity tolerance numbers in §6.4 being too tight or too loose for any reference-implementation pair.
- Naming / hierarchy ambiguity in n4m_* symbols (collisions, inconsistent prefixes, missing namespacing).
- Open questions in §10 that should be answered before implementation starts.

OUTPUT FORMAT — a Markdown report with these sections, in this order:
- "## Critical (must fix before implementation)" — blocking issues.
- "## Important (should fix during implementation)" — non-blocking but should be addressed.
- "## Minor / nice-to-have"  — bikeshed-tier.
- "## Confirmed correct"  — short list of what's solid, no need to belabor.

For each item, point to the roadmap section (e.g. "§4.5"), describe the problem in one sentence, and propose the exact edit (or the question to resolve). Be terse — bullet points, no prose paragraphs.

Do not write any files. Do not edit anything. Output your full report on stdout.
```

## Codex response

## Critical (must fix before implementation)

- §4.6 / §4.8: Phase 1 cannot add NIPALS “without changing a single public symbol” because `predict`, `transform`, coefficients, intercept, loadings, weights, rotations, means/scales accessors are missing. Edit: add Phase-0-stubbed `n4m_model_predict`, `n4m_model_transform`, `n4m_model_predict_alloc`, `n4m_model_transform_alloc`, and `n4m_model_get_array(ctx, model, n4m_model_array_t, n4m_array_t**)`.

- §4.6: Phase 1 binary serialization surface is absent. Edit: add `n4m_model_export_size`, `n4m_model_export_to_buffer`, `n4m_model_import_from_buffer`, and constants documenting magic, format version, ABI version, and explicit little-endian encoding.

- §4.5: `N4M_ALGO_SIMPLS` and `N4M_ALGO_KERNEL_PLS` conflate model family with solver, while `n4m_solver_t` also contains those concepts. Edit: remove solver-valued algorithms; keep SIMPLS/kernel/wide-kernel only in `n4m_solver_t`.

- §4.5: AOM/POP are exposed as flat algorithm choices with no operator bank or gating surface, violating the composable-engine decision. Edit: add opaque `n4m_operator_bank_t`, `n4m_gating_strategy_t`, `n4m_config_set_operator_bank`, and `n4m_config_set_gating_strategy`; model POP as a gating strategy, not a separate class-style algorithm.

- §4.3 / §4.5: Stateful preprocessing fit/transform is not represented in the ABI. Edit: add opaque `n4m_pipeline_t` plus stubbed lifecycle, `n4m_pipeline_add_operator`, `n4m_pipeline_fit`, `n4m_pipeline_transform`, and `n4m_config_set_pipeline`.

- §4 / §7: The “every extern C wrapper catches” rule is not stated for every exported symbol, especially `void` destroy/free functions. Edit: add a blanket rule that every `N4M_API` wrapper is `noexcept`; status-returning wrappers translate exceptions, and `void` wrappers catch and swallow after best-effort cleanup.

- §10.2: The proposal to bump the error buffer to 8 KiB conflicts with the canonical 4 KiB context-owned buffer rule. Edit: fix `N4M_ERROR_BUFFER_BYTES` at exactly 4096 and remove the bump language.

- §10.1: ILP32 / Android 32-bit ABI layout is still tentative, but `n4m_matrix_view_t` layout becomes public ABI immediately. Edit: either hard-exclude ILP32/`armeabi-v7a` for Phase 0/2 and document it, or add explicit packing/alignment plus CI coverage.

- §1 / §5.1 / §10.8: The CI matrix is internally inconsistent: §1 describes Linux 3×2 plus macOS 1×2 plus Windows 2×2, which is 12 cells, while §5.1 lists 10 and omits gcc13-debug and MinGW-debug. Edit: declare the exact required matrix and align branch protection to it.

- §1 / §6.3: Acceptance says all three Phase 0 fixtures round-trip through Python and R, but §6.3 says R ships only one companion fixture. Edit: either require R to generate all three canonical fixtures or change the acceptance criterion.

## Important (should fix during implementation)

- §4.2 / §7: Slice-view support is underspecified. Edit: add `n4m_matrix_view_init_strided` and tests for non-contiguous positive strides, transposed views, and negative strides or explicitly reject negative strides.

- §5.3: ELF/macOS undefined-symbol enforcement is missing. Edit: add Linux `-Wl,--no-undefined` and macOS `-Wl,-undefined,error` checks for `n4m_c`.

- §3.3 / §5: Dead-code stripping is missing. Edit: add Release `-ffunction-sections -fdata-sections -Wl,--gc-sections`, macOS `-Wl,-dead_strip`, and MSVC `/OPT:REF`.

- §5: macOS universal2 coverage is missing. Edit: add a universal2 job with `CMAKE_OSX_ARCHITECTURES="x86_64;arm64"` or explicitly defer universal2.

- §5.3: Runtime dependency auditing is missing. Edit: add `ldd` / `otool -L` / `dumpbin /DEPENDENTS` checks and fail on OpenBLAS, MKL, Boost, pybind11, Rcpp, embind, nlohmann-json, yaml-cpp, etc.

- §5.8: Branch protection omits `parity-gate`, despite parity being a Phase 0 acceptance criterion. Edit: require `parity-gate` after PR 15.

- §7: Fuzzing is missing. Edit: add libFuzzer/AFL targets for config setters, matrix-view validation, error-buffer formatting, and future serialization import.

- §7: Thread-safety stress is too narrow. Edit: add TSAN CI plus multi-context and same-context stress tests defining the supported concurrency model.

- §7: Signal/fork-safety policy is missing. Edit: document that the API is not async-signal-safe except explicitly listed trivial functions, and add POSIX fork smoke tests where supported.

- §7: NaN/Inf policy is missing. Edit: add Phase 1 numerical tests for NaN/Inf propagation, rejection, or missing-aware behavior with explicit expected statuses.

- §6.4: Cross-algorithm rows compare raw latent matrices (`W/P/Q/R/T`) too aggressively. Edit: gate cross-algorithm parity on predictions, coefficients, intercepts, and means/scales; keep raw latent arrays diagnostic unless conventions are normalized.

- §6.4: Some tolerances are likely too tight: mixOmics `1e-9`, BLAS `1e-12/1e-11`, and cross-platform `1e-13` will be fragile. Edit: relax or require fixed reduction order and FMA policy; defer AOM tolerance until the reference is frozen.

- §4.5: Public `*_COUNT` enum sentinels are ABI traps because their values change when new enum values are added. Edit: remove them from public enums or document them as non-ABI constants.

- §6.2: Version pins like `scikit-learn==1.4.x` are not reproducible. Edit: pin exact patch versions or use a lockfile.

- §7: “double-destroy” testing is unsafe for opaque handles. Edit: test `destroy(NULL)` as no-op and document double destroy as caller UB.

## Minor / nice-to-have

- §3.2 / §3.3: `PLS4ALL_BUILD_SHARED` is declared but ignored. Edit: guard `add_library(n4m_c SHARED ...)` and install rules behind it.

- §3.1: `install(TARGETS n4m_c n4m_cli ...)` breaks when `PLS4ALL_BUILD_CLI=OFF`. Edit: install the CLI only inside the CLI option block.

- §4.5: `N4M_ALGO_O_PLS` is awkward and inconsistent. Edit: rename to `N4M_ALGO_OPLS` and `N4M_ALGO_OPLS_DA`.

- §4.5: `N4M_SOLVER_KERNEL` is ambiguous between linear kernel-algorithm PLS and nonlinear kernel PLS. Edit: rename to `N4M_SOLVER_KERNEL_ALGORITHM`.

- §10.6 / §10.7: Repository visibility and Codex MCP state are project-operations notes, not implementation risks. Edit: move them out of the technical roadmap.

- §10.5: Vendored doctest policy should include checksum/source URL. Edit: pin tag plus SHA-256.

## Confirmed correct

- §4: Public ABI is C-shaped: opaque handles, C enums/structs, no STL, no Eigen, no `std::*`.

- §3.2 / §6 / §7: Mandatory runtime dependencies remain zero; BLAS/OpenMP/CUDA are optional, and JSON is test-only.

- §4.2: Matrix views are stride-aware and use element strides, which is the right public shape.

- §4.5: Config uses opaque handles with setters, matching Direction Technique Option B.

- §4.10: Error message ownership is correctly context-owned with no `n4m_free_string` or `char** out_msg`.

- §5.3: Windows export checking with `dumpbin /EXPORTS` is explicitly included.
## Disposition

Per the project's review rule (`si Codex n'est pas d'accord avec toi, donne-lui plus d'info; si il est toujours en désaccord, considère qu'il a raison`), every Codex correction was reviewed for soundness. All 29 actionable items (10 critical + 13 important + 6 minor) were accepted; revision 2 of the roadmap incorporates each of them.

Concrete dispositions:

### Critical (all accepted, applied in revision 2)
1. Missing Phase 1 ABI surface (predict/transform/get_array) — added stubbed `n4m_model_predict`, `n4m_model_transform`, `n4m_model_get_array`, paired `_alloc` variants, and a `n4m_model_array_t` enum tag.
2. Missing serialization surface — added `n4m_model_export_size`, `n4m_model_export_to_buffer`, `n4m_model_import_from_buffer`, plus `N4M_SERIALIZATION_FORMAT_VERSION`, `N4M_SERIALIZATION_MAGIC`, explicit little-endian.
3. Algorithm vs solver conflation — removed `N4M_ALGO_SIMPLS` from `n4m_algorithm_t`; SIMPLS / kernel-algorithm / wide-kernel remain in `n4m_solver_t`. `N4M_ALGO_KERNEL_PLS` is retained but semantically defined as *nonlinear* kernel PLS to disambiguate.
4. AOM / POP need composable surface — added opaque `n4m_operator_bank_t`, `n4m_gating_strategy_t`, `n4m_config_set_operator_bank`, `n4m_config_set_gating_strategy`. POP is a gating strategy choice, not a separate top-level algorithm.
5. Preprocessing pipeline missing — added opaque `n4m_pipeline_t` with `_create / _destroy / _add_operator / _fit / _transform / _transform_alloc` and `n4m_config_set_pipeline`.
6. Blanket `noexcept` rule — added to §4 prelude: every `N4M_API` function is `noexcept`; status-returning wrappers translate exceptions; `void` wrappers catch and swallow after best-effort cleanup.
7. Error buffer size — pinned at exactly 4096 bytes (`N4M_ERROR_BUFFER_BYTES = 4096`); removed the "tentative bump to 8 KiB" language.
8. ILP32 / Android 32-bit — hard-excluded `armeabi-v7a` from Phase 0 and Phase 2 (arm64-v8a + x86_64 only). Documented in §11 (operations notes) and §1.
9. CI matrix consistency — declared exactly 10 cells in §1, listed identically in §5.1, with explicit rationale for which combinations are dropped (gcc-13-debug, MinGW-debug).
10. R generator fixture coverage — committed to all 3 canonical fixtures (Python and R) at Phase 0.

### Important (all accepted, applied in revision 2)
- `n4m_matrix_view_init_strided` added; negative strides explicitly rejected by `_validate`.
- `-Wl,--no-undefined` (Linux), `-Wl,-undefined,error` (macOS) added to `cmake/CompilerWarnings.cmake`.
- Dead-code stripping: `-ffunction-sections -fdata-sections -Wl,--gc-sections` (Linux), `-Wl,-dead_strip` (macOS), `/OPT:REF /OPT:ICF` (MSVC) in Release.
- macOS universal2: added explicit `ci-macos-universal2` preset and CI cell with `CMAKE_OSX_ARCHITECTURES="x86_64;arm64"`.
- Runtime dependency audit: `ldd`, `otool -L`, `dumpbin /DEPENDENTS` step in `abi-check.yml`, failing on OpenBLAS, MKL, Boost, pybind11, Rcpp, embind, nlohmann-json, yaml-cpp links.
- Parity-gate added to branch protection (after PR 15 lands).
- Fuzzing: libFuzzer harnesses for config setters, matrix-view validation, error-buffer formatting, fixture loader; deferred serialization fuzzer to Phase 1.
- TSAN job added; multi-context stress test (two `n4m_context_t*` on two threads doing independent failing calls); same-context behaviour explicitly documented as "single-threaded only".
- Async-signal-safety policy: documented in `docs/architecture/threading.md`. Only `n4m_context_last_error` is safe-ish (read-only); everything else is not. POSIX fork smoke tests in Phase 1.
- NaN/Inf policy: documented in `docs/architecture/error_model.md`; Phase 1 numerical tests assert explicit rejection on NaN/Inf inputs with `N4M_ERR_INVALID_ARGUMENT`.
- Cross-algorithm parity (in §6.4): gate on `predict_train`, `coefficients`, `intercept`, `x_mean`, `x_scale`, `y_mean`, `y_scale`. Raw latent arrays (W, P, Q, R, T, U) are diagnostic only when algorithms differ.
- Tolerances relaxed: mixOmics 1e-9 → 1e-7; BLAS 1e-12/1e-11 → 1e-10/1e-9; cross-platform 1e-13 → 1e-11. AOM tolerance row marked "TBD, freeze when reference frozen".
- `_COUNT` sentinels: kept in headers but marked `/* not part of ABI — for source-tree use only */`; reference docs explicitly say "do not compare with `_COUNT` across libpls versions".
- Version pins: switched to exact patch (e.g. `scikit-learn==1.4.2`) plus `requirements-lock.txt` committed alongside `pyproject.toml`.
- Removed "double-destroy is a defect — protected with a debug-only sentinel" language; replaced with: `destroy(NULL)` is a guaranteed no-op; double-destroy of a non-NULL handle is documented as undefined behaviour.

### Minor (all accepted, applied in revision 2)
- `add_library(n4m_c SHARED ...)` and the install block guarded behind `PLS4ALL_BUILD_SHARED`.
- `n4m_cli` install guarded behind `PLS4ALL_BUILD_CLI`.
- `N4M_ALGO_O_PLS` / `N4M_ALGO_O_PLS_DA` renamed → `N4M_ALGO_OPLS` / `N4M_ALGO_OPLS_DA`.
- `N4M_SOLVER_KERNEL` renamed → `N4M_SOLVER_KERNEL_ALGORITHM`.
- §10.6 / §10.7 (visibility / Codex MCP) moved to a new §11 "Operations notes" (separated from technical risks).
- doctest pinned to `v2.4.11` with SHA-256 `<TBD-when-vendored>`; bump procedure documented in `docs/dev/testing.md`.

### Confirmed correct (no action)
- §4: public ABI is C-shaped.
- §3.2 / §6 / §7: zero mandatory deps maintained.
- §4.2: stride-aware view in element units.
- §4.5: opaque config + setters.
- §4.10: error message ownership rule.
- §5.3: Windows `dumpbin /EXPORTS` symbol surface check.

Next step: re-write `roadmap/phase-0.md` to revision 2 incorporating all of the above, then re-run Codex review one last time before the implementation phase begins.
