# Codex review · Phase 0 · PR 2 · cmake-skeleton

- **Date**: 2026-05-14
- **Codex CLI**: v0.125.0
- **Model**: gpt-5.5 (reasoning_effort=medium)
- **Subject**: PR 2 — top-level `CMakeLists.txt`, `CMakePresets.json`, `cmake/*.cmake`, `cpp/include/pls4all/p4a_export.h.in`, `cpp/src/CMakeLists.txt`, `cpp/src/{core,c_api}/placeholder.cpp`.
- **Outcome**: 1 critical + 6 important + 4 minor in the first round; all addressed, second round flagged 1 new minor (LTO doc) which was also fixed. Final verdict: **approved for commit** (the only remaining "partial" item — Windows zero-export DLL risk — is acceptable because Windows CI is not enabled until PR 12, by which time PRs 3-7 have added real `p4a_*` symbols).

## Round 1 — first review

### Verdict: not-approved

### Critical
- `CMakePresets.json` declared `"version": 6` while the project requires CMake 3.21. Schema v6 needs CMake 3.25+. → fixed by dropping to schema `"version": 3`.

### Important
- Windows zero-export DLL risk: `pls4all_c` has no `P4A_API` exports until PR 7. *Accepted as-is*: Windows MSVC CI lands in PR 12, by which time PRs 3-7 have added real `p4a_*` symbols.
- Sanitizer presets missing `CC`/`CXX`. → fixed by setting `clang-16` env on each sanitizer preset and adding a Linux condition.
- `ci-windows-mingw-release` missing `CC`/`CXX` env. → fixed (`CC=gcc, CXX=g++`).
- MinGW `.def` export gating missing. → fixed by extending the gate to `(MSVC OR MINGW)`.
- `CMAKE_SOURCE_DIR` usage breaks `add_subdirectory` consumers. → fixed by global rename to `PROJECT_SOURCE_DIR` / `PROJECT_BINARY_DIR`.
- `pls4all_c_static` excluded from install/export deviates from roadmap §3.1. → fixed by promoting `pls4all_core` from `STATIC` to `OBJECT` library — both `pls4all_c` and `pls4all_c_static` consume it via `$<TARGET_OBJECTS:pls4all_core>` so neither install needs the core artefact.
- `ci-coverage` missing test preset. → fixed.

### Minor
- Both `PLS4ALL_BUILD_SHARED=OFF` and `PLS4ALL_BUILD_STATIC=OFF` silently valid. → fixed with explicit `FATAL_ERROR`.
- `PLS4ALL_BUILD_FUZZ` checked `CMAKE_C_COMPILER_ID` but harnesses are C++. → fixed to `CMAKE_CXX_COMPILER_ID`.
- Dead-code stripping flags keyed off `CMAKE_C_COMPILER_ID`. → fixed to `CMAKE_CXX_COMPILER_ID`.
- `install(p4a_export.h OPTIONAL)` masks regressions. → fixed by removing `OPTIONAL`.

## Round 2 — re-verify

Codex confirmed every "important" and "minor" item is now resolved, except the Windows zero-export risk which remains "partial" (accepted as a known-late issue for PR 12). One new minor introduced:

- `PLS4ALL_ENABLE_LTO` option-help said "auto-on in Release" but implementation only enabled IPO when the option was `ON`. → fixed by removing the misleading auto-on phrasing and adding a proper toolchain-feature check with a warning when IPO is unsupported.

## Local validation

- `cmake --preset dev-release && cmake --build --preset dev-release --parallel` → 6 build steps, no warnings.
- `nm -D --defined-only build/dev-release/cpp/src/libp4a.so` → empty (correct: PR 2 has zero `p4a_*` symbols).
- `ldd build/dev-release/cpp/src/libp4a.so` → `statically linked` (no dynamic deps, confirms the zero-mandatory-deps promise).
- `cmake --install build/dev-release --prefix /tmp/p4a-install` installs:
  - `lib/libp4a.so{,.0,.0.1.0}`
  - `lib/libp4a_static.a`
  - `include/pls4all/p4a_export.h`
  - `lib/cmake/pls4all/Pls4allConfig.cmake` + `Pls4allConfigVersion.cmake` + `Pls4allTargets.cmake` + `Pls4allTargets-release.cmake`
