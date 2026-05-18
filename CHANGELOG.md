# Changelog

All notable changes to chemometrics4all will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to dual semver: **project version** in `C4A_PROJECT_VERSION_*` and **ABI version** in `C4A_ABI_VERSION_*` (both in `cpp/include/chemometrics4all/c4a_version.h`).

## [Unreleased]

## [0.1.0+abi.1.1.0] — Phase 1 PCG64 RNG

### Added

- **PCG64 random number generator** with bit-exact NumPy parity:
  - `cpp/src/core/common/rng_pcg64.{h,c}` — pure-C PCG-XSL-RR 128/64 implementation with native `__uint128_t` (gcc/clang) and MSVC two-uint64 fallback. Full NumPy `SeedSequence(seed)` expansion via SplitMix64-style 4-stage pool mixer.
  - `cpp/src/core/common/ziggurat.c` + `ziggurat_constants.h` — vendored NumPy 1.26.4 Ziggurat tables (BSD-3-Clause, license-compatible with CeCILL-2.1) with 256-region Marsaglia–Tsang algorithm for `standard_normal_fill`.
  - `cpp/src/c_api/c_api_rng.cpp` — 7 C ABI wrappers (`c4a_rng_pcg64_{create, destroy, set_seed, uint64, uint64_fill, standard_normal_fill, advance}`) behind opaque handle `c4a_rng_pcg64_state_t`. Try/catch wrapped, null-pointer-safe, `destroy(NULL)` no-op.
- **Parity fixture** `parity/fixtures/_rng_pcg64_stream_v1.json` (848 kB) — 5 seeds × (4096 uint64 + 4096 standard_normal) generated with `numpy==1.26.4` (pinned per CONTRIBUTING.md). Doubles encoded as big-endian hex to avoid JSON float-precision loss.
- **13 new tests** in `cpp/tests/test_rng_pcg64.cpp`: 3 functional (null-safety, set_seed-resets-stream, advance-matches-repeats) + 5 uint64 byte-exact parity (one per seed) + 5 standard_normal IEEE-754 bit-exact parity (one per seed). Hand-rolled JSON parser keeps the zero-dep policy.
- **Documentation** `docs/algorithms/rng_pcg64.md` — algorithm, API, vendoring provenance, parity policy.

### Fixed

- `cmake/CompilerWarnings.cmake` — split C/C++ warning flags. C++-only flags (`-Wnon-virtual-dtor`, `-Wold-style-cast`, `-Woverloaded-virtual`, `-Wzero-as-null-pointer-constant`) now wrapped in `$<COMPILE_LANGUAGE:CXX>` generator expression so they no longer trigger "valid for C++/ObjC++ but not for C" warnings on `.c` TUs.

### ABI

- Project version stays 0.1.0; **ABI bumps to 1.1.0** (additive: 7 new symbols, no breaking changes). Total exported symbols: 29 → **36** (`c4a_rng_pcg64_*` × 7 added).

### Review

- Opus post-review at `docs/reviews/phase-1/opus-post.md`. Verdict: ACCEPT. Two pre-commit nits fixed (fixture provenance, C-vs-C++ warnings). One medium issue fixed (algorithm doc). Five low-priority items tracked for Phase 2+.

### Tests

- 20/20 passing: 7 phase-0 smoke + 13 phase-1 RNG (3 functional + 10 NumPy parity).


## [0.1.0+abi.1.0.0] — Phase 0 bootstrap

### Added

- C++17 core skeleton mirroring `pls4all`'s architecture: `cpp/src/core/{config,context,matrix_view,status,version}.cpp` + `cpp/src/c_api/{c_api_context,c_api_matrix,c_api_version}.cpp`.
- Public C ABI header `cpp/include/chemometrics4all/c4a.h` (Phase 0 surface: status codes, dtype/backend enums, matrix view, context lifecycle, version queries). Full operator surface to be added phase by phase.
- CMake build system inherited from pls4all with 20+ presets in `CMakePresets.json` (dev-debug, dev-release, ci-*, blas-on, sanitizer-*). Zero mandatory dependencies in the reference build.
- Tiny CLI `chemometrics4all_cli` with `--version`, `--abi-info`, `--selfcheck`.
- Hand-rolled zero-dependency test harness in `cpp/tests/main.cpp` + `cpp/tests/harness.hpp`, with 7 Phase 0 smoke tests (all passing).
- GitHub Actions workflows inherited from pls4all (ci, parity-gate, docs, abi-check, coverage, sanitizers, version-sync) — to be re-enabled after Phase 1 once the new C ABI surface is wired.
- `ROADMAP.md` with the 27-phase plan, `ARCHITECTURE.md` with the load-bearing decisions, `CONTRIBUTING.md` with the Codex+Opus review workflow.

### Stripped from the pls4all template

- All PLS algorithm code in `cpp/src/core/` (50+ files: `pls_*`, `simpls*`, `opls*`, `aom_*`, `*_selection.cpp`, `model.cpp`, `pipeline.cpp`, `config.{cpp,hpp}` …).
- All PLS-specific C ABI files in `cpp/src/c_api/` (`c_api_{config,model,pipeline,operator_bank,gating_strategy,validation_plan,aom_selection,method_result,component_coefficients}.cpp`).
- All PLS test fixtures and ABI test suites in `cpp/tests/{abi,fixtures}/`.
- `bindings/` entirely (rebuilt in Phases 22-25 for Python / R / MATLAB / JS-WASM).
- `benchmarks/` entirely (rebuilt in Phase 26).
- `parity/python_generator/` and `parity/r_generator/` (rebuilt in Phase 1+ with c4a adapters; `parity/schema/` and `parity/tolerances.md` retained as inherited contract).
- `docs/methods/` (90 PLS algorithm docs), `docs/parity_audit_2026_05/`, `docs/reviews/phase-0/` (old pls4all reviews), `docs/benchmarks/cross_binding*.md`.
- Pls4all-specific top-level docs (`Overview.md`, `Direction_Technique.md`, `DISTRIBUTION.md`, `Backlog.md`) — replaced by `ARCHITECTURE.md`, `ROADMAP.md`.
- 97+ pls4all `roadmap/phase-*.md` files — replaced by c4a-scoped phases.
- `CI workflows except `ci.yml` moved to `.github/workflows.disabled/` (`parity-gate`, `release-python`, `abi-check`, `version-sync`, `coverage`, `sanitizers`, `docs`). Each is re-enabled at the phase that wires its inputs.

### ABI

- Initial ABI version `1.0.0`. `cpp/abi/expected_symbols_{linux,macos,windows}.txt` populated with the 29 actually-exported `c4a_*` symbols from `libc4a.so.1.0.0` (status / backend / dtype / matrix_view / context lifecycle / version queries).

### Opus post-review

- Independent Opus review completed (transcript at `docs/reviews/phase-0/opus-post.md`). 9 high-confidence issues identified and fixed before commit. 5 medium/low-priority items deferred to Phase 1+ with explicit tracking in the roadmap.

[Unreleased]: https://github.com/GBeurier/chemometrics4all/compare/v0.1.0...HEAD
[0.1.0+abi.1.0.0]: https://github.com/GBeurier/chemometrics4all/releases/tag/v0.1.0
