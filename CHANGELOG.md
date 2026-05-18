# Changelog

All notable changes to chemometrics4all will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to dual semver: **project version** in `C4A_PROJECT_VERSION_*` and **ABI version** in `C4A_ABI_VERSION_*` (both in `cpp/include/chemometrics4all/c4a_version.h`).

## [Unreleased]

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
