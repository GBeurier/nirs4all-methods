# Changelog

All notable changes to `pls4all` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Next phase: broaden the algorithm family beyond NIPALS PLS regression, with
reference parity and benchmarks for every added method.

## [0.3.0-simpls] — 2026-05-14

SIMPLS core increment.

### Added

- Dependency-free SIMPLS regression solver for PLS1 / PLS2 behind
  `P4A_SOLVER_SIMPLS`.
- Deterministic NumPy SIMPLS parity fixtures plus C++ parity coverage for
  predictions, coefficients, preprocessing statistics, latent arrays,
  transform scores and serialization round-trips.
- CLI selfcheck smoke for the SIMPLS fit / predict path.

### Changed

- Project version is now `0.3.0+abi.1.0.0`; the C ABI remains unchanged.
- The model import path now accepts serialized NIPALS and SIMPLS models.

## [0.2.0-phase1] — 2026-05-14

Phase 1 — PLS CPU reference. Roadmap at [`roadmap/phase-1.md`](roadmap/phase-1.md).

### Added

- Dependency-free reference CPU NIPALS implementation for PLS regression
  (PLS1 / PLS2) behind the existing Phase 0 C ABI.
- Live `p4a_model_predict`, `p4a_model_transform`, allocated output arrays,
  fitted-model array accessors and model dimension getters.
- Binary `P4AM` serialization v1 with little-endian fields and checksum
  validation for export/import round-trips.
- Phase 1 C++ parity tests generated from the three checked-in synthetic
  scikit-learn fixtures.

### Changed

- Unsupported model modes now return `P4A_ERR_UNSUPPORTED`; fitted-model
  functions no longer return `P4A_ERR_NOT_IMPLEMENTED` for the supported
  NIPALS regression path.

## [0.1.0-phase0] — 2026-05-14

First milestone: the public C ABI is feature-complete, exercised by a 60-test suite, and reachable from a Python binding. No PLS algorithm is implemented yet — every function that needs a fitted model returns `P4A_ERR_NOT_IMPLEMENTED`. Phase 1 plugs in NIPALS without changing a single public symbol.

### Added

- Public C ABI header (`cpp/include/pls4all/p4a.h`, `p4a_version.h`, generated `p4a_export.h`) declaring **96 `p4a_*` symbols** across status / dtype / backend / matrix-view / context / config / operator-bank / gating-strategy / pipeline / model / serialization / array.
- Static asserts pinning every public enum to 4 bytes and `p4a_matrix_view_t` to 48 bytes (LP64 / LLP64).
- CMake project (CMake 3.21+) with hidden default visibility, `-Wl,--no-undefined`, dead-code stripping, undefined-symbol enforcement on every supported toolchain.
- `CMakePresets.json` with 11 CI matrix presets matching the GitHub Actions labels 1:1, plus sanitizer (ASAN / UBSAN / TSAN / ASAN+UBSAN), coverage, and cross-compile presets (Emscripten, Android arm64 / x86_64).
- Reference CPU implementations of status / version / context / config / matrix-view / operator-bank / gating-strategy / pipeline lifecycles, with try/catch boundary protection on every `P4A_API` wrapper and a per-context 4096-byte thread-safe error buffer.
- `pls4all_cli` introspection tool: `--version`, `--abi-info`, `--selfcheck`.
- Hand-rolled zero-dependency test harness (`cpp/tests/{harness.hpp, main.cpp}`) plus 8 ABI test files: lifecycle, config setters, status codes, version, matrix view, error messages, OOM paths, model stubs. **60 / 60 tests pass.**
- GitHub Actions workflows: `ci.yml` (11-cell build matrix), `abi-check.yml` (symbol surface + runtime-dep audit on Linux / macOS / Windows), `sanitizers.yml` (ASAN / UBSAN / TSAN / combined), `coverage.yml` (informational), `parity-gate.yml` (fixture-determinism + Python binding smoke).
- Parity infrastructure: `parity/schema/fixture_schema_v1.{json,md}`, `parity/tolerances.md` (11 pair-wise rows), 3 deterministic synthetic fixtures with SHA-256 manifest, Python generator pinned via `requirements-lock.txt` exposing `generate-fixtures --regenerate` / `--check`.
- Python binding (`bindings/python/`) — ctypes loader, Pythonic `Context`, `Config`, `Pls4allError`, `Backend`, `Dtype`, `Status`.
- Skeleton READMEs for the R, MATLAB, JavaScript/WASM, and Android bindings (full implementations land in Phase 2).
- Documentation skeleton under `docs/` with stubs for architecture, ABI reference, binding guides, parity methodology, and the development workflow.
- Codex review transcripts under `docs/reviews/phase-0/` for the roadmap (rev 1 → 2 → 3) and PRs 2-4 (`docs/reviews/phase-0/000{1,2,3,4,5}-*.md`).

### Known limitations

- Every function that requires a fitted model returns `P4A_ERR_NOT_IMPLEMENTED`. NIPALS PLS1 / PLS2 land in Phase 1 (PR-1, immediately after this tag).
- The R-side parity generator is stubbed — only the Python side produces fixtures at Phase 0.
- Bindings beyond Python ship only README placeholders. Full bindings land in Phase 2 once Phase 1 / 3 / 4 supply real algorithms.
- Concurrency / fuzz tests are deferred to a follow-up PR.
