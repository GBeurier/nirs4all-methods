# Changelog

All notable changes to `pls4all` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Next phase: broaden the algorithm family beyond NIPALS PLS regression, with
reference parity and benchmarks for every added method.

## [0.69.0-pls-extensions] — 2026-05-15

Phases 8 – 15. Smallest viable additions to algorithm taxonomy sections §7,
§11, §12 (partial), §13, §17, §18, §19. All shipped as internal C++ kernels;
public ABI exposure deferred to the binding tranche.

### Added

- **§7 Sparse PLS** — `fit_pls_sparse_simpls` (soft-threshold on the SIMPLS
  loading direction; `cfg.sparsity_lambda`). `P4A_ALGO_SPARSE_PLS` is now
  dispatched through `fit_model`.
- **§13 Domain-invariant PLS** — `fit_di_pls` projects the SIMPLS direction
  away from the source-target mean difference at each component;
  `cfg.di_lambda` controls the penalty.
- **§12 Recursive PLS (moving window)** — `fit_predict_recursive_pls` in
  `cpp/src/core/recursive_pls.cpp`. Each sample at position ≥ `window_size`
  is predicted from a SIMPLS model fitted on the previous `window_size` rows.
- **§17 Diagnostics** — `pls_hotelling_t2`, `pls_q_residuals`, `pls_dmodx`
  in `cpp/src/core/pls_diagnostics.cpp`.
- **§18 One-SE rule** — `cross_validate_component_prefixes` now records
  per-fold RMSE values; new `select_one_se_components` picks the smallest
  competitive component count.
- **§19 Process monitoring** — `pls_monitoring_fit` /
  `pls_monitoring_evaluate` in `cpp/src/core/pls_monitoring.cpp`. Empirical
  percentile thresholds for T² and Q-residuals at a configurable
  α level, plus per-sample alarm flags.
- **§11 Just-in-time PLS** — documented as already shipped by
  `fit_predict_lw_pls` (k-NN + uniform-weight local SIMPLS).
- Internal `Config` extended with `sparsity_lambda`, `kernel_type`,
  `kernel_gamma`, `kernel_coef0`, `kernel_degree`, `di_lambda`,
  `recursive_forgetting`.
- 19 new C++ ABI / core tests (228 total).

### Deferred (open phases at this release)

- **§10.2 Non-linear kernel PLS** (RBF / polynomial / sigmoid) — needs a
  Gram-matrix dual path and a Model field for kernel parameters.
- **§8 O2PLS** — needs bi-directional OPLS infrastructure.
- **§9 N-PLS** — needs tensor unfolding utilities.

### Changed

- Project version is now `0.69.0+abi.1.1.0`. C ABI surface unchanged at
  `1.1.0`; the new fields on `Config` are internal-only.

## [0.68.0-comprehensive-benchmark] — 2026-05-15

Phase 7b–7e. Comprehensive performance benchmark matrix.

### Added

- Python ctypes wrappers `pls4all.Model` / `pls4all.ModelArrayKind` with
  NumPy zero-copy `p4a_matrix_view_t` constructors. Fit / predict /
  transform / get_array. Non-copyable owning handles.
- `pls4all_cli --bench` subcommand: deterministic synthetic dataset,
  fit + predict timing per algorithm, CSV-parseable output. Recognized
  algos: pls_nipals, pls_orthogonal_scores, pls_simpls,
  pls_kernel_algorithm, pls_wide_kernel, pls_svd, pls_power,
  pls_randomized_svd, pcr_svd.
- `bindings/r/pls4all/` minimum viable R package: `.Call` gateway,
  fit/predict for the 9 shipped PLS regression solvers, R-level
  wrappers, version queries, `R_registerRoutines` init, Makevars +
  Makevars.win. No Rcpp dependency.
- `benchmarks/runners/pls_regression.py` PLS regression benchmark
  runner with 9 algos × `(n_samples, n_features)` cells. Smoke gated
  at 200x100 and 500x100; full matrix on demand via `--scale full`.
- `benchmarks/runners/matrix.py` cross-language performance matrix
  driver: native C++ (CLI) / pls4all-Python / pls4all-R / sklearn
  reference. Records median + min wall times per row, language status
  per column, CPU pinning environment variables. Smoke at
  200x100/500x100 × 2 algos; full matrix is 9 × 5 × 3 = 135 cells.
- `benchmarks/results/{pls_regression,matrix}/accuracy.csv` committed
  and gated by `python benchmarks/run.py --check`. Timing CSVs +
  summaries written but not gated.
- Updated `benchmarks/README.md` with the run / pinning commands.
- Phase notes in `roadmap/phase-7b-…md`, `phase-7c-…md`,
  `phase-7d-…md`, `phase-7e-…md`.
- Overview status snapshot at the top of `Overview.md` mapping each
  taxonomy section to the current pls4all delivery state.

### Changed

- `bindings/python/src/pls4all/_config.py` exposes `center_x`,
  `scale_x`, `center_y`, `scale_y`, `store_scores` as Python
  properties (the corresponding ctypes prototypes were also added in
  Phase 6f and are reused).
- Benchmark output reorganized under `benchmarks/results/<benchmark>/`
  per-suite subdirectories.
- Project version is now `0.68.0+abi.1.1.0`. C ABI unchanged from
  `1.1.0`; no new public symbols. Only Python / CLI / R bindings and
  the benchmark suite changed.

## [0.67.0-benchmark-foundation] — 2026-05-15

Phase 7a. Benchmark foundation.

### Added

- `benchmarks/` directory with the deterministic Python driver
  `benchmarks/run.py` (`--check` mode + `--repeats N`).
- `benchmarks/runners/_harness.py` exposing `AccuracyRecord`,
  `TimingRecord`, `median_time_ms`, and the summary-table formatter.
- `benchmarks/runners/aom_global.py`: AOM-PLS global selection runner that
  drives `p4a_aom_global_select` through the Python ctypes wrapper and
  mirrors the dataset + folds into `nirs4all/bench/AOM_v0/aompls`.
- Four committed test cases (smooth-monotonic 9x6, 12x8, 16x10 plus a
  detrend-favoured 14x9). The committed accuracy CSV is gated by
  `--check`; current run reports 0.0 absolute RMSE delta against the
  oracle across all four cases.
- Committed `benchmarks/results/aom_global_accuracy.csv` (gated),
  `aom_global_timing.csv` and `aom_global_summary.md` (informational).
- `benchmarks/README.md` describing run command, gated vs informational
  split, and the contract for adding a new runner.

### Changed

- Project version is now `0.67.0+abi.1.1.0`. The C ABI surface is
  unchanged from `1.1.0`; benchmarks consume the existing public AOM/POP
  ABI only.

## [0.66.0-public-aom-pop-abi] — 2026-05-15

Phase 6f. Public C ABI for AOM/POP selection.

### Added

- Opaque handle `p4a_validation_plan_t` with create / destroy /
  `set_n_samples` / `add_fold` / `get_n_samples` / `get_n_folds`.
- Opaque handle `p4a_aom_global_result_t` plus entry point
  `p4a_aom_global_select` and typed accessors for `n_operators`,
  `max_components`, `selected_operator_index`, `selected_n_components`,
  `best_score`, `operator_kinds` (`p4a_operator_kind_t*`),
  `operator_scores`, `rmse_curves` and `predictions`.
- Opaque handle `p4a_aom_per_component_result_t` plus entry point
  `p4a_aom_per_component_select` and typed accessors for `n_operators`,
  `max_components`, `selected_n_components`, `best_score`,
  `operator_kinds` (`p4a_operator_kind_t*`), `selected_operator_indices`
  (`int32_t*`), `component_scores`, `prefix_scores` and `predictions`.
- Python ctypes wrappers `OperatorBank`, `OperatorKind`,
  `ValidationPlan`, `AomGlobalResult`, `AomPerComponentResult`,
  `aom_global_select`, `aom_per_component_select`. Result wrappers are
  non-copyable and copy the C-owned buffers into Python `list`s.
- Python `Config` properties for `center_x`, `scale_x`, `center_y`,
  `scale_y`, `store_scores` (already exposed by the C ABI).
- Driver script `bindings/python/smoke_aom_pop.py` that drives every
  shipped AOM/POP fixture through the public C ABI.
- ABI / parity tests in `cpp/tests/abi/test_validation_plan_abi.cpp`,
  `cpp/tests/abi/test_aom_global_public_abi.cpp`,
  `cpp/tests/abi/test_aom_pop_public_abi.cpp`.

### Changed

- Project version is now `0.66.0+abi.1.1.0`; the C ABI bumps to
  `1.1.0` (additive only, 28 new `p4a_*` symbols).
- Internal `validate_plan` in `cpp/src/core/aom_selection.cpp` now
  rejects out-of-range, duplicated or train/test-overlapping fold
  indices with `P4A_ERR_INVALID_ARGUMENT`. Plan/X row mismatch now
  returns `P4A_ERR_SHAPE_MISMATCH`, documented in the public header.

## [0.4.0-svd] — 2026-05-14

SVD solver increment.

### Added

- Dependency-free SVD regression solver for PLS1 / PLS2 behind
  `P4A_SOLVER_SVD`, using exact covariance SVD directions with regression
  deflation.
- Deterministic NumPy SVD parity fixtures plus C++ parity coverage for
  predictions, coefficients, preprocessing statistics, latent arrays,
  transform scores and serialization round-trips.
- CLI selfcheck smoke for the SVD fit / predict path.

### Changed

- Project version is now `0.4.0+abi.1.0.0`; the C ABI remains unchanged.
- The model import path now accepts serialized NIPALS, SIMPLS and SVD models.

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
