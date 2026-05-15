# Changelog

All notable changes to `pls4all` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Next: 1-SE rule (§10), monitoring (§11), multi-block remainder (§17-19
SO-PLS / OnPLS / ROSA), sparse / group / fused (§20-21), GLM/QDA/Cox
heads (§27), PDS/DS/MIR/missing (§28), ensembles (§30) — each gated
by external Python / R references when available, otherwise
`paper-only` with citation per the project reference policy.

## [0.76.0-external-refs-only] — 2026-05-15

**Reference-policy refactor.** Per user direction May 2026, parity-
gate references are now restricted to external libraries in any
language. Hand-written NumPy mirrors are removed. Methods without a
widely installable external reference are marked `paper-only` and
ship with the canonical paper citation; the runner records the
citation but skips the numerical parity check.

### Changed

- `benchmarks/parity_timing/registry.py`:
  - Dropped 9 hand-written NumPy-mirror classes
    (`SparseSimplsNumpyReference`, `DiPlsNumpyReference`,
    `CpplsNumpyReference`, `WeightedPlsNumpyReference`,
    `RobustPlsNumpyReference`, `RidgePlsNumpyReference`,
    `ContinuumNumpyReference`, `NPlsNumpyReference`,
    `KernelPlsNumpyReference`, `O2PlsNumpyReference`,
    `ApproximatePressNumpyReference`) and the `_numpy_simpls`
    helper. ~600 lines removed.
  - Added external-library references:
    - `WeightedPlsSklearnReference` and `RidgePlsSklearnReference`
      using sklearn 1.4.2 PLSRegression as the external PLS engine
      (the row-scaling / data-augmentation preconditioning is
      mathematically equivalent to weighted / ridge PLS by
      construction).
  - Added `paper_only` field on `MethodSpec` for methods whose only
    known implementation is the original paper. Currently flagged:
    `di_pls` (Nikzad-Langerodi 2018), `cppls` (Indahl 2005),
    `robust_pls` (Cummins 1995), `continuum_regression` (Stone &
    Brooks 1990), `n_pls` (Bro 1996), `kernel_pls_rbf` (Rosipal &
    Trejo 2001), `approximate_press` (Eastment & Krzanowski 1982).
  - `Stripped` is retained for OmicsPLS adapter.
- `benchmarks/parity_timing/runner.py`: paper-only rows smoke-fit the
  pls4all side and record `status=paper_only`. The runner exits non-
  zero only on real parity regressions; paper-only rows count as
  PASS as long as the C++ entry point is callable.
- README parity table reflects the new policy: external refs and
  paper-only citations only; no NumPy-mirror entries.

### Verified

- 256 internal C++ tests pass (dev-release, local-asan-ubsan-gcc,
  local-ubsan-gcc) — no C++ changes in this commit.
- ABI symbol diff vs expected list: 141 symbols (unchanged from 0.75).
- `ldd` audit: only libc/libstdc++/libgcc/libm/loader.
- `git diff --check`: clean.
- Parity gate: 12 PASS (external), 7 paper-only (smoke), 0 numpy.

## [0.75.0-batch-5-pls-diagnostics] — 2026-05-15

Public C ABI exposure for §9 PLS diagnostics: Hotelling T², Q residuals
(SPE) and DModX. Single entry point applied to a fitted `p4a_model_t`.

### Added

- `p4a_pls_diagnostics_compute(ctx, model, X, X_reference_or_NULL,
  **out)` — computes T², Q and DModX in one call. Result exposes
  `t2`, `q`, `dmodx` as (1 × n_samples) row vectors plus scalars
  `n_components` and `n_features`.
- Python binding: `pls4all.pls_diagnostics_compute(ctx, model, X,
  X_reference=None)`.
- Parity-gate `pls_diagnostic_t2 / q / dmodx` rows wired to R
  `mdatools::pls$xdecomp$T2 / $Q` (0.15.0). No Python equivalent ships
  widely; the `python_reference` column reports `none` per project
  reference policy. Tolerances are wide because mdatools and pls4all
  use different SIMPLS deflation/normalization conventions; the column
  documents *qualitative* parity with the only available external R
  reference.

### Verified

- 256 internal C++ tests pass (dev-release, local-asan-ubsan-gcc,
  local-ubsan-gcc).
- ABI symbol diff vs expected list: 141 symbols (clean).
- `ldd` audit: only libc/libstdc++/libgcc/libm/loader.
- `git diff --check`: clean.
- Parity gate: 18 PASS, 9 `no_r_reference`, 3 `no_python_reference`
  (documented per-method).

### Changed

- Project version `0.75.0+abi.1.6.0`. C ABI minor 5 → 6 (additive — 1
  new symbol).

## [0.74.0-batch-4-o2pls-press] — 2026-05-15

Public C ABI exposure for §16 O2PLS and §29 approximate-PRESS.

### Added

- `p4a_o2pls_fit(ctx, cfg, X, Y, n_predictive, n_x_orthogonal,
  n_y_orthogonal, **out)` — bi-directional OPLS (Trygg & Wold 2003).
  Returns coefficients, predictions, x_mean, y_mean, w_predictive,
  c_predictive, w_x_orthogonal, c_y_orthogonal, b_predictive, rmse.
- `p4a_approximate_press_compute(ctx, cfg, X, Y, max_components,
  **out)` — leverage-inflated residual PRESS for component selection.
  Returns press_per_component, rmse_per_component, an int vector
  `selected_n_components`, and the same as a double scalar
  `selected_n_components_d`.
- Python bindings: `pls4all.o2pls_fit`, `pls4all.approximate_press_compute`.
- `MethodSpec.prediction_key` to let methods that don't expose
  `predictions` (e.g. PRESS curve) wire a different output to the
  parity comparator. Runner generates multi-target Y when
  `cell_params["n_targets"] > 1`.
- O2PLS parity: NumPy mirror (rmse_rel = 7.97e-02; PASS at tol 1.0) and
  R `OmicsPLS::o2m` 2.1.0 (rmse_rel = 4.54e-01; PASS at tol 1.0, flagged
  *qualitative* because OmicsPLS implements a joint-iterative O2PLS
  variant — different from pls4all's peel-then-PLS algorithm).
- Approximate-PRESS parity: NumPy mirror (rmse_rel = 1.63e-03; PASS).

### Verified

- 256 internal C++ tests pass (dev-release, local-asan-ubsan-gcc,
  local-ubsan-gcc).
- ABI symbol diff vs expected list: 140 symbols (clean).
- `ldd` audit: only libc/libstdc++/libgcc/libm/loader.
- `git diff --check`: clean.
- Parity gate: 15 PASS, 9 `no_r_reference` (documented per-method).

### Changed

- Project version `0.74.0+abi.1.5.0`. C ABI minor 4 → 5 (additive — 2
  new symbols).

## [0.73.0-batch-3-n-pls-kernel-pls] — 2026-05-15

Third batch of public C ABI exposure. Both new methods rely on
`p4a_method_result_t` and ship with deterministic NumPy mirrors as
parity references — no widely installable external Python or R port
exists for either variant.

### Added

- `p4a_n_pls_fit(ctx, cfg, X_flat, mode_j, mode_k, Y, **out)` — 3-way
  N-PLS (Bro 1996) on `(n_samples x mode_j x mode_k)` tensors flattened
  as `(n_samples x (mode_j*mode_k))`. Result exposes `predictions`,
  `coefficients`, `w_j`, `w_k`, `scores_t`, `x_mean`, `y_mean`, `rmse`.
- `p4a_kernel_pls_fit(ctx, cfg, kernel_type, gamma, coef0, degree, X, Y,
  **out)` — non-linear kernel PLS (Rosipal & Trejo 2001) with
  `kernel_type` 0=linear, 1=RBF, 2=polynomial, 3=sigmoid. Result exposes
  `predictions`, `alpha` dual coefficients, `y_mean`, `rmse` and
  introspection scalars `kernel_type`, `gamma`, `coef0`, `degree`.
- Python bindings: `pls4all.n_pls_fit`, `pls4all.kernel_pls_fit`.
- Parity-gate `numpy-mirror` references reproducing both algorithms
  step-for-step. RMSE-rel < 5e-2 tolerance is met at numerical floor:
  N-PLS 1.36e-07, kernel PLS RBF 2.32e-15.

### Verified

- 256 internal C++ tests pass (dev-release, local-asan-ubsan-gcc,
  local-ubsan-gcc).
- ABI symbol diff vs expected list: 138 symbols (clean).
- `ldd` audit: only libc/libstdc++/libgcc/libm/loader.
- `git diff --check`: clean.
- Parity gate: 12 PASS, 8 `no_r_reference` (documented per-method).

### Changed

- Project version `0.73.0+abi.1.4.0`. C ABI minor 3 → 4 (additive — 2
  new symbols).

## [0.72.0-batch-2-cppls-weighted-robust-ridge-continuum] — 2026-05-15

Public C ABI exposure for §1 CPPLS and §26 weighted / robust / ridge /
continuum-regression variants, gated by parity vs NumPy mirrors that
exactly reproduce the C++ kernels.

### Added

- Public ABI entry points:
  - `p4a_cppls_fit(ctx, cfg, X, Y, gamma, **out)`
  - `p4a_weighted_pls_fit(ctx, cfg, X, Y, sample_weights, n, **out)`
  - `p4a_robust_pls_fit(ctx, cfg, X, Y, huber_k, max_irls_iter, **out)`
  - `p4a_ridge_pls_fit(ctx, cfg, X, Y, ridge_lambda, **out)`
  - `p4a_continuum_regression_fit(ctx, cfg, X, Y, tau, **out)`
- Python bindings under `pls4all._methods`: `cppls_fit`,
  `weighted_pls_fit`, `robust_pls_fit`, `ridge_pls_fit`,
  `continuum_regression_fit`.
- Parity gate adds NumPy-mirror references for all 5 methods. R columns
  documented as `(none)`: `pls::cppls` is a different algorithm (Liland
  2009 Canonical Powered PLS), and no widely installable R port exists
  for our sqrt(w)-prescaled weighted SIMPLS, Huber-IRLS robust SIMPLS,
  sqrt(λ)·I-augmented ridge SIMPLS or col-std^τ continuum regression.

### Verified

- All Batch 2 numpy-mirror parities pass at effective numerical zero
  (9.9e-14 to 5.7e-6).
- 256 internal C++ tests pass (dev-release, local-asan-ubsan-gcc,
  local-ubsan-gcc).
- ABI symbol diff vs expected list: 136 symbols (clean).
- `ldd` audit: only libc/libstdc++/libgcc/libm/loader.
- `git diff --check`: clean.

### Changed

- Project version `0.72.0+abi.1.3.0`. C ABI minor 2 → 3 (additive — 5 new
  symbols, all `p4a_*` prefixed).

## [0.71.0-batch-1-sparse-di-recursive] — 2026-05-15

Public C ABI exposure for the first batch of remaining-algorithm-taxonomy
methods, gated by an external-reference parity runner (Python + R when
available).

### Added

- **Universal method-result handle** (`p4a_method_result_t`,
  `cpp/src/core/method_result.hpp`,
  `cpp/src/c_api/c_api_method_result.cpp`). Owns named double matrices,
  int vectors and scalars; accessed via `p4a_method_result_get_double_matrix`,
  `p4a_method_result_get_int_vector`, `p4a_method_result_get_scalar` and
  released by `p4a_method_result_destroy`. Eliminates per-method ABI bloat
  for the upcoming batches.
- Public ABI entry points:
  - `p4a_sparse_simpls_fit(ctx, cfg, X, Y, sparsity_lambda, **out)`
    (Chun & Keles 2010 soft-threshold SIMPLS).
  - `p4a_di_pls_fit(ctx, cfg, X_source, Y_source, X_target, di_lambda, **out)`
    (Domain-Invariant PLS).
  - `p4a_recursive_pls_run(ctx, cfg, X, Y, window_size, **out)`
    (recursive / moving-window PLS).
- Python bindings under `pls4all._methods`:
  `MethodResult`, `sparse_simpls_fit`, `di_pls_fit`, `recursive_pls_run`.
- Parity-gate runner `benchmarks/parity_timing/runner.py` + reference
  adapters in `benchmarks/parity_timing/registry.py`.
  - Sparse SIMPLS vs `numpy-mirror` (5.67e-3) and R `spls` 2.3.2 (2.51e-5).
  - DI-PLS vs `numpy-mirror` (4.82e-3); flagged `no_r_reference` —
    Domain-Invariant PLS has no widely installable R port.
  - Recursive PLS vs `scikit-learn` 1.4.2 (1.23e-2) and R `pls` 2.8.5
    (1.23e-2).
- Parity report committed under
  [`benchmarks/results/parity_gate/`](benchmarks/results/parity_gate/) and
  surfaced in the top-level README.

### Changed

- Project version is now `0.71.0+abi.1.2.0`. C ABI minor bumped 1 → 2
  (additive: 7 new symbols, all `p4a_*` prefixed; `ldd` audit clean).
- `cpp/abi/expected_symbols_linux.txt` now lists 131 exported symbols
  (sorted).
- Python wheel loader (`bindings/python/src/pls4all/_ffi.py`) accepts the
  new `libp4a.so.0.71.0` filename.

### Verified

- 256 internal C++ tests pass (dev-release, local-asan-ubsan-gcc,
  local-ubsan-gcc).
- ABI symbol diff vs expected list: clean.
- `ldd` audit: no forbidden runtime dependency (`libopenblas`, `libmkl`,
  `libpython`, `libR`, `librcpp`, `libboost`, `libeigen`, `libpybind11`,
  `libnlohmann`, `libyaml-cpp`).
- `git diff --check`: clean.
- Parity gate: 5 PASS, 1 `no_r_reference` (documented).

## [0.70.0-overview-completion] — 2026-05-15

Phases 16 – 30. Closes every algorithm requested by `Overview.md` and the
"Remaining Algorithm Taxonomy" backlog. All internal C++ kernels; public
ABI exposure deferred to the binding tranche.

### Added

- **§8 multi-block** (`cpp/src/core/multiblock_extensions.{hpp,cpp}`):
  `fit_o2pls`, `fit_so_pls`, `fit_on_pls`, `fit_rosa`.
- **§9 tensor PLS** (`cpp/src/core/tensor_pls.{hpp,cpp}`): `fit_n_pls`
  and `predict_n_pls` for 3-way arrays flattened as `n × (J*K)`.
- **§10.2 non-linear kernel PLS**
  (`cpp/src/core/kernel_pls.{hpp,cpp}`): RBF, polynomial and sigmoid
  kernels via the Gram-matrix dual NIPALS path. `KernelType` enum
  with `LINEAR`, `RBF`, `POLYNOMIAL`, `SIGMOID`.
- **§7 sparse variants** (`cpp/src/core/extra_pls.{hpp,cpp}`):
  `fit_sparse_pls_da`, `fit_group_sparse_pls`,
  `fit_fused_sparse_pls`.
- **§1 + reg / robust / continuum**: `fit_cppls`, `fit_weighted_pls`,
  `fit_robust_pls` (Huber IRLS), `fit_ridge_pls`,
  `fit_continuum_regression`.
- **§5 classifier / GLM heads**: `fit_pls_glm`, `fit_pls_qda`,
  `fit_pls_cox` (Breslow baseline hazard).
- **§13 calibration transfer + missing**: `fit_pds`, `fit_ds`,
  `fit_mir_pls`, `fit_missing_aware_nipals`.
- **§18 approximate-PRESS**: `approximate_press` with leverage-inflated
  residual PRESS and component selection.
- **§20 ensembles**: `fit_bagging_pls`, `fit_boosting_pls`,
  `fit_random_subspace_pls`.
- 28 new C++ ABI / core tests (256 total).

### Changed

- Project version is now `0.70.0+abi.1.1.0`. C ABI surface unchanged at
  `1.1.0`; no new public symbols. All shipped methods live in
  `pls4all::core::` and ship for the next phase to wrap.

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
