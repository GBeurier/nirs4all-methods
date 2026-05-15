# Phase 8 - 15 - PLS extensions (§7, §17, §18, §19, §11, §13, §12)

Status: shipped locally as `phase-8-to-15-pls-extensions`.

This batch closes the smallest viable additions to the algorithm taxonomy
sections §7, §11, §12 (partial), §13, §17, §18 and §19 of Overview.md.
Each extension lives in the **internal** C++ core. The public C ABI,
Python / R bindings and benchmark coverage will be added in a follow-up
binding tranche.

Delivered:

- **Phase 8 — Sparse PLS (§7)**: `fit_pls_sparse_simpls` in
  `cpp/src/core/model.cpp` extends SIMPLS with soft-thresholding on the
  loading direction. `cfg.sparsity_lambda` controls the L1 threshold;
  `0.0` falls back to plain SIMPLS. `P4A_ALGO_SPARSE_PLS` is now
  dispatched through `fit_model`. ABI tests in
  `cpp/tests/abi/test_sparse_pls.cpp` (4 cases).

- **Phase 9 — PLS diagnostics (§17)**: new files
  `cpp/src/core/pls_diagnostics.hpp` / `.cpp` exposing
  `pls_hotelling_t2`, `pls_q_residuals`, `pls_dmodx` against a fitted
  Model. Tests in `cpp/tests/abi/test_pls_diagnostics.cpp` (5 cases).

- **Phase 10 — One-SE rule (§18)**: extended
  `cross_validate_component_prefixes` to record per-fold RMSE values
  and added `select_one_se_components` to pick the smallest
  component count within one standard error of the best. Tests in
  `cpp/tests/abi/test_model_selection.cpp` (2 new cases).

- **Phase 11 — Process monitoring (§19)**: `pls_monitoring_fit` and
  `pls_monitoring_evaluate` in
  `cpp/src/core/pls_monitoring.{hpp,cpp}`. Empirical (percentile)
  thresholds at level α for T² and Q-residuals, plus per-sample
  alarm flags. Tests in `cpp/tests/abi/test_pls_monitoring.cpp`
  (3 cases).

- **Phase 13 — Domain-invariant PLS (§13)**: `fit_di_pls` in
  `cpp/src/core/model.cpp` adds the source-target difference vector
  to the SIMPLS direction update. `cfg.di_lambda` controls the
  penalty strength; `0.0` recovers plain SIMPLS. Tests in
  `cpp/tests/abi/test_di_pls.cpp` (3 cases).

- **Phase 14 — Just-in-time PLS (§11)**: already shipped as the
  existing `fit_predict_lw_pls` in `cpp/src/core/lw_pls.cpp` (uniform
  weights across the k nearest neighbours). The "weighted" LW-PLS
  variant with a smooth distance kernel is documented as the
  deferred follow-up.

- **Phase 15 — Recursive PLS (§12)**:
  `cpp/src/core/recursive_pls.{hpp,cpp}` ships a moving-window
  variant: each sample at position ≥ `window_size` is predicted by a
  SIMPLS model fitted on the previous `window_size` rows. Tests in
  `cpp/tests/abi/test_recursive_pls.cpp` (2 cases).

Deferred (kept open for the next tranche):

- **Phase 12 — Non-linear kernel PLS (§10.2)**: full RBF / polynomial
  / sigmoid Gram-matrix path requires a new model field for the
  kernel parameters and a dual prediction path. Currently the
  declared `fit_pls_regression_nonlinear_kernel` slot in `model.hpp`
  is reserved but unimplemented. The polynomial feature-map shortcut
  was rejected because it requires storing the expansion meta in
  the Model serialization.
- **Phase 16 — O2PLS (§8 extension)**: bi-directional OPLS needs
  parallel X-orthogonal and Y-orthogonal score series and a joint
  predictive direction. Open.
- **Phase 17 — N-PLS (§9)**: 3-way tensor PLS via Bro's algorithm
  needs tensor unfolding utilities not currently in the core. Open.

Local gate (green):

- 228 C++ ABI / core tests in dev-release.
- UBSAN: 228 / 228.
- ASAN+UBSAN: 228 / 228.
- ABI symbol diff unchanged (all extensions are internal; the public
  ABI is unmodified at this version).

Trade-offs documented in this push:

- These extensions are tested against **deterministic reference
  computations done inside the test code itself** (e.g. comparing
  sparse SIMPLS with lambda=0 to plain SIMPLS, comparing DI-PLS with
  lambda=0 to plain SIMPLS, checking diagnostics against
  reconstruction identities). The canonical
  `parity/python_generator/` JSON / `cpp_header.py` flow is NOT
  extended for these new methods — the pinned generator venv is
  unavailable on this host (sklearn 1.4.2 does not install with
  Python 3.13). Migration to the canonical parity-fixture pipeline is
  deferred to a future tranche once the pinned venv is restored.
- `Config` grew the fields `sparsity_lambda`, `kernel_type`,
  `kernel_gamma`, `kernel_coef0`, `kernel_degree`, `di_lambda` and
  `recursive_forgetting`. These are internal — no public setter is
  added yet, so the C ABI remains at `1.1.0`.
