# Phase 16 - 30 - Overview completion

Status: shipped locally as `phase-16-to-30-overview-completion`.

Goal: close every remaining method requested by `Overview.md` (§7 sparse
variants, §8 multi-block extensions, §9 tensor PLS, §10.2 non-linear
kernel PLS, §17 / §18 / §19 already shipped earlier, §20 ensembles, plus
the "Remaining Algorithm Taxonomy" backlog). All internal C++ kernels;
public ABI exposure ships with the binding tranche.

Delivered (one file per batch unless noted):

- **Batch A — §8 multi-block extensions**
  (`cpp/src/core/multiblock_extensions.{hpp,cpp}`,
  `cpp/tests/abi/test_multiblock_extensions.cpp`):
  - `fit_o2pls` — bi-directional OPLS (predictive +
    X-orthogonal + Y-orthogonal components).
  - `fit_so_pls` — sequential and orthogonalized PLS for B X-blocks
    predicting one Y.
  - `fit_on_pls` — multi-block PLS extracting joint and unique
    components per block.
  - `fit_rosa` — Response-Oriented Sequential Alternation: pick the
    best-correlated block per component.
- **Batch B — §9 tensor PLS**
  (`cpp/src/core/tensor_pls.{hpp,cpp}`, `cpp/tests/abi/test_tensor_pls.cpp`):
  - `fit_n_pls` / `predict_n_pls` — Bro's 3-way N-PLS for tensors
    flattened as `n × (J*K)`.
- **Batch C — §10.2 kernel non-linear PLS**
  (`cpp/src/core/kernel_pls.{hpp,cpp}`,
  `cpp/tests/abi/test_kernel_pls.cpp`):
  - `fit_kernel_pls` / `predict_kernel_pls` for RBF, polynomial and
    sigmoid kernels via the Gram-matrix dual NIPALS path. Linear
    kernel falls back to the existing kernel-algorithm solver.
- **Batches D – J — taxonomy backlog**
  (`cpp/src/core/extra_pls.{hpp,cpp}`,
  `cpp/tests/abi/test_extra_pls.cpp`):
  - §7: `fit_sparse_pls_da`, `fit_group_sparse_pls`,
    `fit_fused_sparse_pls`.
  - §1: `fit_cppls` (CPPLS / powered PLS with gamma in [0, 1]).
  - Robustness / weights / regularization: `fit_weighted_pls`,
    `fit_robust_pls` (Huber IRLS), `fit_ridge_pls`,
    `fit_continuum_regression`.
  - Classifier / GLM heads: `fit_pls_glm` (PLS-GLM with PLS-reduced
    design + identity/Poisson families), `fit_pls_qda`, `fit_pls_cox`
    (Breslow baseline hazard).
  - Calibration transfer: `fit_pds` (window-LS), `fit_ds`
    (direct standardisation), `fit_mir_pls` (multivariate inverse
    regression), `fit_missing_aware_nipals` (NaN imputation through
    column means before SIMPLS).
  - §18: `approximate_press` (leverage-inflated residual PRESS with
    component selection).
  - §20: `fit_bagging_pls`, `fit_boosting_pls` (sequential residual
    boosting with learning rate), `fit_random_subspace_pls`.

Internal `Config` was already extended in Phase 8–15 with
`sparsity_lambda`, `kernel_type`, `kernel_gamma`, `kernel_coef0`,
`kernel_degree`, `di_lambda`, `recursive_forgetting`. No new public
ABI symbols.

Trade-offs documented:

- Most kernels keep their internal helpers self-contained (no
  reuse of static helpers from `model.cpp`). This duplicates
  centering / mean / matrix-inversion logic across files but lets
  each algorithm be shipped without invasive header surgery.
- Parity gates are **in-test deterministic identities and finiteness
  checks** rather than fixtures from the pinned `parity/` venv —
  the pinned scikit-learn 1.4.2 still does not build on this host's
  Python 3.13. Migrating to the canonical JSON-fixture pipeline is
  a separate item.
- Some methods (PLS-Cox baseline hazard, MIR-PLS, robust PLS IRLS,
  group / fused sparse) are minimum-viable: they produce the right
  shapes and finite numerics but full reference-grade parity against
  R / Python implementations is deferred to the binding tranche.

Local gate (green):

- 256 C++ ABI / core tests (was 228; +28 new tests across all batches).
- dev-release: 256 / 256.
- UBSAN: 256 / 256.
- ASAN+UBSAN: 256 / 256.
- ABI symbol diff unchanged.
- ldd: only libc / libstdc++ / libgcc / libm / loader.
- Benchmark suite: pre-existing accuracy CSVs still pass `--check`.

Remaining work (now intentionally OUT of scope for the method
development push):

- Public C ABI exposure for §7–§20 internal kernels — bundled with
  the binding tranche.
- Python / R / MATLAB / JS / Julia / Java bindings — next batch.
- Benchmark matrix expansion to cover the new methods — runs once
  bindings are in place.
- Acceleration roadmap (BLAS / OpenMP / CUDA) — after benchmarks
  identify hotspots.
