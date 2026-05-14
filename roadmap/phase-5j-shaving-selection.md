# Phase 5j — Shaving-PLS recursive elimination

Status: shipped locally as `phase-5j-shaving-selection`.

Goal: add a deterministic shaving-PLS wrapper selector that recursively removes
low-score variables while scoring every retained subset with k-fold CV over the
existing NIPALS PLS regression chassis.

Delivered:

- Python parity fixture `synthetic_shaving_pls_recursive_v1`.
- C++ internal `select_by_shaving` kernel.
- Per-step PLS coefficient scoring over the retained feature set.
- Deterministic score-ascending variable shaving down to `min_features`.
- Candidate subset scoring through the internal k-fold CV engine.
- ABI/core tests against Python/sklearn-derived coefficient score paths,
  retained counts, RMSE path, best step and exact selected indices.

Deferred:

- Public C ABI wrappers for shaving result buffers.
- BVE-PLS, T2-PLS and WVC-PLS wrapper families.
- Batched/OpenMP evaluation for larger recursive-elimination paths.
