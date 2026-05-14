# Phase 5k — BVE-PLS backward variable elimination

Status: shipped locally as `phase-5k-bve-selection`.

Goal: add a deterministic backward variable elimination wrapper selector that
tests every possible one-variable removal at each step and scores candidates by
k-fold CV RMSE over the existing NIPALS PLS regression chassis.

Delivered:

- Python parity fixture `synthetic_bve_pls_backward_v1`.
- C++ internal `select_by_bve` kernel.
- Greedy one-variable-removal search scored by the internal k-fold CV engine.
- Candidate RMSE matrix, chosen RMSE path, retained counts, removed indices,
  best step and exact selected subset.
- ABI/core tests against Python/sklearn-derived candidate paths, chosen RMSE
  path, retained counts, removed indices, best step and exact selected indices.

Deferred:

- Public C ABI wrappers for BVE result buffers.
- T2-PLS and WVC-PLS wrapper families.
- Batched/OpenMP evaluation for larger backward-elimination paths.
