# Phase 5m — WVC-PLS weighted variable contribution

Status: shipped locally as `phase-5m-wvc-selection`.

Goal: add a deterministic WVC-PLS selector for numeric single-response
regression, mirroring the `plsVarSel` WVC2 recurrence and exposing top-k
selection from the final component scores.

Delivered:

- Python parity fixture `synthetic_wvc_pls_numeric_v1`.
- C++ internal `select_by_wvc` kernel.
- X standardization, SVD-based component weights, X/Y deflation and normalized
  weighted-variable-contribution score matrix.
- Deterministic final-component top-k selection with score-desc/index-asc tie
  handling.
- ABI/core tests against NumPy/plsVarSel-derived weights, loadings, WVC scores,
  final scores and exact selected indices.

Deferred:

- Public C ABI wrappers for WVC result buffers.
- WVC thresholded refit mode and factor/dummy-response mode.
- Batched/OpenMP evaluation for WVC parameter sweeps.
