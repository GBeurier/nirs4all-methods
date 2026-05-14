# Phase 5u — ST-PLS score-threshold selection

Status: shipped locally as `phase-5u-st-selection`.

Goal: add a deterministic ST-PLS selector that normalizes original-scale PLS
coefficient scores, evaluates a threshold path with a min-selected fallback,
scores every threshold subset by k-fold CV RMSE, and returns the best subset.

Delivered:

- Python parity fixture `synthetic_st_pls_threshold_v1`.
- C++ internal `select_by_st` kernel.
- Deterministic score-threshold path with stable ranking fallback when a
  threshold selects too few variables.
- Threshold RMSE path, selected-count trace, selected masks, final ranking and
  exact selected-index result.
- ABI/core tests against the Python/sklearn reference for normalized scores,
  threshold path outputs, masks, ranking and invalid-request handling.

Deferred:

- Public C ABI wrappers for ST result buffers.
- Alternative ST scoring sources, especially VIP, selectivity-ratio and WVC.
- Batched/OpenMP CV evaluation for threshold grids.
