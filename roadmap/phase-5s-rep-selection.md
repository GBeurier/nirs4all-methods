# Phase 5s — REP-PLS recursive elimination

Status: shipped locally as `phase-5s-rep-selection`.

Goal: add a deterministic REP-PLS selector that repeatedly fits PLS on the
active variable set, removes a fixed number of weakest original-scale
coefficient-score variables, scores every retained subset by k-fold CV RMSE,
and returns the best retained subset along the path.

Delivered:

- Python parity fixture `synthetic_rep_pls_recursive_v1`.
- C++ internal `select_by_rep` kernel.
- Deterministic fixed-count recursive elimination with coefficient-score
  ranking and stable tie-breaking.
- Removed-count / removed-index trace plus retained counts, RMSE path and best
  subset.
- ABI/core tests against the Python/sklearn reference for score matrix, RMSE
  path, removed trace and selected indices.

Deferred:

- Public C ABI wrappers for REP result buffers.
- REP variants using VIP/selectivity/WVC scores instead of coefficient scores.
- Batched/OpenMP CV evaluation for larger recursive paths.
