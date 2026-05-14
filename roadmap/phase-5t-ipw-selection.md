# Phase 5t — IPW-PLS iterative predictor weighting

Status: shipped locally as `phase-5t-ipw-selection`.

Goal: add a deterministic IPW-PLS selector that iteratively reweights
original-scale PLS coefficient scores, records the score and weight paths,
scores the top-k subset at each iteration by k-fold CV RMSE, and returns the
best subset along the path.

Delivered:

- Python parity fixture `synthetic_ipw_pls_reweighted_v1`.
- C++ internal `select_by_ipw` kernel.
- Deterministic coefficient-score reweighting with damping, floor-clamped
  weights, stable ranking tie-breaking and top-k subset scoring.
- Score path, weight path, per-iteration RMSE path, final ranking and exact
  selected-index trace.
- ABI/core tests against the Python/sklearn reference for path matrices,
  ranking, selected indices and invalid-request handling.

Deferred:

- Public C ABI wrappers for IPW result buffers.
- IPW variants that drive reweighting from VIP, selectivity-ratio or WVC scores.
- Batched/OpenMP CV evaluation for larger IPW grids.
