# Phase 5r — thresholded/factor WVC selection

Status: shipped locally as `phase-5r-wvc-threshold-selection`.

Goal: extend the WVC-PLS selector with deterministic threshold/factor rules:
compute the WVC2 score ranking, derive an effective threshold from a fixed
score floor and a factor of the mean score, and keep all variables passing the
threshold with a minimum-selected fallback.

Delivered:

- Python parity fixture `synthetic_wvc_threshold_factor_v1`.
- C++ internal `select_by_wvc_threshold` kernel built on the existing WVC2
  score computation.
- Deterministic ranked-index output, mean score, effective threshold and
  selected-variable list.
- ABI/core tests against the Python/NumPy reference for final scores, ranking,
  threshold scalar fields and selected indices.

Deferred:

- Public C ABI wrappers for WVC threshold result buffers.
- Alternative quantile/MAD WVC threshold policies.
- Batched threshold sweeps for model-selection workflows.
