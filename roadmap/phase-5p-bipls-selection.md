# Phase 5p — biPLS backward interval selection

Status: shipped locally as `phase-5p-bipls-selection`.

Goal: add a deterministic backward interval PLS selector that partitions the
feature axis into contiguous intervals, removes intervals one at a time by
cross-validated RMSE, and returns the best active interval set along the path.

Delivered:

- Python parity fixture `synthetic_bipls_backward_intervals_v1`.
- C++ internal `select_by_bipls` kernel.
- Deterministic contiguous interval partitioning with a final partial interval
  when the feature count is not a multiple of `interval_width`.
- Fold-stable backward interval elimination with deterministic tie-breaking by
  interval index.
- ABI/core tests against the Python/sklearn reference for interval partition,
  removal path, active counts, RMSE path, best step and selected indices.

Deferred:

- Public C ABI wrappers for biPLS result buffers.
- siPLS exhaustive / heuristic interval-combination search.
- Batched/OpenMP CV evaluation for larger interval grids.
