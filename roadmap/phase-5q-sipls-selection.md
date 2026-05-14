# Phase 5q — siPLS interval-combination selection

Status: shipped locally as `phase-5q-sipls-selection`.

Goal: add a deterministic synergy interval PLS selector that partitions the
feature axis into contiguous intervals, exhaustively scores fixed-size
interval combinations by cross-validated RMSE, and returns the best
synergistic interval subset.

Delivered:

- Python parity fixture `synthetic_sipls_interval_combinations_v1`.
- C++ internal `select_by_sipls` kernel.
- Deterministic contiguous interval partitioning with lexicographic exhaustive
  interval-combination enumeration.
- Stable tie-breaking by candidate order, equivalent to RMSE then
  lexicographic interval tuple.
- ABI/core tests against the Python/sklearn reference for interval partition,
  candidate matrix, RMSE path, best combination and selected indices.

Deferred:

- Public C ABI wrappers for siPLS result buffers.
- Pruned / heuristic siPLS search for large interval grids.
- Batched/OpenMP CV evaluation for high-combination workloads.
