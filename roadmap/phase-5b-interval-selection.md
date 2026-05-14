# Phase 5b — Interval selection

Status: shipped as `phase-5b-interval-selection`.

Goal: add the deterministic interval-CV primitive used by moving-window PLS and
iPLS-style variable selection.

Delivered:

- Python parity fixture `synthetic_interval_selection_moving_window_v1`.
- C++ internal `cross_validate_intervals` kernel over contiguous feature
  windows.
- Deterministic k-fold NIPALS PLS refits per window with RMSE/metric ranking.
- ABI/core tests against sklearn-derived metrics, RMSE vector and exact best
  interval.

Deferred:

- Public C ABI wrappers for interval-selection results.
- biPLS / siPLS interval-combination search.
- Monte-Carlo stability selectors and wrapper/metaheuristic methods.
