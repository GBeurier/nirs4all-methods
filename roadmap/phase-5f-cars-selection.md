# Phase 5f — CARS-PLS competitive adaptive reweighted sampling

Status: shipped locally as `phase-5f-cars-selection`.

Goal: add a deterministic CARS-PLS primitive for wrapper-style variable
selection over the existing NIPALS PLS regression and k-fold CV chassis.

Delivered:

- Python parity fixture `synthetic_cars_pls_competitive_v1`.
- C++ internal `select_by_cars` kernel.
- Per-iteration PLS coefficient scoring over the active feature set.
- Deterministic exponential retention schedule down to `min_features`.
- Candidate subset scoring through the internal k-fold CV engine.
- ABI/core tests against Python/sklearn-derived coefficient score paths,
  retained counts, RMSE path, best iteration and exact selected indices.

Deferred:

- Public C ABI wrappers for CARS result buffers.
- SCARS stochastic resampling and random-frog posterior inclusion frequencies.
- GA-PLS / shaving-style metaheuristics.
