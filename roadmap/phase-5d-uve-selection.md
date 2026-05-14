# Phase 5d — UVE artificial-variable selection

Status: shipped as `phase-5d-uve-selection`.

Goal: add a deterministic UVE / MCUVE primitive that thresholds real-feature
coefficient stability against artificial reference variables.

Delivered:

- Python parity fixture `synthetic_uve_artificial_variables_v1`.
- C++ internal `select_by_uve` kernel that augments `X` with deterministic
  SplitMix64 reference variables.
- Per-feature real/noise stability scores, max-noise threshold and exact
  selected real-feature indices.
- ABI/core tests against sklearn-derived stability scores and thresholded
  selected indices.

Deferred:

- Public C ABI wrappers for UVE result buffers.
- EMCUVE random-variable ensembles and randomisation tests.
- Wrapper/metaheuristic selectors: CARS, SPA, Random Frog and GA-PLS.
