# Phase 5c — Stability selection

Status: shipped as `phase-5c-stability-selection`.

Goal: add the deterministic Monte-Carlo coefficient-stability primitive used by
MCUVE-style variable selection.

Delivered:

- Python parity fixture `synthetic_coefficient_stability_mcuve_v1`.
- C++ internal `select_by_coefficient_stability` kernel over validation-plan
  Monte-Carlo folds.
- Original-scale coefficient mean/std aggregation and per-feature
  max-abs stability scores.
- ABI/core tests against sklearn-derived coefficient means, standard
  deviations, stability scores and exact top-k indices.

Deferred:

- Public C ABI wrappers for stability-selection results.
- UVE/EMCUVE random-variable references and randomisation tests.
- Wrapper/metaheuristic selectors: CARS, SPA, Random Frog and GA-PLS.
