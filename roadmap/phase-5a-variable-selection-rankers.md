# Phase 5a — Variable-selection rankers

Status: shipped as `phase-5a-variable-selection-rankers`.

Goal: establish the deterministic selection contract before implementing
heavier interval, Monte-Carlo and wrapper methods.

Delivered:

- Python parity fixture `synthetic_variable_selection_rankers_v1`.
- C++ internal `select_variables` kernel for VIP, original-scale coefficient
  magnitude and selectivity-ratio methods.
- Deterministic ranking policy: score descending, index ascending for ties.
- ABI/core tests against sklearn-derived fixture scores and exact top-k
  indices.

Deferred:

- Public C ABI wrappers for variable-selection results.
- iPLS / biPLS / siPLS / moving-window PLS.
- MCUVE / CARS / SPA / Random Frog / GA-PLS and benchmark suites.
