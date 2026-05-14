# Phase 5e — SPA-PLS projection selection

Status: shipped locally as `phase-5e-spa-selection`.

Goal: add a deterministic successive projections algorithm (SPA) primitive for
PLS-guided variable selection.

Delivered:

- Python parity fixture `synthetic_spa_pls_projection_v1`.
- C++ internal `select_by_spa` kernel.
- Seed variable chosen by full-model original-scale PLS coefficient magnitude.
- Successive variables chosen by maximum residual norm after projection onto
  the selected standardized X-column subspace.
- ABI/core tests against Python/sklearn-derived coefficient scores, selection
  step scores and exact selected indices.

Deferred:

- Public C ABI wrappers for SPA result buffers.
- CV-based SPA prefix pruning.
- Wrapper/metaheuristic selectors: CARS, Random Frog and GA-PLS.
