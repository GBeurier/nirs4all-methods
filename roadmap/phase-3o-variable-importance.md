# Phase 3o - Variable Importance

Goal: add deterministic variable-importance kernels for fitted PLS-style models.

## Scope

- Compute variable importance in projection (VIP) from fitted weights, response
  loadings and stored X scores.
- Compute selectivity ratio by reconstructing standardized training X from
  stored scores and X loadings.
- Require `store_scores=1` at fit time so the calculation is explicit and does
  not silently recompute training scores.
- Add a sklearn `PLSRegression` parity fixture for a PLS2 model.

## Non-goals

- No public C ABI expansion yet.
- No per-component coefficient path in this phase.
- No variable-selection thresholding policy yet.

## Acceptance

- Fixture generation is deterministic.
- C++ VIP and selectivity-ratio vectors match sklearn-derived references within
  documented tolerance.
- Missing stored scores and invalid X inputs are rejected deterministically.
- ABI symbol export remains unchanged.
