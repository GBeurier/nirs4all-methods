# Phase 3p - Component Coefficients

Goal: materialise original-scale regression coefficient blocks for each latent
component prefix.

## Scope

- Compute `B_k` for every prefix `k = 1..A` from fitted `W`, `P` and `Q`.
- Recompute `R_k = W_k (P_k.T W_k)^-1` for each prefix rather than slicing the
  full-model rotations.
- Apply the same original-scale `y_scale / x_scale` conversion used by model
  prediction.
- Add a sklearn `PLSRegression` parity fixture for a PLS2 model.

## Non-goals

- No public C ABI expansion yet.
- No automatic component selection policy yet.
- No sparse or interval selector in this phase.

## Acceptance

- Fixture generation is deterministic.
- C++ component-major coefficient blocks match sklearn-derived references within
  documented tolerance.
- Inconsistent fitted-model arrays are rejected deterministically.
- ABI symbol export remains unchanged.
