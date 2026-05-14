# Phase 4m - Multi-Response OPLS / Multi-Class OPLS-DA

Goal: remove the Phase 4l one-column OPLS limitation without changing the C ABI.

## Scope

- `P4A_ALGO_OPLS` and `P4A_ALGO_OPLS_DA` support `Y.cols >= 1`.
- `P4A_SOLVER_NIPALS` and `P4A_DEFLATION_ORTHOGONAL` remain the accepted chassis.
- Multi-response OPLS uses one shared predictive score direction from the dominant singular pair of `X.T @ Y`.
- `n_components` keeps the OPLS meaning: one predictive component plus `n_components - 1` X-orthogonal correction components.
- Predict / transform / fitted-array accessors / binary import-export use the existing model layout with `coefficients` shaped `(n_features, n_targets)` and `y_loadings_q` shaped `(n_targets, n_components)`.

## Parity

- Add one deterministic OPLS PLS2 fixture.
- Add one deterministic multi-class OPLS-DA fixture with one-hot `Y`.
- C++ ABI tests assert predictions, coefficients, intercept, preprocessing statistics, latent arrays, transform scores and serialization round-trip.

## Acceptance

- `P4A_ALGO_OPLS + P4A_SOLVER_NIPALS + P4A_DEFLATION_ORTHOGONAL` fits multi-target `Y`.
- `P4A_ALGO_OPLS_DA + P4A_SOLVER_NIPALS + P4A_DEFLATION_ORTHOGONAL` fits one-hot multi-class `Y`.
- `generate-fixtures --check`, release CTest, CLI selfcheck, ABI symbol diff, dependency audit and sanitizer builds pass locally.
- GitHub CI, ABI Surface, Sanitizers, Coverage and Parity gate pass on `main`.

## Deferred

- One-vs-rest OPLS-DA ensembles with per-class orthogonal components.
- Label-to-dummy helpers and class threshold/calibration utilities.
- External parity against `ropls::opls` once the R parity generator is active.
