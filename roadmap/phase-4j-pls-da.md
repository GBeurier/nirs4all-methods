# Phase 4j - PLS-DA Dummy-Response Scores

Goal: turn the reserved `P4A_ALGO_PLS_DA` ABI enum into a live score model by fitting PLS regression on caller-provided dummy/one-hot `Y`.

## Scope

- `P4A_ALGO_PLS_DA` uses the existing regression-deflation chassis.
- Numeric predictions remain continuous class scores with shape `(n_samples, n_classes_or_dummy_targets)`.
- Bindings can apply argmax / threshold decision rules later without changing the C ABI.
- The core does not validate that `Y` is strictly one-hot; labels and encoding stay binding-side concerns.
- Binary import/export preserves `P4A_ALGO_PLS_DA`.

## Acceptance Criteria

- `P4A_ALGO_PLS_DA` + `P4A_DEFLATION_REGRESSION` fits through the same solver set as PLS regression.
- Two deterministic one-hot fixtures are generated from pinned scikit-learn `PLSRegression` on dummy `Y`.
- C++ ABI tests verify predictions, coefficients, preprocessing statistics, latent arrays, stored scores and serialization round-trips.
- `pls4all_cli --selfcheck`, release CTest, fixture determinism, ABI symbol diff and dependency audit pass locally.
- Release tag `phase-4j-pls-da` points at the final commit.

## Deferred

- Public class-label outputs, confusion matrices and classification metrics.
- PLS-LDA / PLS-logistic / calibrated probability wrappers.
- OPLS-DA and sparse PLS-DA.
