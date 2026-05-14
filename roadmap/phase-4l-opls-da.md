# Phase 4l - Binary OPLS-DA Dummy-Response Scores

Goal: activate the reserved `P4A_ALGO_OPLS_DA` enum for the binary dummy-response case by reusing the Phase 4k OPLS1 orthogonal-correction core.

## Scope

- `P4A_ALGO_OPLS_DA` supports `P4A_SOLVER_NIPALS` and `P4A_DEFLATION_ORTHOGONAL`.
- `Y` must be a single dummy/continuous class-score column. Multi-class one-vs-rest orchestration is deferred until the ABI has an explicit strategy for per-class orthogonal components.
- `n_components` keeps the Phase 4k meaning: one predictive component plus `n_components - 1` orthogonal correction components.
- Predict / transform / fitted-array accessors / binary import-export reuse the same model layout as OPLS.

## Parity

- One deterministic binary fixture is generated from the NumPy OPLS1 recurrence on one-column dummy `Y`.
- C++ ABI tests assert predictions, coefficients, intercept, preprocessing statistics, latent arrays, stored scores and serialization round-trip.

## Acceptance

- `P4A_ALGO_OPLS_DA + P4A_SOLVER_NIPALS + P4A_DEFLATION_ORTHOGONAL` fits one-column dummy targets.
- Multi-column OPLS-DA returns the same deterministic shape error as OPLS.
- `generate-fixtures --check`, release CTest, CLI selfcheck, ABI symbol diff, dependency audit and sanitizer builds pass locally.
- GitHub CI, ABI Surface, Sanitizers, Coverage and Parity gate pass on `main`.

## Deferred

- Multi-class OPLS-DA.
- Label-to-dummy helpers and threshold/calibration utilities.
- External parity against `ropls::opls` once the R parity generator is active.
