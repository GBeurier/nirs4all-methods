# Phase 4k - OPLS1 Orthogonal Corrections

Goal: turn the reserved `P4A_ALGO_OPLS` ABI enum into a live one-target OPLS path without adding public symbols.

## Scope

- `P4A_ALGO_OPLS` supports `P4A_SOLVER_NIPALS` and `P4A_DEFLATION_ORTHOGONAL`.
- `n_components` is interpreted as one predictive component plus `n_components - 1` orthogonal correction components.
- The first score/weight/loading column is predictive; later columns are orthogonal corrections.
- `transform` remains linear by storing raw-space rotations for the predictive and orthogonal score projections.
- Binary import/export preserves OPLS models through the existing format.

## Parity

- Two deterministic single-target fixtures are generated from a NumPy OPLS1 NIPALS recurrence.
- C++ ABI tests assert predictions, coefficients, intercept, preprocessing statistics, latent arrays, stored scores and serialization round-trip.
- OPLS-DA is deliberately deferred; this phase keeps classification semantics out of the regression OPLS core.

## Acceptance

- `P4A_ALGO_OPLS + P4A_SOLVER_NIPALS + P4A_DEFLATION_ORTHOGONAL` fits and predicts for one target.
- Multi-target OPLS returns a deterministic shape error.
- `generate-fixtures --check`, release CTest, CLI selfcheck, ABI symbol diff, dependency audit and sanitizer builds pass locally.
- GitHub CI, ABI Surface, Sanitizers, Coverage and Parity gate pass on `main`.

## Deferred

- OPLS-DA and class-label threshold helpers.
- A separate public orthogonal-component count, if future API review decides `n_components` should mean predictive components only.
