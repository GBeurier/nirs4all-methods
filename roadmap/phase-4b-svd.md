# Phase 4b - SVD Core Solver

## Goal

Add the ABI-reserved `P4A_SOLVER_SVD` path for standard PLS regression without
changing the public C ABI. The solver uses exact covariance-SVD directions and
the same regression deflation, prediction, transform and serialization chassis
as the Phase 1 NIPALS implementation.

## Scope

- `P4A_ALGO_PLS_REGRESSION` + `P4A_SOLVER_SVD` +
  `P4A_DEFLATION_REGRESSION` for PLS1 and PLS2.
- Deterministic NumPy SVD parity fixtures for predictions, coefficients,
  preprocessing statistics and latent arrays.
- C++ ABI tests for fit, predict, transform, stored scores and serialization
  round-trips.
- Python binding enum coverage through the existing `Config.solver` property.

## Out Of Scope

- Nonlinear kernel PLS.
- PLS-SVD canonical transformer semantics.
- Randomized SVD and GPU-accelerated SVD.
- Full Python/R/MATLAB estimator bindings.

## Exit Criteria

- Local `cmake --build`, `ctest`, CLI selfcheck, parity determinism and ABI
  symbol/dependency checks pass.
- GitHub CI, ABI Surface, Sanitizers, Coverage and Parity gate are green.
- Release tag `phase-4b-svd` points at the final commit.
