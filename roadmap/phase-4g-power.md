# Phase 4g - Power-Iteration PLS Solver

## Goal

Activate the ABI-reserved `P4A_SOLVER_POWER` path for PLS regression. It uses
dominant singular-vector power iteration on the residual cross-covariance,
then reuses the existing regression-deflation model chassis.

## Scope

- `P4A_ALGO_PLS_REGRESSION` + `P4A_SOLVER_POWER` +
  `P4A_DEFLATION_REGRESSION` for PLS1 and PLS2.
- Deterministic dependency-free singular-pair power iteration with sign
  canonicalisation.
- Deterministic NumPy fixtures matching the power iteration recurrence.
- C++ ABI tests for fit, predict, transform, stored scores and serialization
  round-trips.
- CLI selfcheck and Python binding smoke coverage.

## Out Of Scope

- Block Krylov / Lanczos variants.
- Randomized SVD.
- GPU or BLAS acceleration.
- Cross-validation prefix scoring.

## Exit Criteria

- Local `cmake --build`, `ctest`, CLI selfcheck, parity determinism and ABI
  symbol/dependency checks pass.
- GitHub CI, ABI Surface, Sanitizers, Coverage and Parity gate are green.
- Release tag `phase-4g-power` points at the final commit.
