# Phase 4h - Randomized SVD PLS Solver

## Goal

Activate the ABI-reserved `P4A_SOLVER_RANDOMIZED_SVD` path for PLS regression.
The first implementation is a deterministic, context-seeded random-start
singular-vector iteration on the residual cross-covariance.

## Scope

- `P4A_ALGO_PLS_REGRESSION` + `P4A_SOLVER_RANDOMIZED_SVD` +
  `P4A_DEFLATION_REGRESSION` for PLS1 and PLS2.
- Dependency-free SplitMix64 random initialisation, seeded by `p4a_context_t`.
- Deterministic NumPy fixtures that mirror the C++ random stream and recurrence.
- C++ ABI tests for fit, predict, transform, stored scores and serialization
  round-trips.
- CLI selfcheck and Python binding smoke coverage.

## Out Of Scope

- Oversampled matrix sketches and range finders.
- Block Krylov / Lanczos acceleration.
- Randomized PCR.
- GPU acceleration.

## Exit Criteria

- Local `cmake --build`, `ctest`, CLI selfcheck, parity determinism and ABI
  symbol/dependency checks pass.
- GitHub CI, ABI Surface, Sanitizers, Coverage and Parity gate are green.
- Release tag `phase-4h-randomized-svd` points at the final commit.
