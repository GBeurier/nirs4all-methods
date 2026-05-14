# Phase 4d - Linear Kernel PLS Solver

## Goal

Add the ABI-reserved `P4A_SOLVER_KERNEL_ALGORITHM` path for linear PLS
regression without changing the public C ABI. This is the reference CPU path
for wide matrices and a stepping stone toward nonlinear kernel PLS.

## Scope

- `P4A_ALGO_PLS_REGRESSION` + `P4A_SOLVER_KERNEL_ALGORITHM` +
  `P4A_DEFLATION_REGRESSION` for PLS1 and PLS2.
- Dependency-free dual score extraction through `K = X X^T`, with primal
  weights reconstructed for the existing prediction / transform chassis.
- Deterministic NumPy kernel-PLS fixtures, including a `p > n` case.
- C++ ABI tests for fit, predict, transform, stored scores and serialization
  round-trips.
- CLI selfcheck coverage.

## Out Of Scope

- Nonlinear kernels (RBF / polynomial / sigmoid).
- Kernel hyperparameter config surface.
- Randomized / GPU kernel acceleration.
- Full Python/R/MATLAB estimator bindings.

## Exit Criteria

- Local `cmake --build`, `ctest`, CLI selfcheck, parity determinism and ABI
  symbol/dependency checks pass.
- GitHub CI, ABI Surface, Sanitizers, Coverage and Parity gate are green.
- Release tag `phase-4d-kernel` points at the final commit.
