# Phase 4e - Wide-Kernel PLS Solver

## Goal

Activate the ABI-reserved `P4A_SOLVER_WIDE_KERNEL` path for wide linear PLS
regression. It shares the dual `K = X X^T` reference kernel with Phase 4d, but
is exposed as a dedicated solver for downstream parity and benchmarking against
wide-kernel implementations.

## Scope

- `P4A_ALGO_PLS_REGRESSION` + `P4A_SOLVER_WIDE_KERNEL` +
  `P4A_DEFLATION_REGRESSION` for PLS1 and PLS2.
- Dedicated deterministic NumPy wide-kernel fixtures with `p >> n` shapes.
- C++ ABI tests for fit, predict, transform, stored scores and serialization
  round-trips.
- CLI selfcheck coverage.

## Out Of Scope

- Nonlinear kernels and kernel hyperparameters.
- Separate optimized memory layout for very large `p`.
- Full binding estimator surfaces.

## Exit Criteria

- Local `cmake --build`, `ctest`, CLI selfcheck, parity determinism and ABI
  symbol/dependency checks pass.
- GitHub CI, ABI Surface, Sanitizers, Coverage and Parity gate are green.
- Release tag `phase-4e-wide-kernel` points at the final commit.
