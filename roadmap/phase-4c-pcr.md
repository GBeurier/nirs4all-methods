# Phase 4c - PCR Baseline

## Goal

Add the ABI-reserved `P4A_ALGO_PCR` path without changing the public C ABI.
PCR is not PLS, but it is a necessary chemometrics baseline for benchmarks and
parity comparisons against R `pls::pcr` / scikit-learn PCA + linear regression.

## Scope

- `P4A_ALGO_PCR` + `P4A_SOLVER_SVD` + `P4A_DEFLATION_REGRESSION` for one or
  many targets.
- Dependency-free PCA/SVD reference path using the existing symmetric Jacobi
  eigensolver.
- Deterministic NumPy PCR fixtures for predictions, coefficients,
  preprocessing statistics and latent arrays.
- C++ ABI tests for fit, predict, transform, stored scores and serialization
  round-trips.
- Python binding enum coverage through `Config.algorithm` and `Config.solver`.

## Out Of Scope

- Cross-validated component selection.
- R `pls::pcr` fixture generation.
- Randomized / GPU PCA.
- Full Python/R/MATLAB estimator bindings.

## Exit Criteria

- Local `cmake --build`, `ctest`, CLI selfcheck, parity determinism and ABI
  symbol/dependency checks pass.
- GitHub CI, ABI Surface, Sanitizers, Coverage and Parity gate are green.
- Release tag `phase-4c-pcr` points at the final commit.
