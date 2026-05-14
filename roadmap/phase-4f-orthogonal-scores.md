# Phase 4f - Orthogonal Scores PLS Solver

## Goal

Activate the ABI-reserved `P4A_SOLVER_ORTHOGONAL_SCORES` path for classical
orthogonal-scores PLS regression. The numerical contract follows the R `pls`
`oscorespls.fit` algorithm, with deterministic NumPy fixtures used in CI.

## Scope

- `P4A_ALGO_PLS_REGRESSION` + `P4A_SOLVER_ORTHOGONAL_SCORES` +
  `P4A_DEFLATION_REGRESSION` for PLS1 and PLS2.
- R `pls`-style score iteration semantics: largest residual Y column
  initialisation for PLS2, score-based convergence, regression deflation.
- Deterministic NumPy fixtures ported from the R `pls` `oscorespls.fit`
  recurrence.
- C++ ABI tests for fit, predict, transform, stored scores and serialization
  round-trips.
- CLI selfcheck coverage.

## Out Of Scope

- Missing-value NIPALS.
- Cross-validation and auto-prefix scoring.
- OPLS / OPLS-DA orthogonal correction components.
- Full binding estimator surfaces.

## Exit Criteria

- Local `cmake --build`, `ctest`, CLI selfcheck, parity determinism and ABI
  symbol/dependency checks pass.
- GitHub CI, ABI Surface, Sanitizers, Coverage and Parity gate are green.
- Release tag `phase-4f-orthogonal-scores` points at the final commit.
