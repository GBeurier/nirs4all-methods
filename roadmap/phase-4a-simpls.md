# Phase 4a — SIMPLS Core Solver

Status: shipped as `0.3.0-simpls`.

This increment extends the Phase 1 fitted-model chassis with a dependency-free
SIMPLS solver for standard PLS regression. It keeps the public C ABI unchanged:
the solver is selected through the existing `p4a_config_set_solver` API with
`P4A_SOLVER_SIMPLS`.

## Scope

- `P4A_ALGO_PLS_REGRESSION` + `P4A_SOLVER_SIMPLS` +
  `P4A_DEFLATION_REGRESSION`.
- PLS1 and PLS2.
- Prediction, transform, fitted-array accessors and binary import/export.
- Deterministic Python parity fixtures generated from a NumPy SIMPLS reference
  ported from the local `nirs4all` / AOM implementation.

## Acceptance

- C++ parity tests cover coefficients, intercepts, preprocessing statistics,
  weights, loadings, rotations, scores and predictions.
- Serialization round-trips preserve SIMPLS predictions.
- The ABI symbol list remains exactly unchanged from Phase 0.
- Runtime dependencies remain limited to libc / libstdc++ / libm / libgcc.

## Deferred

- Kernel PLS, wide-kernel PLS and OPLS.
- Classification wrappers (PLS-DA / OPLS-DA).
- AOM / POP operator-bank integration.
- Benchmarks against R `pls::plsr(method="simpls")` once R is available in CI.
