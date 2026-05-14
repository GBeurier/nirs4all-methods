# Phase 3k - Regression Metrics

Goal: add the internal validation metrics needed by CV, benchmarks and binding
reports before exposing a public metrics ABI.

## Scope

- C++17 internal `core::compute_regression_metrics`.
- Metrics: RMSE, MAE, bias, R2, Q2, observed-vs-predicted slope/intercept,
  RPD and RPIQ.
- Shape, dtype, empty-input and non-finite validation.
- Deterministic constant-response handling.
- NumPy parity fixture and generated C++ header.

## Non-goals

- Public C ABI for metrics.
- Cross-validation orchestration.
- Classification metrics.

## Validation

- `synthetic_metrics_regression_v1` is generated from NumPy formulas.
- C++ tests compare every scalar at `1e-12` absolute/relative tolerance.
- Existing ABI symbol surface remains unchanged.

## Follow-up

- Wire these kernels into k-fold / LOO / holdout CV.
- Expose public metrics result structs once the CV result ABI is designed.
- Add classification metrics for PLS-DA / OPLS-DA.
