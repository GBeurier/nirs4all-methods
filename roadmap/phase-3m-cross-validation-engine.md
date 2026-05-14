# Phase 3m - Cross-Validation Engine

Goal: turn the Phase 3k metrics and Phase 3l validation plans into a reusable,
leakage-safe regression cross-validation kernel.

## Scope

- Add an internal C++ cross-validation engine that accepts a fitted-model config,
  `X`, `Y` and a `ValidationPlan`.
- For each fold, materialise fold-local train/test matrices, fit the requested
  regression model on training rows only, predict test rows only, then stitch the
  out-of-sample predictions back into original sample order.
- Compute aggregate regression metrics by reusing the Phase 3k metric kernel.
- Preserve the fold test-index plan in the result for downstream diagnostics.
- Add sklearn `PLSRegression` parity fixtures for deterministic k-fold PLS1 and
  PLS2 cases.

## Non-goals

- No public C ABI expansion yet; this stays internal until the binding/API shape
  is stable.
- No Kennard-Stone, SPXY, repeated k-fold or Monte-Carlo split generation in this
  phase.
- No parallel fold scheduler yet.

## Acceptance

- Fixture generation is deterministic.
- C++ tests match sklearn out-of-sample predictions and aggregate metrics within
  the documented tolerance.
- Malformed validation plans are rejected before producing partial results.
- ABI symbol export remains unchanged.
