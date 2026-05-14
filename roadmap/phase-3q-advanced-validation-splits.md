# Phase 3q — Advanced Validation Splits

Goal: extend the internal validation-plan layer with deterministic splitters used
by chemometric model selection workflows before exposing the final public tuning
API.

## Scope

- External fold assignments supplied by callers.
- Seeded repeated k-fold splits with balanced fold sizes.
- Seeded Monte-Carlo validation holdouts.
- Kennard-Stone one-fold calibration/validation splits from X-space coverage.
- SPXY one-fold calibration/validation splits from joint X/Y coverage.
- Python parity fixtures for exact train/test index plans.

## Non-goals

- Public C ABI for split configuration.
- Repeated/Monte-Carlo aggregation in the cross-validation result object.
- Stratified classification splitters.
- Group-aware or blocked time-series splitters.

## Acceptance

- Fixture determinism covers all new splitters.
- C++ harness compares train/test offsets and flattened indices exactly.
- No new exported `p4a_*` symbols.
- Existing k-fold CV behavior remains unchanged for exhaustive one-prediction
  plans.
