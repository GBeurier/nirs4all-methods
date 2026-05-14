# Phase 3l - Validation Splits

Goal: add deterministic internal train/test split plans for the future
cross-validation engine.

## Scope

- C++17 internal `core::ValidationPlan`.
- Balanced contiguous k-fold splits.
- Leave-one-out splits.
- Contiguous holdout splits.
- Shape/range validation and deterministic error reporting.
- Python parity fixtures with generated C++ index headers.

## Non-goals

- Public C ABI for validation plans.
- Model fitting across folds.
- Randomized/repeated CV, Kennard-Stone, SPXY or Monte-Carlo CV.

## Validation

- `synthetic_validation_kfold_balanced_v1`.
- `synthetic_validation_leave_one_out_v1`.
- `synthetic_validation_holdout_v1`.
- C++ tests compare exact train/test index offsets and flattened indices.

## Follow-up

- Wire split plans to model fit/predict loops.
- Feed predictions into Phase 3k regression metrics.
- Add deterministic shuffle/repeated splitters with explicit seed contracts.
