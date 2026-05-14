# Phase 3a - Preprocessing Pipeline

Goal: flip the existing `p4a_pipeline_fit` / `p4a_pipeline_transform` ABI from
Phase 0 stubs to a leakage-safe preprocessing pipeline for the first NIRS
operators.

## Scope

- `P4A_OP_IDENTITY`, `P4A_OP_CENTER`, `P4A_OP_AUTOSCALE`,
  `P4A_OP_PARETO_SCALE` and `P4A_OP_SNV`.
- Explicit `fit` / `transform` separation: column statistics are learned from
  the fit matrix and reused on transform matrices.
- `p4a_pipeline_transform_alloc` returns a core-owned `p4a_array_t` using the
  same shape as `X`.
- Unsupported operators remain addable but return `P4A_ERR_NOT_IMPLEMENTED` at
  fit time with a context error.

## Parity

- Add deterministic NumPy fixtures for identity, center, autoscale, Pareto
  scaling and SNV.
- Generate `cpp/tests/fixtures/pipeline_fixtures.hpp` from those fixtures.
- C++ ABI tests assert transform outputs against the NumPy reference and cover
  not-fitted, unsupported-operator and caller-provided-output paths.

## Acceptance

- `p4a_pipeline_fit`, `p4a_pipeline_transform` and
  `p4a_pipeline_transform_alloc` are live for the Phase 3a operator subset.
- `generate-fixtures --check`, release CTest, CLI selfcheck, ABI symbol diff,
  dependency audit and sanitizer builds pass locally.
- GitHub CI, ABI Surface, Sanitizers, Coverage and Parity gate pass on `main`.

## Deferred

- MSC / EMSC, polynomial detrend, Savitzky-Golay, ASLS baseline and OSC.
- Persisting pipeline state inside model serialization.
- Applying `cfg.pipeline` automatically inside `p4a_model_fit`; this needs a
  model-level serialization contract for preprocessing state.
