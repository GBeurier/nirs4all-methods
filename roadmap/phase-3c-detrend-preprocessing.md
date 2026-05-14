# Phase 3c - Polynomial Detrending

Goal: add row-wise polynomial detrending to the live preprocessing pipeline
using the parameter-copying surface already present in the C ABI.

## Scope

- `P4A_OP_DETREND_POLY` inside `p4a_pipeline_fit` /
  `p4a_pipeline_transform`.
- Zero parameters means degree 1. One parameter selects an integer degree in
  `[0, 5]`.
- Wavelength positions are normalised to `[-1, 1]`; each row is least-squares
  fitted against a polynomial basis and the fitted trend is subtracted.
- Degree must be smaller than the number of columns.

## Parity

- Add one deterministic NumPy fixture using `np.linalg.lstsq` for degree-2
  detrending.
- Regenerate `cpp/tests/fixtures/pipeline_fixtures.hpp` with operator params.
- C++ pipeline parity tests pass copied params through `p4a_pipeline_add_operator`.

## Acceptance

- `P4A_OP_DETREND_POLY` fits and transforms with default and explicit degree
  paths.
- `generate-fixtures --check`, release CTest, CLI selfcheck, ABI symbol diff,
  dependency audit and sanitizer builds pass locally.
- GitHub CI, ABI Surface, Sanitizers, Coverage and Parity gate pass on `main`.

## Deferred

- Savitzky-Golay smoothing / derivatives.
- Baseline algorithms such as ASLS.
- Wavelength-axis metadata; current detrending uses evenly spaced feature
  indices normalised to `[-1, 1]`.
