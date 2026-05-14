# Phase 3b - MSC Preprocessing

Goal: add multiplicative scatter correction to the live preprocessing pipeline
without changing the C ABI.

## Scope

- `P4A_OP_MSC` inside `p4a_pipeline_fit` / `p4a_pipeline_transform`.
- The fitted reference spectrum is the column-wise mean spectrum of the fit
  matrix after preceding pipeline operators have been applied.
- Transform regresses each sample against that fitted reference spectrum and
  returns `(x - intercept) / slope`.
- Degenerate reference spectra or per-row zero slopes return
  `P4A_ERR_NUMERICAL_FAILURE` with a context error.

## Parity

- Add one deterministic NumPy MSC fixture and regenerate the pipeline fixture
  header.
- C++ pipeline parity tests consume the generated fixture alongside identity,
  center, autoscale, Pareto and SNV.

## Acceptance

- `P4A_OP_MSC` fits and transforms through both caller-owned and allocated
  output paths.
- `generate-fixtures --check`, release CTest, CLI selfcheck, ABI symbol diff,
  dependency audit and sanitizer builds pass locally.
- GitHub CI, ABI Surface, Sanitizers, Coverage and Parity gate pass on `main`.

## Deferred

- EMSC with polynomial / external interferent terms.
- Optional user-provided MSC reference spectrum.
- Model-fit integration of fitted preprocessing state.
