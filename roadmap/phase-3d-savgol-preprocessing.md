# Phase 3d - Savitzky-Golay Preprocessing

Goal: add Savitzky-Golay smoothing and derivative operators to the live
preprocessing pipeline while keeping the C ABI unchanged.

## Scope

- `P4A_OP_SAVGOL_SMOOTH` inside `p4a_pipeline_fit` /
  `p4a_pipeline_transform`.
- `P4A_OP_SAVGOL_DERIVATIVE` inside the same fitted pipeline contract.
- Stateless fitted operator state containing validated window length,
  polynomial degree, derivative order and delta.
- Nearest-edge padding and SciPy parity fixtures generated with
  `scipy.signal.savgol_filter(..., mode="nearest")`.

## Parameter Contract

- Smooth:
  - zero params: window `5`, polynomial degree `2`;
  - two params: `window`, `poly_degree`.
- Derivative:
  - zero params: window `5`, polynomial degree `2`, derivative order `1`,
    delta `1.0`;
  - three or four params: `window`, `poly_degree`, `derivative_order`,
    optional `delta`.
- Window length must be an odd integer in `[3, 501]`.
- Polynomial degree must be non-negative and smaller than the window length.
- Derivative order is `1` or `2` for `P4A_OP_SAVGOL_DERIVATIVE` and must not
  exceed the polynomial degree.
- Delta must be finite and positive.

## Acceptance

- C++ ABI tests cover the default derivative path, smoothing of a quadratic
  interior point and invalid parameter failures.
- Python/SciPy fixture parity covers smoothing and first-derivative outputs.
- `pls4all_cli --selfcheck` exercises a smooth + derivative pipeline.
- CI, parity gate, ABI surface, coverage and sanitizers are green before tag.

## Deferred

- External interferent spectra for EMSC; fitted-reference polynomial EMSC
  shipped in Phase 3e.
- ASLS baseline correction.
- OSC supervised preprocessing shipped in Phase 3i; EPO still requires
  dedicated supervised preprocessing work.
