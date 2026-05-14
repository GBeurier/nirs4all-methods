# Phase 3j - EPO Preprocessing

Goal: activate `P4A_OP_EPO` as a supervised external parameter
orthogonalization operator inside the fitted preprocessing pipeline.

## Scope

- One-component EPO reference implementation.
- `Y` is interpreted as the external/interferent parameter matrix.
- The fitted direction is the dominant left singular direction of centered
  `X.T @ Y`.
- Transform applies the fitted projection `X - ((X - mean) @ direction) *
  direction.T` without needing `Y`.
- Parameters:
  - zero params: max_iter `100`, tol `1e-10`;
  - two params: `max_iter`, `tol`.
- Validation:
  - `Y` rows must match `X` rows;
  - `Y` must have at least one column;
  - `max_iter` integer in `[1, 1000]`;
  - `tol` finite and positive;
  - zero external covariance is rejected.

## Acceptance

- ABI tests cover missing `Y`, invalid params and zero external covariance.
- NumPy parity fixture covers supervised one-component EPO against the C++
  pipeline.
- `pls4all_cli --selfcheck` exercises EPO.
- CI, parity gate, ABI surface, coverage and sanitizers are green before tag.

## Deferred

- Multi-component EPO with explicit component count.
- EPO variants based on externally supplied difference spectra.
- Calibration-transfer methods such as PDS and DS.
