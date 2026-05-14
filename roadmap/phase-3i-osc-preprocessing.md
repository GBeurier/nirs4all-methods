# Phase 3i - OSC Preprocessing

Goal: activate `P4A_OP_OSC` as a supervised orthogonal signal correction
operator inside the fitted preprocessing pipeline.

## Scope

- One-component OSC reference implementation.
- `p4a_pipeline_fit` now passes the optional `Y` matrix into the C++ pipeline
  core; unsupervised operators still accept `Y == NULL`.
- OSC requires a non-null `Y` at fit time and stores only training-derived
  filter state for transform-time use.
- The fitted direction is the dominant direction of the part of centered `X`
  orthogonal to centered `Y`.
- Parameters:
  - zero params: max_iter `100`, tol `1e-10`;
  - two params: `max_iter`, `tol`.
- Validation:
  - `Y` rows must match `X` rows;
  - `Y` must have at least one column;
  - `max_iter` integer in `[1, 1000]`;
  - `tol` finite and positive;
  - singular `Y.T @ Y` is rejected.

## Acceptance

- ABI tests cover missing `Y`, invalid params and singular `Y`.
- NumPy parity fixture covers supervised one-component OSC against the C++
  pipeline.
- `pls4all_cli --selfcheck` exercises OSC.
- CI, parity gate, ABI surface, coverage and sanitizers are green before tag.

## Deferred

- Multi-component OSC with explicit component count.
- Alternative OSC algorithms and convergence diagnostics.
- EPO preprocessing.
