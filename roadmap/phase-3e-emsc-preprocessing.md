# Phase 3e - EMSC Preprocessing

Goal: activate `P4A_OP_EMSC` in the live preprocessing pipeline with explicit
fit/transform separation and deterministic NumPy parity.

## Scope

- Fit stores the mean training spectrum after preceding pipeline operators as
  the EMSC reference spectrum.
- Transform solves each row against `[reference, 1, x, x^2, ...]` with
  wavelength positions normalized to `[-1, 1]`.
- Corrected output is `(row - fitted_polynomial_baseline) / multiplicative`.
- Parameters:
  - zero params: polynomial degree `2`;
  - one param: integer polynomial degree in `[0, 5]`.
- The design requires at least `degree + 2` columns.

## Acceptance

- ABI tests cover affine scatter correction against the fitted reference.
- ABI tests cover invalid degree and undersized design failures.
- NumPy parity fixture covers degree-2 EMSC with additive polynomial baseline
  and multiplicative scatter.
- `pls4all_cli --selfcheck` exercises EMSC.
- CI, parity gate, ABI surface, coverage and sanitizers are green before tag.

## Deferred

- External interferent spectra in EMSC.
- Orthogonal signal correction (`P4A_OP_OSC`) and EPO, which require supervised
  `Y` semantics and serialization decisions.
- ASLS baseline correction.
