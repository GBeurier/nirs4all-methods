# Phase 3f - ASLS Baseline Preprocessing

Goal: activate `P4A_OP_ASLS_BASELINE` in the live preprocessing pipeline for
common NIRS baseline correction.

## Scope

- Dense reference implementation of asymmetric least-squares baseline
  correction using the second-difference penalty matrix.
- Row-wise iterative weights:
  - low weight `p` for positive residual peaks;
  - high weight `1 - p` for points at or below the baseline.
- Corrected output is `row - baseline`.
- Parameters:
  - zero params: lambda `100000`, asymmetry `0.001`, iterations `10`;
  - three params: `lambda`, `asymmetry`, `iterations`.
- Validation:
  - lambda finite and positive;
  - asymmetry finite and in `(0, 1)`;
  - iterations integer in `[1, 100]`;
  - at least 3 columns.

## Acceptance

- ABI tests cover constant-baseline removal and parameter validation.
- NumPy parity fixture covers peaked spectra on a curved baseline.
- `pls4all_cli --selfcheck` exercises ASLS.
- CI, parity gate, ABI surface, coverage and sanitizers are green before tag.

## Deferred

- Sparse/banded solver optimization for long spectra.
- OSC / EPO supervised preprocessing.
- Wavelet denoising and Norris-Williams derivative variants.
