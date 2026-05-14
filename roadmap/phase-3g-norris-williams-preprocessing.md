# Phase 3g - Norris-Williams Preprocessing

Goal: activate `P4A_OP_NORRIS_WILLIAMS` as a gap-segment derivative operator
inside the fitted preprocessing pipeline.

## Scope

- Deterministic gap-segment derivative filters for derivative orders 1 to 4.
- Default params: segment size `5`, gap size `3`, derivative order `1`.
- Explicit params: `segment`, `gap`, `derivative_order`.
- Segment and gap sizes must be odd integers in `[1, 101]`.
- Filter length is bounded to `501`.
- Edge handling uses nearest-value extension, matching the Phase 3d
  Savitzky-Golay edge convention.
- Coefficients follow the standard gap-segment form:
  - order 1: `[-1_segment, gap_zeros, +1_segment] / segment`;
  - higher orders use binomial finite-difference segment blocks.

## Acceptance

- ABI tests cover first-order gap derivative output and parameter validation.
- NumPy parity fixture covers the default segment/gap/order path.
- `pls4all_cli --selfcheck` exercises Norris-Williams preprocessing.
- CI, parity gate, ABI surface, coverage and sanitizers are green before tag.

## Deferred

- Wavelength-spacing scaling (`delta.wav`) as an explicit param.
- Wavelet denoising.
- OSC / EPO supervised preprocessing.
