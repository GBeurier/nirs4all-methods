# Phase 6c - AOM strict-linear operator tranche

Status: shipped locally as `phase-6c-aom-strict-operators`.

Goal: align the internal AOM operator path with the bench reference under
`nirs4all/bench/AOM_v0/aompls`, rather than reusing the general preprocessing
pipeline where boundary semantics differ.

Delivered:

- Internal strict-linear AOM operator module.
- Bench-parity transforms for identity, polynomial detrend, zero-padded
  Savitzky-Golay smoothing/derivatives, finite differences and
  Norris-Williams gap derivatives.
- Public enum value `P4A_OP_FINITE_DIFFERENCE` for AOM operator banks.
- Dedicated fixture `synthetic_aom_strict_operators_v1`.
- Wider global AOM-SIMPLS fixture
  `synthetic_aom_global_simpls_sg_cv_v1`, covering identity, SG, FD,
  Norris-Williams and detrend in the same selector.
- AOM global selection now uses strict AOM kernels instead of the general
  preprocessing pipeline semantics.

Deferred:

- Strict-linear Whittaker and FCK operators.
- Public C ABI wrappers for AOM selection result buffers.
- Per-component / soft / sparse / superblock AOM selection.
- AOM-NIPALS and covariance/adjoint engines.
