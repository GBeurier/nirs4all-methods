# Phase 6d - AOM Whittaker and FCK strict operators

Status: shipped locally as `phase-6d-aom-whittaker-fck`.

Goal: extend the internal strict-linear AOM operator path to the next
`nirs4all/bench/AOM_v0/aompls` tranche: Whittaker smoothing and fractional
convolutional kernels.

Delivered:

- Public enum values `P4A_OP_WHITTAKER` and `P4A_OP_FCK` for AOM operator
  banks.
- Internal Whittaker smoother matching the bench `(I + lambda D^T D) z = x`
  operator.
- Internal FCK strict operator matching the bench zero-padded fractional
  convolution kernel.
- Dedicated fixture `synthetic_aom_extended_strict_operators_v1`.
- Global AOM-SIMPLS fixture
  `synthetic_aom_global_simpls_fck_whittaker_cv_v1`, covering identity,
  Whittaker, FCK and detrend in the same selector.
- Invalid-parameter tests for Whittaker/FCK strict operator requests.

Deferred:

- Public C ABI wrappers for AOM selection result buffers.
- POP-PLS / per-component AOM selection.
- Soft / sparse / superblock AOM selection.
- AOM-NIPALS and covariance/adjoint engines.
