# Phase 6e - POP per-component AOM selection

Status: shipped locally as `phase-6e-aom-pop-selection`.

Goal: add the first POP-PLS tranche against
`nirs4all/bench/AOM_v0/aompls`: greedy per-component operator selection with
SIMPLS covariance scoring and original-space orthogonalization.

Delivered:

- Internal `select_aom_per_component` kernel.
- Strict-operator matrix path for covariance application and adjoint-vector
  projection.
- Bench-compatible CV scorer for POP candidate sequences, including the
  AOM_v0 PLS1 residual broadcasting semantics used by the oracle.
- Dedicated fixture `synthetic_aom_pop_simpls_covariance_cv_v1`.
- ABI test coverage for selected operator sequence, component candidate
  scores, prefix scores and full-fit predictions.

Deferred:

- Public C ABI wrappers for AOM/POP selection result buffers.
- POP holdout, approximate-PRESS and one-SE variants.
- POP-NIPALS and materialized-engine variants.
- Soft / sparse / superblock AOM selection.
