# Phase 5n — EMCUVE ensemble MC-UVE selection

Status: shipped locally as `phase-5n-emcuve-selection`.

Goal: add a deterministic Ensemble Monte-Carlo UVE selector that combines
multiple artificial-variable UVE members with a vote rule, while preserving the
existing internal-only variable-selection surface.

Delivered:

- Python parity fixture `synthetic_emcuve_pls_ensemble_v1`.
- C++ internal `select_by_emcuve` kernel.
- Deterministic ensemble seed schedule over MC-UVE/UVE members.
- Per-feature vote frequencies, mean real-variable stability scores, per-member
  noise thresholds and exact selected-variable ordering.
- ABI/core tests against the Python/sklearn reference for vote frequencies,
  stability means, thresholds and selected indices.

Deferred:

- Public C ABI wrappers for EMCUVE result buffers.
- Robust EMCUVE variants using median / median absolute deviation stability
  descriptors.
- Batched/OpenMP evaluation for larger ensemble sizes.

