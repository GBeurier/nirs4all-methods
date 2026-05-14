# Phase 6a — AOM preprocessing-bank primitive

Status: shipped locally as `phase-6a-aom-preprocessing`.

Goal: add the first internal Phase 6 building block: apply a fitted
`OperatorBank` as an AOM preprocessing bank, expose every operator output, the
gating weights and the mixed transform, and keep the scope below full AOM-PLS
model fitting until the bench reference parity fixtures are wired.

Delivered:

- Python parity fixtures `synthetic_aom_soft_preprocessing_v1` and
  `synthetic_aom_hard_preprocessing_v1`.
- C++ internal `apply_aom_preprocessing` kernel.
- Soft equal-weight gating across every operator in the bank.
- Hard first-operator gating for deterministic discrete-selection plumbing.
- Operator-major transformed blocks, operator-kind trace and final mixed
  transform outputs.
- ABI/core tests against the NumPy fixture and invalid-request handling.

Reference policy:

- Full AOM-PLS parity is anchored on `nirs4all/bench/AOM_v0/aompls`, especially
  global selection curves, one-SE component prefix selection and predictions.
- The packaged `nirs4all` model surface is not the parity oracle for AOM-PLS.

Deferred:

- Public C ABI wrappers for AOM preprocessing result buffers.
- Global AOM-SIMPLS fitted model state and predict path.
- Sparse gating and POP per-component operator selection.
- Bench-derived parity fixtures for AOM selection curves and predictions.
