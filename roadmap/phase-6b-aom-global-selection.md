# Phase 6b — AOM global SIMPLS selection

Status: shipped locally as `phase-6b-aom-global-selection`.

Goal: add the first full AOM selection kernel: score a global operator bank by
cross-validated SIMPLS prefix RMSE, select one operator plus component count,
and produce full-fit predictions. The parity oracle is the bench implementation
under `nirs4all/bench/AOM_v0/aompls`.

Delivered:

- Bench-derived parity fixture `synthetic_aom_global_simpls_cv_v1`.
- C++ internal `select_aom_global` kernel.
- Explicit-fold CV scoring with per-operator/per-component RMSE curves.
- Deterministic best operator and component-prefix selection.
- Full-fit prediction path for the selected operator/component count.
- ABI/core tests against bench AOM_v0 selection curves, scores and predictions.

Scope note:

- The shipped parity tranche covers identity + polynomial detrend strict-linear
  operators. Savitzky-Golay in bench AOM_v0 uses a strict zero-padded operator,
  while the existing pls4all preprocessing pipeline keeps its current
  preprocessing semantics; strict bench-operator ports are deferred to the next
  AOM operator-parity tranche.

Deferred:

- Public C ABI wrappers for AOM selection result buffers.
- Strict-linear bench SG / finite-difference / Norris-Williams operator ports.
- One-SE rule and covariance/hybrid selection criteria.
- POP per-component operator selection.
