# Phase 5h — SCARS-PLS stability adaptive reweighted sampling

Status: shipped locally as `phase-5h-scars-selection`.

Goal: add a deterministic SCARS-style wrapper selector that combines
calibration subsampling, stability-weighted CARS retention and k-fold CV subset
scoring over the existing NIPALS PLS regression chassis.

Delivered:

- Python parity fixture `synthetic_scars_pls_stability_v1`.
- C++ internal `select_by_scars` kernel.
- SplitMix64-seeded deterministic calibration-row subsampling.
- Per-iteration PLS coefficient scoring over sampled rows and active features.
- Stability-normalized feature scores accumulated across adaptive retention
  iterations.
- Candidate subset scoring through the internal k-fold CV engine.
- ABI/core tests against Python/sklearn-derived coefficient score paths,
  stability scores, retained counts, RMSE path, sample count, best iteration
  and exact selected indices.

Deferred:

- Public C ABI wrappers for SCARS result buffers.
- GA-PLS and shaving/BVE-style metaheuristics.
- Repeated-run uncertainty summaries for SCARS stability scores.
