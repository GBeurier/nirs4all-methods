# Phase 5l — T2-PLS loading-weight selection

Status: shipped locally as `phase-5l-t2-selection`.

Goal: add a deterministic Hotelling T2 variable selector over PLS loading
weights, mirroring the `plsVarSel` T2-PLS selection rule while scoring subsets
through the existing k-fold CV engine.

Delivered:

- Python parity fixture `synthetic_t2_pls_hotelling_v1`.
- C++ internal `select_by_t2` kernel.
- Hotelling T2 scores from fitted PLS loading weights with rounded T2/UCL values.
- Alpha-specific UCL thresholds, top-k fallback and selected masks/counts.
- k-fold CV scoring for every alpha-selected subset, with best-error and
  minimum-set selections.
- ABI/core tests against Python/sklearn-derived T2 scores, UCLs, RMSE path,
  selected masks/counts and exact selected indices.

Deferred:

- Public C ABI wrappers for T2 result buffers.
- WVC-PLS wrapper family.
- Batched/OpenMP evaluation for larger alpha grids.
