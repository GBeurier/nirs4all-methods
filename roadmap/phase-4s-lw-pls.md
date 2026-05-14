# Phase 4s — LW-PLS

Goal: add a deterministic internal local-window PLS kernel as the first
locally-weighted PLS building block.

## Scope

- Autoscale X globally for neighbor search.
- Select k nearest neighbors per sample with stable distance/index tie-breaks.
- Refit a local NIPALS PLS model on each local window.
- Produce training predictions and the exact neighbor index plan.
- Add Python parity against the same kNN plan plus sklearn
  `PLSRegression`.

## Non-goals

- Public C ABI exposure.
- Persisted LW-PLS prediction models.
- Continuous kernel weighting, adaptive bandwidths or KD-tree acceleration.

## Acceptance

- Generated fixture validates local-window predictions and neighbor indices.
- Invalid neighbor counts and shape mismatches are rejected.
- No exported `p4a_*` symbols are added.
