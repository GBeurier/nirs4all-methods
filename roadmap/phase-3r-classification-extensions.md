# Phase 3r — Classification Metric Extensions

Goal: complete the Phase 3 internal classification validation kernels beyond
binary threshold metrics.

## Scope

- Multiclass confusion matrices for integer class labels and score matrices.
- Per-class sensitivity, specificity, precision, F1 and one-vs-rest AUC.
- Macro averages across classes.
- Micro precision, recall and F1.
- Binary fixed-bin calibration curves for probability-like scores.
- NumPy parity fixtures for multiclass metrics and calibration bins.

## Non-goals

- Public C ABI exposure.
- Class weighting, sample weighting or stratified split generation.
- Probabilistic calibration fitting such as Platt scaling or isotonic
  regression.

## Acceptance

- Exact fixture determinism from the pinned Python generator.
- C++ harness validates all scalar summaries, per-class rows, confusion counts
  and calibration bins.
- Existing binary classification metric parity remains unchanged.
- No exported `p4a_*` symbols are added.
