# Phase 3n - Classification Metrics

Goal: add deterministic binary classification metric kernels for PLS-DA and
OPLS-DA style continuous score outputs.

## Scope

- Accept binary 0/1 labels and continuous scores through `p4a_matrix_view_t`.
- Compute confusion counts at an explicit threshold.
- Report sensitivity, specificity, balanced accuracy, accuracy, precision, F1,
  Matthews correlation coefficient and ROC AUC.
- Use average-rank AUC so tied scores are deterministic.
- Add a NumPy parity fixture with integer labels, continuous scores and tied
  scores.

## Non-goals

- No public C ABI expansion yet.
- No multi-class macro/micro aggregation yet.
- No threshold optimisation or calibration curves yet.

## Acceptance

- Fixture generation is deterministic.
- C++ metrics match NumPy references within documented tolerance.
- Invalid shapes, dtypes, labels, thresholds, non-finite scores and one-class
  labels are rejected deterministically.
- ABI symbol export remains unchanged.
