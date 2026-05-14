# Phase 4p — PLS-LDA

Goal: add a deterministic internal PLS-LDA classifier kernel for the advanced
discriminant-analysis track.

## Scope

- Fit PLS dummy-response scores from labelled classes.
- Fit pooled-covariance LDA in the latent score space.
- Produce training decision scores and class predictions.
- Add Python parity against sklearn PLS scores plus a NumPy LDA reference.

## Non-goals

- Public C ABI exposure.
- Prediction on external matrices through a persisted PLS-LDA model.
- Shrinkage LDA, QDA or logistic calibration.

## Acceptance

- Generated fixture validates predictions and decision scores.
- Invalid class counts, missing classes and shape mismatches are rejected.
- No exported `p4a_*` symbols are added.
