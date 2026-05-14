# Phase 4q — PLS-logistic

Goal: add a deterministic internal PLS-logistic classifier kernel for the
advanced discriminant-analysis track.

## Scope

- Fit PLS dummy-response scores from labelled classes.
- Fit a baseline-parameterized multinomial logistic model in latent score space.
- Use fixed L2 ridge regularization for stable Newton updates.
- Produce training decision scores, class probabilities, class predictions and
  latent-space logistic coefficients.
- Add Python parity against sklearn PLS scores plus a NumPy logistic reference.

## Non-goals

- Public C ABI exposure.
- Prediction on external matrices through a persisted PLS-logistic model.
- Alternative solvers, class weights or probability calibration.

## Acceptance

- Generated fixture validates predictions, decision scores, probabilities,
  intercepts and logistic coefficients.
- Invalid class counts, missing classes and shape mismatches are rejected.
- No exported `p4a_*` symbols are added.
