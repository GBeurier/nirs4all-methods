# Phase 4r — MB-PLS

Goal: add a deterministic internal multi-block PLS kernel for block-structured
spectroscopy and chemometrics workflows.

## Scope

- Accept an explicit block-size partition over X columns.
- Autoscale each feature within its block.
- Weight each block by `1 / sqrt(block_width)` before fitting the shared PLS
  model.
- Fit NIPALS PLS regression on the weighted block matrix.
- Return training predictions, original-scale coefficients, intercept,
  original feature means/scales and block weights.
- Add Python parity against a NumPy block transform plus sklearn
  `PLSRegression(scale=False)`.

## Non-goals

- Public C ABI exposure.
- Persisted MB-PLS prediction models.
- Super-score / hierarchical MB-PLS variants or per-block latent diagnostics.

## Acceptance

- Generated fixture validates predictions, original-scale coefficients,
  intercept, feature preprocessing stats and block weights.
- Invalid block layouts and shape mismatches are rejected.
- No exported `p4a_*` symbols are added.
