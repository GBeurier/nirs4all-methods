# Phase 4n - PLSSVD

Goal: turn the reserved `P4A_ALGO_PLS_SVD` ABI enum into a live direct
cross-covariance score model without adding public symbols.

## Scope

- `P4A_ALGO_PLS_SVD` + `P4A_SOLVER_SVD` + `P4A_DEFLATION_CANONICAL`.
- Rank guard follows PLSSVD semantics: `n_components <= min(n_features, n_targets, n_samples)`.
- `transform` returns X scores from the dominant singular vectors of `X.T @ Y`.
- `predict` is a deterministic pls4all convention: latent projection through
  `X @ W @ V.T`, scaled back to the response space.
- Predict / transform / fitted-array accessors / binary import-export use the
  existing model layout with `weights_W` as X singular vectors and
  `y_loadings_Q` as Y singular vectors.

## Parity

- Add two deterministic PLSSVD fixtures generated from a NumPy mirror of
  scikit-learn 1.4 `PLSSVD`.
- C++ ABI tests assert projection predictions, preprocessing statistics,
  weights, loadings, rotations, transform scores and serialization round-trip.
- Python binding smoke exercises `Algorithm.PLS_SVD`, `Solver.SVD` and
  `Deflation.CANONICAL`.

## Acceptance

- `P4A_ALGO_PLS_SVD + P4A_SOLVER_SVD + P4A_DEFLATION_CANONICAL` fits
  multi-target `Y`.
- `generate-fixtures --check`, release CTest, CLI selfcheck, ABI symbol diff,
  dependency audit and sanitizer builds pass locally.
- GitHub CI, ABI Surface, Sanitizers, Coverage and Parity gate pass on `main`.

## Deferred

- Public Y-score transform output through a dedicated binding-level API.
- External parity against downstream PLSSVD APIs once the R generator is active.
- Benchmarking PLSSVD against scikit-learn and vectorised NumPy baselines.
