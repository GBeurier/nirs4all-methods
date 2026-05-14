# Phase 5g — Random Frog PLS subset sampling

Status: shipped locally as `phase-5g-random-frog-selection`.

Goal: add a deterministic Random Frog-style wrapper selector that can reuse the
existing NIPALS PLS regression, k-fold CV and fitted-coefficient chassis.

Delivered:

- Python parity fixture `synthetic_random_frog_pls_v1`.
- C++ internal `select_by_random_frog` kernel.
- SplitMix64-seeded add / remove / swap subset proposals.
- Metropolis-style acceptance driven by k-fold RMSE.
- Inclusion-frequency ranking with deterministic score and index tie-breaks.
- ABI/core tests against Python/sklearn-derived global scores, inclusion
  frequencies, RMSE path, subset sizes, best subset and exact selected indices.

Deferred:

- Public C ABI wrappers for Random Frog result buffers.
- SCARS and GA-PLS metaheuristics.
- Posterior inclusion uncertainty summaries and repeated-chain aggregation.
