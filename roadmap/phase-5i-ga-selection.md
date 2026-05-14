# Phase 5i — GA-PLS deterministic population search

Status: shipped locally as `phase-5i-ga-selection`.

Goal: add a deterministic GA-PLS wrapper selector that evolves bounded feature
subsets with k-fold CV fitness over the existing NIPALS PLS regression chassis.

Delivered:

- Python parity fixture `synthetic_ga_pls_wrapper_v1`.
- C++ internal `select_by_ga` kernel.
- SplitMix64-seeded population initialization from fitted PLS coefficient
  scores.
- Deterministic elitism, score-biased crossover and add/remove/swap mutation.
- Per-generation best RMSE, mean RMSE, best subset size and population
  inclusion-frequency summaries.
- ABI/core tests against Python/sklearn-derived global scores, inclusion
  frequencies, RMSE paths, subset sizes and exact selected indices.

Deferred:

- Public C ABI wrappers for GA-PLS result buffers.
- Shaving, BVE, T2 and WVC wrapper families.
- Larger population stress tests and batched/OpenMP fitness evaluation.
