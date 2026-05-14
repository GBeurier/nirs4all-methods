# Phase 5o — PLS randomization-test selection

Status: shipped locally as `phase-5o-randomization-selection`.

Goal: add a deterministic PLS variable selector based on randomization tests:
fit the observed model, build a null distribution from deterministic
permutations of `Y`, compute empirical p-values per variable, and select the
features passing `alpha`.

Delivered:

- Python parity fixture `synthetic_randomization_pls_permutation_v1`.
- C++ internal `select_by_randomization_test` kernel.
- Deterministic SplitMix64 Fisher-Yates permutations of response rows.
- Original-scale PLS coefficient magnitude scores, empirical smoothed p-values,
  exceedance counts and exact selected-variable ordering.
- ABI/core tests against the Python/sklearn reference for observed scores,
  p-values, exceedance counts and selected indices.

Deferred:

- Public C ABI wrappers for randomization-test result buffers.
- Alternative null models based on response sign flips or residual
  permutations.
- Batched/OpenMP permutation evaluation.

