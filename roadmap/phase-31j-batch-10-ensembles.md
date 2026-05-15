# Phase 31j — Batch 10: §20 PLS ensembles

**Status:** shipped — local tag `phase-31j-batch-10-ensembles`
(`0.80.0+abi.1.10.0`).

## Methods (all paper-only)

| Method | Internal kernel | Public C ABI entry | Paper |
|--------|-----------------|--------------------|-------|
| Bagging PLS | `fit_bagging_pls` | `p4a_bagging_pls_fit` | Breiman 1996 |
| Boosting PLS | `fit_boosting_pls` | `p4a_boosting_pls_fit` | Friedman 2001 |
| Random-subspace PLS | `fit_random_subspace_pls` | `p4a_random_subspace_pls_fit` | Ho 1998 |

## Why paper-only

Ensemble methods use a random number generator to draw bootstrap
indices, sample weights, or feature subsets. pls4all uses
`std::mt19937` seeded by `uint64_t seed`; sklearn uses NumPy's
`PCG64`; R `caret` uses R's Mersenne-Twister. Even with the same
seed, the bit streams differ, so the bootstrap indices / feature
subsets are different. Numerical parity is therefore impossible
without sharing the exact RNG implementation — and would force pls4all
to depend on a specific external RNG (which violates the zero-deps
policy).

The papers are cited as the algorithmic reference; the parity gate
smoke-fits the pls4all entry point to confirm it runs.

## ABI delta

- 3 new public symbols, total 154 (sorted, all `p4a_*` prefixed).
- C ABI minor 9 → 10.

## Local gate

- 256 internal C++ tests.
- ABI symbol diff: clean.
- `git diff --check`: clean.
- Parity gate: 13 external PASS, 19 paper-only smoke PASS, 0 numpy.
