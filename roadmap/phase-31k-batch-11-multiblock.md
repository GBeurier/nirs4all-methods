# Phase 31k — Batch 11: §17-19 multi-block remainder

**Status:** shipped — local tag `phase-31k-batch-11-multiblock`
(`0.81.0+abi.1.11.0`).

## Methods (all paper-only)

| Method | Internal kernel | Public C ABI entry | Paper |
|--------|-----------------|--------------------|-------|
| SO-PLS | `fit_so_pls` | `p4a_so_pls_fit` | Næs et al. 2011 |
| OnPLS | `fit_on_pls` | `p4a_on_pls_fit` | Löfstedt & Trygg 2011 |
| ROSA | `fit_rosa` | `p4a_rosa_fit` | Liland et al. 2016 |

## Why paper-only

R `multiblock` ships `sopls`, `onpls` and `rosa`, but its transitive
dep `HDANOVA` fails to build in the minimal conda R env on this host.
No Python port is widely installable.

The C ABI accepts an array of `p4a_matrix_view_t` block descriptors
plus per-block component counts; per-block outputs (loadings, scores,
coefficients) are namespaced by block index via the universal result
handle.

## ABI delta

- 3 new public symbols, total 157.
- C ABI minor 10 → 11.

## Local gate

- 256 internal C++ tests.
- Parity gate: 13 external PASS, 22 paper-only smoke PASS, 0 numpy.
