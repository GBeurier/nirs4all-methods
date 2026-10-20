# `linear_sum_assignment` — n4m.optimize.linear_sum_assignment

_Namespace_: **`n4m.optimize`** · _Fully-qualified_: `n4m.optimize.linear_sum_assignment` · _Catalog id_: `diagnostics.linear_sum_assignment`

## API surface

**C ABI (ABI 2):** [`n4m_linear_sum_assignment`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/metrics.h#L282). Use the linked public header for the exact signature, configuration, and result handles.

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Crouse (2016), *On implementing 2D rectangular assignment algorithms*, IEEE Transactions on Aerospace and Electronic Systems 52(4), 1679–1696, https://doi.org/10.1109/TAES.2016.140952; Kuhn (1955), *The Hungarian method for the assignment problem*, Naval Research Logistics Quarterly 2, 83–97, https://doi.org/10.1002/nav.3800020109.

### Mathematical principle

For a cost matrix $C$ with $r$ rows and $c$ columns, the kernel finds a one-to-one assignment of $\min(r,c)$ row/column pairs that minimizes (or maximizes) $\sum C_{i,\sigma(i)}$, by successive shortest augmenting paths with dual variables. A $+\infty$ cost forbids a pair.

### Appropriate uses

Matching clusters across partitions by maximum total Jaccard overlap, and other one-to-one assignment problems.

### Limits and validation

Inside the declared numerical domain (nonzero |cost| in [2^-256, 2^256], spread at most 2^32; other inputs are refused) duals and path costs are exact 192-bit fixed-point integers, so the result is an exact optimum even where SciPy's double arithmetic loses small differences next to large costs. Among optimal assignments, the returned one follows SciPy's tie order (columns scanned from last to first, unassigned columns first on equal reduced cost), so equal-cost solutions are reproducible but arbitrary. NaN and $-\infty$ costs are invalid; a matrix without a feasible assignment returns `N4M_ERR_NUMERICAL_FAILURE`. Cost is $O(rc\min(r,c))$ exact-arithmetic operations and is limited to 1e8.

### Implementation

The ABI-2 symbol `n4m_linear_sum_assignment` in `n4m/metrics.h` is a port of SciPy 1.18.1 `scipy.optimize.linear_sum_assignment` (BSD-3-Clause) in `cpp/src/core/optimization/_vendored/rectangular_lsap.cpp`, wrapped in JS/WASM by `linearSumAssignment`. `parity/fixtures/cluster_matching_null_v1.json` pins SciPy assignments on square, wide, tall, tied, forbidden and maximized matrices, checked exactly by `cpp/tests/test_cluster_matching_null.cpp`.

### Sources and provenance

https://doi.org/10.1109/TAES.2016.140952; https://doi.org/10.1002/nav.3800020109; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/optimization/_vendored/rectangular_lsap.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/parity/fixtures/cluster_matching_null_v1.json

## Catalog note

Rectangular linear sum assignment ported from SciPy 1.18.1 scipy.optimize.linear_sum_assignment (Crouse shortest augmenting path, BSD-3-Clause, cpp/src/core/optimization/_vendored), with SciPy tie order and exact 192-bit fixed-point duals, so the result is an exact optimum; rule-13 numerical domain. Reference gate: parity/generators/cluster_matching_null_v1.py -> parity/fixtures/cluster_matching_null_v1.json, cpp/tests/test_cluster_matching_null.cpp.


_See also_: [methods index](index.md).