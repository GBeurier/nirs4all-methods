# `energy_distance` — n4m.stats.energy_distance

_Namespace_: **`n4m.stats`** · _Fully-qualified_: `n4m.stats.energy_distance` · _Catalog id_: `diagnostics.energy_distance`

## API surface

**C ABI (ABI 2):** [`n4m_stats_energy_distance`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/metrics.h#L702). Use the linked public header for the exact signature, configuration, and result handles.

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Székely & Rizzo (2004), *Testing for equal distributions in high dimension*, InterStat 5; Rizzo & Székely (2016), *Energy distance*, WIREs Computational Statistics 8(1), 27–38, https://doi.org/10.1002/wics.1375; Székely & Rizzo (2013), *Energy statistics: A class of statistics based on distances*, Journal of Statistical Planning and Inference 143(8), 1249–1272, https://doi.org/10.1016/j.jspi.2013.03.018.

### Mathematical principle

With $M_{ab}$ the mean Euclidean distance over all ordered row pairs of groups $a$ and $b$ (a V-statistic, so within a group the zero self-pairs count), the energy distance between the empirical distributions is $E_{ab}=2M_{ab}-M_{aa}-M_{bb}\ge 0$. The K-sample statistic is $T=\sum_{a<b}\frac{n_an_b}{n_a+n_b}E_{ab}$, as R `energy::eqdist.etest`. The Monte Carlo test permutes the group labels and reports $(1+b)/(1+B)$ with $b$ the draws where $T_{perm}\ge T-\sqrt{\epsilon}|T|$; draws within the rounding estimate of that threshold are counted as undecidable, and a conservative p-value counts them as ties.

### Appropriate uses

Comparing the distributions of spectra (or of other coordinates) across instruments, campaigns or partitions without a bandwidth or kernel to choose.

### Limits and validation

Rows must be exchangeable under the null. Euclidean distances weight every column equally and are scale dependent. Supported domain: finite values, nonzero $|x|$ in $[2^{-256}, 2^{256}]$ and a spread of at most $2^{32}$; other inputs return `N4M_ERR_UNSUPPORTED`. Distances come from local pairwise differences; sums and the combination are double-double and each $E$ carries a conservative rounding estimate (not a certified bound). Budgets (20 million pairs, 50 million cells, 10 million group pairs, 2.5 billion operations) are checked before allocation.

### Implementation

The ABI-2 symbol `n4m_stats_energy_distance` in `n4m/metrics.h` is implemented in `cpp/src/c_api/c_api_preflight.cpp` and wrapped in JS/WASM by `statsEnergyDistance`. `parity/generators/domain_shift_v1.py` writes `parity/fixtures/domain_shift_v1.json` (SciPy `cdist`, exact rational energies and permutation statistics), checked by `cpp/tests/test_domain_shift.cpp`.

### Sources and provenance

https://doi.org/10.1002/wics.1375; https://doi.org/10.1016/j.jspi.2013.03.018; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_preflight.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/parity/fixtures/domain_shift_v1.json

## Catalog note

Energy distance (Szekely-Rizzo) between groups of rows (Euclidean, V-statistic mean distances, symmetric, with conservative rounding estimates) and the K-sample energy statistic sum n_a n_b/(n_a+n_b) E_ab of R energy::eqdist.etest, with a seeded Monte Carlo label permutation test (PCG64, numpy.random.default_rng parity; declared tie rule T_perm >= T - sqrt(eps)|T|, numerically undecidable draws counted separately with a conservative p-value). Supported domain: finite values, nonzero |x| in [2^-256, 2^256], spread <= 2^32 (refused otherwise). Reference gate: parity/generators/domain_shift_v1.py -> parity/fixtures/domain_shift_v1.json (scipy cdist, exact rational energies and permutation statistics), cpp/tests/test_domain_shift.cpp.


_See also_: [methods index](index.md).