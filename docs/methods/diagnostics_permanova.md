# `permanova` — n4m.stats.permanova

_Namespace_: **`n4m.stats`** · _Fully-qualified_: `n4m.stats.permanova` · _Catalog id_: `diagnostics.permanova`

## API surface

**C ABI (ABI 2):** [`n4m_stats_permanova`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/metrics.h#L323). Use the linked public header for the exact signature, configuration, and result handles.

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Anderson (2001), *A new method for non-parametric multivariate analysis of variance*, Austral Ecology 26, 32–46, https://doi.org/10.1111/j.1442-9993.2001.01070.pp.x; McArdle & Anderson (2001), *Fitting multivariate models to community data: a comment on distance-based redundancy analysis*, Ecology 82, 290–297, https://doi.org/10.1890/0012-9658(2001)082[0290:FMMTCD]2.0.CO;2; Anderson & ter Braak (2003), *Permutation tests for multi-factorial analysis of variance*, Journal of Statistical Computation and Simulation 73, 85–113, https://doi.org/10.1080/00949650215733.

### Mathematical principle

With Euclidean distances the PERMANOVA sums of squares are traces of projections of the centered coordinates. For a term given strata in the additive model strata + term, $SS_{term}=SS(strata+term)-SS(strata)$ is computed from the reduced normal equations $M_{ab}=\delta_{ab}n_a-\sum_s n_{as}n_{bs}/n_s$ after dropping one reference level per connected component of the level-stratum occupancy graph, whose count $c$ gives the identifiable df $=$ levels $-\,c$ (0 when the term is aliased with the strata). pseudo-$F=(SS_{term}/df_{term})/(SS_{res}/df_{res})$. The Monte Carlo test permutes rows only inside their stratum, counts permuted $F\ge F_{obs}-\sqrt{\epsilon}$ as vegan `anova.cca`, and reports $(1+b)/(1+B)$.

### Appropriate uses

Exploratory variance partition of a multivariate structure (for example PCA scores of spectra) by categorical design factors such as instrument or campaign, with explicit detection of factors that cannot be separated from the others.

### Limits and validation

The test assumes exchangeable rows inside each stratum: it tests the term together with any interaction with the strata, and a stratum where the term is constant contributes nothing. A significant result can reflect dispersion as well as location differences (see PERMDISP). Only the Euclidean metric on the supplied coordinates is offered. Terms above 2048 levels and workloads above 250 million operations return `N4M_ERR_UNSUPPORTED`.

### Implementation

The ABI-2 symbol `n4m_stats_permanova` in `n4m/metrics.h` is implemented in `cpp/src/c_api/c_api_preflight.cpp` and wrapped in JS/WASM by `statsPermanova`. Draws reuse the PCG64 generator seeded like `numpy.random.default_rng`. `parity/generators/permanova_v1.py` writes `parity/fixtures/permanova_v1.json` (NumPy least squares on full one-hot designs, scikit-bio `permanova`, NumPy permutation counts), checked by `cpp/tests/test_permanova.cpp`.

### Sources and provenance

https://doi.org/10.1111/j.1442-9993.2001.01070.pp.x; https://doi.org/10.1890/0012-9658(2001)082[0290:FMMTCD]2.0.CO;2; https://doi.org/10.1080/00949650215733; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_preflight.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/parity/fixtures/permanova_v1.json

## Catalog note

PERMANOVA (Anderson 2001) of one categorical term on Euclidean distances between rows, optionally given strata (additive strata + term SS, the vegan adonis2 by='margin' SS for one other factor), with exact identifiable df from the connected components of the level-stratum graph and a seeded within-strata Monte Carlo test (PCG64, numpy.random.default_rng parity). Reference gate: parity/generators/permanova_v1.py -> parity/fixtures/permanova_v1.json, cpp/tests/test_permanova.cpp.


_See also_: [methods index](index.md).