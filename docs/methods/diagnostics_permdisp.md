# `permdisp` — n4m.stats.permdisp

_Namespace_: **`n4m.stats`** · _Fully-qualified_: `n4m.stats.permdisp` · _Catalog id_: `diagnostics.permdisp`

## API surface

**C ABI (ABI 2):** [`n4m_stats_permdisp`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/metrics.h#L338). Use the linked public header for the exact signature, configuration, and result handles.

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Anderson (2006), *Distance-based tests for homogeneity of multivariate dispersions*, Biometrics 62, 245–253, https://doi.org/10.1111/j.1541-0420.2005.00440.x.

### Mathematical principle

Each row's Euclidean distance $z_i$ to its group centroid is computed, then a one-way ANOVA $F$ compares the groups' mean distances. As in vegan `betadisper(type = "centroid")` and `permutest`, the Monte Carlo test permutes the ANOVA residuals $z_i-\bar z_{g(i)}$ over all rows and counts permuted $F$ at least the observed one.

### Appropriate uses

Reporting whether groups differ in spread, alongside a PERMANOVA whose location test is sensitive to dispersion differences.

### Limits and validation

The residual-permutation test is approximate. Dispersion differences neither validate nor invalidate exchangeability or a PERMANOVA result. Only centroids are offered (not the spatial median variant). Workloads above 250 million operations return `N4M_ERR_UNSUPPORTED`.

### Implementation

The ABI-2 symbol `n4m_stats_permdisp` in `n4m/metrics.h` is implemented in `cpp/src/c_api/c_api_preflight.cpp` and wrapped in JS/WASM by `statsPermdisp`. SciPy `f_oneway`, scikit-bio `permdisp` and NumPy permutation counts are pinned in `parity/fixtures/permanova_v1.json`, checked by `cpp/tests/test_permanova.cpp`.

### Sources and provenance

https://doi.org/10.1111/j.1541-0420.2005.00440.x; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_preflight.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/parity/fixtures/permanova_v1.json

## Catalog note

PERMDISP (Anderson 2006) as vegan betadisper(type='centroid') + permutest: Euclidean distances to group centroids, one-way ANOVA F and a seeded Monte Carlo permutation of the ANOVA residuals (PCG64, numpy.random.default_rng parity). Reference gate shared with diagnostics.permanova (parity/fixtures/permanova_v1.json, cpp/tests/test_permanova.cpp).


_See also_: [methods index](index.md).