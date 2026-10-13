# `bh_adjust` — n4m.metrics.multiple_testing.bh_adjust

_Namespace_: **`n4m.metrics.multiple_testing`** · _Fully-qualified_: `n4m.metrics.multiple_testing.bh_adjust` · _Catalog id_: `diagnostics.bh_adjust`

## API surface

**C ABI (ABI 2):** [`n4m_stats_bh_adjust`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/metrics.h#L289). Use the linked public header for the exact signature, configuration, and result handles.

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Benjamini & Hochberg (1995), *Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing*, Journal of the Royal Statistical Society B 57(1), 289–300, https://doi.org/10.1111/j.2517-6161.1995.tb02031.x; Benjamini & Yekutieli (2001), *The control of the false discovery rate in multiple testing under dependency*, Annals of Statistics 29(4), 1165–1188, https://doi.org/10.1214/aos/1013699998.

### Mathematical principle

For one family of $m$ p-values sorted ascending as $p_{(1)}\le\dots\le p_{(m)}$, the adjusted value of rank $i$ is $\min(1,\min_{j\ge i} m\,p_{(j)}/j)$, as R `p.adjust(method = "BH")` and statsmodels `multipletests(method="fdr_bh")`.

### Appropriate uses

Publishing an exploratory false-discovery-rate view of a hypothesis family next to a family-wise (Holm) decision rule.

### Limits and validation

BH controls the false discovery rate under independence or positive regression dependence (PRDS) of the test statistics, not the family-wise error rate. The caller defines the family. Inputs must lie in $[0,1]$ and $m\ge 1$.

### Implementation

The ABI-2 symbol `n4m_stats_bh_adjust` in `n4m/metrics.h` is implemented in `cpp/src/c_api/c_api_preflight.cpp` and wrapped in JS/WASM by `statsBhAdjust`. statsmodels and SciPy `false_discovery_control` reference values are pinned in `cpp/tests/test_preflight.cpp`.

### Sources and provenance

https://doi.org/10.1111/j.2517-6161.1995.tb02031.x; https://doi.org/10.1214/aos/1013699998; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/c_api/c_api_preflight.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/tests/test_preflight.cpp

## Catalog note

Benjamini-Hochberg step-up adjusted p-values for one hypothesis family, matching R p.adjust(method = "BH") and statsmodels multipletests(method = "fdr_bh").


_See also_: [methods index](index.md).