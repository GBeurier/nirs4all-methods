# `local_density` — n4m.diagnostics.local_density

_Namespace_: **`n4m.diagnostics`** · _Fully-qualified_: `n4m.diagnostics.local_density` · _Catalog id_: `diagnostics.local_density`

## API surface

**C ABI (ABI 2):** [`n4m_procedure_run`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/estimator.h#L8). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):** `from n4m.roles import LocalDensity`

**Signature:** [`LocalDensity(*, valid_mask = (), mode = 'training', n_neighbors = 20)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/roles/_generated.py#L1994)

**R (source-verified):** [`n4m_local_density(valid_mask = c(), mode = "training", n_neighbors = 20L)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/estimator_roles_generated.R).

```r
library(n4m)
result <- n4m_local_density()
```

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `valid_mask` | `—` | `()` |
| `mode` | `—` | `'training'` |
| `n_neighbors` | `—` | `20` |

## Explanations

### Bibliographic source

Breunig, Kriegel, Ng and Sander (2000), LOF: Identifying Density-Based Local Outliers. https://doi.org/10.1145/335191.335388.

### Mathematical principle

Use the existing native LOF training neighborhoods and reachability densities. Training LOF excludes self; novelty scores use a separately fitted reference neighborhood. The score is mean neighbor reachability density divided by the row density, with stable distance/index ties and fixed k capped by reference rows minus one.

### Appropriate uses

Comparing local density under an explicit representation, population and Euclidean distance protocol; exposing genuine LOF separately from kNN distance-ratio proxies.

### Limits and validation

Fixed-k tie breaking is not an all-distance-tie expansion. Duplicate or degenerate density neighborhoods can be undefined and are reported unavailable rather than clean. Complete data are required; distance over differing masked coordinate subsets is not inferred. LOF is neither a calibrated anomaly probability nor a physical cause label.

### Implementation

The generic wrapper reuses cpp/src/core/filters/_vendored/lof.c. Only native training score access was added; the old filter/novelty behavior is unchanged. Independent sklearn training and novelty witnesses qualify the finite nondegenerate cases.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/estimator/intrinsic_diagnostics.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/tests/test_intrinsic_diagnostics.py; https://github.com/GBeurier/nirs4all-methods/blob/main/docs/intrinsic_diagnostics_contract.v1.json

## Catalog note

Versioned descriptive protocol with native aggregates and explicit per-row availability; see docs/intrinsic_diagnostics_contract.v1.json.


_See also_: [methods index](index.md).