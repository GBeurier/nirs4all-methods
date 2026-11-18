# `spectral_population_events_qc` — n4m.diagnostics.spectral_population_events_qc

_Namespace_: **`n4m.diagnostics`** · _Fully-qualified_: `n4m.diagnostics.spectral_population_events_qc` · _Catalog id_: `diagnostics.spectral_population_events_qc`

## API surface

**C ABI:** no standalone exported symbol is declared for this method.

**Python (verified public re-export):** `from n4m.roles import SpectralPopulationEventsQc`

**Signature:** [`SpectralPopulationEventsQc(*, valid_mask = (), axis_basis = 'index', axis_values = (), outlier_threshold = 8, min_spectra = 5)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/roles/_generated.py#L2182)

**R (source-verified):** [`n4m_spectral_population_events_qc(valid_mask = c(), axis_basis = "index", axis_values = c(), outlier_threshold = 8.0, min_spectra = 5L)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/estimator_roles_generated.R).

```r
library(n4m)
result <- n4m_spectral_population_events_qc()
```

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `valid_mask` | `—` | `()` |
| `axis_basis` | `—` | `'index'` |
| `axis_values` | `—` | `()` |
| `outlier_threshold` | `—` | `8` |
| `min_spectra` | `—` | `5` |

## Explanations

### Bibliographic source

A versioned descriptive threshold protocol: at each band position, the normal-scaled median absolute deviation (Rousseeuw and Croux, 1993, Alternatives to the Median Absolute Deviation, https://doi.org/10.1080/01621459.1993.10476408) of the derivatives across spectra. NumPy defines the independent difference oracle: https://numpy.org/doc/stable/reference/generated/numpy.diff.html.

### Mathematical principle

Use the same first and second derivatives as the row-local event protocol, including signed physical slopes and the irregular-spacing second derivative. For each pair (j,j+1) or triple (j,j+1,j+2), take the rows where all bands are valid; with at least min_spectra rows, center is their median and scale is 1.4826 times their MAD. The scale floor is the median scale over evaluated positions of the same derivative order; an event is abs(d-center) > threshold*max(scale,floor). Events are attributed to band j+1. Row fractions divide by the row's evaluated positions.

### Appropriate uses

Flagging narrow spikes and steps that are atypical relative to the other spectra at the same band position, so that absorption features shared by the population are not reported as events, and locating events along the axis with per-band counts.

### Limits and validation

Atypicality is relative to the supplied population: a mixture of instruments, products or acquisition settings widens the per-position scale and hides events, while a feature present in only a few spectra can be flagged. An event is not proof of cosmic rays, detector splices or any physical cause. Positions with fewer than min_spectra eligible spectra are not evaluated, and zero population spread makes the derivative order undefined rather than clean. Missing bands are not bridged.

### Implementation

One mask-aware native generic procedure sharing the declared-axis derivative helper of the row-local protocol, which is unchanged. Independent NumPy oracles validate masks, irregular physical axes, per-band attribution and pooled denominators; smooth multi-band synthetic spectra and injected spike/step witnesses check the intended contrast with row-local scaling.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/estimator/intrinsic_diagnostics.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/tests/test_intrinsic_diagnostics.py; https://github.com/GBeurier/nirs4all-methods/blob/main/docs/intrinsic_diagnostics_contract.v1.json

## Catalog note

Versioned descriptive protocol: derivative events relative to the across-spectra distribution at each band position; see docs/intrinsic_diagnostics_contract.v1.json.


_See also_: [methods index](index.md).