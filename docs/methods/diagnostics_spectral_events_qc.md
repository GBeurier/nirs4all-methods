# `spectral_events_qc` — n4m.diagnostics.spectral_events_qc

_Namespace_: **`n4m.diagnostics`** · _Fully-qualified_: `n4m.diagnostics.spectral_events_qc` · _Catalog id_: `diagnostics.spectral_events_qc`

## API surface

**C ABI (ABI 2):** [`n4m_procedure_run`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/estimator.h#L8). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):** `from n4m.roles import SpectralEventsQc`

**Signature:** [`SpectralEventsQc(*, valid_mask = (), axis_basis = 'index', axis_values = (), outlier_threshold = 8, clip_lower = None, clip_upper = None, clip_tolerance = 0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/roles/_generated.py#L2126)

**R (source-verified):** [`n4m_spectral_events_qc(valid_mask = c(), axis_basis = "index", axis_values = c(), outlier_threshold = 8.0, clip_lower = NULL, clip_upper = NULL, clip_tolerance = 0.0)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/estimator_roles_generated.R).

```r
library(n4m)
result <- n4m_spectral_events_qc()
```

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `valid_mask` | `—` | `()` |
| `axis_basis` | `—` | `'index'` |
| `axis_values` | `—` | `()` |
| `outlier_threshold` | `—` | `8` |
| `clip_lower` | `—` | `None` |
| `clip_upper` | `—` | `None` |
| `clip_tolerance` | `—` | `0` |

## Explanations

### Bibliographic source

A versioned descriptive threshold protocol: native first/second differences, normal-scaled MAD with a sample-standard-deviation fallback. NumPy defines the independent difference oracle: https://numpy.org/doc/stable/reference/generated/numpy.diff.html.

### Mathematical principle

An event exceeds threshold times row-local robust scale about the derivative median. Physical slopes divide by signed axis spacing and second derivatives account for irregular spacing; index differences are a separate mode. Native pooled counts divide only by eligible pairs/triples. Sensor clipping compares valid values to declared bounds.

### Appropriate uses

Detecting narrow local irregularities and declared sensor-limit occupancy while preserving valid-band denominators and distinguishing observed-extrema occupancy.

### Limits and validation

A derivative event is not proof of detector splice or cosmic-ray causation. Fewer than three eligible differences cannot establish the row scale. Missing bands are not bridged. Unknown sensor bounds make clipping unavailable; observed extrema alone do not prove clipping.

### Implementation

Native generic procedure, signed monotonic physical axis and explicit typed masks. Synthetic offset/gain, injected jumps/spikes, irregular/reversed axes and bounded-sensor witnesses are independently checked before cross-binding transport parity.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/estimator/intrinsic_diagnostics.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/tests/test_intrinsic_diagnostics.py; https://github.com/GBeurier/nirs4all-methods/blob/main/docs/intrinsic_diagnostics_contract.v1.json

## Catalog note

Versioned descriptive protocol with native aggregates and explicit per-row availability; see docs/intrinsic_diagnostics_contract.v1.json.


_See also_: [methods index](index.md).