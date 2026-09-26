# `signal_type_detector` — n4m.transform.signal_conversion.signal_type_detector

_Namespace_: **`n4m.transform.signal_conversion`** · _Fully-qualified_: `n4m.transform.signal_conversion.signal_type_detector` · _Catalog id_: `utilities.signal_type_detector`

## API surface

**C ABI (ABI 2):** [`n4m_transform_signal_type_detector`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/transform/signal_conversion.h#L75). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):** `from n4m.transform.signal_conversion import signal_type_detector`

**Signature:** [`signal_type_detector(X, wavelengths = None, confidence_threshold: float = 0.7)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_impl/native.py#L10648)

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `X` | `—` | `required` |
| `wavelengths` | `—` | `None` |
| `confidence_threshold` | `float` | `0.7` |

## Explanations

### Bibliographic source

No canonical scientific paper defines these thresholds. This is an n4m compatibility heuristic, and `signal_type_detector.c` is the authoritative specification.

### Mathematical principle

NaN-skipping global min, max, mean, and standard deviation first detect centered, standardized, or derivative-like data and return `UNKNOWN`. Otherwise range/mean rules score absorbance, fractional/percent reflectance, and fractional/percent transmittance. Optional wavelength cues near 1450, 1940, and 2500 nm adjust scores; confidence is the best score divided by the score sum and is thresholded.

### Appropriate uses

Advisory detection of likely raw signal units before choosing absorbance or percentage conversion in an ingestion workflow.

### Limits and validation

Reflectance and transmittance ranges overlap, so classification is inherently ambiguous. Water-band cues assume wavelengths in nm and representative NIR coverage. The enum contains additional types that this heuristic does not currently score.

### Implementation

`n4m.transform.signal_conversion.signal_type_detector` wraps `n4m_transform_signal_type_detector`; all thresholds and the 256-byte reason string are defined in `cpp/src/core/utilities/signal_type_detector.c`.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/utilities/signal_type_detector.c


_See also_: [methods index](index.md).