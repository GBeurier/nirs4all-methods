# `splice_correction` — n4m.transform.signal_conversion.splice_correction

_Namespace_: **`n4m.transform.signal_conversion`** · _Fully-qualified_: `n4m.transform.signal_conversion.splice_correction` · _Catalog id_: `preprocessing.signal_conversion.splice_correction`

## API surface

**C ABI:** no standalone exported symbol is declared for this method.

**Python (verified public re-export):** `from n4m.roles import SpliceCorrection`

**Signature:** [`SpliceCorrection(*, splices = None, reference_segment = 1, method = 'multiplicative', window = 0, taper = False)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/roles/_generated.py#L4474)

**R (source-verified):** [`n4m_splice_correction(splices = NULL, reference_segment = 1L, method = "multiplicative", window = 0.0, taper = FALSE)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/estimator_roles_generated.R).

```r
library(n4m)
result <- n4m_splice_correction()
```

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `splices` | `—` | `None` |
| `reference_segment` | `—` | `1` |
| `method` | `—` | `'multiplicative'` |
| `window` | `—` | `0` |
| `taper` | `—` | `False` |

## Explanations

### Bibliographic source

Hueni & Bialek (2017), *Cause, effect, and correction of field spectroradiometer interchannel radiometric steps*, IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing 10(4), 1542–1551, https://doi.org/10.1109/JSTARS.2016.2625043.

### Mathematical principle

The axis is cut into detector segments at `splices` (the first band of each following detector). Segments are corrected outward from `reference_segment`: the mean level of the bands within `window` axis units of the splice on the corrected neighbour's side is matched by the segment's own mean level, by their ratio (multiplicative) or difference (additive); `window` 0 uses the two adjacent bands. With `taper`, the correction decreases linearly with wavelength from full at the splice to none at the far end of the segment.

### Appropriate uses

Removing interchannel steps of multi-detector field spectroradiometers (ASD, Spectral Evolution, SVC) after reflectance is computed from raw spectra.

### Limits and validation

An empirical level match: it does not model the temperature-dependent detector response that causes the steps, and it trusts the reference detector. Vendor software may blend overlapping detectors differently (Spectral Evolution DARWin output differs by up to a few percent reflectance near the splices). A segment whose factor is not finite, and the segments beyond it, are NaN.

### Implementation

The native `preprocessing.signal_conversion.splice_correction` transformer is exposed as SpliceCorrection in Python and JS and as n4m_splice_correction in R; its state is the source axis (project 1.5.2, ABI 2.24.0 unchanged). External-reference parity: identical to specdal `jump_correct` (additive, every reference segment) and within 4e-16 relative of spectrolab `match_sensors` (multiplicative, tapered).

### Sources and provenance

https://doi.org/10.1109/JSTARS.2016.2625043; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/signal_conversion/splice_correction.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/parity/spectral_conversions/generate.py

## Catalog note

Detector splice (jump) correction of multi-detector spectra (ASD, Spectral Evolution, SVC) through the shared n4m_estimator_* role API, available since project 1.5.2 with ABI 2.24.0 unchanged. splices are the first wavelengths of each following detector; segments are corrected outward from reference_segment (0-based), each matched to its corrected neighbour by the ratio (multiplicative) or difference (additive) of the mean levels of the bands within window axis units of the splice on each side (window 0: the two adjacent bands). taper spreads the correction linearly from full at the splice to none at the far end of the segment. Additive with window 0 is the specdal jump_correct convention; multiplicative with taper is spectrolab match_sensors. A segment whose factor is not finite, and those beyond it, are NaN. The fitted state is the source axis.


_See also_: [methods index](index.md).