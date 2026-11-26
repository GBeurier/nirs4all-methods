# `reflectance_from_references` — n4m.transform.signal_conversion.reflectance_from_references

_Namespace_: **`n4m.transform.signal_conversion`** · _Fully-qualified_: `n4m.transform.signal_conversion.reflectance_from_references` · _Catalog id_: `preprocessing.signal_conversion.reflectance_from_references`

## API surface

**C ABI:** no standalone exported symbol is declared for this method.

**Python (verified public re-export):** `from n4m.roles import ReflectanceFromReferences`

**Signature:** [`ReflectanceFromReferences(*, white_reference = None, dark_reference = (), as_percent = False)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/roles/_generated.py#L4452)

**R (source-verified):** [`n4m_reflectance_from_references(white_reference = NULL, dark_reference = c(), as_percent = FALSE)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/estimator_roles_generated.R).

```r
library(n4m)
result <- n4m_reflectance_from_references()
```

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `white_reference` | `—` | `None` |
| `dark_reference` | `—` | `()` |
| `as_percent` | `—` | `False` |

## Explanations

### Bibliographic source

Milton, Schaepman, Anderson, Kneubühler & Fox (2009), *Progress in field spectroscopy*, Remote Sensing of Environment 113, S92–S109, https://doi.org/10.1016/j.rse.2007.08.001.

### Mathematical principle

Per band, $R=s\,(X-D)/(W-D)$ with $X$ the sample signal, $W$ the white (or reference-panel radiance) spectrum, $D$ the optional dark spectrum and $s=100$ when `as_percent`. The same ratio gives transmittance against a blank reference. White and dark are parameters of the input width.

### Appropriate uses

Computing reflectance or transmittance from raw spectrometer counts when the file stores the sample, white and dark spectra (AvaSoft, OceanView, ASD, Spectral Evolution, SVC), for example before log(1/R) or Kubelka–Munk.

### Limits and validation

The counts must share integration time, averaging and gain, or be normalized (radiance, normalized DN); the method does not rescale them. Bands whose $W-D$ is not a finite positive number give NaN; bands with a reference at noise level give meaningless ratios. No panel calibration, nonlinearity, stray-light or detector splice correction is applied (see SpliceCorrection).

### Implementation

The native `preprocessing.signal_conversion.reflectance_from_references` transformer is exposed as ReflectanceFromReferences in Python and JS and as n4m_reflectance_from_references in R (project 1.5.2, ABI 2.24.0 unchanged). External-reference parity: identical to `prospectr::readASD` and `asdreader::get_spectra` reflectance on three real ASD files; vendor comparison (parity/results/spectral_conversions_vendor.json): identical to AvaSoft 6/7 binary files and OceanView output, within the 2-decimal rounding of SVC files; AvaSoft 8 text exports differ by up to 0.6 % (undocumented vendor processing).

### Sources and provenance

https://doi.org/10.1016/j.rse.2007.08.001; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/signal_conversion/reference_conversions.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/parity/spectral_conversions/vendor_check.py

## Catalog note

Reflectance or transmittance from raw references through the shared n4m_estimator_* role API, available since project 1.5.2 with ABI 2.24.0 unchanged: R = scale (X - D) / (W - D) per column, with W the white (or reference radiance) spectrum and D the optional dark spectrum, both parameters of the input width. as_percent sets scale = 100. Columns whose W - D is not finite and positive give NaN. Width-only WTD1 state; N4ME replays the references.


_See also_: [methods index](index.md).