# `axis_unit_conversion` — n4m.transform.resampling.axis_unit_conversion

_Namespace_: **`n4m.transform.resampling`** · _Fully-qualified_: `n4m.transform.resampling.axis_unit_conversion` · _Catalog id_: `preprocessing.resampling.axis_unit_conversion`

## API surface

**C ABI:** no standalone exported symbol is declared for this method.

**Python (verified public re-export):** `from n4m.roles import AxisUnitConversion`

**Signature:** [`AxisUnitConversion(*, source_unit = 'nm', target_unit = 'cm-1', spectral_density = False)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/roles/_generated.py#L3867)

**R (source-verified):** [`n4m_axis_unit_conversion(source_unit = "nm", target_unit = "cm-1", spectral_density = FALSE)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/estimator_roles_generated.R).

```r
library(n4m)
result <- n4m_axis_unit_conversion()
```

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `source_unit` | `—` | `'nm'` |
| `target_unit` | `—` | `'cm-1'` |
| `spectral_density` | `—` | `False` |

## Explanations

### Bibliographic source

IUPAC (2007), *Quantities, Units and Symbols in Physical Chemistry* (Green Book), 3rd ed., section 2.7: wavenumber $\tilde\nu=1/\lambda$ in vacuum; spectral densities transform with $|d\lambda/d\tilde\nu|$; https://doi.org/10.1039/9781847557889.

### Mathematical principle

Axis values are converted between nm, µm and cm⁻¹ with the exact constants $\tilde\nu[\mathrm{cm^{-1}}]=10^7/\lambda[\mathrm{nm}]=10^4/\lambda[\mu m]$ and $\lambda[\mathrm{nm}]=1000\,\lambda[\mu m]$; columns are permuted so the converted axis is strictly ascending. Values are unchanged unless `spectral_density`, which multiplies a density per source unit by the Jacobian $|dx_{source}/dx_{target}|$ (for nm to cm⁻¹, $\lambda^2/10^7$).

### Appropriate uses

Opening FT-NIR spectra recorded in cm⁻¹ in software that expects nm (or the reverse) before resampling to the target grid.

### Limits and validation

Wavelengths are taken as vacuum wavelengths: no air-to-vacuum correction is applied (about 0.03 %, 0.7 nm at 2500 nm). Absorbance, reflectance, transmittance and counts are per point and must not use `spectral_density`; radiance and irradiance per unit wavelength must. The converted grid is no longer evenly spaced. Axis values must be finite, positive and distinct.

### Implementation

The native `preprocessing.resampling.axis_unit_conversion` transformer is exposed as AxisUnitConversion in Python and JS and as n4m_axis_unit_conversion in R; its state is the source axis and the converted axis is the fit result entry `axis` (project 1.5.2, ABI 2.24.0 unchanged). External-reference parity: axis values within 3e-16 relative of astropy `spectral()` and densities within 7e-16 of astropy `spectral_density()` for all six unit pairs.

### Sources and provenance

https://doi.org/10.1039/9781847557889; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/resampling/axis_unit_conversion.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/parity/spectral_conversions/generate.py

## Catalog note

Spectral axis unit conversion between vacuum wavelengths (nm, um) and wavenumbers (cm-1) through the shared n4m_estimator_* role API, available since project 1.5.2 with ABI 2.24.0 unchanged. Fit takes the source axis; transform permutes columns so the converted axis is strictly ascending, published as the fit result entry axis (with source_columns). Values are unchanged unless spectral_density, which multiplies by the Jacobian |d source / d target| for densities per axis unit. No air-to-vacuum correction. Axis values must be finite, positive and distinct; same units only sort the axis.


_See also_: [methods index](index.md).