# `from_kubelka_munk` — n4m.transform.signal_conversion.from_kubelka_munk

_Namespace_: **`n4m.transform.signal_conversion`** · _Fully-qualified_: `n4m.transform.signal_conversion.from_kubelka_munk` · _Catalog id_: `preprocessing.signal_conversion.from_kubelka_munk`

## API surface

**C ABI:** no standalone exported symbol is declared for this method.

**Python (verified public re-export):** `from n4m.roles import FromKubelkaMunk`

**Signature:** [`FromKubelkaMunk(*, is_percent = False)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/roles/_generated.py#L4407)

**R (source-verified):** [`n4m_from_kubelka_munk(is_percent = FALSE)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/estimator_roles_generated.R).

```r
library(n4m)
result <- n4m_from_kubelka_munk()
```

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

### Parameters

| Name | Type | Default |
|---|---|---|
| `is_percent` | `—` | `False` |

## Explanations

### Bibliographic source

Kubelka & Munk (1931), *Ein Beitrag zur Optik der Farbanstriche*, Zeitschrift für Technische Physik 12, 593–601; English translation: https://doi.org/10.1002/col.5080010404.

### Mathematical principle

The remission function $F=(1-R)^2/(2R)$ is inverted on $R\in(0,1]$: $R$ is the smaller root of $R^2-2(1+F)R+1=0$, evaluated as $R=1/(1+F+\sqrt{F(F+2)})$, which avoids the cancellation of $1+F-\sqrt{F(F+2)}$ for large $F$. `is_percent` scales the output by 100.

### Appropriate uses

Returning Kubelka–Munk spectra to diffuse reflectance, for example before computing apparent absorbance log(1/R) for software that expects it.

### Limits and validation

Only the infinite-layer two-flux model is inverted; its physical assumptions are those of the forward transform. Negative $F$ has no reflectance and gives NaN, NaN propagates and $F=+\infty$ gives $R=0$. A forward transform that clamped $R$ to $[\epsilon,1-\epsilon]$ is not undone.

### Implementation

The native `preprocessing.signal_conversion.from_kubelka_munk` transformer is exposed as FromKubelkaMunk through the generated Python and JS roles and as n4m_from_kubelka_munk in R, with width-only WTD1 state and no dedicated C symbols (project 1.5.2, ABI 2.24.0 unchanged). External-reference parity: a bracketing `stats::uniroot` solve over `mdatools::prep.ref2km` agrees within 4e-16 relative (parity/spectral_conversions, replayed by bindings/python/tests/test_spectral_conversions.py).

### Sources and provenance

https://doi.org/10.1002/col.5080010404; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/preprocessing/signal_conversion/reference_conversions.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/parity/spectral_conversions/generate.py

## Catalog note

Inverse Kubelka-Munk function through the shared n4m_estimator_* role API, available since project 1.5.2 with ABI 2.24.0 unchanged: R = 1 / (1 + F + sqrt(F (F + 2))), the root in (0, 1] of F = (1 - R)^2 / (2 R), written without cancellation. is_percent scales the output by 100. F < 0 gives NaN; NaN propagates. Width-only WTD1 state.


_See also_: [methods index](index.md).