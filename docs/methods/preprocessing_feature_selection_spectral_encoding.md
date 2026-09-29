# `spectral_encoding` — n4m.decomposition.spectral_encoding

_Namespace_: **`n4m.decomposition`** · _Fully-qualified_: `n4m.decomposition.spectral_encoding` · _Catalog id_: `preprocessing.feature_selection.spectral_encoding`

## API surface

**C ABI (ABI 2):** [`n4m_decomposition_spectral_create`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/decomposition.h#L19) · [`n4m_decomposition_spectral_destroy`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/decomposition.h#L23) · [`n4m_decomposition_spectral_fit`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/decomposition.h#L24) · [`n4m_decomposition_spectral_output_cols`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/decomposition.h#L26) · [`n4m_decomposition_spectral_transform`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/decomposition.h#L28) · [`n4m_decomposition_spectral_export_affine`](https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/decomposition.h#L32). Use the linked public header for the exact signature, configuration, and result handles.

**Python (verified public re-export):** `from n4m.decomposition import LVSE`

**Signature:** [`LVSE(width = 64, rank = 4, overlap = 0.0, standardize = True, snv = False)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/src/n4m/_spectral_encoding.py#L98)

**R (source-verified):** [`lvse_fit(X, width = 64L, rank = 4L, overlap = 0, standardize = TRUE, snv = FALSE)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/r/n4m/R/spectral_encoding.R).

```r
library(n4m)
result <- lvse_fit(X)
```

**MATLAB / Octave (source-verified):** [`LVSE(width, rank, overlap, standardize, snv)`](https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/matlab/+n4m/LVSE.m).

The source signature has additional required inputs, so no example call is fabricated.

### Parameters

| Name | Type | Default |
|---|---|---|
| `width` | `—` | `64` |
| `rank` | `—` | `4` |
| `overlap` | `—` | `0.0` |
| `standardize` | `—` | `True` |
| `snv` | `—` | `False` |

## Explanations

### Bibliographic source

No single canonical paper specifies this combined n4m lifecycle. The exact local subspace and global nonnegative constructions are defined by the current C++ implementation and its spectral-encoding contract.

### Mathematical principle

LVSE centers and optionally scales each channel on training rows, then uses exact local Gram SVD bases over fixed or overlapping windows; small singular modes are zeroed and the retained coordinates form a fitted spectral representation. GCU subtracts training channel minima, applies a global scale and fits nonnegative factors with NNDSVDa initialization and cyclic coordinate descent. Query rows reuse the fitted bases and scaling instead of refitting them.

### Appropriate uses

Representing NIR spectra by local subspaces or global nonnegative factors before a downstream estimator, with encoder window, rank and overlap chosen inside training-fold validation.

### Limits and validation

Fitting an encoder before the data split leaks validation spectra. Overlapping LVSE windows do not generally form one orthogonal projector; SNV-enabled LVSE and GCU are non-affine in raw inputs. Exact SVD on all supplied rows does not promise bitwise parity with stochastic or subsampled research prototypes.

### Implementation

The `n4m_decomposition_spectral_*` C ABI owns the fitted lifecycle. `cpp/src/core/spectral_encoding.cpp` implements the numerical transforms; Python, R, MATLAB and WASM bindings marshal data to that same native engine.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/spectral_encoding.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/docs/methods/spectral_encoding.md

## Catalog note

Shared lifecycle for LVSE (kind 0) and GCU (kind 1). Python exposes LVSE and GCU; R exposes lvse_fit and gcu_fit; WASM exposes SpectralEncoder. Exact SVD, no implicit subsampling. See docs/methods/spectral_encoding.md.


_See also_: [methods index](index.md).