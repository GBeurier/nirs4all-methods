# Phase 6 — Wavelets & denoising preprocessings

## Goal

Add a wavelet transform tier to the n4m preprocessing toolkit, covering
single-level DWT, multi-level decomposition + denoising, statistical
feature extraction, and DWT-flatten + PCA / SVD dimensionality
reduction.  All six operators stay within the nirs4all-methods
"pure C, no deps" constraint by vendoring filter banks and the
boundary-handling kernels.

## Operators (6 stateless / stateful mix)

| Operator           | Algorithm                                                      | Lifecycle |
|--------------------|----------------------------------------------------------------|-----------|
| **Wavelet**        | Single-level DWT, returns approximation \|\| detail per row    | stateless |
| **Haar**           | Wavelet with `family = haar`, `mode = periodization`           | stateless |
| **WaveletDenoise** | Multi-level DWT + universal threshold + reconstruction         | stateless |
| **WaveletFeatures**| Multi-level DWT + per-band {mean, std, energy, entropy}        | stateless |
| **WaveletPCA**     | DWT-flatten + PCA (FlexiblePCA-style component selection)      | stateful  |
| **WaveletSVD**     | DWT-flatten + Truncated SVD (FlexibleSVD-style selection)      | stateful  |

The two stateful operators reuse the shared SVD primitive in
`cpp/src/core/common/svd.c`.  The shared low-level engines for the
filter banks, single-level DWT/IDWT and multi-level
`wavedec` / `waverec` live in
`cpp/src/core/common/wavelet_kernels.{c,h}` so future operators (e.g.
wavelet packets, stationary wavelet) can build on the same kernels.

## Supported families

- **haar**  (L = 2)
- **db4**   (Daubechies-4, L = 8)
- **sym4**  (Symlet-4, L = 8)
- **coif1** (Coiflet-1, L = 6)

Filter banks are vendored from PyWavelets 1.6.0 (validated by
`parity/python_generator/scripts/validate_frozen_wavelets_ref.py`).

## Supported boundary modes

- **periodization** (period = $n$; odd-length signals padded
  internally with one copy of the last sample, mirroring the PyWavelets
  convention).
- **symmetric**     (edge-repeat reflection).
- **zero**          (zero pad).

The PyWavelets `reflect`, `antisymmetric` and `smooth` modes are
deferred to a later phase — they are not required by any nirs4all-tier
example and would inflate the parity surface unnecessarily.

## C ABI surface added (27 symbols)

```c
/* Wavelet (stateless) */
n4m_pp_wavelet_{create,destroy,output_cols,transform}                  /* 4 */
/* Haar (stateless) */
n4m_pp_haar_{create,destroy,output_cols,transform}                      /* 4 */
/* WaveletDenoise (stateless) */
n4m_pp_wavelet_denoise_{create,destroy,transform}                       /* 3 */
/* WaveletFeatures (stateless) */
n4m_pp_wavelet_features_{create,destroy,output_cols,transform}          /* 4 */
/* WaveletPCA (stateful) */
n4m_pp_wavelet_pca_{create,destroy,fit,transform,is_fitted,output_cols} /* 6 */
/* WaveletSVD (stateful) */
n4m_pp_wavelet_svd_{create,destroy,fit,transform,is_fitted,output_cols} /* 6 */
```

Wrappers live in `cpp/src/c_api/c_api_wavelets.cpp`; declarations move
to `n4m.h` §16 (the Phase 6 slot in the §-numbering gap) at integration
time.

## Enum companions

```c
typedef enum {
    N4M_PP_WAVELET_HAAR  = 0,
    N4M_PP_WAVELET_DB4   = 1,
    N4M_PP_WAVELET_SYM4  = 2,
    N4M_PP_WAVELET_COIF1 = 3
} n4m_pp_wavelet_family_t;

typedef enum {
    N4M_PP_WAVELET_BOUNDARY_PERIODIZATION = 0,
    N4M_PP_WAVELET_BOUNDARY_SYMMETRIC     = 1,
    N4M_PP_WAVELET_BOUNDARY_ZERO          = 2
} n4m_pp_wavelet_boundary_t;

typedef enum {
    N4M_PP_WAVELET_THRESHOLD_SOFT = 0,
    N4M_PP_WAVELET_THRESHOLD_HARD = 1
} n4m_pp_wavelet_threshold_t;

typedef enum {
    N4M_PP_WAVELET_NOISE_MEDIAN = 0,
    N4M_PP_WAVELET_NOISE_STD    = 1
} n4m_pp_wavelet_noise_t;
```

## Parity

Frozen NumPy reference under
`parity/python_generator/src/n4m_parity_wavelets_ref/` — independent of
PyWavelets so upstream changes can't drift the fixtures.  Validation
once against PyWavelets 1.6.0 via
`parity/python_generator/scripts/validate_frozen_wavelets_ref.py`.

Tolerance budget (per the Phase 6 brief):

| Operator         | Abs tolerance | Rel tolerance |
|------------------|---------------|---------------|
| Wavelet / Haar   | 1e-10         | 1e-11         |
| WaveletDenoise   | 1e-9          | 1e-10         |
| WaveletFeatures  | 1e-12         | 1e-13         |
| WaveletPCA / SVD | 1e-10         | 1e-11         |

Fixtures (one per operator):

- `parity/fixtures/wavelet_v1.json`           (4 cases)
- `parity/fixtures/haar_v1.json`              (1 case)
- `parity/fixtures/wavelet_denoise_v1.json`   (3 cases)
- `parity/fixtures/wavelet_features_v1.json`  (3 cases)
- `parity/fixtures/wavelet_pca_v1.json`       (3 cases)
- `parity/fixtures/wavelet_svd_v1.json`       (3 cases)

## Out of scope (v1)

- Bigger wavelet families (db8, sym8, coif3 etc.) — deferred to v2 when
  filter banks for the extended set get vendored.
- Per-level PCA / SVD (the nirs4all reference fits a separate PCA per
  decomposition band).  v1 fits a single PCA / SVD on the concatenated
  coefficient stack, which is the simpler flatten-first variant
  documented in the Phase 6 brief.
- Top-K coefficient retention in `WaveletFeatures` (nirs4all exposes
  this as an extra column block).  v1 keeps only the four canonical
  statistics per band; the top-K block can be added in a future
  expansion without breaking the ABI.

## PyWavelets caveats

- For odd-length inputs in `periodization` mode, PyWavelets internally
  pads with one extra copy of the trailing sample before periodising
  (`n_eff = n + (n & 1)`).  The n4m engine and the frozen reference
  reproduce this convention exactly.
- The `symmetric` mode in PyWavelets corresponds to NumPy's
  `np.pad(x, L-1, mode="symmetric")` — *edge-repeat* reflection
  (`...,b,a,a,b,c,...`), not the no-edge-repeat reflection used by
  scipy.signal `savgol_filter(mode='mirror')`.

## Files

- Engines: `cpp/src/core/preprocessing/wavelets/{wavelet,haar,wavelet_denoise,wavelet_features,wavelet_pca,wavelet_svd}.{c,h}`
- Filter banks + DWT / IDWT helpers: `cpp/src/core/common/wavelet_kernels.{c,h}`
- C ABI wrappers: `cpp/src/c_api/c_api_wavelets.cpp`
- Tests: `cpp/tests/test_preprocessing_wavelets.cpp`
  (chained from `register_preprocessing_feature_selection_tests` because
   `main.cpp` is frozen for this batch).
- Frozen NumPy ref: `parity/python_generator/src/n4m_parity_wavelets_ref/`
- Fixture generator: `parity/python_generator/scripts/generate_phase6_fixtures.py`
- Validation script: `parity/python_generator/scripts/validate_frozen_wavelets_ref.py`
- Docs: `docs/algorithms/{wavelet,haar,wavelet_denoise,wavelet_features,wavelet_pca,wavelet_svd}.md`
