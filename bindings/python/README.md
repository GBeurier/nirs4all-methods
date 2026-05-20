# chemometrics4all — Python binding

`ctypes`-based binding for [`libc4a`](../../cpp/), the C ABI of
chemometrics4all.

## Tiers

* **Tier 1 (`chemometrics4all._ffi`)** — raw `ctypes.CDLL` access to the 498
  `c4a_*` symbols. Loads the shared library from
  `CHEMOMETRICS4ALL_LIB_PATH`, the wheel's bundled `lib/` directory, or the
  development build tree (`build/dev-debug/cpp/src/`).

* **Tier 2 (`chemometrics4all.python`)** — ABI-close NumPy functions that
  allocate the matching `c4a_*` handle, run the public C ABI call, return NumPy
  arrays, and destroy the handle. This is the dashboard `C4A.python` backend.

* **Tier 3 (`chemometrics4all.sklearn`)** — hand-written
  scikit-learn-compatible wrappers (`BaseEstimator`, `TransformerMixin`,
  `fit`, `transform`, `fit_transform`, and `inverse_transform`
  where the C ABI exposes a real inverse) over the main non-model NIRS
  operators already present in the C ABI. This is the dashboard
  `C4A.sklearn` backend:

  | Family           | Classes                                                              |
  | ---------------- | -------------------------------------------------------------------- |
  | Preprocessings   | `SNV`, `LSNV`, `RNV`, `AreaNormalization`, `Normalize`, `SimpleScale`, |
  |                  | `LogTransform`, `MSC`, `EMSC`, `BaselineCenter`, `Derivate`,          |
  |                  | `SavitzkyGolay`, `FirstDerivative`, `SecondDerivative`,               |
  |                  | `NorrisWilliams`, `Gaussian`, `ToAbsorbance`, `FromAbsorbance`,       |
  |                  | `PercentToFraction`, `FractionToPercent`, `KubelkaMunk`              |
  | Baseline         | `Detrend`, `AsLS`, `AirPLS`, `ArPLS`, `ModPoly`, `IModPoly`,         |
  |                  | `SNIP`, `RollingBall`, `IAsLS`, `BEADS`                              |
  | Feature extraction | `OSC`, `EPO`, `FlexiblePCA`, `FlexibleSVD`, `FCKStaticTransformer`  |
  | Resampling       | `CropTransformer`, `ResampleTransformer`, `Resampler`,               |
  |                  | `IntegerKBinsDiscretizer`, `RangeDiscretizer`                        |
  | Wavelets         | `Wavelet`, `Haar`, `WaveletDenoise`, `WaveletFeatures`,              |
  |                  | `WaveletPCA`, `WaveletSVD`                                           |
  | Splitters        | `KennardStoneSplitter`, `SPXYSplitter`, `SPXYFoldSplitter`,          |
  |                  | `SPXYGroupFoldSplitter`, `KMeansSplitter`, `KBinsStratifiedSplitter`,|
  |                  | `BinnedStratifiedGroupKFoldSplitter`, `SystematicCircularSplitter`,  |
  |                  | `SPlitSplitter`                                                      |
  | Filters          | `YOutlierFilter`, `XOutlierFilter`, `HighLeverageFilter`,            |
  |                  | `SpectralQualityFilter`, `CompositeFilter`                           |
  | Augmentation     | Full `c4a_aug_*` surface from noise/drift, wavelength/spectral,      |
  |                  | mixup/physical/environmental, edge/spline/random phases              |
  | Advanced         | `DirectStandardization`, `PiecewiseDirectStandardization`,           |
  |                  | `WeightedSNV`, `VariableSortingNormalization`, alignment helpers,    |
  |                  | and lightweight feature selectors backed by C ABI handles            |
  | Metrics          | `rmse`, `mae`, `bias`, `sep`, `rpd`, `rpiq`, `r2`, `nrmse`,          |
  |                  | `hotelling_t2`, `q_residuals`, `transfer_metrics`                    |

## binding_parity gate

`tests/test_binding_parity.py` is the Gate 1 contract: for every covered
operator, the binding output must match the libc4a reference fixture in
`parity/fixtures/<op>_v1.json` to within `tolerance=1e-6`.

## Install + test

```bash
cd bindings/python
pip install -e .
CHEMOMETRICS4ALL_LIB_PATH=/path/to/libc4a.so.1.9.0 \
    python -m pytest tests/test_binding_parity.py -v
```
