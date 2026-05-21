# External Reference Audit

This audit records which benchmark rows are executable external parity gates and
which rows are kept only as scientific context. A reference gate may be either a
native external implementation or a contract adapter around external numerical
primitives; adapters are allowed only to map shapes, windows, or comparison
contracts, not to compare C++ against a second local N4M implementation.

`nirs4all` is treated as a third-party Python reference. During development the
benchmark resolves it from the sibling checkout; once the next `nirs4all`
release is tagged the reference should be pinned to the published PyPI package.

## Current State

| Metric | Value |
|--------|-------|
| Public benchmarked methods | 66 |
| Methods with at least one executable external reference row | 66 |
| Methods left as `reference_needed` in the dashboard | 0 |
| Full matrix used by the dashboard | `benchmarks/cross_binding/results/full_matrix.csv` |
| Full matrix status | 999 / 999 timed cells, 0 parity failures, 0 divergences above `1e-6` |
| nirs4all-methods bindings in the dashboard | C++ ABI, `N4M.python`, `N4M.sklearn`, R |

## New Or Previously Missing Gates

| Method | Executable reference gate | Contract notes |
|--------|---------------------------|----------------|
| Robust direct standardization | `sklearn.linear_model.LinearRegression(iterative residual trimming)` | Composite gate: scikit-learn owns the least-squares fits; the adapter applies N4M's documented residual-trimming schedule. No native robust-DS package with this exact contract was found. |
| Piecewise direct standardization | `sklearn.linear_model.LinearRegression(local windows)` | Same-contract local OLS PDS gate. `pycaltransfer.pds_pls_transfer_fit` remains context because it is the PDS-PLS variant. |
| Score-augmented projection standardization | `sklearn.decomposition.PCA + sklearn.linear_model.LinearRegression` | Composite gate matching N4M's PCA-score augmentation and OLS transfer contract. The SA-PBS paper is public, but no installable native package was found. |
| Local centering | `sklearn.preprocessing.StandardScaler(with_std=False)` | scikit-learn supplies the source and target centering primitives. |
| Weighted SNV | `statsmodels.stats.weightstats.DescrStatsW` | statsmodels supplies weighted mean and variance for the row-wise SNV contract. |
| Variable sorting normalization | `sklearn.feature_selection.r_regression + statsmodels.stats.weightstats.DescrStatsW` | scikit-learn supplies the variable scores; statsmodels supplies the weighted normalization statistics. |
| Piecewise SNV | `nirs4all.StandardNormalVariate(ddof=0) per interval` | nirs4all supplies SNV; the adapter applies it independently over N4M intervals. |
| Piecewise MSC | `sklearn.linear_model.LinearRegression(MSC per interval)`, `prospectr::msc per interval` | scikit-learn is the strict all-size gate. prospectr is also gated on small matrices; on ill-conditioned 16-column windows its Armadillo normal-equation solve drifts above `1e-6`, so it is capped to small widths. |
| Localized MSC | `sklearn.linear_model.LinearRegression(sliding local MSC)`, `prospectr::msc sliding interval` | scikit-learn is the strict all-size gate. prospectr gates small-width rows. The previous SciPy raw moving-sum comparator is kept only as context because cancellation in near-singular windows produces unstable differences. |
| Cross-correlation alignment | `scipy.ndimage.shift(cross-correlation shift search)` | SciPy supplies the tested integer shifts and edge-clamped interpolation. |
| Icoshift alignment | `scipy.ndimage.shift(interval-correlation shift search)` | SciPy supplies interval-local shifts and edge-clamped interpolation. |
| Correlation optimized warping | `cowarp.warp(segment_length=16, slack=2)` | `cowarp` supplies the external dynamic-programming COW contract; N4M was changed to match that contract. |
| Interval generator | `sklearn.compose.ColumnTransformer(passthrough intervals)` | scikit-learn supplies interval column selection; the adapter returns N4M's list-of-blocks shape. |

## Existing Active Gates

| Family | External references |
|--------|---------------------|
| Scatter and normalization | `prospectr`, `nirs4all`, `sklearn` |
| Baselines | `pybaselines` |
| Wavelets | `PyWavelets`, `nirs4all` |
| Calibration transfer | `pycaltransfer`, `sklearn`, `nirs4all` |
| Alignment | `dtw-python`, `cowarp`, `scipy` |
| Filters and selectors | `sklearn`, `nirs4all` |
| Resampling and splitters | `scipy`, `sklearn`, `nirs4all` |

## Context Rows Kept In The Dashboard

| Method | Context reference | Why it is not a parity gate |
|--------|-------------------|-----------------------------|
| MSC | `nirs4all.MultiplicativeScatterCorrection(scale=False)` | The current local nirs4all checkout uses a historical column-regression variant; `prospectr::msc` is the active same-contract gate. |
| Piecewise direct standardization | `pycaltransfer.pds_pls_transfer_fit` | pycaltransfer exposes PDS-PLS; N4M's active contract is local OLS PDS and is gated by scikit-learn. |
| Localized MSC | `scipy.ndimage.convolve1d(local MSC window sums)` | SciPy supplies moving-window sums, but raw sums are numerically unstable on near-singular local MSC windows; `sklearn.linear_model.LinearRegression` is the strict gate. |
| Detrend | `prospectr::detrend` | prospectr combines the NIR detrend convention with package-specific behavior; SciPy/nirs4all cover the active N4M contract. |
| Wavelet / Haar | `nirs4all` detail-resampled wavelet operators | nirs4all returns resampled detail coefficients; `PyWavelets.dwt(..., mode="periodization")` gates the active N4M output. |
| Savitzky-Golay | `nirs4all.SavitzkyGolay(default edge mode)` | nirs4all does not expose N4M's mirror edge mode; SciPy and prospectr cover the same-contract gates. |

## External Source Notes

- `pycaltransfer` documents DS, PDS, EPO, slope/bias, and related calibration-transfer routines: <https://pypi.org/project/pycaltransfer/>
- SA-PBS is described in "Enhancing standardization through score-augmented projection-based calibration transfer": <https://doi.org/10.1016/j.chemolab.2023.105026>
- VSN is described in "Variable sorting for normalization": <https://iris.uniroma1.it/handle/11573/1352525>
- PDS-PLS references and worked examples commonly model each local window with PLS, which is why they are not the same contract as N4M's local OLS PDS gate.
