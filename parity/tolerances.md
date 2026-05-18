# Parity tolerances

Per-category tolerances enforced by `chemometrics4all_tests` against the
frozen Python reference fixtures under `parity/fixtures/`. Each row pairs a
reference implementation (frozen NumPy / SciPy / scikit-learn / PyWavelets /
pybaselines reference, version-pinned via `parity/python_generator/
requirements-lock.txt`) with the corresponding chemometrics4all C kernel.

Numbers in **bold** are the relaxations applied after Opus / Codex review.

| Row key | Reference (A) | chemometrics4all (B) | Operators / kernels | abs_tol | rel_tol | Comparison set | Notes |
|---|---|---|---|---|---|---|---|
| `c4a-scatter-preprocessing-stateless`            | NumPy / scikit-learn 1.4.2 stateless preprocessing reference | `c4a_pp_*_apply` (stateless ops) | identity / center / normalize / SNV / RNV / area / LSNV / log_transform | 1e-12 | 1e-12 | transform outputs | operator parity |
| `c4a-scatter-preprocessing-stateful`             | NumPy / scikit-learn 1.4.2 stateful preprocessing reference | `c4a_pp_*_fit` + `_transform` | baseline-center / MSC / EMSC / Pareto / simple-scale (stateful family) | 1e-12 | 1e-12 | transform outputs | stateful operator parity |
| `c4a-derivatives-smoothing`                      | NumPy / SciPy 1.11.4 reference | `c4a_pp_derivate_*`, `c4a_pp_savgol_*`, `c4a_pp_gaussian_*`, `c4a_pp_norris_williams_*` | derivate (1st / 2nd) / Savitzky-Golay / Gaussian filter / Norris-Williams | 1e-12 | 1e-12 | transform outputs | smoothing parity |
| `c4a-baselines-asls-family`                      | Frozen pybaselines 1.1.4 reference | `c4a_pp_{asls,airpls,arpls,iasls,beads}_*` | AsLS / AirPLS / ArPLS / IAsLS / BEADS | **1e-11** | **1e-10** | transform outputs | LDLT vs SuperLU pivot differences (Phase 5a/5b) |
| `c4a-baselines-polynomial`                       | Frozen NumPy / pybaselines reference | `c4a_pp_{detrend,modpoly,imodpoly,snip,rolling_ball}_*` | Detrend / ModPoly / IModPoly / SNIP / RollingBall | 1e-12 | 1e-12 | transform outputs | polynomial baseline parity |
| `c4a-signal-conversion`                          | NumPy reference (Beer-Lambert / Kubelka-Munk / percent ↔ fraction) | `c4a_pp_{to_absorbance,from_absorbance,kubelka_munk,pct_to_frac,frac_to_pct}_apply` | A↔R conversion / Kubelka-Munk / unit scaling | 1e-12 | 1e-12 | transform outputs | signal-conversion parity (Phase 7) |
| `c4a-orthogonalization`                          | NumPy reference port of nirs4all OSC / EPO | `c4a_pp_{osc,epo}_*` | Orthogonal Signal Correction / External Parameter Orthogonalisation | 1e-10 | 1e-10 | transform outputs | orthogonalisation parity (Phase 8) |
| `c4a-feature-selection-pca`                      | scikit-learn 1.4.2 / NumPy thin-SVD reference | `c4a_pp_{flexible_pca,flexible_svd}_*` | PCA (auto / svd / eigh) / thin SVD | 1e-10 | 1e-10 | transform outputs, component vectors | feature-selection parity (Phase 9) |
| `c4a-resampling`                                 | NumPy / SciPy interp reference | `c4a_pp_{resample,resampler,crop}_*` | linear / cubic resample / crop | 1e-12 | 1e-12 | transform outputs | resampling parity (Phase 10) |
| `c4a-splitters-deterministic`                    | NumPy PCG64 reference | `c4a_split_{kennard_stone,spxy,spxy_fold,spxy_g_fold,kmeans,kbins_stratified,binned_strat_group_kfold,systematic_circular,split_splitter}_*` | nine splitter algorithms | exact | exact | train / test index plans (int64) | byte-exact integer indices for fixed seed (Phase 11) |
| `c4a-filters-y-outliers`                         | NumPy reference | `c4a_filter_y_outlier_fit_get_mask` | IQR / Z-score / Percentile / MAD | exact | exact | keep-mask (uint8), exclusion counts | byte-exact masks (Phase 12) |
| `c4a-filters-meta`                               | NumPy reference (hat-matrix / PCA-leverage / quality thresholds) | `c4a_filter_{leverage,quality,composite}_*` | leverage (hat / PCA-score) / quality / composite (ANY / ALL) | 1e-10 | 1e-10 | keep-mask + per-row scores | meta-filter parity (Phase 14) |
| `c4a-signal-type-detection`                      | NumPy heuristic reference | `c4a_signal_detect` | Absorbance / Reflectance / Transmittance / Kubelka-Munk / log(1/R) / log(1/T) | exact | exact | detected enum + confidence ± 1e-12 | signal-type parity (Phase 19) |
| `c4a-nirs-metrics`                               | NumPy closed-form reference | `c4a_metric_{rmse,mae,bias,sep,rpd,rpiq,r2,nrmse}` | 8 regression metrics (sklearn / nirs4all convention) | 1e-12 | 1e-12 | scalar metrics | metric parity (Phase 19) |
| `c4a-hotelling-q`                                | NumPy / SciPy F + chi² UCL reference | `c4a_util_{hotelling_t2,q_residuals}` | Hotelling T² + Q-residuals + analytic UCL | 1e-10 | 1e-10 | per-sample T²/Q + UCL | multivariate outlier stats (Phase 19) |
| `c4a-transfer-metrics`                           | NumPy reference port of nirs4all `transfer_metrics` | `c4a_transfer_metrics_compute` | nine alignment metrics (Mahalanobis / Wasserstein / Grassmann / etc.) | 1e-10 | 1e-10 | nine scalar metrics | transfer-metric parity (Phase 20) |
| `c4a-fck-static-transformer`                     | NumPy `convolve1d` reference (nirs4all FCK kernel) | `c4a_pp_fck_static_{create,transform}` | (alpha, sigma) fractional convolution kernel bank | 1e-12 | 1e-13 | per-kernel output bands | FCK parity (Phase 21) |

## Conventions

* Tolerances are sized so the chemometrics4all C kernel passes against the
  frozen Python reference under the locked dependency set
  (`parity/python_generator/requirements-lock.txt`). Bumping a reference
  package may require relaxing the corresponding row — that bump must be
  driven by the parity gate (`run_parity_gate.py`), not by editing this
  table.
* "exact" means int64 / uint8 byte-equality (used for splitter indices and
  filter keep-masks where the C kernel is required to reproduce the Python
  reference output exactly).
* The parity-gate workflow (`.github/workflows/parity-gate.yml`) consumes
  this file informally — the C++ tests carry their own tolerance constants
  per fixture and assert against them directly. This table is the contract
  documentation.
