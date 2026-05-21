# Parity tolerances

Per-category tolerances enforced by `n4m_tests` against the
internal Python parity fixtures under `parity/fixtures/`. These fixtures are
development parity oracles, not external benchmark references. Each row pairs a
fixture source (frozen NumPy / SciPy / scikit-learn / PyWavelets /
pybaselines / selected `nirs4all` operator references, version-pinned via
`parity/python_generator/requirements-lock.txt`) with the corresponding
nirs4all-methods C kernel.

Numbers in **bold** are the relaxations applied after Opus / Codex review.

`reference_tol` rows are the Stage-3 (C++ vs internal Python parity fixtures)
tolerances. `binding_tol` is the Stage-4 (binding-vs-libn4m, bit-exact)
tolerance — `1e-6` is the project-wide default
(see `parity/binding_parity.py`). Bindings are expected to forward the
libn4m result without re-implementing it, so the default tolerance is
strict; per-op overrides are documented inline.

Stage 2 reference parity has its own executable policy in
`parity/run_reference_parity.py`. Expected skips and weak external-reference
zones are recorded in `parity/divergences.json`; an unlisted skip is a gate
failure.

| Row key | Fixture source (A) | nirs4all-methods (B) | Operators / kernels | abs_tol | rel_tol | binding_tol | Comparison set | Notes |
|---|---|---|---|---|---|---|---|---|
| `n4m-scatter-preprocessing-stateless`            | NumPy / scikit-learn 1.8.0 stateless preprocessing reference | `n4m_pp_*_apply` (stateless ops) | identity / center / normalize / SNV / RNV / area / LSNV / log_transform | 1e-12 | 1e-12 | 1e-6 | transform outputs | operator parity |
| `n4m-scatter-preprocessing-stateful`             | NumPy / scikit-learn 1.8.0 stateful preprocessing reference | `n4m_pp_*_fit` + `_transform` | baseline-center / MSC / EMSC / Pareto / simple-scale (stateful family) | 1e-12 | 1e-12 | 1e-6 | transform outputs | stateful operator parity |
| `n4m-derivatives-smoothing`                      | NumPy / SciPy 1.17.1 reference | `n4m_pp_derivate_*`, `n4m_pp_savgol_*`, `n4m_pp_gaussian_*`, `n4m_pp_norris_williams_*` | derivate (1st / 2nd) / Savitzky-Golay / Gaussian filter / Norris-Williams | 1e-12 | 1e-12 | 1e-6 | transform outputs | smoothing parity |
| `n4m-baselines-asls-family`                      | Frozen pybaselines-derived reference; current upstream comparator pybaselines 1.2.1 | `n4m_pp_{asls,airpls,arpls,iasls,beads}_*` | AsLS / AirPLS / ArPLS / IAsLS / BEADS | **1e-11** | **1e-10** | 1e-6 | transform outputs | LDLT vs SuperLU pivot differences (Phase 5a/5b) |
| `n4m-baselines-polynomial`                       | Internal NumPy/pybaselines-derived fixture | `n4m_pp_{detrend,modpoly,imodpoly,snip,rolling_ball}_*` | Detrend / ModPoly / IModPoly / SNIP / RollingBall | 1e-12 | 1e-12 | 1e-6 | transform outputs | polynomial baseline parity |
| `n4m-signal-conversion`                          | NumPy reference (Beer-Lambert / Kubelka-Munk / percent ↔ fraction) | `n4m_pp_{to_absorbance,from_absorbance,kubelka_munk,pct_to_frac,frac_to_pct}_apply` | A↔R conversion / Kubelka-Munk / unit scaling | 1e-12 | 1e-12 | 1e-6 | transform outputs | signal-conversion parity (Phase 7) |
| `n4m-orthogonalization`                          | NumPy reference port of nirs4all OSC / EPO | `n4m_pp_{osc,epo}_*` | Orthogonal Signal Correction / External Parameter Orthogonalisation | 1e-10 | 1e-10 | 1e-6 | transform outputs | orthogonalisation parity (Phase 8) |
| `n4m-feature-selection-pca`                      | scikit-learn 1.8.0 / NumPy thin-SVD reference | `n4m_pp_{flexible_pca,flexible_svd}_*` | PCA (auto / svd / eigh) / thin SVD | 1e-10 | 1e-10 | 1e-6 | transform outputs, component vectors | feature-selection parity (Phase 9) |
| `n4m-resampling`                                 | NumPy / SciPy interp reference | `n4m_pp_{resample,resampler,crop}_*` | linear / cubic resample / crop | 1e-12 | 1e-12 | 1e-6 | transform outputs | resampling parity (Phase 10) |
| `n4m-splitters-deterministic`                    | NumPy PCG64 reference | `n4m_split_{kennard_stone,spxy,spxy_fold,spxy_g_fold,kmeans,kbins_stratified,binned_strat_group_kfold,systematic_circular,split_splitter}_*` | nine splitter algorithms | exact | exact | exact | train / test index plans (int64) | byte-exact integer indices for fixed seed (Phase 11) — bindings must forward without reordering |
| `n4m-filters-y-outliers`                         | NumPy reference | `n4m_filter_y_outlier_{fit,apply,is_fitted}` | IQR / Z-score / Percentile / MAD | exact | exact | exact | keep-mask (uint8), exclusion counts | byte-exact masks (Phase 12) — fit/apply split since ABI 1.9.0 |
| `n4m-filters-meta`                               | NumPy reference (hat-matrix / PCA-leverage / quality thresholds) | `n4m_filter_{leverage,quality,composite}_*` | leverage (hat / PCA-score) / quality / composite (ANY / ALL) | 1e-10 | 1e-10 | 1e-6 | keep-mask + per-row scores | meta-filter parity (Phase 14) |
| `n4m-signal-type-detection`                      | NumPy heuristic reference | `n4m_signal_detect` | Absorbance / Reflectance / Transmittance / Kubelka-Munk / log(1/R) / log(1/T) | exact | exact | exact | detected enum + confidence ± 1e-12 | signal-type parity (Phase 19) |
| `n4m-nirs-metrics`                               | NumPy closed-form reference | `n4m_metric_{rmse,mae,bias,sep,rpd,rpiq,r2,nrmse}` | 8 regression metrics (sklearn / nirs4all convention) | 1e-12 | 1e-12 | 1e-6 | scalar metrics | metric parity (Phase 19) |
| `n4m-hotelling-q`                                | NumPy / SciPy F + chi² UCL reference | `n4m_util_{hotelling_t2,q_residuals}` | Hotelling T² + Q-residuals + analytic UCL | 1e-10 | 1e-10 | 1e-6 | per-sample T²/Q + UCL | multivariate outlier stats (Phase 19) |
| `n4m-transfer-metrics`                           | NumPy reference port of nirs4all `transfer_metrics` | `n4m_transfer_metrics_compute` | nine alignment metrics (Mahalanobis / Wasserstein / Grassmann / etc.) | 1e-10 | 1e-10 | 1e-6 | nine scalar metrics | transfer-metric parity (Phase 20) |
| `n4m-fck-static-transformer`                     | NumPy `convolve1d` reference (nirs4all FCK kernel) | `n4m_pp_fck_static_{create,transform}` | (alpha, sigma) fractional convolution kernel bank | 1e-12 | 1e-13 | 1e-6 | per-kernel output bands | FCK parity (Phase 21) |

## Conventions

* The `abs_tol` and `rel_tol` columns are Stage-3 tolerances (C++ vs
  internal Python parity fixtures); they are sized so the nirs4all-methods C kernel
  passes against the internal parity fixture under the locked dependency set
  (`parity/python_generator/requirements-lock.txt`). Bumping a reference
  package may require relaxing the corresponding row — that bump must be
  driven by the parity gate (`run_parity_gate.py`), not by editing this
  table.
* The `binding_tol` column is the Stage-4 tolerance (per-binding vs
  libn4m, comparator at `parity/binding_parity.py`). The project-wide
  default is `1e-6` (matches pls4all). For deterministic-by-construction
  ops (splitters, filter masks, signal-type detection) the binding is
  required to be bit-exact (`binding_tol = exact`). Stage 4 is a no-op
  until at least one binding ships in Phase 22.
* "exact" means int64 / uint8 byte-equality (used for splitter indices and
  filter keep-masks where the C kernel — and any binding — is required to
  reproduce the Python reference output exactly).
* The parity-gate workflow (`.github/workflows/parity-gate.yml`) consumes
  this file informally — the C++ tests and Stage-2 reference runner carry
  their own tolerance constants per fixture and assert against them directly.
  This table is the contract documentation until the tolerance manifest is
  generated from a single machine-readable source.
