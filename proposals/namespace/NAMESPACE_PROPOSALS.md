# Namespace organization proposals for `nirs4all-methods` (`libn4m`)

**Status:** decision-ready proposal. Pick one scheme; this document is the input to that decision. **No source code is changed here.**

**Scope:** all **208** catalogued methods (`catalog/methods/*.yaml`), ground-truthed against
`proposals/namespace/_method_inventory.json` (208 unique ids). The two schemes below
each map **all 208** methods exactly once — the mapping tables were generated and
validated programmatically by `proposals/namespace/_build_proposals.py`
(`mappedStats == mappedMl == 208`, zero missing, zero double-mapped).

## Verification (coverage audit)

Re-checked by re-parsing the §A.2 / §B.2 mapping tables straight out of this markdown and diffing
their method-id sets against `_method_inventory.json`. All counts below were computed, not asserted:

| Quantity | Count |
|---|---|
| Inventory records (`_method_inventory.json`) | 208 |
| Inventory **unique** ids | 208 |
| Method ids mapped in **Scheme A** (§A.2) | 208 |
| Method ids mapped in **Scheme B** (§B.2) | 208 |
| Orphans (inventory id missing from a table) | 0 (both schemes) |
| Duplicates (id appearing twice in a table) | 0 (both schemes) |
| Fully-qualified-name collisions (two ids → same `n4m.<ns>.<leaf>`) | 0 (both schemes) |

**Coverage confirmed:** `inventory_total (208) == mapped_in_stats (208) == mapped_in_ml (208)`, with
zero orphans, zero duplicates, and zero name collisions in either scheme. Structural counts also
verified against the tables: Scheme A = 12 top-levels / 31 leaf namespaces; Scheme B = 12 top-levels
/ 39 leaf namespaces; every leaf node in the §A.1 / §B.1 trees is populated and every namespace used
in a mapping table appears in its tree (no orphan nodes either direction). The two schemes are
genuinely distinct framings, not relabeled twins: four Scheme-A top-levels fan out across multiple
Scheme-B top-levels (`outliers` → `feature_selection` + `outlier_detection`; `regression` →
`estimators` + `ensemble`; `transfer` → `domain_adaptation` + `transform`; `validation` → `metrics`
+ `utils`) and three Scheme-B top-levels gather from multiple Scheme-A top-levels (`estimators` ←
`regression` + `discriminant` + `multiblock`; `feature_selection` ← `outliers` + `variable_selection`;
`transform` ← `preprocessing` + `transfer`).

---

## 0. Why this is needed (problem recap)

The clean 9-category / 44-family / `category.subcategory.method` taxonomy that the catalog proves
is well-formed for all 208 methods **exists only as YAML filenames**. The actual shipped surface
throws it away on all three faces:

- **C ABI** — the 11 per-category headers (`preprocessing.h`, `models.h`, `selection.h`, ...) are
  empty ~39-line stubs that just `#include "n4m/n4m.h"` with a `TODO`. The real surface is two flat
  headers: `n4m.h` (2838 lines, ~498 `N4M_API` decls) + `pls.h` (2373 lines, ~204 decls), organized
  only by 29 historical `Phase N` build-order banners. Header placement even contradicts the catalog
  (`cow_align` → `n4m.h`, `ridge` → `pls.h`).
- **Python** — `bindings/python/src/n4m/python.py` is one 10311-line flat file: 170 free functions,
  0 classes, 0 subpackages. `__init__.py` then dumps ~200 names (sklearn classes, ~45
  `Native*Regressor` AOM classes, bare function aliases) into the flat top level. Discovery is
  `dir()`/grep, not navigation.
- **R / others** — inherit the same flatness.

Both schemes replace the two flat dumps with a **navigable namespace**. They differ in the *axis*
of organization:

| | Scheme A — STATS / CHEMOMETRICS | Scheme B — ML / DL PIPELINE |
|---|---|---|
| Organizing axis | **statistical/mathematical nature** of the method | **pipeline ROLE + estimator API** |
| Audience | chemometricians, statisticians, R users | scikit-learn / PyTorch users |
| Mirrors | R `pls`, `mdatools`, `ChemoSpec`, `prospectr` | sklearn module layout + DL augmentation libs |
| Top-level count | 12 | 12 |
| Leaf-namespace count | 31 | 39 |
| Key question it answers | "what *kind of statistics* is this?" | "*where in my pipeline* does this go?" |

A method's **`kind`** (derived in the inventory: `stateful_transformer` 65, `applier` 44,
`estimator` 38, `selector` 25, `splitter` 6, `metric` 5, `utility_fn` 6, `aom_operator` 3, plus 16
`null`/Python-only) and its **`category`** are the two levers both schemes pull on. Scheme A leans on
mathematical family; Scheme B leans on `kind` (estimator API role).

---

# Scheme A — STATS / CHEMOMETRICS

Organized by **statistical and mathematical nature**, the way the chemometrics literature and the
canonical R packages carve the space. The top tier is the set of **analysis families** a
chemometrician already thinks in (latent-variable regression, penalized/sparse regression,
discriminant analysis, multiblock, variable selection, signal preprocessing, calibration transfer,
resampling/validation, diagnostics), not a single `chemometrics.` umbrella — this matches how `pls`
exposes `plsr`/`pcr`, `mdatools` exposes `pls`/`pca`/`simca`/`mcr`, and `prospectr` exposes
preprocessing + sample selection as peers.

## A.1 Namespace tree (rationale per node)

```
n4m
├── regression                         Supervised continuous-response calibration (the core of NIRS chemometrics).
│   ├── latent                         Latent-variable regression: PLS/PCR/CPPLS/kernel-PLS/continuum/GPR/tensor/LWPLS.
│   ├── penalized                      Coefficient-shrinkage / robust-loss regression (ridge, weighted, robust PLS).
│   ├── sparse                         Sparsity-inducing latent regression (sparse SIMPLS, fused/group sparse PLS).
│   ├── ensemble                       Resampling ensembles of latent regressors (bagging/boosting/subspace/stack).
│   └── glm_survival                   Non-Gaussian response families & survival (PLS-GLM, PLS-Cox).
├── discriminant                       Discriminant analysis & supervised classification on latent scores.
├── multiblock                         Multiblock / multiway latent models (MB-PLS, SO/ON/O2-PLS, ROSA, mid-IR PLS).
├── transfer                           Calibration transfer & instrument standardization (DS/PDS/DI-PLS/EPO/slope-bias).
├── variable_selection                 Wavelength / variable selection: wrappers, ranker, interval generator.
├── preprocessing                      Signal pre-treatment of the spectral matrix (the prospectr/ChemoSpec world).
│   ├── scatter                        Scatter correction (SNV/MSC/EMSC/VSN families + local/piecewise variants).
│   ├── baseline                       Baseline estimation & removal (AsLS/airPLS/arPLS/SNIP/rolling-ball/detrend...).
│   ├── smoothing_derivative           Smoothing & finite/SG/Norris-Williams derivatives.
│   ├── wavelet                        Wavelet transforms, denoising, and wavelet feature extraction.
│   ├── scaling                        Per-feature scaling / normalization / log.
│   ├── alignment                      Peak/axis alignment & warping (COW/DTW/icoshift/xcorr).
│   ├── resampling                     Wavelength-axis resampling, cropping, binning/discretizing.
│   ├── signal_conversion              Radiometric conversions (absorbance/reflectance/Kubelka-Munk/unit scaling).
│   ├── osc                            Orthogonal signal correction (response-orthogonal filtering).
│   └── specialized                    Domain-specific static preprocessing (FCK).
├── decomposition                      Unsupervised latent projections used as features (flexible PCA / SVD).
├── outliers                           Sample/variable screening: leverage, Q-residuals, Hotelling T2, filters.
├── resampling                         Calibration/validation *design*: sample selection & data splitting.
├── validation                         Model diagnostics, validation metrics, CV scoring helpers, moments/sweep.
├── optimization                       Operator/preprocessing-space search & hyperparameter optimization.
│   └── aom                            AOM / POP families: sweeps, superblocks, blenders, staged campaigns.
└── simulation                         Physically/statistically motivated spectral perturbation (data augmentation).
    ├── scatter                        Simulated scatter/particle-size/batch effects.
    ├── baseline_drift                 Simulated drift & path-length variation.
    ├── instrument                     Simulated instrument artefacts & environmental effects.
    ├── noise                          Additive/multiplicative/heteroscedastic/spike noise.
    ├── wavelength                     Simulated wavelength shift/stretch/warp.
    ├── intensity                      Intensity/band perturbations & spline-based reshaping.
    └── mixing                         Sample mixing (mixup) & generic random ops.
```

**Design notes.**
- `regression` is deliberately the largest top-level (22 methods across 5 leaves) because
  latent-variable regression is the centre of gravity of the field; sub-splitting by *estimator
  mathematics* (latent vs penalized vs sparse vs ensemble vs GLM) is exactly the `pls` /
  `mdatools` mental model.
- `transfer` is promoted to a **top-level** family (not buried in preprocessing) because in
  chemometrics calibration transfer is its own literature; it deliberately co-locates
  transfer *models* (`di_pls`) with transfer *preprocessors* (`ds`, `pds`, `epo`, `slope_bias`,
  the four `direct_standardization` variants) and the `transfer_metrics` diagnostic — the user
  thinks "transfer", not "is this a model or a preprocessor".
- `preprocessing` keeps the literature's sub-families intact (scatter / baseline / derivative /
  wavelet …), which is precisely how ChemoSpec and prospectr document themselves.
- `optimization.aom` keeps the whole AOM/POP product line in one mathematically-coherent place
  (operator-space optimization), rather than scattering its 18 members across regression and
  pipeline nodes.
- `simulation` is the chemometrician's name for augmentation: physically-motivated perturbation of
  the spectral matrix. Sub-nodes follow the *physical origin* of the perturbation.

## A.2 Complete mapping table (208 / 208)

Every method id → fully-qualified Python namespace (`n4m.<ns>`). C-ABI prefix and R prefix follow
the same `<ns>` token (see §A.3). Rows are grouped by namespace.

| method id (catalog) | Scheme A namespace | leaf name |
|---|---|---|
| `preprocessing.feature_selection.flexible_pca` | `n4m.decomposition` | `flexible_pca` |
| `preprocessing.feature_selection.flexible_svd` | `n4m.decomposition` | `flexible_svd` |
| `models.classification.pls_lda` | `n4m.discriminant` | `pls_lda` |
| `models.classification.pls_logistic` | `n4m.discriminant` | `pls_logistic` |
| `models.classification.pls_qda` | `n4m.discriminant` | `pls_qda` |
| `models.sparse.sparse_pls_da` | `n4m.discriminant` | `sparse_pls_da` |
| `models.multiblock.mb_pls` | `n4m.multiblock` | `mb_pls` |
| `models.multiblock.mir_pls` | `n4m.multiblock` | `mir_pls` |
| `models.multiblock.o2pls` | `n4m.multiblock` | `o2pls` |
| `models.multiblock.on_pls` | `n4m.multiblock` | `on_pls` |
| `models.multiblock.rosa` | `n4m.multiblock` | `rosa` |
| `models.multiblock.so_pls` | `n4m.multiblock` | `so_pls` |
| `aom_pop.aom_chain_fixed_fit` | `n4m.optimization.aom` | `aom_chain_fixed_fit` |
| `aom_pop.aom_chain_ridge_pls` | `n4m.optimization.aom` | `aom_chain_ridge_pls` |
| `aom_pop.aom_chain_screen_refit` | `n4m.optimization.aom` | `aom_chain_screen_refit` |
| `aom_pop.aom_chain_sweep` | `n4m.optimization.aom` | `aom_chain_sweep` |
| `aom_pop.aom_pls` | `n4m.optimization.aom` | `aom_pls` |
| `aom_pop.aom_pls_superblock` | `n4m.optimization.aom` | `aom_pls_superblock` |
| `aom_pop.aom_preprocessing` | `n4m.optimization.aom` | `aom_preprocessing` |
| `aom_pop.aom_ridge_pls_superblock` | `n4m.optimization.aom` | `aom_ridge_pls_superblock` |
| `aom_pop.aom_staged_chain_campaign` | `n4m.optimization.aom` | `aom_staged_chain_campaign` |
| `aom_pop.aom_sweep` | `n4m.optimization.aom` | `aom_sweep` |
| `aom_pop.operator_pls_stack` | `n4m.optimization.aom` | `operator_pls_stack` |
| `aom_pop.pop_pls` | `n4m.optimization.aom` | `pop_pls` |
| `aom_pop.ridge_active_superblock` | `n4m.optimization.aom` | `ridge_active_superblock` |
| `aom_pop.ridge_blender` | `n4m.optimization.aom` | `ridge_blender` |
| `aom_pop.ridge_global` | `n4m.optimization.aom` | `ridge_global` |
| `aom_pop.ridge_mkl_superblock` | `n4m.optimization.aom` | `ridge_mkl_superblock` |
| `aom_pop.ridge_superblock` | `n4m.optimization.aom` | `ridge_superblock` |
| `aom_pop.robust_hpo` | `n4m.optimization.aom` | `robust_hpo` |
| `filters.composite` | `n4m.outliers` | `composite` |
| `filters.correlation` | `n4m.outliers` | `correlation` |
| `filters.high_leverage` | `n4m.outliers` | `high_leverage` |
| `filters.spectral_quality` | `n4m.outliers` | `spectral_quality` |
| `filters.variance` | `n4m.outliers` | `variance` |
| `filters.x_outlier` | `n4m.outliers` | `x_outlier` |
| `filters.y_outlier` | `n4m.outliers` | `y_outlier` |
| `utilities.hotelling_t2` | `n4m.outliers` | `hotelling_t2` |
| `utilities.q_residuals` | `n4m.outliers` | `q_residuals` |
| `preprocessing.alignment.cow_align` | `n4m.preprocessing.alignment` | `cow_align` |
| `preprocessing.alignment.dtw_align` | `n4m.preprocessing.alignment` | `dtw_align` |
| `preprocessing.alignment.icoshift_align` | `n4m.preprocessing.alignment` | `icoshift_align` |
| `preprocessing.alignment.xcorr_align` | `n4m.preprocessing.alignment` | `xcorr_align` |
| `preprocessing.baselines.airpls` | `n4m.preprocessing.baseline` | `airpls` |
| `preprocessing.baselines.arpls` | `n4m.preprocessing.baseline` | `arpls` |
| `preprocessing.baselines.asls` | `n4m.preprocessing.baseline` | `asls` |
| `preprocessing.baselines.beads` | `n4m.preprocessing.baseline` | `beads` |
| `preprocessing.baselines.detrend` | `n4m.preprocessing.baseline` | `detrend` |
| `preprocessing.baselines.iasls` | `n4m.preprocessing.baseline` | `iasls` |
| `preprocessing.baselines.imodpoly` | `n4m.preprocessing.baseline` | `imodpoly` |
| `preprocessing.baselines.modpoly` | `n4m.preprocessing.baseline` | `modpoly` |
| `preprocessing.baselines.rolling_ball` | `n4m.preprocessing.baseline` | `rolling_ball` |
| `preprocessing.baselines.saps` | `n4m.preprocessing.baseline` | `saps` |
| `preprocessing.baselines.snip` | `n4m.preprocessing.baseline` | `snip` |
| `preprocessing.orthogonalization.osc` | `n4m.preprocessing.osc` | `osc` |
| `preprocessing.resampling.crop` | `n4m.preprocessing.resampling` | `crop` |
| `preprocessing.resampling.kbins_discretizer` | `n4m.preprocessing.resampling` | `kbins_discretizer` |
| `preprocessing.resampling.range_discretizer` | `n4m.preprocessing.resampling` | `range_discretizer` |
| `preprocessing.resampling.resample_transformer` | `n4m.preprocessing.resampling` | `resample_transformer` |
| `preprocessing.resampling.resampler` | `n4m.preprocessing.resampling` | `resampler` |
| `preprocessing.scaling.baseline` | `n4m.preprocessing.scaling` | `baseline` |
| `preprocessing.scaling.log_transform` | `n4m.preprocessing.scaling` | `log_transform` |
| `preprocessing.scaling.normalize` | `n4m.preprocessing.scaling` | `normalize` |
| `preprocessing.scaling.simple_scale` | `n4m.preprocessing.scaling` | `simple_scale` |
| `preprocessing.scatter.area_normalization` | `n4m.preprocessing.scatter` | `area_normalization` |
| `preprocessing.scatter.emsc` | `n4m.preprocessing.scatter` | `emsc` |
| `preprocessing.scatter.local_centering` | `n4m.preprocessing.scatter` | `local_centering` |
| `preprocessing.scatter.local_snv` | `n4m.preprocessing.scatter` | `local_snv` |
| `preprocessing.scatter.localized_msc` | `n4m.preprocessing.scatter` | `localized_msc` |
| `preprocessing.scatter.msc` | `n4m.preprocessing.scatter` | `msc` |
| `preprocessing.scatter.piecewise_msc` | `n4m.preprocessing.scatter` | `piecewise_msc` |
| `preprocessing.scatter.piecewise_snv` | `n4m.preprocessing.scatter` | `piecewise_snv` |
| `preprocessing.scatter.robust_snv` | `n4m.preprocessing.scatter` | `robust_snv` |
| `preprocessing.scatter.snv` | `n4m.preprocessing.scatter` | `snv` |
| `preprocessing.scatter.vsn` | `n4m.preprocessing.scatter` | `vsn` |
| `preprocessing.scatter.weighted_snv` | `n4m.preprocessing.scatter` | `weighted_snv` |
| `preprocessing.signal_conversion.fraction_to_percent` | `n4m.preprocessing.signal_conversion` | `fraction_to_percent` |
| `preprocessing.signal_conversion.from_absorbance` | `n4m.preprocessing.signal_conversion` | `from_absorbance` |
| `preprocessing.signal_conversion.kubelka_munk` | `n4m.preprocessing.signal_conversion` | `kubelka_munk` |
| `preprocessing.signal_conversion.percent_to_fraction` | `n4m.preprocessing.signal_conversion` | `percent_to_fraction` |
| `preprocessing.signal_conversion.to_absorbance` | `n4m.preprocessing.signal_conversion` | `to_absorbance` |
| `preprocessing.derivatives.derivate` | `n4m.preprocessing.smoothing_derivative` | `derivate` |
| `preprocessing.derivatives.first_derivative` | `n4m.preprocessing.smoothing_derivative` | `first_derivative` |
| `preprocessing.derivatives.norris_williams` | `n4m.preprocessing.smoothing_derivative` | `norris_williams` |
| `preprocessing.derivatives.savitzky_golay` | `n4m.preprocessing.smoothing_derivative` | `savitzky_golay` |
| `preprocessing.derivatives.second_derivative` | `n4m.preprocessing.smoothing_derivative` | `second_derivative` |
| `preprocessing.smoothing.gaussian` | `n4m.preprocessing.smoothing_derivative` | `gaussian` |
| `preprocessing.specialized.fck_static` | `n4m.preprocessing.specialized` | `fck_static` |
| `preprocessing.wavelets.haar` | `n4m.preprocessing.wavelet` | `haar` |
| `preprocessing.wavelets.wavelet` | `n4m.preprocessing.wavelet` | `wavelet` |
| `preprocessing.wavelets.wavelet_denoise` | `n4m.preprocessing.wavelet` | `wavelet_denoise` |
| `preprocessing.wavelets.wavelet_features` | `n4m.preprocessing.wavelet` | `wavelet_features` |
| `preprocessing.wavelets.wavelet_pca` | `n4m.preprocessing.wavelet` | `wavelet_pca` |
| `preprocessing.wavelets.wavelet_svd` | `n4m.preprocessing.wavelet` | `wavelet_svd` |
| `models.ensembles.bagging_pls` | `n4m.regression.ensemble` | `bagging_pls` |
| `models.ensembles.boosting_pls` | `n4m.regression.ensemble` | `boosting_pls` |
| `models.ensembles.moment_stack` | `n4m.regression.ensemble` | `moment_stack` |
| `models.ensembles.random_subspace_pls` | `n4m.regression.ensemble` | `random_subspace_pls` |
| `models.heads.pls_cox` | `n4m.regression.glm_survival` | `pls_cox` |
| `models.heads.pls_glm` | `n4m.regression.glm_survival` | `pls_glm` |
| `models.local.lw_pls` | `n4m.regression.latent` | `lw_pls` |
| `models.pls.cppls` | `n4m.regression.latent` | `cppls` |
| `models.pls.kernel` | `n4m.regression.latent` | `kernel` |
| `models.pls.pcr` | `n4m.regression.latent` | `pcr` |
| `models.pls.pls_fit_simple` | `n4m.regression.latent` | `pls_fit_simple` |
| `models.regularized.continuum_regression` | `n4m.regression.latent` | `continuum_regression` |
| `models.regularized.ridge_pls` | `n4m.regression.latent` | `ridge_pls` |
| `models.specialized.ecr` | `n4m.regression.latent` | `ecr` |
| `models.specialized.gpr_pls` | `n4m.regression.latent` | `gpr_pls` |
| `models.specialized.missing_aware_nipals` | `n4m.regression.latent` | `missing_aware_nipals` |
| `models.specialized.recursive` | `n4m.regression.latent` | `recursive` |
| `models.specialized.tensor_pls` | `n4m.regression.latent` | `tensor_pls` |
| `models.regularized.ridge` | `n4m.regression.penalized` | `ridge` |
| `models.regularized.robust_pls` | `n4m.regression.penalized` | `robust_pls` |
| `models.regularized.weighted_pls` | `n4m.regression.penalized` | `weighted_pls` |
| `models.sparse.fused_sparse_pls` | `n4m.regression.sparse` | `fused_sparse_pls` |
| `models.sparse.group_sparse_pls` | `n4m.regression.sparse` | `group_sparse_pls` |
| `models.sparse.sparse_simpls` | `n4m.regression.sparse` | `sparse_simpls` |
| `splitters.binned_strat_group_kfold` | `n4m.resampling` | `binned_strat_group_kfold` |
| `splitters.kbins_stratified` | `n4m.resampling` | `kbins_stratified` |
| `splitters.kennard_stone` | `n4m.resampling` | `kennard_stone` |
| `splitters.kmeans` | `n4m.resampling` | `kmeans` |
| `splitters.split_splitter` | `n4m.resampling` | `split_splitter` |
| `splitters.spxy` | `n4m.resampling` | `spxy` |
| `splitters.spxy_fold` | `n4m.resampling` | `spxy_fold` |
| `splitters.spxy_g_fold` | `n4m.resampling` | `spxy_g_fold` |
| `splitters.systematic_circular` | `n4m.resampling` | `systematic_circular` |
| `augmentation.drift.linear_drift` | `n4m.simulation.baseline_drift` | `linear_drift` |
| `augmentation.drift.path_length` | `n4m.simulation.baseline_drift` | `path_length` |
| `augmentation.drift.poly_drift` | `n4m.simulation.baseline_drift` | `poly_drift` |
| `augmentation.edge_artifacts.detector_rolloff` | `n4m.simulation.instrument` | `detector_rolloff` |
| `augmentation.edge_artifacts.edge_artifacts` | `n4m.simulation.instrument` | `edge_artifacts` |
| `augmentation.edge_artifacts.edge_curvature` | `n4m.simulation.instrument` | `edge_curvature` |
| `augmentation.edge_artifacts.stray_light` | `n4m.simulation.instrument` | `stray_light` |
| `augmentation.edge_artifacts.truncated_peak` | `n4m.simulation.instrument` | `truncated_peak` |
| `augmentation.environmental.moisture` | `n4m.simulation.instrument` | `moisture` |
| `augmentation.environmental.temperature` | `n4m.simulation.instrument` | `temperature` |
| `augmentation.spectral.band_mask` | `n4m.simulation.intensity` | `band_mask` |
| `augmentation.spectral.band_perturb` | `n4m.simulation.intensity` | `band_perturb` |
| `augmentation.spectral.channel_dropout` | `n4m.simulation.intensity` | `channel_dropout` |
| `augmentation.spectral.gauss_jitter` | `n4m.simulation.intensity` | `gauss_jitter` |
| `augmentation.spectral.local_clip` | `n4m.simulation.intensity` | `local_clip` |
| `augmentation.spectral.magnitude_warp` | `n4m.simulation.intensity` | `magnitude_warp` |
| `augmentation.spectral.unsharp_mask` | `n4m.simulation.intensity` | `unsharp_mask` |
| `augmentation.splines.spline_curve_simplification` | `n4m.simulation.intensity` | `spline_curve_simplification` |
| `augmentation.splines.spline_smoothing` | `n4m.simulation.intensity` | `spline_smoothing` |
| `augmentation.splines.spline_x_perturbations` | `n4m.simulation.intensity` | `spline_x_perturbations` |
| `augmentation.splines.spline_x_simplification` | `n4m.simulation.intensity` | `spline_x_simplification` |
| `augmentation.splines.spline_y_perturbations` | `n4m.simulation.intensity` | `spline_y_perturbations` |
| `augmentation.mixup.local_mixup` | `n4m.simulation.mixing` | `local_mixup` |
| `augmentation.mixup.mixup` | `n4m.simulation.mixing` | `mixup` |
| `augmentation.random.random_x_op` | `n4m.simulation.mixing` | `random_x_op` |
| `augmentation.random.rotate_translate` | `n4m.simulation.mixing` | `rotate_translate` |
| `augmentation.noise.gaussian_noise` | `n4m.simulation.noise` | `gaussian_noise` |
| `augmentation.noise.hetero_noise` | `n4m.simulation.noise` | `hetero_noise` |
| `augmentation.noise.multiplicative_noise` | `n4m.simulation.noise` | `multiplicative_noise` |
| `augmentation.noise.spike_noise` | `n4m.simulation.noise` | `spike_noise` |
| `augmentation.scattering.batch_effect` | `n4m.simulation.scatter` | `batch_effect` |
| `augmentation.scattering.dead_band` | `n4m.simulation.scatter` | `dead_band` |
| `augmentation.scattering.emsc_distort` | `n4m.simulation.scatter` | `emsc_distort` |
| `augmentation.scattering.instrument_broaden` | `n4m.simulation.scatter` | `instrument_broaden` |
| `augmentation.scattering.particle_size` | `n4m.simulation.scatter` | `particle_size` |
| `augmentation.scattering.scatter_sim_msc` | `n4m.simulation.scatter` | `scatter_sim_msc` |
| `augmentation.wavelength.local_warp` | `n4m.simulation.wavelength` | `local_warp` |
| `augmentation.wavelength.wavelength_shift` | `n4m.simulation.wavelength` | `wavelength_shift` |
| `augmentation.wavelength.wavelength_stretch` | `n4m.simulation.wavelength` | `wavelength_stretch` |
| `models.transfer.di_pls` | `n4m.transfer` | `di_pls` |
| `models.transfer.ds` | `n4m.transfer` | `ds` |
| `models.transfer.pds` | `n4m.transfer` | `pds` |
| `preprocessing.orthogonalization.epo` | `n4m.transfer` | `epo` |
| `preprocessing.transfer.direct_standardization` | `n4m.transfer` | `direct_standardization` |
| `preprocessing.transfer.piecewise_direct_standardization` | `n4m.transfer` | `piecewise_direct_standardization` |
| `preprocessing.transfer.robust_direct_standardization` | `n4m.transfer` | `robust_direct_standardization` |
| `preprocessing.transfer.slope_bias` | `n4m.transfer` | `slope_bias` |
| `utilities.transfer_metrics` | `n4m.transfer` | `transfer_metrics` |
| `diagnostics.approximate_press` | `n4m.validation` | `approximate_press` |
| `diagnostics.model_selection` | `n4m.validation` | `model_selection` |
| `diagnostics.pls_diagnostics` | `n4m.validation` | `pls_diagnostics` |
| `diagnostics.pls_monitoring` | `n4m.validation` | `pls_monitoring` |
| `diagnostics.regression_metrics` | `n4m.validation` | `regression_metrics` |
| `utilities.moments` | `n4m.validation` | `moments` |
| `utilities.signal_type_detector` | `n4m.validation` | `signal_type_detector` |
| `utilities.sweep` | `n4m.validation` | `sweep` |
| `selection.bipls` | `n4m.variable_selection` | `bipls` |
| `selection.bve` | `n4m.variable_selection` | `bve` |
| `selection.cars` | `n4m.variable_selection` | `cars` |
| `selection.emcuve` | `n4m.variable_selection` | `emcuve` |
| `selection.ga` | `n4m.variable_selection` | `ga` |
| `selection.interval` | `n4m.variable_selection` | `interval` |
| `selection.ipw` | `n4m.variable_selection` | `ipw` |
| `selection.irf` | `n4m.variable_selection` | `irf` |
| `selection.iriv` | `n4m.variable_selection` | `iriv` |
| `selection.pso` | `n4m.variable_selection` | `pso` |
| `selection.random_frog` | `n4m.variable_selection` | `random_frog` |
| `selection.randomization` | `n4m.variable_selection` | `randomization` |
| `selection.rep` | `n4m.variable_selection` | `rep` |
| `selection.scars` | `n4m.variable_selection` | `scars` |
| `selection.shaving` | `n4m.variable_selection` | `shaving` |
| `selection.sipls` | `n4m.variable_selection` | `sipls` |
| `selection.spa` | `n4m.variable_selection` | `spa` |
| `selection.st` | `n4m.variable_selection` | `st` |
| `selection.stability` | `n4m.variable_selection` | `stability` |
| `selection.t2` | `n4m.variable_selection` | `t2` |
| `selection.uve` | `n4m.variable_selection` | `uve` |
| `selection.variable_select` | `n4m.variable_selection` | `variable_select` |
| `selection.vip_spa` | `n4m.variable_selection` | `vip_spa` |
| `selection.vissa` | `n4m.variable_selection` | `vissa` |
| `selection.wvc` | `n4m.variable_selection` | `wvc` |

**Per-node counts (A):** `variable_selection` 25 · `optimization.aom` 18 · `regression.latent` 12 ·
`preprocessing.scatter` 12 · `simulation.intensity` 12 · `preprocessing.baseline` 11 · `outliers` 9 ·
`resampling` 9 · `transfer` 9 · `validation` 8 · `simulation.instrument` 7 · `multiblock` 6 ·
`preprocessing.smoothing_derivative` 6 · `preprocessing.wavelet` 6 · `simulation.scatter` 6 ·
`preprocessing.resampling` 5 · `preprocessing.signal_conversion` 5 · `discriminant` 4 ·
`preprocessing.scaling` 4 · `regression.ensemble` 4 · `simulation.mixing` 4 · `simulation.noise` 4 ·
`preprocessing.alignment` 4 · `regression.penalized` 3 · `regression.sparse` 3 ·
`simulation.baseline_drift` 3 · `simulation.wavelength` 3 · `decomposition` 2 ·
`regression.glm_survival` 2 · `preprocessing.osc` 1 · `preprocessing.specialized` 1. **Total 208.**

## A.3 API-surface implications (three faces)

### C ABI
- **Header split**: one umbrella `n4m/n4m.h` (types, context, status, `n4m_matrix_view_t`, the
  ownership/free contract) plus **one header per top-level family**:
  `n4m/regression.h`, `n4m/discriminant.h`, `n4m/multiblock.h`, `n4m/transfer.h`,
  `n4m/variable_selection.h`, `n4m/preprocessing.h`, `n4m/decomposition.h`, `n4m/outliers.h`,
  `n4m/resampling.h`, `n4m/validation.h`, `n4m/optimization.h`, `n4m/simulation.h`. Large families
  (`preprocessing`, `simulation`) get **second-level sub-headers** included by the family umbrella
  (e.g. `n4m/preprocessing/scatter.h`). This finally fills the 11 empty stub headers with real
  declarations — and replaces the wrong `pls.h`/`n4m.h` split.
- **Symbol prefix convention**: `n4m_<family>_<method>_<verb>(...)`, e.g.
  `n4m_regression_ridge_fit`, `n4m_preprocessing_snv_transform`,
  `n4m_variable_selection_cars_select`, `n4m_transfer_pds_fit`, `n4m_optimization_aom_sweep_run`,
  `n4m_simulation_gaussian_noise_apply`. The existing terse symbols (`n4m_ridge_fit`,
  `n4m_pp_snv_create`) become the **frozen ABI** and stay exported; the namespaced names are
  *additional* thin aliases (or a documented rename behind an ABI-version bump). Either way the
  family is now legible from the symbol.
- **ABI snapshot impact**: adding alias symbols is an additive ABI change → bump
  `N4M_ABI_VERSION_MINOR`, regenerate all three `cpp/abi/expected_symbols_*.txt`, log in
  `docs/abi/changes_log.md`. The numeric core in `cpp/src/core/` is untouched.

### Python (`replaces the 373 KB flat python.py`)
- `n4m/` becomes a **package of subpackages** mirroring the tree:
  `n4m/regression/{latent,penalized,sparse,ensemble,glm_survival}.py`, `n4m/discriminant.py`,
  `n4m/multiblock.py`, `n4m/transfer.py`, `n4m/variable_selection.py`,
  `n4m/preprocessing/{scatter,baseline,...}.py`, `n4m/decomposition.py`, `n4m/outliers.py`,
  `n4m/resampling.py`, `n4m/validation.py`, `n4m/optimization/aom.py`,
  `n4m/simulation/{scatter,baseline_drift,...}.py`.
- Discovery becomes navigation: `from n4m.regression.penalized import Ridge`,
  `from n4m.preprocessing.scatter import SNV`, `from n4m.variable_selection import CARS`,
  `from n4m.transfer import PDS`.
- The current `python.py` free functions are split into these modules unchanged in behaviour; the
  sklearn estimator classes move next to their function. `n4m/__init__.py` stops dumping ~200 names
  flat — it exposes only the subpackages (and, optionally, a curated `n4m.api` convenience surface).
- **R users targeting this scheme** read it as the natural home: `n4m.regression.latent.pls` is
  the `pls::plsr` analogue, `n4m.discriminant` is the `mdatools::simca` analogue.

### R
- Function convention `n4m_<family>_<method>()`, matching the C symbol stem so R docs and C docs
  line up: `n4m_regression_ridge()`, `n4m_preprocessing_snv()`, `n4m_variable_selection_cars()`,
  `n4m_transfer_pds()`. Because chemometricians are the R audience, the family-first naming reads
  like the existing R chemometrics packages.
- The slim **`pls4all`** CRAN subset maps cleanly to `n4m.regression.*` + `n4m.preprocessing.*` +
  `n4m.validation.*` (it is exactly the latent-regression + preprocessing slice).

## A.4 Pros / cons / migration / edge cases

**Pros**
- Matches the **mental model of the target audience** (chemometricians/statisticians) and the R
  package ecosystem; lowest "where do I look?" cost for that audience.
- Co-locates mathematically-related methods that the current surface scatters: all transfer in one
  place, all AOM in one place, all latent regression in one place.
- `regression.*` sub-splitting communicates the *estimator mathematics* (penalized vs sparse vs
  ensemble) which is genuinely useful information about the method.

**Cons**
- `kind` is **not** the organizing axis, so a pipeline author ("I need a transform here") must know
  that, e.g., scatter correction lives under `preprocessing` while scatter *augmentation* lives
  under `simulation` — a sklearn user finds this less obvious than Scheme B.
- `transfer` mixes `kind`s (estimators + transformers + a metric) under one node — coherent for a
  chemometrician, slightly surprising for an API-role purist.
- Two different nodes named `scatter` (`preprocessing.scatter` vs `simulation.scatter`) and two
  `resampling`-ish ideas (`preprocessing.resampling` = axis resampling vs top-level `resampling` =
  sample/CV design). Disambiguated by their parent, but worth a doc callout.

**Migration cost:** medium. Mechanical file-split of `python.py` + new C headers + alias symbols.
The numeric core and the frozen terse ABI symbols are untouched, so no parity fixtures change.

**Collision / edge-case decisions (flagged):**
| method | tension | resolution in Scheme A |
|---|---|---|
| `models.sparse.sparse_pls_da` | sparse regression vs discriminant | → `discriminant` (its *task* is classification; the "sparse" trait is secondary). |
| `models.classification.pls_logistic` | classifier vs GLM head | → `discriminant` (binary classification is its primary use; GLM lineage noted). |
| `models.heads.pls_glm` / `pls_cox` | regression vs classification vs survival | → `regression.glm_survival` (non-Gaussian/time-to-event responses kept together). |
| `preprocessing.orthogonalization.epo` | preprocessing vs transfer | → `transfer` (EPO's purpose is removing between-instrument/condition variance). |
| `preprocessing.orthogonalization.osc` | preprocessing vs transfer | → `preprocessing.osc` (response-orthogonal *signal* correction, not instrument transfer). |
| `models.transfer.di_pls` | estimator vs transfer | → `transfer` (it is a transfer *model*; the family wins over the API role). |
| `preprocessing.feature_selection.flexible_pca/_svd` | name says "feature_selection" but it is decomposition | → `decomposition` (PCA/SVD projection, **not** variable selection). |
| `selection.interval` | transformer vs selector | → `variable_selection` (it generates candidate intervals for selection). |
| `filters.correlation` / `filters.variance` | outlier filter vs feature filter | → `outliers` (kept with the other `filters.*` quality/screening tools for one-stop screening). |
| `utilities.moments` / `utilities.sweep` | low-level helper | → `validation` (diagnostic/scoring helpers; flagged as utility-grade). |

---

# Scheme B — ML / DL PIPELINE

Organized by **pipeline role and estimator API**, mirroring the scikit-learn module layout
(`sklearn.preprocessing`, `sklearn.decomposition`, `sklearn.feature_selection`, estimators,
`sklearn.ensemble`, `sklearn.model_selection`, `sklearn.compose`/`pipeline`, `sklearn.metrics`)
plus DL data-augmentation conventions. The top tier answers **"where in my pipeline does this go?"**
The inventory's `kind` field (estimator / stateful_transformer / applier / selector / splitter /
metric) is the primary lever.

## B.1 Namespace tree (rationale per node)

```
n4m
├── transform                          fit/transform stateful or stateless feature transforms (sklearn.preprocessing).
│   ├── scatter                        Scatter-correction transformers (SNV/MSC/EMSC/VSN ...).
│   ├── baseline                       Baseline-removal transformers.
│   ├── smoothing                      Smoothing & derivative transformers.
│   ├── wavelet                        Wavelet transformers / feature extractors.
│   ├── scaling                        Scaler/normalizer/log transformers.
│   ├── alignment                      Alignment/warping transformers.
│   ├── resampling                     Axis resample/crop/bin transformers.
│   ├── signal_conversion              Radiometric-conversion transformers.
│   ├── orthogonalization              OSC / EPO orthogonalizing transformers.
│   └── specialized                    Specialized static transformer (FCK).
├── decomposition                      Unsupervised projection transformers (PCA/SVD) — sklearn.decomposition.
├── feature_selection                  Variable selection — sklearn.feature_selection.
│   ├── wrapper                        Model-wrapped search selectors (CARS/GA/VISSA/UVE/...).
│   ├── filter                         Univariate/filter selectors (correlation, variance).
│   ├── ranking                        Importance ranking selector (variable_select).
│   └── interval                       Interval generator for interval-based selection.
├── augmentation                       DL-style data augmentation appliers (apply-only, no fit).
│   ├── noise                          Noise injection.
│   ├── scattering                     Scatter/particle/batch distortion.
│   ├── drift                          Baseline/path-length drift.
│   ├── instrument                     Instrument-artefact & environmental augmentation.
│   ├── wavelength                     Wavelength shift/stretch/warp.
│   ├── spectral                       Intensity/band perturbation.
│   ├── splines                        Spline-based reshaping/perturbation.
│   └── mixup                          Mixup + generic random ops.
├── estimators                         Supervised predictors (fit/predict) — sklearn estimators.
│   ├── regression
│   │   ├── linear                     Latent/penalized linear regressors (PLS/PCR/ridge/CPPLS/GPR/...).
│   │   └── sparse                     Sparse regressors (sparse SIMPLS, fused/group sparse PLS).
│   ├── classification                 Classifiers (PLS-LDA/QDA/logistic, sparse-PLS-DA).
│   ├── glm_survival                   Generalized-response & survival estimators (PLS-GLM, PLS-Cox).
│   ├── multiblock                     Multiblock estimators (MB/SO/ON/O2-PLS, ROSA, mid-IR PLS).
│   └── local                          Locally-weighted regressor (LW-PLS).
├── ensemble                           Meta-estimators over base learners — sklearn.ensemble.
├── model_selection                    CV splitters & sample-selection designs — sklearn.model_selection.
│   └── split                          Splitter/fold/sample-selection objects.
├── compose                            Pipeline composition & staged campaigns — sklearn.compose/pipeline.
│   ├── aom_search                     AOM/POP operator search & single-operator selection/fit.
│   ├── aom_superblock                 AOM superblock / stack / blender estimators.
│   └── aom_campaign                   Multi-stage screen→refit AOM campaigns.
├── domain_adaptation                  Calibration transfer = domain adaptation (DS/PDS/DI-PLS/EPO-style).
├── outlier_detection                  Outlier/novelty detection & sample filters — sklearn.ensemble/neighbors lineage.
├── metrics                            Scoring & diagnostics — sklearn.metrics.
│   ├── scoring                        Regression metrics, PRESS, model-selection rules.
│   └── diagnostics                    PLS diagnostics & monitoring.
└── utils                              Low-level helpers that aren't estimators/transformers/metrics.
```

**Design notes.**
- `transform` vs `augmentation` is the key sklearn-vs-DL split: `transform.*` are
  fit/transform objects you put in a `Pipeline`; `augmentation.*` are apply-only training-time
  perturbations (every member has `kind=applier`). This is the distinction a PyTorch/sklearn user
  expects and the current flat surface destroys.
- `estimators` is split first by **task** (regression / classification / glm_survival /
  multiblock / local), then regression by **regularization style** — mirroring how sklearn groups
  `linear_model` vs `cross_decomposition` but presented by predictive task.
- `compose` is where the AOM/POP product line lives, because to an ML user the AOM machinery *is*
  pipeline composition + automated preprocessing search (cf. sklearn `Pipeline` + `GridSearchCV`).
  It is sub-split into `aom_search` (find/fit one operator chain), `aom_superblock` (concatenated-
  block estimators), and `aom_campaign` (multi-stage screen→refit).
- `domain_adaptation` is the ML-native name for calibration transfer and is kept as its own
  top-level (sklearn has no first-class transfer module, so this also signals the project's
  differentiator).
- `outlier_detection` collects the `filters.*` sample screeners + the Q/T2 utilities, matching
  sklearn's outlier-detection framing.

## B.2 Complete mapping table (208 / 208)

| method id (catalog) | Scheme B namespace | leaf name |
|---|---|---|
| `augmentation.drift.linear_drift` | `n4m.augmentation.drift` | `linear_drift` |
| `augmentation.drift.path_length` | `n4m.augmentation.drift` | `path_length` |
| `augmentation.drift.poly_drift` | `n4m.augmentation.drift` | `poly_drift` |
| `augmentation.edge_artifacts.detector_rolloff` | `n4m.augmentation.instrument` | `detector_rolloff` |
| `augmentation.edge_artifacts.edge_artifacts` | `n4m.augmentation.instrument` | `edge_artifacts` |
| `augmentation.edge_artifacts.edge_curvature` | `n4m.augmentation.instrument` | `edge_curvature` |
| `augmentation.edge_artifacts.stray_light` | `n4m.augmentation.instrument` | `stray_light` |
| `augmentation.edge_artifacts.truncated_peak` | `n4m.augmentation.instrument` | `truncated_peak` |
| `augmentation.environmental.moisture` | `n4m.augmentation.instrument` | `moisture` |
| `augmentation.environmental.temperature` | `n4m.augmentation.instrument` | `temperature` |
| `augmentation.mixup.local_mixup` | `n4m.augmentation.mixup` | `local_mixup` |
| `augmentation.mixup.mixup` | `n4m.augmentation.mixup` | `mixup` |
| `augmentation.random.random_x_op` | `n4m.augmentation.mixup` | `random_x_op` |
| `augmentation.random.rotate_translate` | `n4m.augmentation.mixup` | `rotate_translate` |
| `augmentation.noise.gaussian_noise` | `n4m.augmentation.noise` | `gaussian_noise` |
| `augmentation.noise.hetero_noise` | `n4m.augmentation.noise` | `hetero_noise` |
| `augmentation.noise.multiplicative_noise` | `n4m.augmentation.noise` | `multiplicative_noise` |
| `augmentation.noise.spike_noise` | `n4m.augmentation.noise` | `spike_noise` |
| `augmentation.scattering.batch_effect` | `n4m.augmentation.scattering` | `batch_effect` |
| `augmentation.scattering.dead_band` | `n4m.augmentation.scattering` | `dead_band` |
| `augmentation.scattering.emsc_distort` | `n4m.augmentation.scattering` | `emsc_distort` |
| `augmentation.scattering.instrument_broaden` | `n4m.augmentation.scattering` | `instrument_broaden` |
| `augmentation.scattering.particle_size` | `n4m.augmentation.scattering` | `particle_size` |
| `augmentation.scattering.scatter_sim_msc` | `n4m.augmentation.scattering` | `scatter_sim_msc` |
| `augmentation.spectral.band_mask` | `n4m.augmentation.spectral` | `band_mask` |
| `augmentation.spectral.band_perturb` | `n4m.augmentation.spectral` | `band_perturb` |
| `augmentation.spectral.channel_dropout` | `n4m.augmentation.spectral` | `channel_dropout` |
| `augmentation.spectral.gauss_jitter` | `n4m.augmentation.spectral` | `gauss_jitter` |
| `augmentation.spectral.local_clip` | `n4m.augmentation.spectral` | `local_clip` |
| `augmentation.spectral.magnitude_warp` | `n4m.augmentation.spectral` | `magnitude_warp` |
| `augmentation.spectral.unsharp_mask` | `n4m.augmentation.spectral` | `unsharp_mask` |
| `augmentation.splines.spline_curve_simplification` | `n4m.augmentation.splines` | `spline_curve_simplification` |
| `augmentation.splines.spline_smoothing` | `n4m.augmentation.splines` | `spline_smoothing` |
| `augmentation.splines.spline_x_perturbations` | `n4m.augmentation.splines` | `spline_x_perturbations` |
| `augmentation.splines.spline_x_simplification` | `n4m.augmentation.splines` | `spline_x_simplification` |
| `augmentation.splines.spline_y_perturbations` | `n4m.augmentation.splines` | `spline_y_perturbations` |
| `augmentation.wavelength.local_warp` | `n4m.augmentation.wavelength` | `local_warp` |
| `augmentation.wavelength.wavelength_shift` | `n4m.augmentation.wavelength` | `wavelength_shift` |
| `augmentation.wavelength.wavelength_stretch` | `n4m.augmentation.wavelength` | `wavelength_stretch` |
| `aom_pop.aom_chain_screen_refit` | `n4m.compose.aom_campaign` | `aom_chain_screen_refit` |
| `aom_pop.aom_staged_chain_campaign` | `n4m.compose.aom_campaign` | `aom_staged_chain_campaign` |
| `aom_pop.aom_chain_fixed_fit` | `n4m.compose.aom_search` | `aom_chain_fixed_fit` |
| `aom_pop.aom_chain_sweep` | `n4m.compose.aom_search` | `aom_chain_sweep` |
| `aom_pop.aom_pls` | `n4m.compose.aom_search` | `aom_pls` |
| `aom_pop.aom_preprocessing` | `n4m.compose.aom_search` | `aom_preprocessing` |
| `aom_pop.aom_sweep` | `n4m.compose.aom_search` | `aom_sweep` |
| `aom_pop.pop_pls` | `n4m.compose.aom_search` | `pop_pls` |
| `aom_pop.ridge_global` | `n4m.compose.aom_search` | `ridge_global` |
| `aom_pop.robust_hpo` | `n4m.compose.aom_search` | `robust_hpo` |
| `aom_pop.aom_chain_ridge_pls` | `n4m.compose.aom_superblock` | `aom_chain_ridge_pls` |
| `aom_pop.aom_pls_superblock` | `n4m.compose.aom_superblock` | `aom_pls_superblock` |
| `aom_pop.aom_ridge_pls_superblock` | `n4m.compose.aom_superblock` | `aom_ridge_pls_superblock` |
| `aom_pop.operator_pls_stack` | `n4m.compose.aom_superblock` | `operator_pls_stack` |
| `aom_pop.ridge_active_superblock` | `n4m.compose.aom_superblock` | `ridge_active_superblock` |
| `aom_pop.ridge_blender` | `n4m.compose.aom_superblock` | `ridge_blender` |
| `aom_pop.ridge_mkl_superblock` | `n4m.compose.aom_superblock` | `ridge_mkl_superblock` |
| `aom_pop.ridge_superblock` | `n4m.compose.aom_superblock` | `ridge_superblock` |
| `preprocessing.feature_selection.flexible_pca` | `n4m.decomposition` | `flexible_pca` |
| `preprocessing.feature_selection.flexible_svd` | `n4m.decomposition` | `flexible_svd` |
| `models.transfer.di_pls` | `n4m.domain_adaptation` | `di_pls` |
| `models.transfer.ds` | `n4m.domain_adaptation` | `ds` |
| `models.transfer.pds` | `n4m.domain_adaptation` | `pds` |
| `preprocessing.transfer.direct_standardization` | `n4m.domain_adaptation` | `direct_standardization` |
| `preprocessing.transfer.piecewise_direct_standardization` | `n4m.domain_adaptation` | `piecewise_direct_standardization` |
| `preprocessing.transfer.robust_direct_standardization` | `n4m.domain_adaptation` | `robust_direct_standardization` |
| `preprocessing.transfer.slope_bias` | `n4m.domain_adaptation` | `slope_bias` |
| `utilities.transfer_metrics` | `n4m.domain_adaptation` | `transfer_metrics` |
| `models.ensembles.bagging_pls` | `n4m.ensemble` | `bagging_pls` |
| `models.ensembles.boosting_pls` | `n4m.ensemble` | `boosting_pls` |
| `models.ensembles.moment_stack` | `n4m.ensemble` | `moment_stack` |
| `models.ensembles.random_subspace_pls` | `n4m.ensemble` | `random_subspace_pls` |
| `models.classification.pls_lda` | `n4m.estimators.classification` | `pls_lda` |
| `models.classification.pls_logistic` | `n4m.estimators.classification` | `pls_logistic` |
| `models.classification.pls_qda` | `n4m.estimators.classification` | `pls_qda` |
| `models.sparse.sparse_pls_da` | `n4m.estimators.classification` | `sparse_pls_da` |
| `models.heads.pls_cox` | `n4m.estimators.glm_survival` | `pls_cox` |
| `models.heads.pls_glm` | `n4m.estimators.glm_survival` | `pls_glm` |
| `models.local.lw_pls` | `n4m.estimators.local` | `lw_pls` |
| `models.multiblock.mb_pls` | `n4m.estimators.multiblock` | `mb_pls` |
| `models.multiblock.mir_pls` | `n4m.estimators.multiblock` | `mir_pls` |
| `models.multiblock.o2pls` | `n4m.estimators.multiblock` | `o2pls` |
| `models.multiblock.on_pls` | `n4m.estimators.multiblock` | `on_pls` |
| `models.multiblock.rosa` | `n4m.estimators.multiblock` | `rosa` |
| `models.multiblock.so_pls` | `n4m.estimators.multiblock` | `so_pls` |
| `models.pls.cppls` | `n4m.estimators.regression.linear` | `cppls` |
| `models.pls.kernel` | `n4m.estimators.regression.linear` | `kernel` |
| `models.pls.pcr` | `n4m.estimators.regression.linear` | `pcr` |
| `models.pls.pls_fit_simple` | `n4m.estimators.regression.linear` | `pls_fit_simple` |
| `models.regularized.continuum_regression` | `n4m.estimators.regression.linear` | `continuum_regression` |
| `models.regularized.ridge` | `n4m.estimators.regression.linear` | `ridge` |
| `models.regularized.ridge_pls` | `n4m.estimators.regression.linear` | `ridge_pls` |
| `models.regularized.robust_pls` | `n4m.estimators.regression.linear` | `robust_pls` |
| `models.regularized.weighted_pls` | `n4m.estimators.regression.linear` | `weighted_pls` |
| `models.specialized.ecr` | `n4m.estimators.regression.linear` | `ecr` |
| `models.specialized.gpr_pls` | `n4m.estimators.regression.linear` | `gpr_pls` |
| `models.specialized.missing_aware_nipals` | `n4m.estimators.regression.linear` | `missing_aware_nipals` |
| `models.specialized.recursive` | `n4m.estimators.regression.linear` | `recursive` |
| `models.specialized.tensor_pls` | `n4m.estimators.regression.linear` | `tensor_pls` |
| `models.sparse.fused_sparse_pls` | `n4m.estimators.regression.sparse` | `fused_sparse_pls` |
| `models.sparse.group_sparse_pls` | `n4m.estimators.regression.sparse` | `group_sparse_pls` |
| `models.sparse.sparse_simpls` | `n4m.estimators.regression.sparse` | `sparse_simpls` |
| `filters.correlation` | `n4m.feature_selection.filter` | `correlation` |
| `filters.variance` | `n4m.feature_selection.filter` | `variance` |
| `selection.interval` | `n4m.feature_selection.interval` | `interval` |
| `selection.variable_select` | `n4m.feature_selection.ranking` | `variable_select` |
| `selection.bipls` | `n4m.feature_selection.wrapper` | `bipls` |
| `selection.bve` | `n4m.feature_selection.wrapper` | `bve` |
| `selection.cars` | `n4m.feature_selection.wrapper` | `cars` |
| `selection.emcuve` | `n4m.feature_selection.wrapper` | `emcuve` |
| `selection.ga` | `n4m.feature_selection.wrapper` | `ga` |
| `selection.ipw` | `n4m.feature_selection.wrapper` | `ipw` |
| `selection.irf` | `n4m.feature_selection.wrapper` | `irf` |
| `selection.iriv` | `n4m.feature_selection.wrapper` | `iriv` |
| `selection.pso` | `n4m.feature_selection.wrapper` | `pso` |
| `selection.random_frog` | `n4m.feature_selection.wrapper` | `random_frog` |
| `selection.randomization` | `n4m.feature_selection.wrapper` | `randomization` |
| `selection.rep` | `n4m.feature_selection.wrapper` | `rep` |
| `selection.scars` | `n4m.feature_selection.wrapper` | `scars` |
| `selection.shaving` | `n4m.feature_selection.wrapper` | `shaving` |
| `selection.sipls` | `n4m.feature_selection.wrapper` | `sipls` |
| `selection.spa` | `n4m.feature_selection.wrapper` | `spa` |
| `selection.st` | `n4m.feature_selection.wrapper` | `st` |
| `selection.stability` | `n4m.feature_selection.wrapper` | `stability` |
| `selection.t2` | `n4m.feature_selection.wrapper` | `t2` |
| `selection.uve` | `n4m.feature_selection.wrapper` | `uve` |
| `selection.vip_spa` | `n4m.feature_selection.wrapper` | `vip_spa` |
| `selection.vissa` | `n4m.feature_selection.wrapper` | `vissa` |
| `selection.wvc` | `n4m.feature_selection.wrapper` | `wvc` |
| `diagnostics.pls_diagnostics` | `n4m.metrics.diagnostics` | `pls_diagnostics` |
| `diagnostics.pls_monitoring` | `n4m.metrics.diagnostics` | `pls_monitoring` |
| `diagnostics.approximate_press` | `n4m.metrics.scoring` | `approximate_press` |
| `diagnostics.model_selection` | `n4m.metrics.scoring` | `model_selection` |
| `diagnostics.regression_metrics` | `n4m.metrics.scoring` | `regression_metrics` |
| `splitters.binned_strat_group_kfold` | `n4m.model_selection.split` | `binned_strat_group_kfold` |
| `splitters.kbins_stratified` | `n4m.model_selection.split` | `kbins_stratified` |
| `splitters.kennard_stone` | `n4m.model_selection.split` | `kennard_stone` |
| `splitters.kmeans` | `n4m.model_selection.split` | `kmeans` |
| `splitters.split_splitter` | `n4m.model_selection.split` | `split_splitter` |
| `splitters.spxy` | `n4m.model_selection.split` | `spxy` |
| `splitters.spxy_fold` | `n4m.model_selection.split` | `spxy_fold` |
| `splitters.spxy_g_fold` | `n4m.model_selection.split` | `spxy_g_fold` |
| `splitters.systematic_circular` | `n4m.model_selection.split` | `systematic_circular` |
| `filters.composite` | `n4m.outlier_detection` | `composite` |
| `filters.high_leverage` | `n4m.outlier_detection` | `high_leverage` |
| `filters.spectral_quality` | `n4m.outlier_detection` | `spectral_quality` |
| `filters.x_outlier` | `n4m.outlier_detection` | `x_outlier` |
| `filters.y_outlier` | `n4m.outlier_detection` | `y_outlier` |
| `utilities.hotelling_t2` | `n4m.outlier_detection` | `hotelling_t2` |
| `utilities.q_residuals` | `n4m.outlier_detection` | `q_residuals` |
| `preprocessing.alignment.cow_align` | `n4m.transform.alignment` | `cow_align` |
| `preprocessing.alignment.dtw_align` | `n4m.transform.alignment` | `dtw_align` |
| `preprocessing.alignment.icoshift_align` | `n4m.transform.alignment` | `icoshift_align` |
| `preprocessing.alignment.xcorr_align` | `n4m.transform.alignment` | `xcorr_align` |
| `preprocessing.baselines.airpls` | `n4m.transform.baseline` | `airpls` |
| `preprocessing.baselines.arpls` | `n4m.transform.baseline` | `arpls` |
| `preprocessing.baselines.asls` | `n4m.transform.baseline` | `asls` |
| `preprocessing.baselines.beads` | `n4m.transform.baseline` | `beads` |
| `preprocessing.baselines.detrend` | `n4m.transform.baseline` | `detrend` |
| `preprocessing.baselines.iasls` | `n4m.transform.baseline` | `iasls` |
| `preprocessing.baselines.imodpoly` | `n4m.transform.baseline` | `imodpoly` |
| `preprocessing.baselines.modpoly` | `n4m.transform.baseline` | `modpoly` |
| `preprocessing.baselines.rolling_ball` | `n4m.transform.baseline` | `rolling_ball` |
| `preprocessing.baselines.saps` | `n4m.transform.baseline` | `saps` |
| `preprocessing.baselines.snip` | `n4m.transform.baseline` | `snip` |
| `preprocessing.orthogonalization.epo` | `n4m.transform.orthogonalization` | `epo` |
| `preprocessing.orthogonalization.osc` | `n4m.transform.orthogonalization` | `osc` |
| `preprocessing.resampling.crop` | `n4m.transform.resampling` | `crop` |
| `preprocessing.resampling.kbins_discretizer` | `n4m.transform.resampling` | `kbins_discretizer` |
| `preprocessing.resampling.range_discretizer` | `n4m.transform.resampling` | `range_discretizer` |
| `preprocessing.resampling.resample_transformer` | `n4m.transform.resampling` | `resample_transformer` |
| `preprocessing.resampling.resampler` | `n4m.transform.resampling` | `resampler` |
| `preprocessing.scaling.baseline` | `n4m.transform.scaling` | `baseline` |
| `preprocessing.scaling.log_transform` | `n4m.transform.scaling` | `log_transform` |
| `preprocessing.scaling.normalize` | `n4m.transform.scaling` | `normalize` |
| `preprocessing.scaling.simple_scale` | `n4m.transform.scaling` | `simple_scale` |
| `preprocessing.scatter.area_normalization` | `n4m.transform.scatter` | `area_normalization` |
| `preprocessing.scatter.emsc` | `n4m.transform.scatter` | `emsc` |
| `preprocessing.scatter.local_centering` | `n4m.transform.scatter` | `local_centering` |
| `preprocessing.scatter.local_snv` | `n4m.transform.scatter` | `local_snv` |
| `preprocessing.scatter.localized_msc` | `n4m.transform.scatter` | `localized_msc` |
| `preprocessing.scatter.msc` | `n4m.transform.scatter` | `msc` |
| `preprocessing.scatter.piecewise_msc` | `n4m.transform.scatter` | `piecewise_msc` |
| `preprocessing.scatter.piecewise_snv` | `n4m.transform.scatter` | `piecewise_snv` |
| `preprocessing.scatter.robust_snv` | `n4m.transform.scatter` | `robust_snv` |
| `preprocessing.scatter.snv` | `n4m.transform.scatter` | `snv` |
| `preprocessing.scatter.vsn` | `n4m.transform.scatter` | `vsn` |
| `preprocessing.scatter.weighted_snv` | `n4m.transform.scatter` | `weighted_snv` |
| `preprocessing.signal_conversion.fraction_to_percent` | `n4m.transform.signal_conversion` | `fraction_to_percent` |
| `preprocessing.signal_conversion.from_absorbance` | `n4m.transform.signal_conversion` | `from_absorbance` |
| `preprocessing.signal_conversion.kubelka_munk` | `n4m.transform.signal_conversion` | `kubelka_munk` |
| `preprocessing.signal_conversion.percent_to_fraction` | `n4m.transform.signal_conversion` | `percent_to_fraction` |
| `preprocessing.signal_conversion.to_absorbance` | `n4m.transform.signal_conversion` | `to_absorbance` |
| `preprocessing.derivatives.derivate` | `n4m.transform.smoothing` | `derivate` |
| `preprocessing.derivatives.first_derivative` | `n4m.transform.smoothing` | `first_derivative` |
| `preprocessing.derivatives.norris_williams` | `n4m.transform.smoothing` | `norris_williams` |
| `preprocessing.derivatives.savitzky_golay` | `n4m.transform.smoothing` | `savitzky_golay` |
| `preprocessing.derivatives.second_derivative` | `n4m.transform.smoothing` | `second_derivative` |
| `preprocessing.smoothing.gaussian` | `n4m.transform.smoothing` | `gaussian` |
| `preprocessing.specialized.fck_static` | `n4m.transform.specialized` | `fck_static` |
| `preprocessing.wavelets.haar` | `n4m.transform.wavelet` | `haar` |
| `preprocessing.wavelets.wavelet` | `n4m.transform.wavelet` | `wavelet` |
| `preprocessing.wavelets.wavelet_denoise` | `n4m.transform.wavelet` | `wavelet_denoise` |
| `preprocessing.wavelets.wavelet_features` | `n4m.transform.wavelet` | `wavelet_features` |
| `preprocessing.wavelets.wavelet_pca` | `n4m.transform.wavelet` | `wavelet_pca` |
| `preprocessing.wavelets.wavelet_svd` | `n4m.transform.wavelet` | `wavelet_svd` |
| `utilities.moments` | `n4m.utils` | `moments` |
| `utilities.signal_type_detector` | `n4m.utils` | `signal_type_detector` |
| `utilities.sweep` | `n4m.utils` | `sweep` |

**Per-node counts (B):** `feature_selection.wrapper` 23 · `estimators.regression.linear` 14 ·
`transform.scatter` 12 · `transform.baseline` 11 · `model_selection.split` 9 ·
`compose.aom_search` 8 · `compose.aom_superblock` 8 · `domain_adaptation` 8 · `outlier_detection` 7 ·
`augmentation.instrument` 7 · `augmentation.spectral` 7 · `transform.smoothing` 6 ·
`transform.wavelet` 6 · `augmentation.scattering` 6 · `estimators.multiblock` 6 ·
`transform.resampling` 5 · `transform.signal_conversion` 5 · `augmentation.splines` 5 ·
`ensemble` 4 · `estimators.classification` 4 · `augmentation.mixup` 4 · `augmentation.noise` 4 ·
`transform.alignment` 4 · `transform.scaling` 4 · `estimators.regression.sparse` 3 ·
`metrics.scoring` 3 · `augmentation.drift` 3 · `augmentation.wavelength` 3 · `utils` 3 ·
`decomposition` 2 · `estimators.glm_survival` 2 · `feature_selection.filter` 2 ·
`metrics.diagnostics` 2 · `transform.orthogonalization` 2 · `compose.aom_campaign` 2 ·
`estimators.local` 1 · `feature_selection.interval` 1 · `feature_selection.ranking` 1 ·
`transform.specialized` 1. **Total 208.**

## B.3 API-surface implications (three faces)

### C ABI
- **Header split** by top-level role: `n4m/transform.h`, `n4m/decomposition.h`,
  `n4m/feature_selection.h`, `n4m/augmentation.h`, `n4m/estimators.h`, `n4m/ensemble.h`,
  `n4m/model_selection.h`, `n4m/compose.h`, `n4m/domain_adaptation.h`, `n4m/outlier_detection.h`,
  `n4m/metrics.h`, `n4m/utils.h`, included by the `n4m/n4m.h` umbrella. `transform`,
  `augmentation`, and `estimators` get second-level sub-headers (e.g.
  `n4m/transform/scatter.h`, `n4m/estimators/regression.h`).
- **Symbol prefix**: `n4m_<role>_<method>_<verb>`:
  `n4m_transform_snv_create/_transform`, `n4m_estimators_ridge_fit`,
  `n4m_feature_selection_cars_select`, `n4m_augmentation_gaussian_noise_apply`,
  `n4m_compose_aom_sweep_run`, `n4m_model_selection_kennard_stone_split`,
  `n4m_domain_adaptation_pds_fit`. As in Scheme A the existing terse symbols stay as the frozen ABI
  and the role-prefixed names are additive aliases (minor-ABI bump + snapshot regen).
- This mapping has the nice property that the C verb already encodes `kind`
  (`_create/_transform` ↔ transform, `_fit/_predict` ↔ estimator, `_select` ↔ selector,
  `_split` ↔ splitter, `_apply` ↔ augmentation), so role-first headers and verb suffixes reinforce
  each other.

### Python (`replaces the 373 KB flat python.py`)
- `n4m/` mirrors sklearn so muscle memory transfers 1:1:
  `from n4m.transform.scatter import SNV`,
  `from n4m.estimators.regression.linear import Ridge`,
  `from n4m.feature_selection.wrapper import CARS`,
  `from n4m.augmentation.noise import GaussianNoise`,
  `from n4m.model_selection.split import KennardStoneSplitter`,
  `from n4m.compose.aom_search import AOMSweep`,
  `from n4m.domain_adaptation import PDS`.
- The ~45 `Native*Regressor` AOM classes land in `n4m.compose.*` next to their function. sklearn
  estimator classes (`SNV`, `EMSC`, `KennardStoneSplitter`, `SavitzkyGolay`, `XOutlierFilter`, ...)
  land in the role module their API implies — exactly where a sklearn user expects them.
- `n4m/__init__.py` exposes the subpackages only; no flat dump. A thin `n4m.pipeline`/`n4m.compose`
  helper can re-expose the AOM campaigns as sklearn-compatible meta-estimators.

### R
- Function convention `n4m_<role>_<method>()`: `n4m_transform_snv()`,
  `n4m_estimators_ridge()`, `n4m_feature_selection_cars()`, `n4m_model_selection_kennard_stone()`.
  Less idiomatic for the R chemometrics crowd than Scheme A (R users don't think "transform" vs
  "estimator"), but unambiguous and consistent with the C/Python faces.

## B.4 Pros / cons / migration / edge cases

**Pros**
- **Lowest friction for the largest likely user base** (sklearn/PyTorch): the layout, the import
  paths, and even the C verb suffixes line up with `sklearn.*`. "Where does this go in my pipeline?"
  is answered by the top-level node.
- `transform` vs `augmentation` split is genuinely informative (fit/transform pipeline step vs
  apply-only training perturbation) and is invisible in the current flat surface.
- `compose.*` gives the AOM/POP machinery the framing it deserves (automated pipeline search), and
  the sub-split (search / superblock / campaign) tracks how the methods are actually used.
- C verb suffix already encodes `kind`, so headers + symbols are self-consistent.

**Cons**
- Splits some chemometrically-unified families across nodes: scatter *correction*
  (`transform.scatter`) is far from scatter *augmentation* (`augmentation.scattering`); calibration
  transfer is split between `domain_adaptation` (models + standardization) while OSC/EPO
  orthogonalization sits in `transform.orthogonalization` — a chemometrician may find transfer
  "scattered" (see edge cases).
- `estimators.regression.linear` is a 14-member catch-all; "linear" is a loose label for
  kernel-PLS/GPR/tensor-PLS (kept there because their *API* is a continuous regressor and they are
  not penalized/sparse/ensemble).
- R audience fit is weaker than Scheme A.

**Migration cost:** medium, essentially identical mechanical effort to Scheme A (file split + new
headers + alias symbols; core and frozen symbols untouched).

**Collision / edge-case decisions (flagged):**
| method | tension | resolution in Scheme B |
|---|---|---|
| `models.sparse.sparse_pls_da` | sparse-regression API vs classification task | → `estimators.classification` (predict = class label). |
| `models.classification.pls_logistic` | logistic = GLM vs classifier | → `estimators.classification` (used as a classifier). |
| `models.heads.pls_glm` / `pls_cox` | not plain regressor/classifier | → `estimators.glm_survival` (kept distinct from the two standard tasks). |
| `models.pls.kernel` / `gpr_pls` / `tensor_pls` | non-linear, but regressor API | → `estimators.regression.linear` (continuous-output estimator; "linear" reflects the module, not the kernel). |
| `models.transfer.di_pls` | estimator vs domain adaptation | → `domain_adaptation` (its purpose is cross-domain calibration). |
| `preprocessing.orthogonalization.epo` | transform vs domain adaptation | → `transform.orthogonalization` (kept with OSC as a fit/transform step). **Note divergence from Scheme A**, where EPO sits in `transfer`. |
| `preprocessing.transfer.*` (DS/PDS-style standardization) | transform vs domain adaptation | → `domain_adaptation` (they learn a source→target map; treated as adaptation, not generic transform). |
| `preprocessing.feature_selection.flexible_pca/_svd` | misleading family name | → `decomposition` (projection transformers, not selection). |
| `filters.correlation` / `filters.variance` | outlier filter vs feature filter | → `feature_selection.filter` (they drop *features*, the sklearn filter-method definition). **Note divergence from Scheme A**, which keeps them in `outliers`. |
| `filters.composite/high_leverage/spectral_quality/x_outlier/y_outlier` | drop *samples* | → `outlier_detection` (sample-level screening). |
| `aom_pop.*` (18) | estimator vs pipeline | → `compose.*` (automated preprocessing/operator search = pipeline composition). |
| `utilities.moments` / `sweep` | helper | → `utils` (not an estimator/transformer/metric). |

---

## C. Side-by-side comparison

| Axis | Scheme A — STATS / CHEMOMETRICS | Scheme B — ML / DL PIPELINE |
|---|---|---|
| **Primary lever** | mathematical family | `kind` / estimator API role |
| **Top-levels (12)** | regression, discriminant, multiblock, transfer, variable_selection, preprocessing, decomposition, outliers, resampling, validation, optimization, simulation | transform, decomposition, feature_selection, augmentation, estimators, ensemble, model_selection, compose, domain_adaptation, outlier_detection, metrics, utils |
| **Leaf namespaces** | 31 | 39 |
| **Biggest node** | `variable_selection` (25) | `feature_selection.wrapper` (23) |
| **Augmentation home** | `simulation.*` (physical origin) | `augmentation.*` (DL convention) |
| **Preprocessing home** | `preprocessing.*` (one family, lit. sub-families) | `transform.*` (sklearn name) |
| **AOM/POP home** | `optimization.aom` (one node) | `compose.{aom_search,aom_superblock,aom_campaign}` |
| **Calibration transfer** | unified top-level `transfer` (models+preproc+metric+EPO) | split: `domain_adaptation` (models+DS/PDS) **+** `transform.orthogonalization` (OSC/EPO) |
| **Feature/quality filters** | all in `outliers` | split: feature filters → `feature_selection.filter`, sample filters → `outlier_detection` |
| **Audience fit** | chemometricians / R users ★★★ ; sklearn users ★★ | sklearn / PyTorch users ★★★ ; R chemometricians ★★ |
| **Mirrors** | `pls`, `mdatools`, `ChemoSpec`, `prospectr` | `sklearn`, DL augmentation libs |
| **C header self-consistency** | family headers; symbol stem = family | role headers; symbol verb already encodes role ★ |
| **Migration cost** | medium | medium (same mechanical effort) |
| **Main weakness** | pipeline-role not explicit; two `scatter` nodes | transfer family split across 2 nodes; "linear" catch-all |

**Where the two schemes genuinely diverge on placement** (not just renaming):
1. **EPO** — A: `transfer`; B: `transform.orthogonalization`.
2. **DS/PDS/slope-bias standardization** — A: `transfer`; B: `domain_adaptation` (same intent,
   different parent name) — but A also pulls in the *models* `di_pls/ds/pds` to the same node, while
   B keeps OSC/EPO out.
3. **`filters.correlation` / `filters.variance`** — A: `outliers`; B: `feature_selection.filter`.
4. **Augmentation** — A scatters it by physical origin under `simulation`; B keeps the catalog's
   augmentation sub-families under `augmentation`.
5. **AOM** — A: single `optimization.aom`; B: 3-way `compose.*` split by usage pattern.

---

## D. Recommendation

**Adopt Scheme B (ML / DL pipeline) as the primary public namespace, and ship Scheme A's family
labels as a secondary, non-authoritative "chemometrics view" / alias layer.**

Rationale:
1. **Audience weight.** `nirs4all-methods` is consumed first through its Python binding (and the
   wider nirs4all ecosystem is sklearn-shaped: `SpectroDataset`, pipelines, `run()/predict()`).
   The plurality of users arrive with sklearn/PyTorch muscle memory, and Scheme B makes
   `from n4m.transform.scatter import SNV` / `from n4m.estimators.regression.linear import Ridge`
   immediately discoverable.
2. **Self-consistency across faces.** In Scheme B the C verb suffix already encodes the role
   (`_create/_transform`, `_fit`, `_select`, `_split`, `_apply`), so role-first headers, symbols,
   and Python subpackages reinforce one another with no extra convention to remember. This directly
   attacks the current pathology (build-phase-ordered flat headers) at every layer.
3. **The AOM/POP line gets honest framing.** `compose.*` (search / superblock / campaign) tells a
   user what the 18 AOM methods *are* (automated pipeline search), which the flat dump and even the
   catalog's `aom_pop` bucket obscure.
4. **A is cheap to keep as a view.** Because the catalog already carries family metadata and the two
   schemes share leaf names, the chemometrics view can be a generated alias module
   (`n4m.chemometrics.regression.latent.pls -> n4m.estimators.regression.linear.pls_fit_simple`,
   etc.) plus R docs that group by family — giving the R/chemometrics audience their preferred
   navigation **without** maintaining two real trees.

**Implementation guardrails (independent of which scheme wins):**
- Keep the existing terse ABI symbols as the **frozen** surface; add namespaced symbols as additive
  aliases behind a single `N4M_ABI_VERSION_MINOR` bump, regenerate all three
  `cpp/abi/expected_symbols_*.txt`, and log in `docs/abi/changes_log.md`.
- Fill the 11 empty stub headers with real declarations under the chosen top-level split; retire the
  `n4m.h` vs `pls.h` distinction (it already contradicts the catalog).
- Split `python.py` mechanically into the subpackages with **no behavioural change**; preserve the
  current flat names for one release as deprecated top-level re-exports (the legacy `aom_pls`,
  `pop_pls`, `moments_train_from_heldout` aliases must survive — they are documented public aliases).
- Drive the whole thing from the catalog: add an explicit `namespace` (and optional
  `namespace_alias`) field per method YAML so the tree is generated, not hand-maintained, and the
  `_build_proposals.py` mapping becomes the single source of truth.

---

## Appendix — provenance & reproducibility

- Ground truth: `proposals/namespace/_method_inventory.json` (208 records).
- Mapping generator + validator: `proposals/namespace/_build_proposals.py`
  (asserts every id mapped exactly once; `STATS` and `ML` each = 208; emits `_mappings.json`).
- Rendered tables: `proposals/namespace/_stats_table.md`, `proposals/namespace/_ml_table.md`
  (208 rows each), `proposals/namespace/_full_table.tsv` (joint id → A-ns / B-ns / since_abi /
  python-binding / edge-flag).
- **Maturity signal** (`since_abi`): 188 methods are `1.0.0`; **20 are newer** (1.12–1.21),
  concentrated in `aom_pop` (15 of 18) plus `pcr`, `ridge`, `moment_stack`, `moments`, `sweep`. In
  both schemes these are the "emerging" members of their node — a candidate for a per-method
  `status: stable|emerging` badge rather than a separate namespace tier.
- **Open seams in the catalog itself** (not blockers, but worth fixing alongside): 10 methods carry
  **0 ABI symbols** (9 Python-only AOM superblocks + `moment_stack`); 16 methods have `kind=null`;
  only 30/208 have a Python binding entry; `parity.references` is empty in all 208. The namespace
  refactor is a natural moment to backfill `kind`, `namespace`, and `status`.
