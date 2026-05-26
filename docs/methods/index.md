# Methods catalogue

Every method in the library — the PLS / selection algorithms from `benchmarks.parity_timing.registry.METHODS` and the C++ preprocessing / augmentation / filter / splitter operators from the n4m binding — documented with parameters, bibliographic source, mathematical principle, binding signatures, and benchmark rows.

_Total methods_: **183** (73 PLS / selection · 110 operators). Grouped by family below.

```{toctree}
:hidden:
:glob:
:maxdepth: 1

*
```

## Core PLS

| Method | Description | Tolerance | Refs |
|--------|-------------|-----------|------|
| [`cppls`](cppls.md) | CPPLS (Canonical Powered PLS, Indahl Liland & Næs 2009) | `0.1` | R |
| [`opls`](opls.md) | Orthogonal PLS (Trygg & Wold 2002) | `0.001` | R |
| [`pcr`](pcr.md) | Principal Components Regression | `1e-06` | Py, R |
| [`pls`](pls.md) | SIMPLS PLS regression baseline | `0.1` | Py, R, ikpls, mixOmics |
| [`recursive_pls`](recursive_pls.md) | Recursive (moving-window) PLS | `0.1` | Py, R |

## Sparse

| Method | Description | Tolerance | Refs |
|--------|-------------|-----------|------|
| [`fused_sparse_pls`](fused_sparse_pls.md) | Fused sparse PLS (§7) | `0.05` | — |
| [`group_sparse_pls`](group_sparse_pls.md) | Group sparse PLS (§7) | `0.05` | Py, R |
| [`sparse_pls_da`](sparse_pls_da.md) | Sparse PLS-DA (§7) | `2.0` | Py, R |
| [`sparse_simpls`](sparse_simpls.md) | Sparse SIMPLS with soft-threshold lambda | `1.0` | Py, R |

## Ensemble

| Method | Description | Tolerance | Refs |
|--------|-------------|-----------|------|
| [`bagging_pls`](bagging_pls.md) | Bagging PLS (§20) | `1e-06` | Py |
| [`boosting_pls`](boosting_pls.md) | Boosting PLS (§20) | `1e-06` | R |
| [`random_subspace_pls`](random_subspace_pls.md) | Random-subspace PLS (§20) | `1e-06` | Py |

## Robust / weighted

| Method | Description | Tolerance | Refs |
|--------|-------------|-----------|------|
| [`robust_pls`](robust_pls.md) | Robust PLS (Partial Robust M-regression, Serneels 2005) | `0.1` | R |
| [`weighted_pls`](weighted_pls.md) | Sample-weighted PLS (sqrt(w)-prescaled NIPALS) | `0.1` | Py |

## Nonlinear / local

| Method | Description | Tolerance | Refs |
|--------|-------------|-----------|------|
| [`continuum_regression`](continuum_regression.md) | Continuum regression (interpolates PLS / OLS) | `1e-06` | Py, R |
| [`gpr_pls`](gpr_pls.md) | GPR-on-PLS — RBF Gaussian Process on PLS scores (§47) | `1e-08` | Py |
| [`kernel_pls_rbf`](kernel_pls_rbf.md) | Non-linear kernel PLS (RBF kernel) | `2.0` | R |
| [`lw_pls`](lw_pls.md) | LW-PLS — Locally-weighted PLS (§17 Phase 4) | `5.0` | Py |

## Multi-block / cross-modal

| Method | Description | Tolerance | Refs |
|--------|-------------|-----------|------|
| [`mb_pls`](mb_pls.md) | MB-PLS — Multi-block PLS (§17 Phase 4) | `2.0` | Py |
| [`mir_pls`](mir_pls.md) | MIR-PLS — Inverse-regression PLS (§13) | `0.05` | — |
| [`n_pls`](n_pls.md) | N-PLS — 3-way tensor PLS (PARAFAC + OLS by default; Bro 1996 opt-in) | `1e-06` | Py |
| [`o2pls`](o2pls.md) | O2PLS — bi-directional OPLS (Trygg & Wold 2003) | `1e-10` | R |
| [`on_pls`](on_pls.md) | OnPLS — Orthogonal multi-block decomposition (§18) | `1e-06` | Py |
| [`rosa`](rosa.md) | ROSA — Response-Oriented Sequential Alternation (§19) | `1e-06` | Py, R |
| [`so_pls`](so_pls.md) | SO-PLS — Sequential & Orthogonalized multi-block PLS (§17) | `1e-06` | R |

## Calibration transfer

| Method | Description | Tolerance | Refs |
|--------|-------------|-----------|------|
| [`di_pls`](di_pls.md) | Domain-invariant PLS | `1e-06` | Py |
| [`ds`](ds.md) | DS — Direct Standardization (§13) | `0.5` | R |
| [`ecr`](ecr.md) | Elastic Component Regression (Phase 50) | `0.001` | Py |
| [`pds`](pds.md) | PDS — Piecewise Direct Standardization (§13) | `0.5` | R |

## Classification & GLM

| Method | Description | Tolerance | Refs |
|--------|-------------|-----------|------|
| [`pls_cox`](pls_cox.md) | PLS-Cox (§5) — Cox PH on PLS scores | `1e-06` | Py |
| [`pls_glm`](pls_glm.md) | PLS-GLM (§5) — softmax/Poisson IRLS on PLS scores | `1e-06` | R |
| [`pls_lda`](pls_lda.md) | PLS-LDA — LDA on PLS scores (§17 Phase 4) | `5.0` | Py |
| [`pls_logistic`](pls_logistic.md) | PLS-Logistic — Logistic regression on PLS scores | `5.0` | Py |
| [`pls_qda`](pls_qda.md) | PLS-QDA (§5) — quadratic discriminant on PLS scores | `1e-06` | Py |

## Missing data

| Method | Description | Tolerance | Refs |
|--------|-------------|-----------|------|
| [`missing_aware_nipals`](missing_aware_nipals.md) | Missing-aware NIPALS PLS (§13) | `10.0` | R |

## Regularised

| Method | Description | Tolerance | Refs |
|--------|-------------|-----------|------|
| [`ridge_pls`](ridge_pls.md) | Ridge-augmented PLS | `0.1` | Py |

## Diagnostic

| Method | Description | Tolerance | Refs |
|--------|-------------|-----------|------|
| [`aom_preprocess`](aom_preprocess.md) | AOM preprocessing pipeline (§17 Phase 4) | `5.0` | Py |
| [`approximate_press`](approximate_press.md) | Approximate-PRESS component selection (§29) | `1e-10` | R |
| [`one_se_rule`](one_se_rule.md) | One-SE component selection rule (§10) | `1e-06` | R |
| [`pls_diagnostic_dmodx`](pls_diagnostic_dmodx.md) | PLS Distance-to-Model X (§9) | `5.0` | R |
| [`pls_diagnostic_q`](pls_diagnostic_q.md) | PLS Q residuals / SPE (§9) | `5.0` | R |
| [`pls_diagnostic_t2`](pls_diagnostic_t2.md) | PLS Hotelling T² (§9) | `10.0` | R |
| [`pls_monitoring`](pls_monitoring.md) | PLS process monitoring (T²/Q thresholds + alarms) (§19) | `10.0` | R |

## Variable selector

| Method | Description | Tolerance | Refs |
|--------|-------------|-----------|------|
| [`bipls_select`](bipls_select.md) | biPLS backward interval elimination (§18 Phase 5p) | `0.7` | R |
| [`bve_select`](bve_select.md) | Backward Variable Elimination (§18 Phase 5k) | `1e-06` | R |
| [`cars_select`](cars_select.md) | CARS competitive adaptive reweighted sampling | `0.0` | R |
| [`emcuve_select`](emcuve_select.md) | EMCUVE ensemble MC-UVE (§18 Phase 5n) | `1e-06` | R |
| [`ga_select`](ga_select.md) | GA-PLS genetic algorithm selection | `1e-06` | R |
| [`interval_select`](interval_select.md) | Interval/iPLS forward selection (§18 Phase 5b) | `1e-06` | R |
| [`ipw_select`](ipw_select.md) | IPW-PLS iterative predictor weighting (§18 Phase 5t) | `1e-06` | R |
| [`irf_select`](irf_select.md) | Interval Random Frog (Phase 52) | `1e-06` | Py |
| [`iriv_select`](iriv_select.md) | Iteratively Retains Informative Variables (Phase 51) | `1e-06` | Py |
| [`pso_select`](pso_select.md) | PSO-PLS — Binary Particle Swarm variable selection (§48) | `1e-06` | Py |
| [`random_frog_select`](random_frog_select.md) | Random Frog selection (§18 Phase 5g) | `1e-06` | Py |
| [`randomization_select`](randomization_select.md) | Randomization-test selector (§18 Phase 5o) | `1e-06` | R |
| [`rep_select`](rep_select.md) | REP-PLS repeated VIP selection (§18 Phase 5s) | `1e-06` | R |
| [`scars_select`](scars_select.md) | SCARS stability + CARS (§18 Phase 5h) | `1e-06` | Py |
| [`shaving_select`](shaving_select.md) | Shaving iterative variable trimming | `1e-06` | R |
| [`sipls_select`](sipls_select.md) | siPLS synergistic interval selection (§18 Phase 5q) | `0.7` | — |
| [`spa_select`](spa_select.md) | SPA Successive Projections (§18 Phase 5e) | `1e-06` | R |
| [`st_select`](st_select.md) | ST-PLS soft-thresholded sparse PLS (§18 Phase 5u) | `1e-06` | R |
| [`stability_select`](stability_select.md) | Stability/MCUVE selection (§18 Phase 5c) | `1e-06` | R |
| [`t2_select`](t2_select.md) | T²-PLS loading-weight selection (§18 Phase 5l) | `1.2` | R |
| [`uve_select`](uve_select.md) | UVE noise-thresholded selection (§18 Phase 5d) | `1e-06` | R |
| [`variable_select_coef`](variable_select_coef.md) | \|Coef\| top-k selection (§18 Phase 5a, method=1) | `1.1` | R |
| [`variable_select_sr`](variable_select_sr.md) | Selectivity-Ratio top-k (§18 Phase 5a, method=2) | `1e-06` | R |
| [`variable_select_vip`](variable_select_vip.md) | VIP top-k variable selection (§18 Phase 5a, method=0) | `1e-06` | R |
| [`vip_spa_select`](vip_spa_select.md) | VIP_SPA — VIP-mask then SPA greedy (Phase 53) | `1e-06` | Py |
| [`vissa_select`](vissa_select.md) | VISSA-PLS — Variable Iterative Space Shrinkage (§49) | `1e-06` | Py |
| [`wvc_select`](wvc_select.md) | WVC weighted-variable-component top-k | `0.7` | R |
| [`wvc_threshold_select`](wvc_threshold_select.md) | WVC threshold-based selection (§18 Phase 5r) | `1e-06` | R |

## Preprocessing

| Operator | Binding | Description |
|----------|---------|-------------|
| [`pp_area`](pp_area.md) | `n4m.sklearn.AreaNormalization` | Per-row area normalisation. |
| [`pp_baseline`](pp_baseline.md) | `n4m.sklearn.BaselineCenter` | Column-mean baseline centering. |
| [`pp_crop`](pp_crop.md) | `n4m.sklearn.CropTransformer` | Slice wavelength columns in the half-open interval ``[start, end)``. |
| [`pp_derivate`](pp_derivate.md) | `n4m.sklearn.Derivate` | Finite-difference derivative along the wavelength axis. |
| [`pp_emsc`](pp_emsc.md) | `n4m.sklearn.EMSC` | Extended Multiplicative Scatter Correction (polynomial). |
| [`pp_first_derivative`](pp_first_derivative.md) | `n4m.sklearn.FirstDerivative` | ``np.gradient(X, delta, axis=1, edge_order=...)`` (shape-preserving). |
| [`pp_frac_to_pct`](pp_frac_to_pct.md) | `n4m.sklearn.FractionToPercent` | Convert fractional reflectance/transmittance to percent. |
| [`pp_from_absorbance`](pp_from_absorbance.md) | `n4m.sklearn.FromAbsorbance` | R = 10**(-A), optionally returned as percent. |
| [`pp_gaussian`](pp_gaussian.md) | `n4m.sklearn.Gaussian` | SciPy-compatible 1-D Gaussian filter along the wavelength axis. |
| [`pp_kbins_disc`](pp_kbins_disc.md) | `n4m.sklearn.IntegerKBinsDiscretizer` | Per-column integer binning using uniform or quantile edges. |
| [`pp_kubelka_munk`](pp_kubelka_munk.md) | `n4m.sklearn.KubelkaMunk` | KM = (1 - R)^2 / (2 R), with R guarded by epsilon. |
| [`pp_log`](pp_log.md) | `n4m.sklearn.LogTransform` | Element-wise logarithm with optional fit-time auto-offset. |
| [`pp_lsnv`](pp_lsnv.md) | `n4m.sklearn.LSNV` | Sliding-window (local) SNV. |
| [`pp_msc`](pp_msc.md) | `n4m.sklearn.MSC` | Multiplicative Scatter Correction. |
| [`pp_normalize`](pp_normalize.md) | `n4m.sklearn.Normalize` | Column-wise normalisation. |
| [`pp_norris_williams`](pp_norris_williams.md) | `n4m.sklearn.NorrisWilliams` | Segment smoothing followed by gap finite differences. |
| [`pp_pct_to_frac`](pp_pct_to_frac.md) | `n4m.sklearn.PercentToFraction` | Convert percent reflectance/transmittance to fraction. |
| [`pp_range_disc`](pp_range_disc.md) | `n4m.sklearn.RangeDiscretizer` | Integer binning against monotonic numeric edges. |
| [`pp_resample`](pp_resample.md) | `n4m.sklearn.ResampleTransformer` | Resize spectra to a fixed column count. |
| [`pp_resampler`](pp_resampler.md) | `n4m.sklearn.Resampler` | Interpolate spectra from a fitted source wavelength grid to a target grid. |
| [`pp_rnv`](pp_rnv.md) | `n4m.sklearn.RNV` | Robust SNV using median + k * MAD. |
| [`pp_savgol`](pp_savgol.md) | `n4m.sklearn.SavitzkyGolay` | scipy.signal.savgol_filter parity. |
| [`pp_second_derivative`](pp_second_derivative.md) | `n4m.sklearn.SecondDerivative` | Two passes of ``np.gradient`` (shape-preserving). |
| [`pp_simple_scale`](pp_simple_scale.md) | `n4m.sklearn.SimpleScale` | Column-wise min-max scaling to ``[0, 1]``. |
| [`pp_snv`](pp_snv.md) | `n4m.sklearn.SNV` | Standard Normal Variate normalisation. |
| [`pp_to_absorbance`](pp_to_absorbance.md) | `n4m.sklearn.ToAbsorbance` | A = -log10(max(R, epsilon)). Optional %-scaling. |

## Baseline correction

| Operator | Binding | Description |
|----------|---------|-------------|
| [`pp_airpls`](pp_airpls.md) | `n4m.sklearn.AirPLS` | Adaptive iteratively reweighted PLS (Zhang 2010). |
| [`pp_arpls`](pp_arpls.md) | `n4m.sklearn.ArPLS` | Asymmetrically reweighted penalized least squares. |
| [`pp_asls`](pp_asls.md) | `n4m.sklearn.AsLS` | Asymmetric Least Squares (Eilers & Boelens 2005). |
| [`pp_beads`](pp_beads.md) | `n4m.sklearn.BEADS` | Baseline estimation and denoising with sparsity. |
| [`pp_detrend`](pp_detrend.md) | `n4m.sklearn.Detrend` | Polynomial baseline subtraction. |
| [`pp_iasls`](pp_iasls.md) | `n4m.sklearn.IAsLS` | Improved asymmetric least-squares baseline correction. |
| [`pp_imodpoly`](pp_imodpoly.md) | `n4m.sklearn.IModPoly` | Improved modified polynomial baseline correction. |
| [`pp_modpoly`](pp_modpoly.md) | `n4m.sklearn.ModPoly` | Modified polynomial baseline correction. |
| [`pp_rolling_ball`](pp_rolling_ball.md) | `n4m.sklearn.RollingBall` | Rolling-ball morphological baseline correction. |
| [`pp_snip`](pp_snip.md) | `n4m.sklearn.SNIP` | Statistics-sensitive nonlinear iterative peak-clipping baseline. |

## Signal transforms

| Operator | Binding | Description |
|----------|---------|-------------|
| [`filter_correlation`](filter_correlation.md) | `n4m.sklearn.CorrelationFilter` | Model-agnostic feature filter by absolute correlation to ``y``. |
| [`filter_variance`](filter_variance.md) | `n4m.sklearn.VarianceFilter` | Model-agnostic feature filter by variance. |
| [`interval_generator`](interval_generator.md) | `n4m.sklearn.IntervalGenerator` | Generate fixed or overlapping wavelength intervals. |
| [`pp_cow_align`](pp_cow_align.md) | `n4m.sklearn.CorrelationOptimizedWarping` | Segment-wise correlation optimized warping approximation. |
| [`pp_direct_standardization`](pp_direct_standardization.md) | `n4m.sklearn.DirectStandardization` | Direct standardization transfer map between paired instruments. |
| [`pp_dtw_align`](pp_dtw_align.md) | `n4m.sklearn.DynamicTimeWarpingAlignment` | Dynamic-time-warping alignment to a fixed-length reference. |
| [`pp_icoshift_align`](pp_icoshift_align.md) | `n4m.sklearn.IcoshiftAlignment` | Interval correlation shifting with fixed-size intervals. |
| [`pp_local_centering`](pp_local_centering.md) | `n4m.sklearn.LocalCentering` | Transfer by subtracting source mean and adding target mean. |
| [`pp_localized_msc`](pp_localized_msc.md) | `n4m.sklearn.LocalizedMSC` | Feature-wise MSC using a moving local wavelength window. |
| [`pp_piecewise_direct_standardization`](pp_piecewise_direct_standardization.md) | `n4m.sklearn.PiecewiseDirectStandardization` | PDS: local regressions mapping source windows to target wavelengths. |
| [`pp_piecewise_msc`](pp_piecewise_msc.md) | `n4m.sklearn.PiecewiseMSC` | Apply MSC independently inside fixed wavelength intervals. |
| [`pp_piecewise_snv`](pp_piecewise_snv.md) | `n4m.sklearn.PiecewiseSNV` | Apply SNV independently inside fixed wavelength intervals. |
| [`pp_robust_direct_standardization`](pp_robust_direct_standardization.md) | `n4m.sklearn.RobustDirectStandardization` | Direct standardization with iterative residual trimming. |
| [`pp_saps`](pp_saps.md) | `n4m.sklearn.ScoreAugmentedProjectionStandardization` | Score-augmented projection standardization inspired by SA-PBS. |
| [`pp_slope_bias`](pp_slope_bias.md) | `n4m.sklearn.SlopeBiasCorrection` | Linear slope/bias correction for transferred predictions. |
| [`pp_vsn`](pp_vsn.md) | `n4m.sklearn.VariableSortingNormalization` | VSN-style data-derived weighted SNV. |
| [`pp_weighted_snv`](pp_weighted_snv.md) | `n4m.sklearn.WeightedSNV` | Weighted standard normal variate normalization. |
| [`pp_xcorr_align`](pp_xcorr_align.md) | `n4m.sklearn.CrossCorrelationAlignment` | Whole-spectrum integer shift chosen by maximum correlation. |

## Feature extraction

| Operator | Binding | Description |
|----------|---------|-------------|
| [`pp_epo`](pp_epo.md) | `n4m.sklearn.EPO` | External Parameter Orthogonalisation. |
| [`pp_fck_static`](pp_fck_static.md) | `n4m.sklearn.FCKStaticTransformer` | Static fractional convolutional kernel bank transformer. |
| [`pp_flex_pca`](pp_flex_pca.md) | `n4m.sklearn.FlexiblePCA` | PCA with integer or explained-variance component selection. |
| [`pp_flex_svd`](pp_flex_svd.md) | `n4m.sklearn.FlexibleSVD` | SVD with integer component selection. |
| [`pp_osc`](pp_osc.md) | `n4m.sklearn.OSC` | Orthogonal Signal Correction. |

## Augmentation

| Operator | Binding | Description |
|----------|---------|-------------|
| [`aug_band_mask`](aug_band_mask.md) | `n4m.sklearn.BandMasking` | Mask random spectral bands with zero-fill or interpolation. |
| [`aug_band_perturb`](aug_band_perturb.md) | `n4m.sklearn.BandPerturbationAugmenter` | Random band-local gain and offset perturbations. |
| [`aug_batch_effect`](aug_batch_effect.md) | `n4m.sklearn.BatchEffectAugmenter` | Random offset, slope and gain batch effects. |
| [`aug_channel_dropout`](aug_channel_dropout.md) | `n4m.sklearn.ChannelDropout` | Randomly drop individual wavelength channels. |
| [`aug_dead_band`](aug_dead_band.md) | `n4m.sklearn.DeadBandAugmenter` | Simulate dead spectral detector bands. |
| [`aug_detector_rolloff`](aug_detector_rolloff.md) | `n4m.sklearn.DetectorRollOffAugmenter` | Detector edge roll-off artifact. |
| [`aug_edge_artifacts`](aug_edge_artifacts.md) | `n4m.sklearn.EdgeArtifactsAugmenter` | Combined edge artifact augmenter. |
| [`aug_edge_curve`](aug_edge_curve.md) | `n4m.sklearn.EdgeCurvatureAugmenter` | Curved edge response artifact. |
| [`aug_emsc_distort`](aug_emsc_distort.md) | `n4m.sklearn.EMSCDistortionAugmenter` | Random EMSC-like multiplicative, additive and polynomial distortion. |
| [`aug_gauss_jitter`](aug_gauss_jitter.md) | `n4m.sklearn.GaussianJitter` | Gaussian smoothing jitter. |
| [`aug_gaussian_noise`](aug_gaussian_noise.md) | `n4m.sklearn.GaussianAdditiveNoise` | Add IID Gaussian noise to each element of ``X``. |
| [`aug_hetero_noise`](aug_hetero_noise.md) | `n4m.sklearn.HeteroscedasticNoiseAugmenter` | Noise whose standard deviation depends on signal magnitude. |
| [`aug_instrument_broaden`](aug_instrument_broaden.md) | `n4m.sklearn.InstrumentalBroadeningAugmenter` | Instrumental spectral broadening via Gaussian convolution. |
| [`aug_linear_drift`](aug_linear_drift.md) | `n4m.sklearn.LinearBaselineDrift` | Add random offset and linear slope drift. |
| [`aug_local_clip`](aug_local_clip.md) | `n4m.sklearn.LocalClip` | Clip random local spectral regions. |
| [`aug_local_mixup`](aug_local_mixup.md) | `n4m.sklearn.LocalMixupAugmenter` | Neighbor-constrained mixup augmentation. |
| [`aug_local_warp`](aug_local_warp.md) | `n4m.sklearn.LocalWarpAugmenter` | Random local wavelength warping. |
| [`aug_magnitude_warp`](aug_magnitude_warp.md) | `n4m.sklearn.MagnitudeWarp` | Smooth multiplicative magnitude warp. |
| [`aug_mixup`](aug_mixup.md) | `n4m.sklearn.MixupAugmenter` | Batch-wise mixup augmentation. |
| [`aug_moisture`](aug_moisture.md) | `n4m.sklearn.MoistureAugmenter` | Water activity and moisture-content spectral perturbation. |
| [`aug_multiplicative_noise`](aug_multiplicative_noise.md) | `n4m.sklearn.MultiplicativeNoise` | Apply per-element multiplicative Gaussian noise. |
| [`aug_particle_size`](aug_particle_size.md) | `n4m.sklearn.ParticleSizeAugmenter` | Particle-size and path-length scattering simulation. |
| [`aug_path_length`](aug_path_length.md) | `n4m.sklearn.PathLengthAugmenter` | Simulate multiplicative path-length variation. |
| [`aug_poly_drift`](aug_poly_drift.md) | `n4m.sklearn.PolynomialBaselineDrift` | Add random polynomial baseline drift. |
| [`aug_random_x_op`](aug_random_x_op.md) | `n4m.sklearn.RandomXOperation` | Random element-wise multiply/add/subtract operation. |
| [`aug_rotate_translate`](aug_rotate_translate.md) | `n4m.sklearn.RotateTranslateAugmenter` | Random rotate/translate spectral augmentation. |
| [`aug_scatter_sim`](aug_scatter_sim.md) | `n4m.sklearn.ScatterSimulationMSC` | MSC-style multiplicative/additive scatter simulation. |
| [`aug_spike_noise`](aug_spike_noise.md) | `n4m.sklearn.SpikeNoise` | Inject random spike artifacts into spectra. |
| [`aug_spline_curve_simplify`](aug_spline_curve_simplify.md) | `n4m.sklearn.SplineCurveSimplificationAugmenter` | Spline curve simplification stub; current C ABI returns NOT_IMPLEMENTED. |
| [`aug_spline_smooth`](aug_spline_smooth.md) | `n4m.sklearn.SplineSmoothingAugmenter` | Spline smoothing augmenter. |
| [`aug_spline_x_perturb`](aug_spline_x_perturb.md) | `n4m.sklearn.SplineXPerturbationAugmenter` | Spline x-axis perturbation augmenter. |
| [`aug_spline_x_simplify`](aug_spline_x_simplify.md) | `n4m.sklearn.SplineXSimplificationAugmenter` | Spline x simplification stub; current C ABI returns NOT_IMPLEMENTED. |
| [`aug_spline_y_perturb`](aug_spline_y_perturb.md) | `n4m.sklearn.SplineYPerturbationAugmenter` | Spline y-axis perturbation augmenter. |
| [`aug_stray_light`](aug_stray_light.md) | `n4m.sklearn.StrayLightAugmenter` | Stray-light edge artifact. |
| [`aug_temperature`](aug_temperature.md) | `n4m.sklearn.TemperatureAugmenter` | Temperature-induced shift, intensity and broadening perturbations. |
| [`aug_truncated_peak`](aug_truncated_peak.md) | `n4m.sklearn.TruncatedPeakAugmenter` | Truncated peaks near spectral edges. |
| [`aug_unsharp_mask`](aug_unsharp_mask.md) | `n4m.sklearn.UnsharpMask` | Random unsharp spectral mask. |
| [`aug_wavelength_shift`](aug_wavelength_shift.md) | `n4m.sklearn.WavelengthShift` | Random spectral shift with linear interpolation. |
| [`aug_wavelength_stretch`](aug_wavelength_stretch.md) | `n4m.sklearn.WavelengthStretch` | Random wavelength-axis stretching. |
| [`pp_haar`](pp_haar.md) | `n4m.sklearn.Haar` | Single-level Haar DWT coefficient transform. |
| [`pp_wavelet`](pp_wavelet.md) | `n4m.sklearn.Wavelet` | Single-level DWT coefficient transform. |
| [`pp_wavelet_denoise`](pp_wavelet_denoise.md) | `n4m.sklearn.WaveletDenoise` | Multi-level DWT VisuShrink denoising. |
| [`pp_wavelet_features`](pp_wavelet_features.md) | `n4m.sklearn.WaveletFeatures` | Multi-level DWT summary features. |
| [`pp_wavelet_pca`](pp_wavelet_pca.md) | `n4m.sklearn.WaveletPCA` | DWT coefficient projection through PCA scores. |
| [`pp_wavelet_svd`](pp_wavelet_svd.md) | `n4m.sklearn.WaveletSVD` | DWT coefficient projection through SVD scores. |

## Splitters

| Operator | Binding | Description |
|----------|---------|-------------|
| [`split_binned_strat_group_kfold`](split_binned_strat_group_kfold.md) | `n4m.sklearn.BinnedStratifiedGroupKFoldSplitter` | Stratified group k-fold splitter after binning continuous ``y``. |
| [`split_kmeans`](split_kmeans.md) | `n4m.sklearn.KMeansSplitter` | K-means++ diversity splitter. |
| [`split_split_splitter`](split_split_splitter.md) | `n4m.sklearn.SPlitSplitter` | SPlit data-twinning splitter. |
| [`split_spxy_fold`](split_spxy_fold.md) | `n4m.sklearn.SPXYFoldSplitter` | SPXY k-fold splitter over paired ``X`` and ``y`` matrices. |
| [`split_spxy_g_fold`](split_spxy_g_fold.md) | `n4m.sklearn.SPXYGroupFoldSplitter` | Group-aware SPXY k-fold splitter. |
| [`split_systematic_circular`](split_systematic_circular.md) | `n4m.sklearn.SystematicCircularSplitter` | Systematic circular split over sorted or ordered targets. |

---

See the [benchmark overview](../benchmarks/overview.md) for how parity and timing are measured, and the [GitHub Pages dashboard](../landing/dashboard.md) for an interactive cross-method comparison.