# Methods catalogue

Every native method in the library, grouped by the `n4m.<role>` namespace (ABI 2.0). Each row links to the method's documentation page and shows its fully-qualified name `n4m.<role>.<sub>...<leaf>`. Parameters, bibliographic sources, mathematical principles, binding signatures, and benchmark rows are on the linked pages.

_Total catalogued native methods_: **209**. Additional Python reference
surfaces are documented where relevant.

```{toctree}
:hidden:
:glob:
:maxdepth: 1

*
```

## transform — fit/transform feature transforms

| Method | Fully-qualified name | Namespace | Refs |
|--------|----------------------|-----------|------|
| [`cow_align`](pp_cow_align.md) | `n4m.transform.alignment.cow_align` | `n4m.transform.alignment` | C |
| [`dtw_align`](pp_dtw_align.md) | `n4m.transform.alignment.dtw_align` | `n4m.transform.alignment` | C |
| [`icoshift_align`](pp_icoshift_align.md) | `n4m.transform.alignment.icoshift_align` | `n4m.transform.alignment` | C |
| [`xcorr_align`](pp_xcorr_align.md) | `n4m.transform.alignment.xcorr_align` | `n4m.transform.alignment` | C |
| [`airpls`](pp_airpls.md) | `n4m.transform.baseline.airpls` | `n4m.transform.baseline` | C |
| [`arpls`](pp_arpls.md) | `n4m.transform.baseline.arpls` | `n4m.transform.baseline` | C |
| [`asls`](pp_asls.md) | `n4m.transform.baseline.asls` | `n4m.transform.baseline` | C |
| [`beads`](pp_beads.md) | `n4m.transform.baseline.beads` | `n4m.transform.baseline` | C |
| [`detrend`](pp_detrend.md) | `n4m.transform.baseline.detrend` | `n4m.transform.baseline` | C |
| [`iasls`](pp_iasls.md) | `n4m.transform.baseline.iasls` | `n4m.transform.baseline` | C |
| [`imodpoly`](pp_imodpoly.md) | `n4m.transform.baseline.imodpoly` | `n4m.transform.baseline` | C |
| [`modpoly`](pp_modpoly.md) | `n4m.transform.baseline.modpoly` | `n4m.transform.baseline` | C |
| [`rolling_ball`](pp_rolling_ball.md) | `n4m.transform.baseline.rolling_ball` | `n4m.transform.baseline` | C |
| [`saps`](pp_saps.md) | `n4m.transform.baseline.saps` | `n4m.transform.baseline` | C |
| [`snip`](pp_snip.md) | `n4m.transform.baseline.snip` | `n4m.transform.baseline` | C |
| [`osc`](pp_osc.md) | `n4m.transform.orthogonalization.osc` | `n4m.transform.orthogonalization` | C |
| [`crop`](pp_crop.md) | `n4m.transform.resampling.crop` | `n4m.transform.resampling` | C |
| [`kbins_discretizer`](pp_kbins_disc.md) | `n4m.transform.resampling.kbins_discretizer` | `n4m.transform.resampling` | C |
| [`range_discretizer`](pp_range_disc.md) | `n4m.transform.resampling.range_discretizer` | `n4m.transform.resampling` | C |
| [`resample_transformer`](pp_resample.md) | `n4m.transform.resampling.resample_transformer` | `n4m.transform.resampling` | C |
| [`resampler`](pp_resampler.md) | `n4m.transform.resampling.resampler` | `n4m.transform.resampling` | C |
| [`baseline_center`](pp_baseline.md) | `n4m.transform.scaling.baseline_center` | `n4m.transform.scaling` | C |
| [`log_transform`](pp_log.md) | `n4m.transform.scaling.log_transform` | `n4m.transform.scaling` | C |
| [`normalize`](pp_normalize.md) | `n4m.transform.scaling.normalize` | `n4m.transform.scaling` | C |
| [`simple_scale`](pp_simple_scale.md) | `n4m.transform.scaling.simple_scale` | `n4m.transform.scaling` | C |
| [`area_normalization`](pp_area.md) | `n4m.transform.scatter.area_normalization` | `n4m.transform.scatter` | C |
| [`emsc`](pp_emsc.md) | `n4m.transform.scatter.emsc` | `n4m.transform.scatter` | C |
| [`local_centering`](pp_local_centering.md) | `n4m.transform.scatter.local_centering` | `n4m.transform.scatter` | C |
| [`local_snv`](pp_lsnv.md) | `n4m.transform.scatter.local_snv` | `n4m.transform.scatter` | C |
| [`localized_msc`](pp_localized_msc.md) | `n4m.transform.scatter.localized_msc` | `n4m.transform.scatter` | C |
| [`msc`](pp_msc.md) | `n4m.transform.scatter.msc` | `n4m.transform.scatter` | C |
| [`piecewise_msc`](pp_piecewise_msc.md) | `n4m.transform.scatter.piecewise_msc` | `n4m.transform.scatter` | C |
| [`piecewise_snv`](pp_piecewise_snv.md) | `n4m.transform.scatter.piecewise_snv` | `n4m.transform.scatter` | C |
| [`robust_snv`](pp_rnv.md) | `n4m.transform.scatter.robust_snv` | `n4m.transform.scatter` | C |
| [`snv`](pp_snv.md) | `n4m.transform.scatter.snv` | `n4m.transform.scatter` | C |
| [`vsn`](pp_vsn.md) | `n4m.transform.scatter.vsn` | `n4m.transform.scatter` | C |
| [`weighted_snv`](pp_weighted_snv.md) | `n4m.transform.scatter.weighted_snv` | `n4m.transform.scatter` | C |
| [`fraction_to_percent`](pp_frac_to_pct.md) | `n4m.transform.signal_conversion.fraction_to_percent` | `n4m.transform.signal_conversion` | C |
| [`from_absorbance`](pp_from_absorbance.md) | `n4m.transform.signal_conversion.from_absorbance` | `n4m.transform.signal_conversion` | C |
| [`kubelka_munk`](pp_kubelka_munk.md) | `n4m.transform.signal_conversion.kubelka_munk` | `n4m.transform.signal_conversion` | C |
| [`percent_to_fraction`](pp_pct_to_frac.md) | `n4m.transform.signal_conversion.percent_to_fraction` | `n4m.transform.signal_conversion` | C |
| [`signal_type_detector`](utilities_signal_type_detector.md) | `n4m.transform.signal_conversion.signal_type_detector` | `n4m.transform.signal_conversion` | C |
| [`to_absorbance`](pp_to_absorbance.md) | `n4m.transform.signal_conversion.to_absorbance` | `n4m.transform.signal_conversion` | C |
| [`derivative`](pp_derivate.md) | `n4m.transform.smoothing.derivative` | `n4m.transform.smoothing` | C |
| [`first_derivative`](pp_first_derivative.md) | `n4m.transform.smoothing.first_derivative` | `n4m.transform.smoothing` | C |
| [`gaussian`](pp_gaussian.md) | `n4m.transform.smoothing.gaussian` | `n4m.transform.smoothing` | C |
| [`norris_williams`](pp_norris_williams.md) | `n4m.transform.smoothing.norris_williams` | `n4m.transform.smoothing` | C |
| [`savitzky_golay`](pp_savgol.md) | `n4m.transform.smoothing.savitzky_golay` | `n4m.transform.smoothing` | C |
| [`second_derivative`](pp_second_derivative.md) | `n4m.transform.smoothing.second_derivative` | `n4m.transform.smoothing` | C |
| [`fck_static`](pp_fck_static.md) | `n4m.transform.specialized.fck_static` | `n4m.transform.specialized` | C |
| [`haar`](pp_haar.md) | `n4m.transform.wavelet.haar` | `n4m.transform.wavelet` | C |
| [`wavelet`](pp_wavelet.md) | `n4m.transform.wavelet.wavelet` | `n4m.transform.wavelet` | C |
| [`wavelet_denoise`](pp_wavelet_denoise.md) | `n4m.transform.wavelet.wavelet_denoise` | `n4m.transform.wavelet` | C |
| [`wavelet_features`](pp_wavelet_features.md) | `n4m.transform.wavelet.wavelet_features` | `n4m.transform.wavelet` | C |
| [`wavelet_pca`](pp_wavelet_pca.md) | `n4m.transform.wavelet.wavelet_pca` | `n4m.transform.wavelet` | C |
| [`wavelet_svd`](pp_wavelet_svd.md) | `n4m.transform.wavelet.wavelet_svd` | `n4m.transform.wavelet` | C |

## augmentation — apply-only training-time perturbations

| Method | Fully-qualified name | Namespace | Refs |
|--------|----------------------|-----------|------|
| [`linear_drift`](aug_linear_drift.md) | `n4m.augmentation.drift.linear_drift` | `n4m.augmentation.drift` | C |
| [`path_length`](aug_path_length.md) | `n4m.augmentation.drift.path_length` | `n4m.augmentation.drift` | C |
| [`poly_drift`](aug_poly_drift.md) | `n4m.augmentation.drift.poly_drift` | `n4m.augmentation.drift` | C |
| [`detector_rolloff`](aug_detector_rolloff.md) | `n4m.augmentation.instrument.detector_rolloff` | `n4m.augmentation.instrument` | C |
| [`edge_artifacts`](aug_edge_artifacts.md) | `n4m.augmentation.instrument.edge_artifacts` | `n4m.augmentation.instrument` | C |
| [`edge_curvature`](aug_edge_curve.md) | `n4m.augmentation.instrument.edge_curvature` | `n4m.augmentation.instrument` | C |
| [`moisture`](aug_moisture.md) | `n4m.augmentation.instrument.moisture` | `n4m.augmentation.instrument` | C |
| [`stray_light`](aug_stray_light.md) | `n4m.augmentation.instrument.stray_light` | `n4m.augmentation.instrument` | C |
| [`temperature`](aug_temperature.md) | `n4m.augmentation.instrument.temperature` | `n4m.augmentation.instrument` | C |
| [`truncated_peak`](aug_truncated_peak.md) | `n4m.augmentation.instrument.truncated_peak` | `n4m.augmentation.instrument` | C |
| [`local_mixup`](aug_local_mixup.md) | `n4m.augmentation.mixup.local_mixup` | `n4m.augmentation.mixup` | C |
| [`mixup`](aug_mixup.md) | `n4m.augmentation.mixup.mixup` | `n4m.augmentation.mixup` | C |
| [`random_x_op`](aug_random_x_op.md) | `n4m.augmentation.mixup.random_x_op` | `n4m.augmentation.mixup` | C |
| [`rotate_translate`](aug_rotate_translate.md) | `n4m.augmentation.mixup.rotate_translate` | `n4m.augmentation.mixup` | C |
| [`gaussian_noise`](aug_gaussian_noise.md) | `n4m.augmentation.noise.gaussian_noise` | `n4m.augmentation.noise` | C |
| [`hetero_noise`](aug_hetero_noise.md) | `n4m.augmentation.noise.hetero_noise` | `n4m.augmentation.noise` | C |
| [`multiplicative_noise`](aug_multiplicative_noise.md) | `n4m.augmentation.noise.multiplicative_noise` | `n4m.augmentation.noise` | C |
| [`spike_noise`](aug_spike_noise.md) | `n4m.augmentation.noise.spike_noise` | `n4m.augmentation.noise` | C |
| [`batch_effect`](aug_batch_effect.md) | `n4m.augmentation.scattering.batch_effect` | `n4m.augmentation.scattering` | C |
| [`dead_band`](aug_dead_band.md) | `n4m.augmentation.scattering.dead_band` | `n4m.augmentation.scattering` | C |
| [`emsc_distort`](aug_emsc_distort.md) | `n4m.augmentation.scattering.emsc_distort` | `n4m.augmentation.scattering` | C |
| [`instrument_broaden`](aug_instrument_broaden.md) | `n4m.augmentation.scattering.instrument_broaden` | `n4m.augmentation.scattering` | C |
| [`particle_size`](aug_particle_size.md) | `n4m.augmentation.scattering.particle_size` | `n4m.augmentation.scattering` | C |
| [`scatter_sim_msc`](aug_scatter_sim.md) | `n4m.augmentation.scattering.scatter_sim_msc` | `n4m.augmentation.scattering` | C |
| [`band_mask`](aug_band_mask.md) | `n4m.augmentation.spectral.band_mask` | `n4m.augmentation.spectral` | C |
| [`band_perturb`](aug_band_perturb.md) | `n4m.augmentation.spectral.band_perturb` | `n4m.augmentation.spectral` | C |
| [`channel_dropout`](aug_channel_dropout.md) | `n4m.augmentation.spectral.channel_dropout` | `n4m.augmentation.spectral` | C |
| [`gauss_jitter`](aug_gauss_jitter.md) | `n4m.augmentation.spectral.gauss_jitter` | `n4m.augmentation.spectral` | C |
| [`local_clip`](aug_local_clip.md) | `n4m.augmentation.spectral.local_clip` | `n4m.augmentation.spectral` | C |
| [`magnitude_warp`](aug_magnitude_warp.md) | `n4m.augmentation.spectral.magnitude_warp` | `n4m.augmentation.spectral` | C |
| [`unsharp_mask`](aug_unsharp_mask.md) | `n4m.augmentation.spectral.unsharp_mask` | `n4m.augmentation.spectral` | C |
| [`spline_curve_simplification`](aug_spline_curve_simplify.md) | `n4m.augmentation.splines.spline_curve_simplification` | `n4m.augmentation.splines` | C |
| [`spline_smoothing`](aug_spline_smooth.md) | `n4m.augmentation.splines.spline_smoothing` | `n4m.augmentation.splines` | C |
| [`spline_x_perturbations`](aug_spline_x_perturb.md) | `n4m.augmentation.splines.spline_x_perturbations` | `n4m.augmentation.splines` | C |
| [`spline_x_simplification`](aug_spline_x_simplify.md) | `n4m.augmentation.splines.spline_x_simplification` | `n4m.augmentation.splines` | C |
| [`spline_y_perturbations`](aug_spline_y_perturb.md) | `n4m.augmentation.splines.spline_y_perturbations` | `n4m.augmentation.splines` | C |
| [`local_warp`](aug_local_warp.md) | `n4m.augmentation.wavelength.local_warp` | `n4m.augmentation.wavelength` | C |
| [`wavelength_shift`](aug_wavelength_shift.md) | `n4m.augmentation.wavelength.wavelength_shift` | `n4m.augmentation.wavelength` | C |
| [`wavelength_stretch`](aug_wavelength_stretch.md) | `n4m.augmentation.wavelength.wavelength_stretch` | `n4m.augmentation.wavelength` | C |

## estimators — supervised predictors (fit/predict)

| Method | Fully-qualified name | Namespace | Refs |
|--------|----------------------|-----------|------|
| [`pls_lda`](pls_lda.md) | `n4m.estimators.classification.pls_lda` | `n4m.estimators.classification` | C |
| [`pls_logistic`](pls_logistic.md) | `n4m.estimators.classification.pls_logistic` | `n4m.estimators.classification` | C |
| [`pls_qda`](pls_qda.md) | `n4m.estimators.classification.pls_qda` | `n4m.estimators.classification` | C |
| [`sparse_pls_da`](sparse_pls_da.md) | `n4m.estimators.classification.sparse_pls_da` | `n4m.estimators.classification` | C |
| [`mb_pls`](mb_pls.md) | `n4m.estimators.multiblock.mb_pls` | `n4m.estimators.multiblock` | C |
| [`mir_pls`](mir_pls.md) | `n4m.estimators.multiblock.mir_pls` | `n4m.estimators.multiblock` | C |
| [`o2pls`](o2pls.md) | `n4m.estimators.multiblock.o2pls` | `n4m.estimators.multiblock` | C |
| [`on_pls`](on_pls.md) | `n4m.estimators.multiblock.on_pls` | `n4m.estimators.multiblock` | C |
| [`rosa`](rosa.md) | `n4m.estimators.multiblock.rosa` | `n4m.estimators.multiblock` | C |
| [`so_pls`](so_pls.md) | `n4m.estimators.multiblock.so_pls` | `n4m.estimators.multiblock` | C |
| [`pls_glm`](pls_glm.md) | `n4m.estimators.regression.glm.pls_glm` | `n4m.estimators.regression.glm` | C |
| [`gpr_pls`](gpr_pls.md) | `n4m.estimators.regression.kernel.gpr_pls` | `n4m.estimators.regression.kernel` | C |
| [`kernel_pls`](kernel_pls_rbf.md) | `n4m.estimators.regression.kernel.kernel_pls` | `n4m.estimators.regression.kernel` | C |
| [`continuum_regression`](continuum_regression.md) | `n4m.estimators.regression.latent.continuum_regression` | `n4m.estimators.regression.latent` | C, Py |
| [`cppls`](cppls.md) | `n4m.estimators.regression.latent.cppls` | `n4m.estimators.regression.latent` | C, Py |
| [`ecr`](ecr.md) | `n4m.estimators.regression.latent.ecr` | `n4m.estimators.regression.latent` | C, Py |
| [`missing_aware_nipals`](missing_aware_nipals.md) | `n4m.estimators.regression.latent.missing_aware_nipals` | `n4m.estimators.regression.latent` | C |
| [`pcr`](pcr.md) | `n4m.estimators.regression.latent.pcr` | `n4m.estimators.regression.latent` | C, Py |
| [`pls`](pls.md) | `n4m.estimators.regression.latent.pls` | `n4m.estimators.regression.latent` | C, Py |
| [`lw_pls`](lw_pls.md) | `n4m.estimators.regression.local.lw_pls` | `n4m.estimators.regression.local` | C |
| [`recursive_pls`](recursive_pls.md) | `n4m.estimators.regression.online.recursive_pls` | `n4m.estimators.regression.online` | C |
| [`ridge`](ridge.md) | `n4m.estimators.regression.regularized.ridge` | `n4m.estimators.regression.regularized` | C, Py |
| [`ridge_pls`](ridge_pls.md) | `n4m.estimators.regression.regularized.ridge_pls` | `n4m.estimators.regression.regularized` | C, Py |
| [`robust_pls`](robust_pls.md) | `n4m.estimators.regression.robust.robust_pls` | `n4m.estimators.regression.robust` | C, Py |
| [`weighted_pls`](weighted_pls.md) | `n4m.estimators.regression.robust.weighted_pls` | `n4m.estimators.regression.robust` | C, Py |
| [`fused_sparse_pls`](fused_sparse_pls.md) | `n4m.estimators.regression.sparse.fused_sparse_pls` | `n4m.estimators.regression.sparse` | C |
| [`group_sparse_pls`](group_sparse_pls.md) | `n4m.estimators.regression.sparse.group_sparse_pls` | `n4m.estimators.regression.sparse` | C |
| [`sparse_simpls`](sparse_simpls.md) | `n4m.estimators.regression.sparse.sparse_simpls` | `n4m.estimators.regression.sparse` | C |
| [`n_pls`](n_pls.md) | `n4m.estimators.regression.tensor.n_pls` | `n4m.estimators.regression.tensor` | C |
| [`pls_cox`](pls_cox.md) | `n4m.estimators.survival.pls_cox` | `n4m.estimators.survival` | C |

## feature_selection — variable selectors

| Method | Fully-qualified name | Namespace | Refs |
|--------|----------------------|-----------|------|
| [`correlation`](filter_correlation.md) | `n4m.feature_selection.filter.correlation` | `n4m.feature_selection.filter` | C |
| [`variance`](filter_variance.md) | `n4m.feature_selection.filter.variance` | `n4m.feature_selection.filter` | C |
| [`interval`](interval_generator.md) | `n4m.feature_selection.interval.interval` | `n4m.feature_selection.interval` | C |
| [`variable_select`](variable_select_vip.md) | `n4m.feature_selection.ranking.variable_select` | `n4m.feature_selection.ranking` | C |
| [`bipls`](bipls_select.md) | `n4m.feature_selection.wrapper.bipls` | `n4m.feature_selection.wrapper` | C |
| [`bve`](bve_select.md) | `n4m.feature_selection.wrapper.bve` | `n4m.feature_selection.wrapper` | C |
| [`cars`](cars_select.md) | `n4m.feature_selection.wrapper.cars` | `n4m.feature_selection.wrapper` | C |
| [`emcuve`](emcuve_select.md) | `n4m.feature_selection.wrapper.emcuve` | `n4m.feature_selection.wrapper` | C |
| [`ga`](ga_select.md) | `n4m.feature_selection.wrapper.ga` | `n4m.feature_selection.wrapper` | C |
| [`ipw`](ipw_select.md) | `n4m.feature_selection.wrapper.ipw` | `n4m.feature_selection.wrapper` | C |
| [`irf`](irf_select.md) | `n4m.feature_selection.wrapper.irf` | `n4m.feature_selection.wrapper` | C |
| [`iriv`](iriv_select.md) | `n4m.feature_selection.wrapper.iriv` | `n4m.feature_selection.wrapper` | C |
| [`pso`](pso_select.md) | `n4m.feature_selection.wrapper.pso` | `n4m.feature_selection.wrapper` | C |
| [`random_frog`](random_frog_select.md) | `n4m.feature_selection.wrapper.random_frog` | `n4m.feature_selection.wrapper` | C |
| [`randomization`](randomization_select.md) | `n4m.feature_selection.wrapper.randomization` | `n4m.feature_selection.wrapper` | C |
| [`rep`](rep_select.md) | `n4m.feature_selection.wrapper.rep` | `n4m.feature_selection.wrapper` | C |
| [`scars`](scars_select.md) | `n4m.feature_selection.wrapper.scars` | `n4m.feature_selection.wrapper` | C |
| [`shaving`](shaving_select.md) | `n4m.feature_selection.wrapper.shaving` | `n4m.feature_selection.wrapper` | C |
| [`sipls`](sipls_select.md) | `n4m.feature_selection.wrapper.sipls` | `n4m.feature_selection.wrapper` | C |
| [`spa`](spa_select.md) | `n4m.feature_selection.wrapper.spa` | `n4m.feature_selection.wrapper` | C |
| [`st`](st_select.md) | `n4m.feature_selection.wrapper.st` | `n4m.feature_selection.wrapper` | C |
| [`stability`](stability_select.md) | `n4m.feature_selection.wrapper.stability` | `n4m.feature_selection.wrapper` | C |
| [`t2`](t2_select.md) | `n4m.feature_selection.wrapper.t2` | `n4m.feature_selection.wrapper` | C |
| [`uve`](uve_select.md) | `n4m.feature_selection.wrapper.uve` | `n4m.feature_selection.wrapper` | C |
| [`vip_spa`](vip_spa_select.md) | `n4m.feature_selection.wrapper.vip_spa` | `n4m.feature_selection.wrapper` | C |
| [`vissa`](vissa_select.md) | `n4m.feature_selection.wrapper.vissa` | `n4m.feature_selection.wrapper` | C |
| [`wvc`](wvc_select.md) | `n4m.feature_selection.wrapper.wvc` | `n4m.feature_selection.wrapper` | C |
| [`wvc_threshold`](wvc_threshold_select.md) | `n4m.feature_selection.wrapper.wvc_threshold` | `n4m.feature_selection.wrapper` | C |

## model_selection — splitters, AOM search/campaign, sweep

| Method | Fully-qualified name | Namespace | Refs |
|--------|----------------------|-----------|------|
| [`aom_chain_screen_refit`](aom_chain_sweep_run.md) | `n4m.model_selection.aom_campaign.aom_chain_screen_refit` | `n4m.model_selection.aom_campaign` | C, Py |
| [`aom_staged_chain_campaign`](aom_staged_chain_campaign.md) | `n4m.model_selection.aom_campaign.aom_staged_chain_campaign` | `n4m.model_selection.aom_campaign` | C, Py |
| [`aom_chain_fixed_fit`](aom_chain_sweep_run.md) | `n4m.model_selection.aom_search.aom_chain_fixed_fit` | `n4m.model_selection.aom_search` | C, Py |
| [`aom_chain_ridge_pls`](aom_chain_ridge_pls.md) | `n4m.model_selection.aom_search.aom_chain_ridge_pls` | `n4m.model_selection.aom_search` | C, Py |
| [`aom_chain_sweep`](aom_chain_sweep_run.md) | `n4m.model_selection.aom_search.aom_chain_sweep` | `n4m.model_selection.aom_search` | C, Py |
| [`aom_pls`](aom_pls.md) | `n4m.model_selection.aom_search.aom_pls` | `n4m.model_selection.aom_search` | C, Py |
| [`aom_preprocessing`](aom_preprocess.md) | `n4m.model_selection.aom_search.aom_preprocessing` | `n4m.model_selection.aom_search` | C, Py |
| [`aom_sweep`](aom_sweep_run.md) | `n4m.model_selection.aom_search.aom_sweep` | `n4m.model_selection.aom_search` | C, Py |
| [`pop_pls`](pop_pls.md) | `n4m.model_selection.aom_search.pop_pls` | `n4m.model_selection.aom_search` | C, Py |
| [`ridge_global`](aom_pop_ridge_global.md) | `n4m.model_selection.aom_search.ridge_global` | `n4m.model_selection.aom_search` | C, Py |
| [`robust_hpo`](aom_robust_hpo.md) | `n4m.model_selection.aom_search.robust_hpo` | `n4m.model_selection.aom_search` | C, Py |
| [`binned_strat_group_kfold`](split_binned_strat_group_kfold.md) | `n4m.model_selection.splitters.binned_strat_group_kfold` | `n4m.model_selection.splitters` | C |
| [`data_twinning`](split_split_splitter.md) | `n4m.model_selection.splitters.data_twinning` | `n4m.model_selection.splitters` | C |
| [`kbins_stratified`](split_kbins_stratified.md) | `n4m.model_selection.splitters.kbins_stratified` | `n4m.model_selection.splitters` | C |
| [`kennard_stone`](split_kennard_stone.md) | `n4m.model_selection.splitters.kennard_stone` | `n4m.model_selection.splitters` | C |
| [`kmeans`](split_kmeans.md) | `n4m.model_selection.splitters.kmeans` | `n4m.model_selection.splitters` | C |
| [`spxy`](split_spxy.md) | `n4m.model_selection.splitters.spxy` | `n4m.model_selection.splitters` | C |
| [`spxy_fold`](split_spxy_fold.md) | `n4m.model_selection.splitters.spxy_fold` | `n4m.model_selection.splitters` | C |
| [`spxy_g_fold`](split_spxy_g_fold.md) | `n4m.model_selection.splitters.spxy_g_fold` | `n4m.model_selection.splitters` | C |
| [`systematic_circular`](split_systematic_circular.md) | `n4m.model_selection.splitters.systematic_circular` | `n4m.model_selection.splitters` | C |
| [`sweep`](sweep_run.md) | `n4m.model_selection.sweep.sweep` | `n4m.model_selection.sweep` | C, Py |

## domain_adaptation — calibration transfer / standardization

| Method | Fully-qualified name | Namespace | Refs |
|--------|----------------------|-----------|------|
| [`di_pls`](di_pls.md) | `n4m.domain_adaptation.invariant.di_pls` | `n4m.domain_adaptation.invariant` | C |
| [`transfer_metrics`](utilities_transfer_metrics.md) | `n4m.domain_adaptation.metrics.transfer_metrics` | `n4m.domain_adaptation.metrics` | C |
| [`epo`](pp_epo.md) | `n4m.domain_adaptation.orthogonalization.epo` | `n4m.domain_adaptation.orthogonalization` | C |
| [`direct_standardization`](pp_direct_standardization.md) | `n4m.domain_adaptation.standardization.direct_standardization` | `n4m.domain_adaptation.standardization` | C |
| [`ds`](ds.md) | `n4m.domain_adaptation.standardization.ds` | `n4m.domain_adaptation.standardization` | C |
| [`pds`](pds.md) | `n4m.domain_adaptation.standardization.pds` | `n4m.domain_adaptation.standardization` | C |
| [`piecewise_direct_standardization`](pp_piecewise_direct_standardization.md) | `n4m.domain_adaptation.standardization.piecewise_direct_standardization` | `n4m.domain_adaptation.standardization` | C |
| [`robust_direct_standardization`](pp_robust_direct_standardization.md) | `n4m.domain_adaptation.standardization.robust_direct_standardization` | `n4m.domain_adaptation.standardization` | C |
| [`slope_bias`](pp_slope_bias.md) | `n4m.domain_adaptation.standardization.slope_bias` | `n4m.domain_adaptation.standardization` | C |

## outlier_detection — sample-level screeners + Q/T²

| Method | Fully-qualified name | Namespace | Refs |
|--------|----------------------|-----------|------|
| [`composite`](filter_composite.md) | `n4m.outlier_detection.composite` | `n4m.outlier_detection` | C |
| [`high_leverage`](filter_leverage.md) | `n4m.outlier_detection.high_leverage` | `n4m.outlier_detection` | C |
| [`hotelling_t2`](utilities_hotelling_t2.md) | `n4m.outlier_detection.hotelling_t2` | `n4m.outlier_detection` | C |
| [`q_residuals`](utilities_q_residuals.md) | `n4m.outlier_detection.q_residuals` | `n4m.outlier_detection` | C |
| [`spectral_quality`](filter_quality.md) | `n4m.outlier_detection.spectral_quality` | `n4m.outlier_detection` | C |
| [`x_outlier`](filter_x_outlier.md) | `n4m.outlier_detection.x_outlier` | `n4m.outlier_detection` | C |
| [`y_outlier`](filter_y_outlier.md) | `n4m.outlier_detection.y_outlier` | `n4m.outlier_detection` | C |

## ensemble — bagging / boosting / stacking / AOM blenders

| Method | Fully-qualified name | Namespace | Refs |
|--------|----------------------|-----------|------|
| [`aom_operator_pls_stack`](aom_operator_pls_stack.md) | `n4m.ensemble.aom_operator_pls_stack` | `n4m.ensemble` | C, Py |
| [`aom_ridge_blender`](aom_ridge_blender.md) | `n4m.ensemble.aom_ridge_blender` | `n4m.ensemble` | C, Py |
| [`bagging_pls`](bagging_pls.md) | `n4m.ensemble.bagging_pls` | `n4m.ensemble` | C |
| [`boosting_pls`](boosting_pls.md) | `n4m.ensemble.boosting_pls` | `n4m.ensemble` | C |
| [`moment_stack`](moment_stack.md) | `n4m.ensemble.moment_stack` | `n4m.ensemble` | C, Py |
| [`random_subspace_pls`](random_subspace_pls.md) | `n4m.ensemble.random_subspace_pls` | `n4m.ensemble` | C |

## compose — AOM operator superblocks

| Method | Fully-qualified name | Namespace | Refs |
|--------|----------------------|-----------|------|
| [`aom_pls_superblock`](aom_pls_superblock.md) | `n4m.compose.aom_superblock.aom_pls_superblock` | `n4m.compose.aom_superblock` | C, Py |
| [`aom_ridge_pls_superblock`](aom_ridge_pls_superblock.md) | `n4m.compose.aom_superblock.aom_ridge_pls_superblock` | `n4m.compose.aom_superblock` | C, Py |
| [`ridge_active_superblock`](aom_pop_ridge_active_superblock.md) | `n4m.compose.aom_superblock.ridge_active_superblock` | `n4m.compose.aom_superblock` | C, Py |
| [`ridge_mkl_superblock`](aom_pop_ridge_mkl_superblock.md) | `n4m.compose.aom_superblock.ridge_mkl_superblock` | `n4m.compose.aom_superblock` | C, Py |
| [`ridge_superblock`](aom_pop_ridge_superblock.md) | `n4m.compose.aom_superblock.ridge_superblock` | `n4m.compose.aom_superblock` | C, Py |

## metrics — scoring + diagnostics

| Method | Fully-qualified name | Namespace | Refs |
|--------|----------------------|-----------|------|
| [`pls_diagnostics`](diagnostics_pls_diagnostics.md) | `n4m.metrics.diagnostics.pls_diagnostics` | `n4m.metrics.diagnostics` | C |
| [`pls_monitoring`](pls_monitoring.md) | `n4m.metrics.diagnostics.pls_monitoring` | `n4m.metrics.diagnostics` | C |
| [`approximate_press`](approximate_press.md) | `n4m.metrics.scoring.approximate_press` | `n4m.metrics.scoring` | C |
| [`one_se_rule`](one_se_rule.md) | `n4m.metrics.scoring.one_se_rule` | `n4m.metrics.scoring` | C |
| [`regression_metrics`](diagnostics_regression_metrics.md) | `n4m.metrics.scoring.regression_metrics` | `n4m.metrics.scoring` | C |

## decomposition — flexible PCA / SVD

| Method | Fully-qualified name | Namespace | Refs |
|--------|----------------------|-----------|------|
| [`flexible_pca`](pp_flex_pca.md) | `n4m.decomposition.flexible_pca` | `n4m.decomposition` | C |
| [`flexible_svd`](pp_flex_svd.md) | `n4m.decomposition.flexible_svd` | `n4m.decomposition` | C |

## lowlevel — sufficient-statistics substrate

| Method | Fully-qualified name | Namespace | Refs |
|--------|----------------------|-----------|------|
| [`moments`](moments.md) | `n4m.lowlevel.moments.moments` | `n4m.lowlevel.moments` | C, Py |

---

See the [benchmark overview](../benchmarks/overview.md) for how parity and timing are measured, and the [GitHub Pages dashboard](../landing/dashboard.md) for an interactive cross-method comparison. The ABI 2.0 namespace migration is documented in [the ABI-2 migration guide](../MIGRATION_ABI2.md).