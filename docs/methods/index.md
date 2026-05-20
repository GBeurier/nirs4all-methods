# Methods catalogue

Every public chemometrics4all method documented with its parameters, bibliographic source, mathematical principle, binding signatures, benchmark comparators/sources, and benchmark rows when available.

_Total methods_: **117**. Grouped by family below.

```{toctree}
:hidden:
:glob:
:maxdepth: 1

*
```

## Preprocessing

| Method | Name | Benchmark | References |
|--------|------|-----------|------------|
| [`area_norm`](area_norm.md) | Area normalization | yes | nirs4all |
| [`baseline_center`](baseline_center.md) | Baseline center | yes | nirs4all |
| [`derivate`](derivate.md) | Derivate | yes | nirs4all, numpy |
| [`emsc`](emsc.md) | EMSC | yes | nirs4all |
| [`first_derivative`](first_derivative.md) | First derivative | yes | nirs4all |
| [`frac_to_pct`](frac_to_pct.md) | Fraction to percent | yes | nirs4all |
| [`from_absorbance`](from_absorbance.md) | From absorbance | yes | nirs4all |
| [`gaussian`](gaussian.md) | Gaussian | yes | nirs4all, scipy |
| [`kubelka_munk`](kubelka_munk.md) | Kubelka-Munk | yes | nirs4all |
| [`localized_msc`](localized_msc.md) | Localized MSC | yes | r.prospectr, scipy, sklearn |
| [`log_transform`](log_transform.md) | Log transform | yes | nirs4all, numpy |
| [`lsnv`](lsnv.md) | LSNV | yes | nirs4all |
| [`msc`](msc.md) | MSC | yes | nirs4all, r.prospectr |
| [`normalize`](normalize.md) | Normalize | yes | nirs4all |
| [`norris_williams`](norris_williams.md) | Norris-Williams | yes | nirs4all |
| [`pct_to_frac`](pct_to_frac.md) | Percent to fraction | yes | nirs4all |
| [`piecewise_msc`](piecewise_msc.md) | Piecewise MSC | yes | r.prospectr, sklearn |
| [`piecewise_snv`](piecewise_snv.md) | Piecewise SNV | yes | nirs4all |
| [`rnv`](rnv.md) | RNV | yes | nirs4all |
| [`savitzky_golay`](savitzky_golay.md) | Savitzky-Golay | yes | nirs4all, r.prospectr, scipy |
| [`second_derivative`](second_derivative.md) | Second derivative | yes | nirs4all |
| [`simple_scale`](simple_scale.md) | Simple scale | yes | nirs4all, sklearn |
| [`snv`](snv.md) | SNV | yes | nirs4all, r.prospectr |
| [`to_absorbance`](to_absorbance.md) | To absorbance | yes | nirs4all |
| [`variable_sorting_normalization`](variable_sorting_normalization.md) | Variable sorting normalization | yes | statsmodels |
| [`weighted_snv`](weighted_snv.md) | Weighted SNV | yes | statsmodels |

## Baseline

| Method | Name | Benchmark | References |
|--------|------|-----------|------------|
| [`airpls`](airpls.md) | AirPLS | yes | pybaselines |
| [`arpls`](arpls.md) | ArPLS | yes | pybaselines |
| [`asls`](asls.md) | AsLS | yes | pybaselines |
| [`beads`](beads.md) | BEADS | yes | pybaselines |
| [`detrend`](detrend.md) | Detrend | yes | nirs4all, r.prospectr, scipy |
| [`iasls`](iasls.md) | IAsLS | yes | pybaselines |
| [`imodpoly`](imodpoly.md) | IModPoly | yes | pybaselines |
| [`modpoly`](modpoly.md) | ModPoly | yes | pybaselines |
| [`rolling_ball`](rolling_ball.md) | Rolling ball | yes | pybaselines |
| [`snip`](snip.md) | SNIP | yes | pybaselines |

## Wavelet

| Method | Name | Benchmark | References |
|--------|------|-----------|------------|
| [`haar`](haar.md) | Haar | yes | nirs4all, pywavelets |
| [`wavelet`](wavelet.md) | Wavelet | yes | nirs4all, pywavelets |
| [`wavelet_denoise`](wavelet_denoise.md) | Wavelet denoise | yes | nirs4all, pywavelets |
| [`wavelet_features`](wavelet_features.md) | Wavelet features | yes | nirs4all, pywavelets |
| [`wavelet_pca`](wavelet_pca.md) | Wavelet PCA | yes | nirs4all, pywavelets |
| [`wavelet_svd`](wavelet_svd.md) | Wavelet SVD | yes | nirs4all, pywavelets |

## Feature extraction

| Method | Name | Benchmark | References |
|--------|------|-----------|------------|
| [`epo`](epo.md) | EPO | yes | nirs4all |
| [`fck_static`](fck_static.md) | FCK static | yes | nirs4all |
| [`flexible_pca`](flexible_pca.md) | Flexible PCA | yes | nirs4all, sklearn |
| [`flexible_svd`](flexible_svd.md) | Flexible SVD | yes | nirs4all, sklearn |
| [`osc`](osc.md) | OSC | yes | nirs4all |

## Resampling

| Method | Name | Benchmark | References |
|--------|------|-----------|------------|
| [`crop`](crop.md) | Crop | yes | nirs4all |
| [`kbins_discretizer`](kbins_discretizer.md) | K-bins discretizer | yes | nirs4all, sklearn |
| [`range_discretizer`](range_discretizer.md) | Range discretizer | yes | nirs4all |
| [`resample`](resample.md) | Resample | yes | nirs4all, scipy |
| [`resampler`](resampler.md) | Wavelength resampler | yes | nirs4all |

## Splitter

| Method | Name | Benchmark | References |
|--------|------|-----------|------------|
| [`kbins_stratified`](kbins_stratified.md) | K-bins stratified | yes | nirs4all, sklearn |
| [`kennard_stone`](kennard_stone.md) | Kennard-Stone | yes | nirs4all |
| [`spxy`](spxy.md) | SPXY | yes | nirs4all |

## Filter

| Method | Name | Benchmark | References |
|--------|------|-----------|------------|
| [`composite_filter`](composite_filter.md) | Composite filter | yes | nirs4all |
| [`high_leverage`](high_leverage.md) | High leverage filter | yes | nirs4all |
| [`spectral_quality`](spectral_quality.md) | Spectral quality filter | yes | nirs4all |
| [`x_outlier_mahalanobis`](x_outlier_mahalanobis.md) | X outlier Mahalanobis | yes | nirs4all, sklearn |
| [`y_outlier_iqr`](y_outlier_iqr.md) | Y outlier IQR | yes | nirs4all |

## Diagnostic

| Method | Name | Benchmark | References |
|--------|------|-----------|------------|
| [`hotelling_t2`](hotelling_t2.md) | Hotelling T2 | yes | sklearn |
| [`q_residuals`](q_residuals.md) | Q residuals | yes | sklearn |
| [`signal_type_detector`](signal_type_detector.md) | Signal type detector | yes | nirs4all |

## Augmentation

| Method | Name | Benchmark | References |
|--------|------|-----------|------------|
| [`aug_band_mask`](aug_band_mask.md) | Band masking augmenter | yes | nirs4all |
| [`aug_band_perturb`](aug_band_perturb.md) | Band perturbation augmenter | yes | nirs4all |
| [`aug_batch_effect`](aug_batch_effect.md) | Batch effect augmenter | yes | nirs4all |
| [`aug_channel_dropout`](aug_channel_dropout.md) | Channel dropout augmenter | yes | nirs4all |
| [`aug_dead_band`](aug_dead_band.md) | Dead band augmenter | yes | nirs4all |
| [`aug_detector_rolloff`](aug_detector_rolloff.md) | Detector roll-off augmenter | yes | nirs4all |
| [`aug_edge_artifacts`](aug_edge_artifacts.md) | Combined edge artifact augmenter | yes | nirs4all |
| [`aug_edge_curve`](aug_edge_curve.md) | Edge curvature augmenter | yes | nirs4all |
| [`aug_emsc_distort`](aug_emsc_distort.md) | EMSC distortion augmenter | yes | nirs4all |
| [`aug_gauss_jitter`](aug_gauss_jitter.md) | Gaussian smoothing jitter | yes | nirs4all |
| [`aug_gaussian_noise`](aug_gaussian_noise.md) | Gaussian additive noise | yes | nirs4all |
| [`aug_hetero_noise`](aug_hetero_noise.md) | Heteroscedastic noise | yes | nirs4all |
| [`aug_instrument_broaden`](aug_instrument_broaden.md) | Instrumental broadening augmenter | yes | nirs4all |
| [`aug_linear_drift`](aug_linear_drift.md) | Linear baseline drift | yes | nirs4all |
| [`aug_local_clip`](aug_local_clip.md) | Local clipping augmenter | yes | nirs4all |
| [`aug_local_mixup`](aug_local_mixup.md) | Local mixup augmenter | yes | nirs4all |
| [`aug_local_warp`](aug_local_warp.md) | Local wavelength warp augmenter | yes | nirs4all |
| [`aug_magnitude_warp`](aug_magnitude_warp.md) | Smooth magnitude warp augmenter | yes | nirs4all |
| [`aug_mixup`](aug_mixup.md) | Mixup augmenter | yes | nirs4all |
| [`aug_moisture`](aug_moisture.md) | Moisture augmenter | yes | nirs4all |
| [`aug_multiplicative_noise`](aug_multiplicative_noise.md) | Multiplicative noise | yes | nirs4all |
| [`aug_particle_size`](aug_particle_size.md) | Particle size augmenter | yes | nirs4all |
| [`aug_path_length`](aug_path_length.md) | Path length augmenter | yes | nirs4all |
| [`aug_poly_drift`](aug_poly_drift.md) | Polynomial baseline drift | yes | nirs4all |
| [`aug_random_x_op`](aug_random_x_op.md) | Random X operation augmenter | yes | nirs4all |
| [`aug_rotate_translate`](aug_rotate_translate.md) | Rotate/translate augmenter | yes | nirs4all |
| [`aug_scatter_sim`](aug_scatter_sim.md) | Scatter simulation MSC | yes | nirs4all |
| [`aug_spike_noise`](aug_spike_noise.md) | Spike noise | yes | nirs4all |
| [`aug_spline_smooth`](aug_spline_smooth.md) | Spline smoothing augmenter | yes | nirs4all |
| [`aug_spline_x_perturb`](aug_spline_x_perturb.md) | Spline X perturbation augmenter | yes | nirs4all |
| [`aug_spline_y_perturb`](aug_spline_y_perturb.md) | Spline Y perturbation augmenter | yes | nirs4all |
| [`aug_stray_light`](aug_stray_light.md) | Stray light augmenter | yes | nirs4all |
| [`aug_temperature`](aug_temperature.md) | Temperature augmenter | yes | nirs4all |
| [`aug_truncated_peak`](aug_truncated_peak.md) | Truncated peak augmenter | yes | nirs4all |
| [`aug_unsharp_mask`](aug_unsharp_mask.md) | Unsharp spectral mask augmenter | yes | nirs4all |
| [`aug_wavelength_shift`](aug_wavelength_shift.md) | Wavelength shift augmenter | yes | nirs4all |
| [`aug_wavelength_spectral`](aug_wavelength_spectral.md) | Wavelength/spectral augmenters | yes | nirs4all |
| [`aug_wavelength_stretch`](aug_wavelength_stretch.md) | Wavelength stretch augmenter | yes | nirs4all |

## Metric

| Method | Name | Benchmark | References |
|--------|------|-----------|------------|
| [`nirs_metrics`](nirs_metrics.md) | NIRS metrics | yes | numpy |
| [`transfer_metrics`](transfer_metrics.md) | Transfer metrics | yes | nirs4all |

## Utility

| Method | Name | Benchmark | References |
|--------|------|-----------|------------|
| [`rng_pcg64`](rng_pcg64.md) | PCG64 RNG | yes | numpy |

---

See the [benchmark overview](../benchmarks/overview.md) and the [interactive dashboard](../landing/dashboard.md).