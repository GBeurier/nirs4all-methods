# Methods catalogue

Every public chemometrics4all method documented with its parameters, bibliographic source, mathematical principle, binding signatures, external references, and benchmark rows when available.

_Total methods_: **80**. Grouped by family below.

```{toctree}
:hidden:
:glob:
:maxdepth: 1

*
```

## Preprocessing

| Method | Name | Benchmark | References |
|--------|------|-----------|------------|
| [`area_norm`](area_norm.md) | Area normalization | yes | nirs4all, numpy, r.base |
| [`baseline_center`](baseline_center.md) | Baseline center | yes | nirs4all |
| [`derivate`](derivate.md) | Derivate | no | — |
| [`emsc`](emsc.md) | EMSC | yes | nirs4all |
| [`first_derivative`](first_derivative.md) | First derivative | yes | nirs4all, numpy |
| [`frac_to_pct`](frac_to_pct.md) | Fraction to percent | yes | numpy, r.base |
| [`from_absorbance`](from_absorbance.md) | From absorbance | yes | numpy, r.base |
| [`gaussian`](gaussian.md) | Gaussian | yes | nirs4all, scipy |
| [`kubelka_munk`](kubelka_munk.md) | Kubelka-Munk | yes | numpy, r.base |
| [`log_transform`](log_transform.md) | Log Transform | no | — |
| [`lsnv`](lsnv.md) | LSNV | yes | nirs4all |
| [`msc`](msc.md) | MSC | yes | nirs4all |
| [`normalize`](normalize.md) | Normalize | yes | numpy, r.base |
| [`norris_williams`](norris_williams.md) | Norris-Williams | yes | nirs4all |
| [`pct_to_frac`](pct_to_frac.md) | Percent to fraction | yes | numpy, r.base |
| [`rnv`](rnv.md) | RNV | yes | numpy, r.base |
| [`savitzky_golay`](savitzky_golay.md) | Savitzky-Golay | yes | nirs4all, scipy |
| [`second_derivative`](second_derivative.md) | Second derivative | yes | nirs4all, numpy |
| [`simple_scale`](simple_scale.md) | Simple scale | yes | numpy, r.base, sklearn |
| [`snv`](snv.md) | SNV | yes | numpy, r.base |
| [`to_absorbance`](to_absorbance.md) | To absorbance | yes | numpy, r.base |

## Baseline

| Method | Name | Benchmark | References |
|--------|------|-----------|------------|
| [`airpls`](airpls.md) | AirPLS | yes | pybaselines |
| [`arpls`](arpls.md) | ArPLS | yes | pybaselines |
| [`asls`](asls.md) | AsLS | yes | pybaselines |
| [`beads`](beads.md) | BEADS | yes | pybaselines |
| [`detrend`](detrend.md) | Detrend | yes | nirs4all, r.stats, scipy |
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
| [`wavelet_pca`](wavelet_pca.md) | WaveletPCA | no | — |
| [`wavelet_svd`](wavelet_svd.md) | WaveletSVD | no | — |

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
| [`composite_filter`](composite_filter.md) | CompositeFilter | no | — |
| [`high_leverage`](high_leverage.md) | HighLeverageFilter | no | — |
| [`spectral_quality`](spectral_quality.md) | SpectralQualityFilter | no | — |
| [`x_outlier_mahalanobis`](x_outlier_mahalanobis.md) | X outlier Mahalanobis | yes | nirs4all, sklearn |
| [`y_outlier_iqr`](y_outlier_iqr.md) | Y outlier IQR | yes | nirs4all |

## Diagnostic

| Method | Name | Benchmark | References |
|--------|------|-----------|------------|
| [`hotelling_t2`](hotelling_t2.md) | Hotelling T² | no | — |
| [`q_residuals`](q_residuals.md) | Q-residuals | no | — |
| [`signal_type_detector`](signal_type_detector.md) | SignalTypeDetector | no | — |

## Augmentation

| Method | Name | Benchmark | References |
|--------|------|-----------|------------|
| [`aug_batch_effect`](aug_batch_effect.md) | BatchEffectAugmenter | no | — |
| [`aug_dead_band`](aug_dead_band.md) | DeadBandAugmenter | no | — |
| [`aug_emsc_distort`](aug_emsc_distort.md) | EMSCDistortionAugmenter | no | — |
| [`aug_gaussian_noise`](aug_gaussian_noise.md) | GaussianAdditiveNoise | no | — |
| [`aug_hetero_noise`](aug_hetero_noise.md) | HeteroscedasticNoiseAugmenter | no | — |
| [`aug_instrument_broaden`](aug_instrument_broaden.md) | InstrumentalBroadeningAugmenter | no | — |
| [`aug_linear_drift`](aug_linear_drift.md) | LinearBaselineDrift | no | — |
| [`aug_local_mixup`](aug_local_mixup.md) | LocalMixupAugmenter | no | — |
| [`aug_mixup`](aug_mixup.md) | MixupAugmenter | no | — |
| [`aug_moisture`](aug_moisture.md) | MoistureAugmenter | no | — |
| [`aug_multiplicative_noise`](aug_multiplicative_noise.md) | MultiplicativeNoise | no | — |
| [`aug_particle_size`](aug_particle_size.md) | ParticleSizeAugmenter | no | — |
| [`aug_path_length`](aug_path_length.md) | PathLengthAugmenter | no | — |
| [`aug_poly_drift`](aug_poly_drift.md) | PolynomialBaselineDrift | no | — |
| [`aug_scatter_sim`](aug_scatter_sim.md) | ScatterSimulationMSC | no | — |
| [`aug_spike_noise`](aug_spike_noise.md) | SpikeNoise | no | — |
| [`aug_temperature`](aug_temperature.md) | TemperatureAugmenter | no | — |
| [`aug_wavelength_spectral`](aug_wavelength_spectral.md) | Phase 16 | no | — |
| [`augmenters_phase18`](augmenters_phase18.md) | Phase 18 augmenters | no | — |

## Metric

| Method | Name | Benchmark | References |
|--------|------|-----------|------------|
| [`nirs_metrics`](nirs_metrics.md) | NIRS regression metrics | no | — |
| [`transfer_metrics`](transfer_metrics.md) | TransferMetricsComputer | no | — |

## Utility

| Method | Name | Benchmark | References |
|--------|------|-----------|------------|
| [`rng_pcg64`](rng_pcg64.md) | PCG64 Random Number Generator | no | — |

---

See the [benchmark overview](../benchmarks/overview.md) and the [interactive dashboard](../landing/dashboard.md).