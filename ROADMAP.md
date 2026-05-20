# nirs4all-methods roadmap

This is the canonical phase index. Each phase has a brief in `roadmap/phase-N-<name>.md`, a pre-implementation Codex review at `docs/reviews/phase-N/codex-pre.md`, a post-implementation Codex review at `docs/reviews/phase-N/codex-post.md`, and an independent Opus review at `docs/reviews/phase-N/opus-post.md`.

## Bootstrap & infrastructure

| # | Phase | Status | Brief |
|--:|-------|:------:|-------|
| 0 | Bootstrap (clone pls4all template, rename p4a→n4m, strip PLS code, smoke build, git+push) | 🟢 | `roadmap/phase-0-bootstrap.md` |
| 1 | Common infrastructure: PCG64+Ziggurat RNG, PCA helper (randomized SVD), banded LDLT solver, wavelet filter banks (Haar/db4/sym4/coif1), B-spline | ⚪ | `roadmap/phase-1-common.md` |

## Preprocessings (52 operators across 9 phases)

| # | Phase | Operators |
|--:|-------|-----------|
| 2 | Stateless scatter | SNV, LocalSNV, RobustSNV, AreaNormalization, Normalize, SimpleScale, LogTransform |
| 3 | Stateful scatter | MSC, EMSC, Baseline (centering), Derivate |
| 4 | Derivatives & smoothing | SavitzkyGolay, FirstDerivative, SecondDerivative, NorrisWilliams, Gaussian |
| 5 | Baseline correction | Detrend, AsLS, AirPLS, ArPLS, IModPoly, ModPoly, SNIP, RollingBall, IAsLS, BEADS, PyBaselineCorrection (dispatcher) |
| 6 | Wavelets & denoising | Wavelet, Haar, WaveletDenoise, WaveletFeatures, WaveletPCA, WaveletSVD |
| 7 | Signal conversion | ToAbsorbance, FromAbsorbance, KubelkaMunk, PercentToFraction, FractionToPercent, SignalTypeConverter |
| 8 | Orthogonalization | OSC (DOSC), EPO |
| 9 | Feature selection | CARS, MCUVE, FlexiblePCA, FlexibleSVD |
| 10 | Resampling, cropping, targets | Resampler, CropTransformer, ResampleTransformer, FlattenPreprocessing, IntegerKBinsDiscretizer, RangeDiscretizer |

## Splitters & filters

| # | Phase | Operators |
|--:|-------|-----------|
| 11 | Splitters | KennardStone, SPXY, SPXYFold, SPXYGFold, KMeans, KBinsStratified, BinnedStratifiedGroupKFold, SystematicCircular, SPlit |
| 12 | Filters: Y-outlier | YOutlierFilter (iqr/zscore/percentile/mad) |
| 13 | Filters: X-outlier | XOutlierFilter (mahalanobis/robust_mahalanobis/pca_residual/pca_leverage/isolation_forest/lof) |
| 14 | Filters: leverage/quality/composite | HighLeverageFilter (hat/pca), SpectralQualityFilter, CompositeFilter |

## Augmentations (39 augmenters across 4 phases)

| # | Phase | Operators |
|--:|-------|-----------|
| 15 | Noise + drift | GaussianAdditive, Multiplicative, Spike, Heteroscedastic noise, LinearBaselineDrift, PolynomialBaselineDrift, PathLength |
| 16 | Wavelength + spectral | WavelengthShift, WavelengthStretch, LocalWavelengthWarp, BandPerturbation, BandMasking, ChannelDropout, GaussianSmoothingJitter, UnsharpMask, SmoothMagnitudeWarp, LocalClipping |
| 17 | Mixup + physical + environmental | Mixup, LocalMixup, ScatterSimulationMSC, ParticleSize, EMSCDistortion, BatchEffect, InstrumentalBroadening, DeadBand, Temperature, Moisture |
| 18 | Edge artifacts + splines + random | DetectorRollOff, StrayLight, EdgeCurvature, TruncatedPeak, EdgeArtifacts, Spline_Smoothing, Spline_X_Perturbations, Spline_Y_Perturbations, Spline_X_Simplification (v2), Spline_Curve_Simplification (v2), Rotate_Translate, Random_X_Operation |

## Utilities & FCK

| # | Phase | Scope |
|--:|-------|-------|
| 19 | Signal type detection + NIRS metrics | SignalTypeDetector (auto-detect A/R/T/KM/log), Hotelling T² + Q-residuals, RPD/RPIQ/SEP/Bias |
| 20 | Transfer metrics | TransferMetricsComputer (CKA, Grassmann, RV, Procrustes, trustworthiness, spread_distance, centroid_distance, evr_source, evr_target) |
| 21 | FCK static transformer | FCKStaticTransformer + fractional_kernel_1d |

## Bindings

| # | Phase | Binding |
|--:|-------|---------|
| 22 | Python tier-1 + tier-2 sklearn | ctypes + sklearn-compatible classes for all 100+ operators; cibuildwheel for manylinux/macOS/Windows |
| 23 | R tier-1 + tier-2 | `.Call` gateway + S3 methods + parsnip/recipes integration |
| 24 | MATLAB | MEX dispatcher + classdefs in `+n4m/` |
| 25 | JS/WASM | Emscripten preset, subset (stateless + lightweight) |

## Benchmarks & release

| # | Phase | Scope |
|--:|-------|-------|
| 26 | Cross-binding benchmark matrix + GitHub Pages + pls4all re-pull | Re-pull pls4all to refresh CRAN/bench infra; 5 benchmark suites; bench-nightly.yml with >20% regression gate; Sphinx docs deploy |

---

Total: **27 phases**. Each phase ends with a green-CI commit, an Opus review transcript, and a user update.
