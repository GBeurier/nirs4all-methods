# SPDX-License-Identifier: CECILL-2.1
"""Smoke coverage for sklearn Tier-2 wrappers not covered by parity tests."""
from __future__ import annotations

import numpy as np
import pytest


def _spectra(rows: int = 12, cols: int = 32):
    wavelengths = np.linspace(1000.0, 1700.0, cols, dtype=np.float64)
    x = np.linspace(0.1, 2.0, rows * cols, dtype=np.float64).reshape(rows, cols)
    return x, wavelengths


def test_public_python_and_sklearn_surfaces_are_distinct():
    import n4m as n4m
    from n4m import python as n4m_python
    from n4m import sklearn as n4m_sklearn

    x, _ = _spectra(rows=4, cols=8)
    direct = n4m_python.snv(x, ddof=1)
    estimator = n4m_sklearn.SNV(ddof=1)
    wrapped = estimator.fit_transform(x)

    assert np.allclose(direct, wrapped)
    assert n4m.python is n4m_python
    assert n4m.sklearn is n4m_sklearn

    try:
        from sklearn.base import BaseEstimator, TransformerMixin
    except Exception:
        pytest.skip("scikit-learn is not installed")

    assert isinstance(estimator, BaseEstimator)
    assert isinstance(estimator, TransformerMixin)


@pytest.mark.parametrize(
    "factory",
    [
        lambda wl: __import__("n4m").MultiplicativeNoise(seed=1),
        lambda wl: __import__("n4m").SpikeNoise(seed=1),
        lambda wl: __import__("n4m").HeteroscedasticNoiseAugmenter(seed=1),
        lambda wl: __import__("n4m").LinearBaselineDrift(seed=1),
        lambda wl: __import__("n4m").PolynomialBaselineDrift(seed=1),
        lambda wl: __import__("n4m").PathLengthAugmenter(seed=1),
        lambda wl: __import__("n4m").WavelengthShift(wavelengths=wl, seed=1),
        lambda wl: __import__("n4m").WavelengthStretch(wavelengths=wl, seed=1),
        lambda wl: __import__("n4m").LocalWarpAugmenter(wavelengths=wl, seed=1),
        lambda wl: __import__("n4m").BandPerturbationAugmenter(seed=1),
        lambda wl: __import__("n4m").BandMasking(seed=1),
        lambda wl: __import__("n4m").ChannelDropout(seed=1),
        lambda wl: __import__("n4m").GaussianJitter(seed=1),
        lambda wl: __import__("n4m").UnsharpMask(seed=1),
        lambda wl: __import__("n4m").MagnitudeWarp(wavelengths=wl, seed=1),
        lambda wl: __import__("n4m").LocalClip(seed=1),
        lambda wl: __import__("n4m").MixupAugmenter(seed=1),
        lambda wl: __import__("n4m").LocalMixupAugmenter(seed=1),
        lambda wl: __import__("n4m").ScatterSimulationMSC(seed=1),
        lambda wl: __import__("n4m").ParticleSizeAugmenter(wavelengths=wl, seed=1),
        lambda wl: __import__("n4m").EMSCDistortionAugmenter(wavelengths=wl, seed=1),
        lambda wl: __import__("n4m").BatchEffectAugmenter(wavelengths=wl, seed=1),
        lambda wl: __import__("n4m").InstrumentalBroadeningAugmenter(wavelengths=wl, seed=1),
        lambda wl: __import__("n4m").DeadBandAugmenter(seed=1),
        lambda wl: __import__("n4m").TemperatureAugmenter(wavelengths=wl, seed=1),
        lambda wl: __import__("n4m").MoistureAugmenter(wavelengths=wl, seed=1),
        lambda wl: __import__("n4m").DetectorRollOffAugmenter(wavelengths=wl, seed=1),
        lambda wl: __import__("n4m").StrayLightAugmenter(wavelengths=wl, seed=1),
        lambda wl: __import__("n4m").EdgeCurvatureAugmenter(wavelengths=wl, seed=1),
        lambda wl: __import__("n4m").TruncatedPeakAugmenter(wavelengths=wl, seed=1),
        lambda wl: __import__("n4m").EdgeArtifactsAugmenter(wavelengths=wl, seed=1),
        lambda wl: __import__("n4m").SplineSmoothingAugmenter(seed=1),
        lambda wl: __import__("n4m").SplineXPerturbationAugmenter(seed=1),
        lambda wl: __import__("n4m").SplineYPerturbationAugmenter(seed=1),
        lambda wl: __import__("n4m").RotateTranslateAugmenter(seed=1),
        lambda wl: __import__("n4m").RandomXOperation(seed=1),
    ],
)
def test_augmenter_wrappers_shape(factory):
    x, wavelengths = _spectra()
    out = factory(wavelengths).fit_transform(x)
    assert out.shape == x.shape
    assert np.isfinite(out).all()


def test_deferred_spline_simplification_wrappers_raise_not_implemented():
    import n4m as n4m

    x, _ = _spectra()
    with pytest.raises(n4m.N4MError):
        n4m.SplineXSimplificationAugmenter(seed=1).fit_transform(x)
    with pytest.raises(n4m.N4MError):
        n4m.SplineCurveSimplificationAugmenter(seed=1).fit_transform(x)


def test_splitter_wrappers_smoke():
    import n4m as n4m

    x, _ = _spectra(rows=24, cols=6)
    y = np.arange(x.shape[0], dtype=np.float64).reshape(-1, 1)
    groups = np.repeat(np.arange(12, dtype=np.int64), 2)

    for train, test in [
        n4m.KMeansSplitter(seed=1).split(x),
        n4m.SystematicCircularSplitter(seed=1).split(y),
        n4m.SPlitSplitter(seed=1).split(x),
    ]:
        assert train.ndim == test.ndim == 1
        assert train.size + test.size == x.shape[0]

    train, test = n4m.SPXYFoldSplitter(n_splits=3).split_fold(x, y, 0)
    assert train.size + test.size == x.shape[0]

    train, test = n4m.SPXYGroupFoldSplitter(n_splits=3).split_fold(x, y, groups, 0)
    assert train.size + test.size == x.shape[0]

    train, test = n4m.BinnedStratifiedGroupKFoldSplitter(
        n_splits=3, n_bins=2, seed=1
    ).split_fold(y, groups, 0)
    assert train.size + test.size == x.shape[0]


def test_filter_meta_wrappers_smoke():
    import n4m as n4m

    x, _ = _spectra(rows=20, cols=4)
    leverage = n4m.HighLeverageFilter().fit(x)
    quality = n4m.SpectralQualityFilter()
    composite = n4m.CompositeFilter(filters=[leverage, quality]).fit(x)

    for filt in [leverage, quality, composite]:
        mask, stats = filt.apply(x)
        assert mask.shape == (x.shape[0],)
        assert stats.n_samples == x.shape[0]


def test_advanced_transfer_and_scatter_wrappers_smoke():
    import n4m as n4m

    x, _ = _spectra(rows=16, cols=12)
    target = 1.05 * x + 0.1

    for op in [
        n4m.DirectStandardization(),
        n4m.RobustDirectStandardization(max_iter=2),
        n4m.PiecewiseDirectStandardization(window_size=5),
        n4m.ScoreAugmentedProjectionStandardization(n_components=3),
        n4m.LocalCentering(),
    ]:
        out = op.fit(x, target).transform(x)
        assert out.shape == x.shape
        assert np.isfinite(out).all()

    y = np.linspace(0.0, 1.0, x.shape[0])
    corrected = n4m.SlopeBiasCorrection().fit(y, 2.0 * y + 1.0).transform(y)
    assert np.allclose(corrected, 2.0 * y + 1.0)

    for op in [
        n4m.WeightedSNV(),
        n4m.VariableSortingNormalization(),
        n4m.PiecewiseSNV(window_size=4),
        n4m.PiecewiseMSC(window_size=4),
        n4m.LocalizedMSC(window_size=5),
    ]:
        out = op.fit_transform(x)
        assert out.shape == x.shape
        assert np.isfinite(out).all()


def test_advanced_alignment_and_selection_wrappers_smoke():
    import n4m as n4m

    x, _ = _spectra(rows=10, cols=10)
    y = np.linspace(0.0, 1.0, x.shape[0])

    for op in [
        n4m.CrossCorrelationAlignment(max_shift=2),
        n4m.IcoshiftAlignment(interval_size=5, max_shift=1),
        n4m.DynamicTimeWarpingAlignment(),
        n4m.CorrelationOptimizedWarping(interval_size=5, max_shift=1),
    ]:
        out = op.fit_transform(x)
        assert out.shape == x.shape
        assert np.isfinite(out).all()

    for selector in [
        n4m.VarianceFilter(top_k=3),
        n4m.CorrelationFilter(top_k=3),
    ]:
        out = selector.fit_transform(x, y)
        assert out.shape == (x.shape[0], 3)

    intervals = n4m.IntervalGenerator(interval_size=4, step=3).fit_transform(x)
    assert intervals
    assert all(block.shape[0] == x.shape[0] for block in intervals)


# Names that must be importable from the top-level ``nirs4all-methods`` module
# for every one of the 32 cross-binding method IDs.
_CROSS_BINDING_SKLEARN_NAMES: dict[str, str] = {
    "log_transform": "LogTransform",
    "derivate": "Derivate",
    "wavelet_pca": "WaveletPCA",
    "wavelet_svd": "WaveletSVD",
    "aug_gaussian_noise": "GaussianAdditiveNoise",
    "aug_multiplicative_noise": "MultiplicativeNoise",
    "aug_spike_noise": "SpikeNoise",
    "aug_hetero_noise": "HeteroscedasticNoiseAugmenter",
    "aug_linear_drift": "LinearBaselineDrift",
    "aug_poly_drift": "PolynomialBaselineDrift",
    "aug_path_length": "PathLengthAugmenter",
    "aug_wavelength_spectral": "aug_wavelength_spectral",
    "aug_mixup": "MixupAugmenter",
    "aug_local_mixup": "LocalMixupAugmenter",
    "aug_scatter_sim": "ScatterSimulationMSC",
    "aug_particle_size": "ParticleSizeAugmenter",
    "aug_emsc_distort": "EMSCDistortionAugmenter",
    "aug_batch_effect": "BatchEffectAugmenter",
    "aug_instrument_broaden": "InstrumentalBroadeningAugmenter",
    "aug_dead_band": "DeadBandAugmenter",
    "aug_temperature": "TemperatureAugmenter",
    "aug_moisture": "MoistureAugmenter",
    "high_leverage": "HighLeverageFilter",
    "spectral_quality": "SpectralQualityFilter",
    "composite_filter": "CompositeFilter",
    "hotelling_t2": "hotelling_t2",
    "q_residuals": "q_residuals",
    "nirs_metrics": "nirs_metrics",
    "signal_type_detector": "signal_type_detector",
    "transfer_metrics": "transfer_metrics",
    "rng_pcg64": "PCG64",
}


def test_cross_binding_sklearn_surface_exposes_all_methods():
    """Every method ID must resolve on the top-level ``nirs4all-methods`` package."""
    import n4m as n4m

    missing = [
        mid
        for mid, name in _CROSS_BINDING_SKLEARN_NAMES.items()
        if not hasattr(n4m, name)
    ]
    assert not missing, f"missing top-level sklearn-surface names: {missing}"
    assert len(_CROSS_BINDING_SKLEARN_NAMES) == 31


def test_cross_binding_wavelength_bundle_helper_matches_python_surface():
    """The sklearn-side wavelength bundle helper reproduces the ABI-close output."""
    import n4m as n4m
    from n4m import python as n4m_python

    x, wl = _spectra(rows=6, cols=16)
    out_py = n4m_python.aug_wavelength_spectral(x, wavelengths=wl, seed=5)
    out_skl = n4m.aug_wavelength_spectral(x, wavelengths=wl, seed=5)
    assert out_py.shape == (x.shape[0], 10 * x.shape[1])
    assert np.allclose(out_py, out_skl)


def test_cross_binding_metric_helpers_smoke():
    """``nirs_metrics`` and ``signal_type_detector`` work from the public surface."""
    import n4m as n4m

    y = np.arange(8, dtype=np.float64)
    metrics = n4m.nirs_metrics(y, y + 0.1)
    assert set(metrics) == {"rmse", "mae", "bias", "sep", "rpd", "rpiq", "r2", "nrmse"}
    assert all(isinstance(v, float) for v in metrics.values())

    x, wl = _spectra(rows=20, cols=24)
    detection = n4m.signal_type_detector(x, wavelengths=wl)
    assert set(detection) == {"type", "confidence", "reason"}
    assert isinstance(detection["type"], int)
    assert 0.0 <= detection["confidence"] <= 1.0
    assert isinstance(detection["reason"], str)
