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
    import chemometrics4all as c4a
    from chemometrics4all import python as c4a_python
    from chemometrics4all import sklearn as c4a_sklearn

    x, _ = _spectra(rows=4, cols=8)
    direct = c4a_python.snv(x, ddof=1)
    estimator = c4a_sklearn.SNV(ddof=1)
    wrapped = estimator.fit_transform(x)

    assert np.allclose(direct, wrapped)
    assert c4a.python is c4a_python
    assert c4a.sklearn is c4a_sklearn

    try:
        from sklearn.base import BaseEstimator, TransformerMixin
    except Exception:
        pytest.skip("scikit-learn is not installed")

    assert isinstance(estimator, BaseEstimator)
    assert isinstance(estimator, TransformerMixin)


@pytest.mark.parametrize(
    "factory",
    [
        lambda wl: __import__("chemometrics4all").MultiplicativeNoise(seed=1),
        lambda wl: __import__("chemometrics4all").SpikeNoise(seed=1),
        lambda wl: __import__("chemometrics4all").HeteroscedasticNoiseAugmenter(seed=1),
        lambda wl: __import__("chemometrics4all").LinearBaselineDrift(seed=1),
        lambda wl: __import__("chemometrics4all").PolynomialBaselineDrift(seed=1),
        lambda wl: __import__("chemometrics4all").PathLengthAugmenter(seed=1),
        lambda wl: __import__("chemometrics4all").WavelengthShift(wavelengths=wl, seed=1),
        lambda wl: __import__("chemometrics4all").WavelengthStretch(wavelengths=wl, seed=1),
        lambda wl: __import__("chemometrics4all").LocalWarpAugmenter(wavelengths=wl, seed=1),
        lambda wl: __import__("chemometrics4all").BandPerturbationAugmenter(seed=1),
        lambda wl: __import__("chemometrics4all").BandMasking(seed=1),
        lambda wl: __import__("chemometrics4all").ChannelDropout(seed=1),
        lambda wl: __import__("chemometrics4all").GaussianJitter(seed=1),
        lambda wl: __import__("chemometrics4all").UnsharpMask(seed=1),
        lambda wl: __import__("chemometrics4all").MagnitudeWarp(wavelengths=wl, seed=1),
        lambda wl: __import__("chemometrics4all").LocalClip(seed=1),
        lambda wl: __import__("chemometrics4all").MixupAugmenter(seed=1),
        lambda wl: __import__("chemometrics4all").LocalMixupAugmenter(seed=1),
        lambda wl: __import__("chemometrics4all").ScatterSimulationMSC(seed=1),
        lambda wl: __import__("chemometrics4all").ParticleSizeAugmenter(wavelengths=wl, seed=1),
        lambda wl: __import__("chemometrics4all").EMSCDistortionAugmenter(wavelengths=wl, seed=1),
        lambda wl: __import__("chemometrics4all").BatchEffectAugmenter(wavelengths=wl, seed=1),
        lambda wl: __import__("chemometrics4all").InstrumentalBroadeningAugmenter(wavelengths=wl, seed=1),
        lambda wl: __import__("chemometrics4all").DeadBandAugmenter(seed=1),
        lambda wl: __import__("chemometrics4all").TemperatureAugmenter(wavelengths=wl, seed=1),
        lambda wl: __import__("chemometrics4all").MoistureAugmenter(wavelengths=wl, seed=1),
        lambda wl: __import__("chemometrics4all").DetectorRollOffAugmenter(wavelengths=wl, seed=1),
        lambda wl: __import__("chemometrics4all").StrayLightAugmenter(wavelengths=wl, seed=1),
        lambda wl: __import__("chemometrics4all").EdgeCurvatureAugmenter(wavelengths=wl, seed=1),
        lambda wl: __import__("chemometrics4all").TruncatedPeakAugmenter(wavelengths=wl, seed=1),
        lambda wl: __import__("chemometrics4all").EdgeArtifactsAugmenter(wavelengths=wl, seed=1),
        lambda wl: __import__("chemometrics4all").SplineSmoothingAugmenter(seed=1),
        lambda wl: __import__("chemometrics4all").SplineXPerturbationAugmenter(seed=1),
        lambda wl: __import__("chemometrics4all").SplineYPerturbationAugmenter(seed=1),
        lambda wl: __import__("chemometrics4all").RotateTranslateAugmenter(seed=1),
        lambda wl: __import__("chemometrics4all").RandomXOperation(seed=1),
    ],
)
def test_augmenter_wrappers_shape(factory):
    x, wavelengths = _spectra()
    out = factory(wavelengths).fit_transform(x)
    assert out.shape == x.shape
    assert np.isfinite(out).all()


def test_deferred_spline_simplification_wrappers_raise_not_implemented():
    import chemometrics4all as c4a

    x, _ = _spectra()
    with pytest.raises(c4a.Chemometrics4allError):
        c4a.SplineXSimplificationAugmenter(seed=1).fit_transform(x)
    with pytest.raises(c4a.Chemometrics4allError):
        c4a.SplineCurveSimplificationAugmenter(seed=1).fit_transform(x)


def test_splitter_wrappers_smoke():
    import chemometrics4all as c4a

    x, _ = _spectra(rows=24, cols=6)
    y = np.arange(x.shape[0], dtype=np.float64).reshape(-1, 1)
    groups = np.repeat(np.arange(12, dtype=np.int64), 2)

    for train, test in [
        c4a.KMeansSplitter(seed=1).split(x),
        c4a.SystematicCircularSplitter(seed=1).split(y),
        c4a.SPlitSplitter(seed=1).split(x),
    ]:
        assert train.ndim == test.ndim == 1
        assert train.size + test.size == x.shape[0]

    train, test = c4a.SPXYFoldSplitter(n_splits=3).split_fold(x, y, 0)
    assert train.size + test.size == x.shape[0]

    train, test = c4a.SPXYGroupFoldSplitter(n_splits=3).split_fold(x, y, groups, 0)
    assert train.size + test.size == x.shape[0]

    train, test = c4a.BinnedStratifiedGroupKFoldSplitter(
        n_splits=3, n_bins=2, seed=1
    ).split_fold(y, groups, 0)
    assert train.size + test.size == x.shape[0]


def test_filter_meta_wrappers_smoke():
    import chemometrics4all as c4a

    x, _ = _spectra(rows=20, cols=4)
    leverage = c4a.HighLeverageFilter().fit(x)
    quality = c4a.SpectralQualityFilter()
    composite = c4a.CompositeFilter(filters=[leverage, quality]).fit(x)

    for filt in [leverage, quality, composite]:
        mask, stats = filt.apply(x)
        assert mask.shape == (x.shape[0],)
        assert stats.n_samples == x.shape[0]


def test_advanced_transfer_and_scatter_wrappers_smoke():
    import chemometrics4all as c4a

    x, _ = _spectra(rows=16, cols=12)
    target = 1.05 * x + 0.1

    for op in [
        c4a.DirectStandardization(),
        c4a.RobustDirectStandardization(max_iter=2),
        c4a.PiecewiseDirectStandardization(window_size=5),
        c4a.ScoreAugmentedProjectionStandardization(n_components=3),
        c4a.LocalCentering(),
    ]:
        out = op.fit(x, target).transform(x)
        assert out.shape == x.shape
        assert np.isfinite(out).all()

    y = np.linspace(0.0, 1.0, x.shape[0])
    corrected = c4a.SlopeBiasCorrection().fit(y, 2.0 * y + 1.0).transform(y)
    assert np.allclose(corrected, 2.0 * y + 1.0)

    for op in [
        c4a.WeightedSNV(),
        c4a.VariableSortingNormalization(),
        c4a.PiecewiseSNV(window_size=4),
        c4a.PiecewiseMSC(window_size=4),
        c4a.LocalizedMSC(window_size=5),
    ]:
        out = op.fit_transform(x)
        assert out.shape == x.shape
        assert np.isfinite(out).all()


def test_advanced_alignment_and_selection_wrappers_smoke():
    import chemometrics4all as c4a

    x, _ = _spectra(rows=10, cols=10)
    y = np.linspace(0.0, 1.0, x.shape[0])

    for op in [
        c4a.CrossCorrelationAlignment(max_shift=2),
        c4a.IcoshiftAlignment(interval_size=5, max_shift=1),
        c4a.DynamicTimeWarpingAlignment(),
        c4a.CorrelationOptimizedWarping(interval_size=5, max_shift=1),
    ]:
        out = op.fit_transform(x)
        assert out.shape == x.shape
        assert np.isfinite(out).all()

    for selector in [
        c4a.VarianceFilter(top_k=3),
        c4a.CorrelationFilter(top_k=3),
    ]:
        out = selector.fit_transform(x, y)
        assert out.shape == (x.shape[0], 3)

    intervals = c4a.IntervalGenerator(interval_size=4, step=3).fit_transform(x)
    assert intervals
    assert all(block.shape[0] == x.shape[0] for block in intervals)


# Names that must be importable from the top-level ``chemometrics4all`` module
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
    """Every method ID must resolve on the top-level ``chemometrics4all`` package."""
    import chemometrics4all as c4a

    missing = [
        mid
        for mid, name in _CROSS_BINDING_SKLEARN_NAMES.items()
        if not hasattr(c4a, name)
    ]
    assert not missing, f"missing top-level sklearn-surface names: {missing}"
    assert len(_CROSS_BINDING_SKLEARN_NAMES) == 31


def test_cross_binding_wavelength_bundle_helper_matches_python_surface():
    """The sklearn-side wavelength bundle helper reproduces the ABI-close output."""
    import chemometrics4all as c4a
    from chemometrics4all import python as c4a_python

    x, wl = _spectra(rows=6, cols=16)
    out_py = c4a_python.aug_wavelength_spectral(x, wavelengths=wl, seed=5)
    out_skl = c4a.aug_wavelength_spectral(x, wavelengths=wl, seed=5)
    assert out_py.shape == (x.shape[0], 10 * x.shape[1])
    assert np.allclose(out_py, out_skl)


def test_cross_binding_metric_helpers_smoke():
    """``nirs_metrics`` and ``signal_type_detector`` work from the public surface."""
    import chemometrics4all as c4a

    y = np.arange(8, dtype=np.float64)
    metrics = c4a.nirs_metrics(y, y + 0.1)
    assert set(metrics) == {"rmse", "mae", "bias", "sep", "rpd", "rpiq", "r2", "nrmse"}
    assert all(isinstance(v, float) for v in metrics.values())

    x, wl = _spectra(rows=20, cols=24)
    detection = c4a.signal_type_detector(x, wavelengths=wl)
    assert set(detection) == {"type", "confidence", "reason"}
    assert isinstance(detection["type"], int)
    assert 0.0 <= detection["confidence"] <= 1.0
    assert isinstance(detection["reason"], str)
