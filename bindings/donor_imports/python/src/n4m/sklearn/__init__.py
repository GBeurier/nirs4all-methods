# SPDX-License-Identifier: CECILL-2.1
"""Scikit-learn-style wrappers around libn4m operators."""
from .advanced import (
    CorrelationFilter,
    CorrelationOptimizedWarping,
    CrossCorrelationAlignment,
    DirectStandardization,
    DynamicTimeWarpingAlignment,
    IcoshiftAlignment,
    IntervalGenerator,
    LocalCentering,
    LocalizedMSC,
    PiecewiseDirectStandardization,
    PiecewiseMSC,
    PiecewiseSNV,
    RobustDirectStandardization,
    ScoreAugmentedProjectionStandardization,
    SlopeBiasCorrection,
    VariableSortingNormalization,
    VarianceFilter,
    WeightedSNV,
)
from .augmentation import (
    BandMasking,
    BandMaskingAugmenter,
    BandPerturbationAugmenter,
    BatchEffectAugmenter,
    ChannelDropout,
    DeadBandAugmenter,
    DetectorRollOffAugmenter,
    EMSCDistortionAugmenter,
    EdgeArtifactsAugmenter,
    EdgeCurvatureAugmenter,
    GaussianAdditiveNoise,
    GaussianJitter,
    GaussianSmoothingJitter,
    Haar,
    HeteroscedasticNoiseAugmenter,
    InstrumentalBroadeningAugmenter,
    LinearBaselineDrift,
    LocalClip,
    LocalMixupAugmenter,
    LocalWarpAugmenter,
    LocalWavelengthWarp,
    MagnitudeWarp,
    MixupAugmenter,
    MoistureAugmenter,
    MultiplicativeNoise,
    ParticleSizeAugmenter,
    PathLengthAugmenter,
    PolynomialBaselineDrift,
    RandomXOperation,
    RotateTranslateAugmenter,
    ScatterSimulationMSC,
    SmoothMagnitudeWarp,
    SplineCurveSimplificationAugmenter,
    SplineSmoothingAugmenter,
    SplineXPerturbationAugmenter,
    SplineXSimplificationAugmenter,
    SplineYPerturbationAugmenter,
    SpikeNoise,
    StrayLightAugmenter,
    TemperatureAugmenter,
    TruncatedPeakAugmenter,
    UnsharpMask,
    UnsharpSpectralMask,
    Wavelet,
    WaveletDenoise,
    WaveletFeatures,
    WaveletPCA,
    WaveletSVD,
    WavelengthShift,
    WavelengthShiftAugmenter,
    WavelengthStretch,
    WavelengthStretchAugmenter,
)
from .baseline import (
    BEADS,
    AirPLS,
    ArPLS,
    AsLS,
    Detrend,
    IAsLS,
    IModPoly,
    ModPoly,
    RollingBall,
    SNIP,
)
from .filters import (
    CompositeFilter,
    HighLeverageFilter,
    SpectralQualityFilter,
    XOutlierFilter,
    YOutlierFilter,
)
from .feature_extraction import (
    EPO,
    FCKStaticTransformer,
    FlexiblePCA,
    FlexibleSVD,
    OSC,
)
from .preprocessing import (
    AreaNormalization,
    BaselineCenter,
    Derivate,
    EMSC,
    FractionToPercent,
    FromAbsorbance,
    Gaussian,
    LSNV,
    LogTransform,
    MSC,
    Normalize,
    NorrisWilliams,
    PercentToFraction,
    RNV,
    SNV,
    FirstDerivative,
    KubelkaMunk,
    SavitzkyGolay,
    SecondDerivative,
    SimpleScale,
    ToAbsorbance,
)
from .resampling import (
    Crop,
    CropTransformer,
    IntegerKBinsDiscretizer,
    KBinsDiscretizer,
    RangeDiscretizer,
    Resample,
    ResampleTransformer,
    Resampler,
)
from .splitters import (
    BinnedStratifiedGroupKFoldSplitter,
    KBinsStratifiedSplitter,
    KennardStoneSplitter,
    KMeansSplitter,
    SPXYFoldSplitter,
    SPXYGroupFoldSplitter,
    SPXYSplitter,
    SPlitSplitter,
    SplitSplitter,
    SystematicCircularSplitter,
)


def aug_wavelength_spectral(X, wavelengths=None, seed: int = 0):
    """Concatenate the 10 wavelength/spectral augmenters into one ``(n, 10*p)``.

    Sklearn-side helper mirroring ``n4m.python.aug_wavelength_spectral``;
    intended as a single-call bundle for scripts using the sklearn surface.
    """
    from .. import python as _py
    return _py.aug_wavelength_spectral(X, wavelengths=wavelengths, seed=seed)


_NIRS_METRIC_NAMES = ("rmse", "mae", "bias", "sep", "rpd", "rpiq", "r2", "nrmse")


def nirs_metrics(y_true, y_pred):
    """Return the eight standard NIRS regression diagnostics as a dict.

    Keys: ``rmse``, ``mae``, ``bias``, ``sep``, ``rpd``, ``rpiq``, ``r2``,
    ``nrmse``. Mirrors ``n4m.python.nirs_metrics`` (which returns
    a NumPy array in the same order) with name-keyed output for ergonomics.
    """
    from .. import metrics as _m
    return {name: getattr(_m, name)(y_true, y_pred) for name in _NIRS_METRIC_NAMES}


def signal_type_detector(X, wavelengths=None, confidence_threshold: float = 0.7):
    """Detect the signal type of ``X`` and return name-keyed diagnostics.

    Returns ``{"type": int, "confidence": float, "reason": str}``. The integer
    matches ``n4m_signal_type_t`` exposed by libn4m.
    """
    import ctypes
    import numpy as np

    from .._errors import check
    from .._ffi import lib
    from .._matrix import as_f64_2d, numpy_to_view

    X_arr = as_f64_2d(X)
    if wavelengths is None:
        wl_ptr = ctypes.POINTER(ctypes.c_double)()
        wl_n = ctypes.c_int64(0)
    else:
        wl_arr = np.ascontiguousarray(
            np.asarray(wavelengths, dtype=np.float64).reshape(-1)
        )
        if wl_arr.size != X_arr.shape[1]:
            raise ValueError("wavelengths length must match X.shape[1]")
        wl_ptr = wl_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        wl_n = ctypes.c_int64(wl_arr.size)
    type_out = ctypes.c_int()
    confidence = ctypes.c_double()
    reason = ctypes.create_string_buffer(256)
    check(
        lib.n4m_signal_detect(
            numpy_to_view(X_arr),
            wl_ptr,
            wl_n,
            ctypes.c_double(float(confidence_threshold)),
            ctypes.byref(type_out),
            ctypes.byref(confidence),
            ctypes.cast(reason, ctypes.c_void_p),
        ),
        "n4m_signal_detect",
    )
    return {
        "type": int(type_out.value),
        "confidence": float(confidence.value),
        "reason": reason.value.decode("utf-8", errors="replace"),
    }


__all__ = [
    "aug_wavelength_spectral",
    "nirs_metrics",
    "signal_type_detector",
    "AirPLS",
    "ArPLS",
    "AreaNormalization",
    "AsLS",
    "BEADS",
    "BandMasking",
    "BandMaskingAugmenter",
    "BandPerturbationAugmenter",
    "BatchEffectAugmenter",
    "BaselineCenter",
    "BinnedStratifiedGroupKFoldSplitter",
    "ChannelDropout",
    "CompositeFilter",
    "CorrelationFilter",
    "CorrelationOptimizedWarping",
    "Crop",
    "CropTransformer",
    "CrossCorrelationAlignment",
    "DeadBandAugmenter",
    "Derivate",
    "DetectorRollOffAugmenter",
    "Detrend",
    "DirectStandardization",
    "DynamicTimeWarpingAlignment",
    "EPO",
    "EMSC",
    "EMSCDistortionAugmenter",
    "EdgeArtifactsAugmenter",
    "EdgeCurvatureAugmenter",
    "FCKStaticTransformer",
    "FirstDerivative",
    "FlexiblePCA",
    "FlexibleSVD",
    "FractionToPercent",
    "FromAbsorbance",
    "Gaussian",
    "GaussianAdditiveNoise",
    "GaussianJitter",
    "GaussianSmoothingJitter",
    "Haar",
    "HeteroscedasticNoiseAugmenter",
    "HighLeverageFilter",
    "IAsLS",
    "IcoshiftAlignment",
    "IModPoly",
    "InstrumentalBroadeningAugmenter",
    "IntegerKBinsDiscretizer",
    "IntervalGenerator",
    "KBinsDiscretizer",
    "KBinsStratifiedSplitter",
    "KennardStoneSplitter",
    "KMeansSplitter",
    "KubelkaMunk",
    "LinearBaselineDrift",
    "LSNV",
    "LocalCentering",
    "LocalClip",
    "LocalMixupAugmenter",
    "LocalWarpAugmenter",
    "LocalWavelengthWarp",
    "LocalizedMSC",
    "LogTransform",
    "MSC",
    "MagnitudeWarp",
    "MixupAugmenter",
    "ModPoly",
    "MoistureAugmenter",
    "MultiplicativeNoise",
    "Normalize",
    "NorrisWilliams",
    "OSC",
    "ParticleSizeAugmenter",
    "PercentToFraction",
    "PathLengthAugmenter",
    "PiecewiseDirectStandardization",
    "PiecewiseMSC",
    "PiecewiseSNV",
    "PolynomialBaselineDrift",
    "RangeDiscretizer",
    "RandomXOperation",
    "RNV",
    "Resample",
    "ResampleTransformer",
    "Resampler",
    "RobustDirectStandardization",
    "SNV",
    "SPXYFoldSplitter",
    "SPXYGroupFoldSplitter",
    "SPXYSplitter",
    "SPlitSplitter",
    "RollingBall",
    "RotateTranslateAugmenter",
    "SNIP",
    "SavitzkyGolay",
    "ScatterSimulationMSC",
    "SecondDerivative",
    "ScoreAugmentedProjectionStandardization",
    "SimpleScale",
    "SlopeBiasCorrection",
    "SmoothMagnitudeWarp",
    "SpectralQualityFilter",
    "SpikeNoise",
    "SplitSplitter",
    "SplineCurveSimplificationAugmenter",
    "SplineSmoothingAugmenter",
    "SplineXPerturbationAugmenter",
    "SplineXSimplificationAugmenter",
    "SplineYPerturbationAugmenter",
    "StrayLightAugmenter",
    "SystematicCircularSplitter",
    "TemperatureAugmenter",
    "ToAbsorbance",
    "TruncatedPeakAugmenter",
    "UnsharpMask",
    "UnsharpSpectralMask",
    "Wavelet",
    "WaveletDenoise",
    "WaveletFeatures",
    "WaveletPCA",
    "WaveletSVD",
    "WavelengthShift",
    "WavelengthShiftAugmenter",
    "WavelengthStretch",
    "WavelengthStretchAugmenter",
    "VariableSortingNormalization",
    "VarianceFilter",
    "WeightedSNV",
    "XOutlierFilter",
    "YOutlierFilter",
]
