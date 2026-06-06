# SPDX-License-Identifier: CECILL-2.1
"""Python binding for libn4m.

The package exposes two public operator surfaces:

* ``n4m.python``: ABI-close NumPy functions over ``libn4m``;
* ``n4m.sklearn``: scikit-learn-compatible estimators.

Top-level imports still re-export the sklearn estimator classes for backwards
compatibility::

    from n4m import SNV, SavitzkyGolay
"""
import ctypes

from . import python, sklearn
from ._errors import N4MError
from ._ffi import (
    ABI_VERSION_MAJOR,
    ABI_VERSION_MINOR,
    ABI_VERSION_PATCH,
    ABI_VERSION_STRING,
    lib,
    library_path,
)
from ._rng import PCG64
from .metrics import (
    bias,
    hotelling_t2,
    mae,
    nrmse,
    q_residuals,
    r2,
    rmse,
    rpd,
    rpiq,
    sep,
    transfer_metrics,
)
from .sklearn import (
    aug_wavelength_spectral,
    nirs_metrics,
    signal_type_detector,
)
from .sklearn import (
    AOMCandidateSpec,
    AOMControlSelector,
    AOMDatasetMetadata,
    AOMEndpointMarginStabilityGate,
    AOMFallbackBlendGate,
    AOMMidPEndpointStack,
    AOMOperatorPLSStack,
    AOMOperatorPLSSpec,
    AOMPreprocessingChain,
    AOMRidgeBlender,
    AOMRobustHPOCompact,
    AOMRobustHPORegressor,
    AOMRobustHPOWide,
    AOMStructuralPolicy,
    AOMStructuralPolicyWithP700BlockLocalAdmission,
    AOMStructuralPolicyWithP700ProtocolUnified,
    AOMStructuralPolicyWithPgt1200Admissions,
    AOMTrueBankEndpointPortfolio,
    EndpointStabilityDecision,
    NativeAOMChainRidgePLSRegressor,
    NativeAOMChainSweepRegressor,
    NativeAOMFixedCandidateRegressor,
    NativeAOMMomentScreenRefitRegressor,
    NativeAOMMomentPLSScreenRefitRegressor,
    NativeAOMMomentRidgeScreenRefitRegressor,
    NativeAOMOperatorPLSStackRegressor,
    NativeAOMPLSRegressor,
    NativeAOMPLSSuperblockRegressor,
    NativeAOMRidgePLSSuperblockRegressor,
    NativeAOMRidgeBlenderRegressor,
    NativeAOMRidgeActiveSuperblockRegressor,
    NativeAOMRidgeGlobalRegressor,
    NativeAOMRidgeMKLSuperblockRegressor,
    NativeAOMRidgeSuperblockRegressor,
    NativeAOMRobustHPORegressor,
    NativeAOMSavgolFocusRegressor,
    NativeAOMScreenRefitRegressor,
    NativeAOMStrictFamilyLiteRegressor,
    NativeAOMStagedChainCampaignRegressor,
    NativeAOMSweepRegressor,
    NativeContinuumRegressionRegressor,
    NativeCPPLSRegressor,
    NativeECRRegressor,
    NativeMomentStackRegressor,
    NativeMomentSweepRegressor,
    NativePCRRegressor,
    NativePLSRegressor,
    NativePOPPLSRegressor,
    NativeRidgeRegressor,
    NativeRidgePLSRegressor,
    NativeRobustPLSRegressor,
    NativeWeightedPLSRegressor,
    build_aom_control_chain_bank,
    EMSC,
    LSNV,
    MSC,
    RNV,
    SNV,
    AirPLS,
    ArPLS,
    AreaNormalization,
    AsLS,
    BEADS,
    BaselineCenter,
    Crop,
    CropTransformer,
    Derivate,
    Detrend,
    EPO,
    FCKStaticTransformer,
    FirstDerivative,
    FlexiblePCA,
    FlexibleSVD,
    FractionToPercent,
    FromAbsorbance,
    Gaussian,
    GaussianAdditiveNoise,
    Haar,
    IAsLS,
    IModPoly,
    IntegerKBinsDiscretizer,
    KBinsDiscretizer,
    KBinsStratifiedSplitter,
    KennardStoneSplitter,
    KubelkaMunk,
    LogTransform,
    ModPoly,
    Normalize,
    NorrisWilliams,
    OSC,
    PercentToFraction,
    RangeDiscretizer,
    RollingBall,
    Resample,
    ResampleTransformer,
    Resampler,
    SNIP,
    SavitzkyGolay,
    SecondDerivative,
    SPXYSplitter,
    SimpleScale,
    ToAbsorbance,
    Wavelet,
    WaveletDenoise,
    WaveletFeatures,
    WaveletPCA,
    WaveletSVD,
    XOutlierFilter,
    YOutlierFilter,
)
from .sklearn import *  # noqa: F403 - re-export full sklearn-style surface
from .sklearn import __all__ as _sklearn_all
from . import aom
from . import moment


def version() -> str:
    """Return the runtime libn4m version string, e.g. ``0.1.0+abi.1.9.0``."""
    raw = ctypes.c_char_p(lib.n4m_get_version_string()).value
    if raw is None:  # pragma: no cover - defensive ABI guard
        raise RuntimeError("libn4m returned a null version string")
    return raw.decode("utf-8")


def abi_version() -> tuple[int, int, int]:
    """Return the loaded libn4m ABI version as ``(major, minor, patch)``."""
    return (
        int(lib.n4m_get_abi_version_major()),
        int(lib.n4m_get_abi_version_minor()),
        int(lib.n4m_get_abi_version_patch()),
    )


aom_robust_hpo = python.aom_robust_hpo
aom_global_select = python.aom_global_select
aom_per_component_select = python.aom_per_component_select
aom_pls = python.aom_pls
pop_pls = python.pop_pls
aom_preprocess = python.aom_preprocess
aom_chain_ridge_pls = python.aom_chain_ridge_pls
aom_sweep_run = python.aom_sweep_run
aom_chain_sweep_run = python.aom_chain_sweep_run
aom_chain_fixed_fit_run = python.aom_chain_fixed_fit_run
aom_candidate_table = python.aom_candidate_table
aom_candidate_operator_summary = python.aom_candidate_operator_summary
aom_candidate_preprocessing_impact = python.aom_candidate_preprocessing_impact
aom_candidate_route_summary = python.aom_candidate_route_summary
aom_candidate_rank_diagnostics = python.aom_candidate_rank_diagnostics
aom_candidate_report_records = python.aom_candidate_report_records
aom_chain_score_campaign = python.aom_chain_score_campaign
aom_chain_screen_refit_campaign = python.aom_chain_screen_refit_campaign
aom_moment_screen_refit_campaign = python.aom_moment_screen_refit_campaign
aom_staged_chain_campaign = python.aom_staged_chain_campaign
aom_screen_refit_candidate_pool = python.aom_screen_refit_candidate_pool
aom_refit_execution_plan = python.aom_refit_execution_plan
aom_refit_candidates = python.aom_refit_candidates
aom_evaluate_candidates = python.aom_evaluate_candidates
aom_load_candidate_report = python.aom_load_candidate_report
aom_save_candidate_report = python.aom_save_candidate_report
build_aom_strict_chain_grid = python.build_aom_strict_chain_grid
iter_aom_strict_chain_grid = python.iter_aom_strict_chain_grid
decode_aom_chains = python.decode_aom_chains
aom_ridge_blender = python.aom_ridge_blender
aom_ridge_active_superblock = python.aom_ridge_active_superblock
aom_ridge_global = python.aom_ridge_global
aom_ridge_mkl_superblock = python.aom_ridge_mkl_superblock
aom_ridge_superblock = python.aom_ridge_superblock
aom_ridge_pls_superblock = python.aom_ridge_pls_superblock
aom_pls_superblock = python.aom_pls_superblock
aom_operator_pls_stack = python.aom_operator_pls_stack
moments = python.moments
moments_train_from_heldout = python.moments_train_from_heldout
moment_screen_backend_recommendation = python.moment_screen_backend_recommendation
sweep_run = python.sweep_run
pls_cross_validate = python.pls_cross_validate
ridge = python.ridge
pls = python.pls
pcr = python.pcr
cppls = python.cppls
weighted_pls = python.weighted_pls
robust_pls = python.robust_pls
ridge_pls = python.ridge_pls
continuum_regression = python.continuum_regression
ecr = python.ecr
moment_stack = python.moment_stack


__all__ = [
    "ABI_VERSION_MAJOR",
    "ABI_VERSION_MINOR",
    "ABI_VERSION_PATCH",
    "ABI_VERSION_STRING",
    "aom",
    "moment",
    "AOMCandidateSpec",
    "AOMControlSelector",
    "AOMDatasetMetadata",
    "AOMEndpointMarginStabilityGate",
    "AOMFallbackBlendGate",
    "AOMMidPEndpointStack",
    "AOMOperatorPLSStack",
    "AOMOperatorPLSSpec",
    "AOMPreprocessingChain",
    "AOMRidgeBlender",
    "AOMRobustHPOCompact",
    "AOMRobustHPORegressor",
    "AOMRobustHPOWide",
    "AOMStructuralPolicy",
    "AOMStructuralPolicyWithP700BlockLocalAdmission",
    "AOMStructuralPolicyWithP700ProtocolUnified",
    "AOMStructuralPolicyWithPgt1200Admissions",
    "AOMTrueBankEndpointPortfolio",
    "build_aom_control_chain_bank",
    "EndpointStabilityDecision",
    "NativeAOMChainSweepRegressor",
    "NativeAOMChainRidgePLSRegressor",
    "NativeAOMFixedCandidateRegressor",
    "NativeAOMMomentScreenRefitRegressor",
    "NativeAOMMomentPLSScreenRefitRegressor",
    "NativeAOMMomentRidgeScreenRefitRegressor",
    "NativeAOMOperatorPLSStackRegressor",
    "NativeAOMPLSRegressor",
    "NativeAOMPLSSuperblockRegressor",
    "NativeAOMRidgePLSSuperblockRegressor",
    "NativeAOMRidgeBlenderRegressor",
    "NativeAOMRidgeActiveSuperblockRegressor",
    "NativeAOMRidgeGlobalRegressor",
    "NativeAOMRidgeMKLSuperblockRegressor",
    "NativeAOMRidgeSuperblockRegressor",
    "NativeAOMRobustHPORegressor",
    "NativeAOMScreenRefitRegressor",
    "NativeAOMStagedChainCampaignRegressor",
    "NativeAOMSweepRegressor",
    "NativeContinuumRegressionRegressor",
    "NativeCPPLSRegressor",
    "NativeECRRegressor",
    "NativeMomentStackRegressor",
    "NativeMomentSweepRegressor",
    "NativePCRRegressor",
    "NativePLSRegressor",
    "NativePOPPLSRegressor",
    "NativeRidgeRegressor",
    "NativeRidgePLSRegressor",
    "NativeRobustPLSRegressor",
    "NativeWeightedPLSRegressor",
    "AirPLS",
    "ArPLS",
    "AreaNormalization",
    "AsLS",
    "BEADS",
    "BaselineCenter",
    "N4MError",
    "Crop",
    "CropTransformer",
    "Derivate",
    "Detrend",
    "EPO",
    "EMSC",
    "FCKStaticTransformer",
    "FirstDerivative",
    "FlexiblePCA",
    "FlexibleSVD",
    "FractionToPercent",
    "FromAbsorbance",
    "Gaussian",
    "GaussianAdditiveNoise",
    "Haar",
    "IAsLS",
    "IModPoly",
    "IntegerKBinsDiscretizer",
    "KBinsDiscretizer",
    "KBinsStratifiedSplitter",
    "KennardStoneSplitter",
    "KubelkaMunk",
    "LSNV",
    "LogTransform",
    "MSC",
    "ModPoly",
    "Normalize",
    "NorrisWilliams",
    "OSC",
    "PCG64",
    "PercentToFraction",
    "python",
    "RangeDiscretizer",
    "RNV",
    "Resample",
    "ResampleTransformer",
    "Resampler",
    "SNV",
    "SPXYSplitter",
    "RollingBall",
    "SNIP",
    "SavitzkyGolay",
    "SecondDerivative",
    "SimpleScale",
    "sklearn",
    "ToAbsorbance",
    "Wavelet",
    "WaveletDenoise",
    "WaveletFeatures",
    "WaveletPCA",
    "WaveletSVD",
    "XOutlierFilter",
    "YOutlierFilter",
    "__version__",
    "abi_version",
    "aom_chain_screen_refit_campaign",
    "aom_chain_ridge_pls",
    "aom_chain_fixed_fit_run",
    "aom_chain_sweep_run",
    "aom_chain_score_campaign",
    "aom_moment_screen_refit_campaign",
    "aom_staged_chain_campaign",
    "aom_screen_refit_candidate_pool",
    "aom_candidate_operator_summary",
    "aom_candidate_preprocessing_impact",
    "aom_candidate_route_summary",
    "aom_candidate_rank_diagnostics",
    "aom_candidate_table",
    "aom_candidate_report_records",
    "aom_evaluate_candidates",
    "aom_global_select",
    "aom_load_candidate_report",
    "aom_operator_pls_stack",
    "aom_per_component_select",
    "aom_preprocess",
    "aom_pls",
    "aom_pls_superblock",
    "aom_refit_execution_plan",
    "aom_ridge_blender",
    "aom_ridge_active_superblock",
    "aom_ridge_global",
    "aom_ridge_mkl_superblock",
    "aom_ridge_pls_superblock",
    "aom_ridge_superblock",
    "aom_robust_hpo",
    "aom_refit_candidates",
    "aom_save_candidate_report",
    "aom_sweep_run",
    "aug_wavelength_spectral",
    "bias",
    "build_aom_strict_chain_grid",
    "continuum_regression",
    "cppls",
    "decode_aom_chains",
    "ecr",
    "hotelling_t2",
    "iter_aom_strict_chain_grid",
    "library_path",
    "mae",
    "moments",
    "moments_train_from_heldout",
    "moment_stack",
    "moment_screen_backend_recommendation",
    "nirs_metrics",
    "nrmse",
    "q_residuals",
    "r2",
    "pop_pls",
    "pcr",
    "pls_cross_validate",
    "pls",
    "ridge",
    "ridge_pls",
    "rmse",
    "robust_pls",
    "rpd",
    "rpiq",
    "sep",
    "signal_type_detector",
    "sweep_run",
    "transfer_metrics",
    "weighted_pls",
    "version",
]

__all__ = sorted(set(__all__) | set(_sklearn_all))

# Derive __version__ from the loaded native library so it stays in sync with the
# C ABI version_string (shape "X.Y.Z+abi.A.B.C") — strip the "+abi.*" suffix.
# Mirrors the pls4all subset; avoids a hardcoded version drifting from the lib.
__version__ = version().split("+", 1)[0]
