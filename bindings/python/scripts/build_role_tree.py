#!/usr/bin/env python3
"""Authoritative builder for the n4m public role-package tree (Phase 3a ii).

Writes the ``n4m.<role>`` packages per ``proposals/namespace/_target_ml_table.tsv``.
Operator / estimator implementation lives in ``n4m._impl`` (re-homed from the old
flat ``python.py`` + ``sklearn/*`` files). The role modules re-export the public
role names. The 10 ``c_surface: none`` AOM-superblock / moment_stack methods stay
Python-only in their namespace module.

Idempotent: re-run after an impl change.
"""
from __future__ import annotations

import os

# scripts/ -> python/ -> bindings/python
PY_BINDINGS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(PY_BINDINGS, "src", "n4m")
H = "# SPDX-License-Identifier: CECILL-2.1\n"

# leaf module path -> list of public class names re-exported verbatim from n4m._impl
CLASS_LEAVES: dict[str, list[str]] = {
    "transform/scatter": [
        "SNV", "MSC", "EMSC", "RNV", "LSNV", "AreaNormalization", "LocalCentering",
        "LocalizedMSC", "PiecewiseMSC", "PiecewiseSNV", "VariableSortingNormalization",
        "WeightedSNV",
    ],
    "transform/baseline": [
        "AirPLS", "ArPLS", "AsLS", "BEADS", "Detrend", "IAsLS", "IModPoly", "ModPoly",
        "RollingBall", "SNIP",
    ],
    "transform/smoothing": [
        "Derivate", "FirstDerivative", "Gaussian", "NorrisWilliams", "SavitzkyGolay",
        "SecondDerivative",
    ],
    "transform/wavelet": [
        "Haar", "Wavelet", "WaveletDenoise", "WaveletFeatures", "WaveletPCA", "WaveletSVD",
    ],
    "transform/signal_conversion": [
        "FractionToPercent", "FromAbsorbance", "KubelkaMunk", "PercentToFraction",
        "ToAbsorbance",
    ],
    "transform/resampling": [
        "CropTransformer", "IntegerKBinsDiscretizer", "RangeDiscretizer",
        "ResampleTransformer", "Resampler",
    ],
    "transform/scaling": ["BaselineCenter", "LogTransform", "Normalize", "SimpleScale"],
    "transform/alignment": [
        "CorrelationOptimizedWarping", "CrossCorrelationAlignment",
        "DynamicTimeWarpingAlignment", "IcoshiftAlignment",
    ],
    "transform/orthogonalization": ["OSC"],
    "transform/specialized": ["FCKStaticTransformer"],
    "augmentation/instrument": [
        "DetectorRollOffAugmenter", "EdgeArtifactsAugmenter", "EdgeCurvatureAugmenter",
        "MoistureAugmenter", "StrayLightAugmenter", "TemperatureAugmenter",
        "TruncatedPeakAugmenter",
    ],
    "augmentation/spectral": [
        "BandMasking", "BandPerturbationAugmenter", "ChannelDropout", "GaussianJitter",
        "LocalClip", "MagnitudeWarp", "UnsharpMask",
    ],
    "augmentation/scattering": [
        "BatchEffectAugmenter", "DeadBandAugmenter", "EMSCDistortionAugmenter",
        "InstrumentalBroadeningAugmenter", "ParticleSizeAugmenter", "ScatterSimulationMSC",
    ],
    "augmentation/splines": [
        "SplineCurveSimplificationAugmenter", "SplineSmoothingAugmenter",
        "SplineXPerturbationAugmenter", "SplineXSimplificationAugmenter",
        "SplineYPerturbationAugmenter",
    ],
    "augmentation/mixup": [
        "LocalMixupAugmenter", "MixupAugmenter", "RandomXOperation",
        "RotateTranslateAugmenter",
    ],
    "augmentation/noise": [
        "GaussianAdditiveNoise", "HeteroscedasticNoiseAugmenter", "MultiplicativeNoise",
        "SpikeNoise",
    ],
    "augmentation/drift": [
        "LinearBaselineDrift", "PathLengthAugmenter", "PolynomialBaselineDrift",
    ],
    "augmentation/wavelength": [
        "LocalWarpAugmenter", "WavelengthShift", "WavelengthStretch",
    ],
    "feature_selection/filter": ["CorrelationFilter", "VarianceFilter"],
    "feature_selection/interval": ["IntervalGenerator"],
    "model_selection/splitters": [
        "BinnedStratifiedGroupKFoldSplitter", "KBinsStratifiedSplitter",
        "KennardStoneSplitter", "KMeansSplitter", "SPlitSplitter", "SPXYFoldSplitter",
        "SPXYGroupFoldSplitter", "SPXYSplitter", "SystematicCircularSplitter",
    ],
    "domain_adaptation/standardization": [
        "DirectStandardization", "PiecewiseDirectStandardization",
        "RobustDirectStandardization", "ScoreAugmentedProjectionStandardization",
        "SlopeBiasCorrection",
    ],
    "domain_adaptation/orthogonalization": ["EPO"],
    "outlier_detection": [
        "CompositeFilter", "HighLeverageFilter", "SpectralQualityFilter",
        "XOutlierFilter", "YOutlierFilter",
    ],
    "decomposition": ["FlexiblePCA", "FlexibleSVD"],
}

# leaf module path -> {public_role_name: impl_name}  (estimator/regressor renames)
ESTIMATOR_LEAVES: dict[str, dict[str, str]] = {
    "estimators/regression/regularized": {
        "Ridge": "NativeRidgeRegressor",
        "RidgePLS": "NativeRidgePLSRegressor",
    },
    "estimators/regression/latent": {
        "PLS": "NativePLSRegressor",
        "PCR": "NativePCRRegressor",
        "CPPLS": "NativeCPPLSRegressor",
        "ContinuumRegression": "NativeContinuumRegressionRegressor",
        "ECR": "NativeECRRegressor",
    },
    "estimators/regression/robust": {
        "WeightedPLS": "NativeWeightedPLSRegressor",
        "RobustPLS": "NativeRobustPLSRegressor",
    },
}

# AOM / moment regressor renames per role module (Native prefix dropped).
AOM_REGRESSOR_LEAVES: dict[str, dict[str, str]] = {
    "model_selection/aom_search": {
        "AOMPLSRegressor": "NativeAOMPLSRegressor",
        "POPPLSRegressor": "NativePOPPLSRegressor",
        "AOMSweepRegressor": "NativeAOMSweepRegressor",
        "AOMChainSweepRegressor": "NativeAOMChainSweepRegressor",
        "AOMRidgeGlobalRegressor": "NativeAOMRidgeGlobalRegressor",
        "AOMFixedCandidateRegressor": "NativeAOMFixedCandidateRegressor",
        "AOMScreenRefitRegressor": "NativeAOMScreenRefitRegressor",
        "AOMRobustHPOSweepRegressor": "NativeAOMRobustHPORegressor",
        "AOMChainRidgePLSRegressor": "NativeAOMChainRidgePLSRegressor",
    },
    "model_selection/aom_campaign": {
        "AOMStagedChainCampaignRegressor": "NativeAOMStagedChainCampaignRegressor",
        "AOMSavgolFocusRegressor": "NativeAOMSavgolFocusRegressor",
        "AOMStrictFamilyLiteRegressor": "NativeAOMStrictFamilyLiteRegressor",
        "AOMMomentScreenRefitRegressor": "NativeAOMMomentScreenRefitRegressor",
        "AOMMomentPLSScreenRefitRegressor": "NativeAOMMomentPLSScreenRefitRegressor",
        "AOMMomentPLSExactScreenRefitRegressor": "NativeAOMMomentPLSExactScreenRefitRegressor",
        "AOMMomentRidgeScreenRefitRegressor": "NativeAOMMomentRidgeScreenRefitRegressor",
    },
    "compose/aom_superblock": {
        "AOMRidgeSuperblockRegressor": "NativeAOMRidgeSuperblockRegressor",
        "AOMRidgeMKLSuperblockRegressor": "NativeAOMRidgeMKLSuperblockRegressor",
        "AOMRidgeActiveSuperblockRegressor": "NativeAOMRidgeActiveSuperblockRegressor",
        "AOMPLSSuperblockRegressor": "NativeAOMPLSSuperblockRegressor",
        "AOMRidgePLSSuperblockRegressor": "NativeAOMRidgePLSSuperblockRegressor",
    },
    "ensemble": {
        "MomentStack": "NativeMomentStackRegressor",
        "MomentSweepRegressor": "NativeMomentSweepRegressor",
        "AOMRidgeBlenderRegressor": "NativeAOMRidgeBlenderRegressor",
        "AOMOperatorPLSStackRegressor": "NativeAOMOperatorPLSStackRegressor",
    },
}

# leaf module path -> {public_name: native_func_name}  (functional re-exports)
FUNC_LEAVES: dict[str, dict[str, str]] = {
    "model_selection/aom_search": {
        "aom_pls": "aom_pls",
        "pop_pls": "pop_pls",
        "aom_global_select": "aom_global_select",
        "aom_per_component_select": "aom_per_component_select",
        "aom_preprocess": "aom_preprocess",
        "aom_sweep_run": "aom_sweep_run",
        "aom_chain_sweep_run": "aom_chain_sweep_run",
        "aom_chain_fixed_fit_run": "aom_chain_fixed_fit_run",
        "aom_chain_ridge_pls": "aom_chain_ridge_pls",
        "aom_ridge_global": "aom_ridge_global",
        "aom_robust_hpo": "aom_robust_hpo",
        "build_aom_strict_chain_grid": "build_aom_strict_chain_grid",
        "iter_aom_strict_chain_grid": "iter_aom_strict_chain_grid",
        "decode_aom_chains": "decode_aom_chains",
        "aom_candidate_table": "aom_candidate_table",
        "aom_candidate_operator_summary": "aom_candidate_operator_summary",
        "aom_candidate_preprocessing_impact": "aom_candidate_preprocessing_impact",
        "aom_candidate_route_summary": "aom_candidate_route_summary",
        "aom_candidate_rank_diagnostics": "aom_candidate_rank_diagnostics",
        "aom_candidate_report_records": "aom_candidate_report_records",
        "aom_evaluate_candidates": "aom_evaluate_candidates",
        "aom_refit_candidates": "aom_refit_candidates",
        "aom_refit_execution_plan": "aom_refit_execution_plan",
        "aom_screen_refit_candidate_pool": "aom_screen_refit_candidate_pool",
        "aom_load_candidate_report": "aom_load_candidate_report",
        "aom_save_candidate_report": "aom_save_candidate_report",
    },
    "model_selection/aom_campaign": {
        "aom_chain_score_campaign": "aom_chain_score_campaign",
        "aom_chain_screen_refit_campaign": "aom_chain_screen_refit_campaign",
        "aom_moment_screen_refit_campaign": "aom_moment_screen_refit_campaign",
        "aom_staged_chain_campaign": "aom_staged_chain_campaign",
    },
    "model_selection/sweep": {
        "sweep_run": "sweep_run",
        "pls_cross_validate": "pls_cross_validate",
        "moment_screen_backend_recommendation": "moment_screen_backend_recommendation",
    },
    "compose/aom_superblock": {
        "aom_pls_superblock": "aom_pls_superblock",
        "aom_ridge_superblock": "aom_ridge_superblock",
        "aom_ridge_active_superblock": "aom_ridge_active_superblock",
        "aom_ridge_mkl_superblock": "aom_ridge_mkl_superblock",
        "aom_ridge_pls_superblock": "aom_ridge_pls_superblock",
    },
    "ensemble": {
        "moment_stack": "moment_stack",
        "aom_ridge_blender": "aom_ridge_blender",
        "aom_operator_pls_stack": "aom_operator_pls_stack",
    },
    "estimators/regression/regularized": {"ridge": "ridge", "ridge_pls": "ridge_pls"},
    "estimators/regression/latent": {
        "pls": "pls", "pcr": "pcr", "cppls": "cppls",
        "continuum_regression": "continuum_regression", "ecr": "ecr",
    },
    "estimators/regression/robust": {"weighted_pls": "weighted_pls", "robust_pls": "robust_pls"},
    "lowlevel/moments": {
        "moments": "moments",
        "moments_train_from_heldout": "moments_train_from_heldout",
    },
    "domain_adaptation/standardization": {
        "direct_standardization": "direct_standardization",
        "piecewise_direct_standardization": "piecewise_direct_standardization",
        "robust_direct_standardization": "robust_direct_standardization",
        "score_augmented_projection_standardization": "score_augmented_projection_standardization",
        "slope_bias_correction": "slope_bias_correction",
    },
    "domain_adaptation/orthogonalization": {"epo": "epo"},
    "domain_adaptation/metrics": {"transfer_metrics": "transfer_metrics"},
    "outlier_detection": {
        "hotelling_t2": "hotelling_t2",
        "q_residuals": "q_residuals",
        "x_outlier_mahalanobis": "x_outlier_mahalanobis",
        "y_outlier_iqr": "y_outlier_iqr",
    },
    "feature_selection/interval": {"interval_generator": "interval_generator"},
    "transform/signal_conversion": {"signal_type_detector": "signal_type_detector"},
    "metrics/scoring": {"nirs_metrics": "nirs_metrics"},
    "augmentation/wavelength": {"aug_wavelength_spectral": "aug_wavelength_spectral"},
}

# leaf module path -> {public_name: func_name} re-exported from n4m._impl.metrics
METRIC_FUNC_LEAVES: dict[str, dict[str, str]] = {
    "metrics/scoring": {
        "rmse": "rmse", "mae": "mae", "bias": "bias", "sep": "sep", "rpd": "rpd",
        "rpiq": "rpiq", "r2": "r2", "nrmse": "nrmse",
    },
}

# Role-name aliases (public name -> impl class) re-exported verbatim from n4m._impl.
# The spec mandates clean role names (e.g. `KennardStone`, not `KennardStoneSplitter`);
# the implementation-suffixed names are kept too for the internal/idiomatic surface.
CLASS_ALIAS_LEAVES: dict[str, dict[str, str]] = {
    "model_selection/splitters": {
        "KennardStone": "KennardStoneSplitter",
        "SPXY": "SPXYSplitter",
        "SPXYFold": "SPXYFoldSplitter",
        "SPXYGroupFold": "SPXYGroupFoldSplitter",
        "KMeans": "KMeansSplitter",
        "KBinsStratified": "KBinsStratifiedSplitter",
        "BinnedStratifiedGroupKFold": "BinnedStratifiedGroupKFoldSplitter",
        "SystematicCircular": "SystematicCircularSplitter",
        "DataTwinning": "SPlitSplitter",
    },
    "compose/aom_superblock": {
        "AOMRidgeSuperblock": "NativeAOMRidgeSuperblockRegressor",
        "AOMRidgeMKLSuperblock": "NativeAOMRidgeMKLSuperblockRegressor",
        "AOMRidgeActiveSuperblock": "NativeAOMRidgeActiveSuperblockRegressor",
        "AOMPLSSuperblock": "NativeAOMPLSSuperblockRegressor",
        "AOMRidgePLSSuperblock": "NativeAOMRidgePLSSuperblockRegressor",
    },
}


# Feature-selection wrapper / ranking classes re-exported from n4m._impl.selection.
SELECTOR_LEAVES: dict[str, list[str]] = {
    "feature_selection/wrapper": [
        "BiPLS", "BVE", "CARS", "EMCUVE", "GA", "IPW", "IRF", "IRIV", "PSO",
        "RandomFrog", "Randomization", "REP", "SCARS", "Shaving", "SiPLS",
        "SPA", "ST", "Stability", "T2", "UVE", "VIPSPA", "VISSA", "WVC",
        "WVCThreshold",
    ],
    "feature_selection/ranking": ["VariableSelect"],
}


# AOM portfolio gate/policy/selector classes (kept, no rename) -> role module.
PORTFOLIO_LEAVES: dict[str, list[str]] = {
    "model_selection/aom_search": [
        "AOMCandidateSpec", "AOMControlSelector", "AOMDatasetMetadata",
        "AOMPreprocessingChain", "AOMRobustHPORegressor", "AOMRobustHPOCompact",
        "AOMRobustHPOWide", "AOMStructuralPolicy",
        "AOMStructuralPolicyWithP700BlockLocalAdmission",
        "AOMStructuralPolicyWithP700ProtocolUnified",
        "AOMStructuralPolicyWithPgt1200Admissions", "build_aom_control_chain_bank",
    ],
    "compose/aom_superblock": ["AOMOperatorPLSSpec", "AOMOperatorPLSStack"],
    "ensemble": [
        "AOMRidgeBlender", "AOMTrueBankEndpointPortfolio", "AOMMidPEndpointStack",
        "AOMFallbackBlendGate", "AOMEndpointMarginStabilityGate",
        "EndpointStabilityDecision",
    ],
}

# every leaf module path -> docstring summary (also defines empty leaves)
LEAF_DOCS: dict[str, str] = {}

# All 49 leaves (some empty for methods n4m has no Python surface for yet).
ALL_LEAVES = [
    "transform/scatter", "transform/baseline", "transform/smoothing", "transform/wavelet",
    "transform/signal_conversion", "transform/resampling", "transform/scaling",
    "transform/alignment", "transform/orthogonalization", "transform/specialized",
    "augmentation/instrument", "augmentation/spectral", "augmentation/scattering",
    "augmentation/splines", "augmentation/mixup", "augmentation/noise",
    "augmentation/drift", "augmentation/wavelength",
    "estimators/regression/latent", "estimators/regression/regularized",
    "estimators/regression/robust", "estimators/regression/kernel",
    "estimators/regression/sparse", "estimators/regression/tensor",
    "estimators/regression/online", "estimators/regression/local",
    "estimators/regression/glm", "estimators/classification",
    "estimators/multiblock", "estimators/survival",
    "feature_selection/wrapper", "feature_selection/filter",
    "feature_selection/interval", "feature_selection/ranking",
    "model_selection/splitters", "model_selection/aom_search",
    "model_selection/aom_campaign", "model_selection/sweep",
    "domain_adaptation/standardization", "domain_adaptation/invariant",
    "domain_adaptation/metrics", "domain_adaptation/orthogonalization",
    "outlier_detection", "ensemble", "compose/aom_superblock",
    "metrics/scoring", "metrics/diagnostics", "decomposition", "lowlevel/moments",
]


def write(path: str, body: str) -> None:
    full = os.path.join(SRC, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as fh:
        fh.write(body)


def render_leaf(leaf: str) -> str:
    lines = [H]
    docs = LEAF_DOCS.get(leaf, "")
    if docs:
        lines.append(f'"""{docs}"""\n')
    all_names: list[str] = []
    imports: list[str] = []
    assigns: list[str] = []
    # verbatim class re-exports
    direct = list(CLASS_LEAVES.get(leaf, [])) + list(PORTFOLIO_LEAVES.get(leaf, []))
    if direct:
        names = sorted(set(direct))
        imports.append("from n4m._impl import (\n    " + ",\n    ".join(names) + ",\n)\n")
        all_names += names
    # feature-selection wrapper / ranking classes from n4m._impl.selection
    selectors = SELECTOR_LEAVES.get(leaf, [])
    if selectors:
        names = sorted(set(selectors))
        imports.append("from n4m._impl.selection import (\n    " + ",\n    ".join(names) + ",\n)\n")
        all_names += names
    # role-name aliases for impl classes (public role name -> impl class name)
    for pub, impl in sorted(CLASS_ALIAS_LEAVES.get(leaf, {}).items()):
        imports.append(f"from n4m._impl import {impl} as {pub}\n")
        all_names.append(pub)
    # renamed estimators
    renames: dict[str, str] = {}
    renames.update(ESTIMATOR_LEAVES.get(leaf, {}))
    renames.update(AOM_REGRESSOR_LEAVES.get(leaf, {}))
    for pub, impl in sorted(renames.items()):
        imports.append(f"from n4m._impl import {impl} as {pub}\n")
        all_names.append(pub)
    # metric function re-exports (from n4m._impl.metrics)
    mfuncs = METRIC_FUNC_LEAVES.get(leaf, {})
    if mfuncs:
        names = sorted(set(mfuncs.values()))
        imports.append("from n4m._impl.metrics import (\n    " + ",\n    ".join(names) + ",\n)\n")
        all_names += list(mfuncs.keys())
    # functional re-exports (from n4m._impl.native)
    funcs = FUNC_LEAVES.get(leaf, {})
    if funcs:
        imports.append("from n4m._impl import native as _native\n")
        for pub, impl in sorted(funcs.items()):
            assigns.append(f"{pub} = _native.{impl}\n")
            all_names.append(pub)
    text = "".join(lines)
    if imports or assigns:
        text += "\n" + "".join(imports)
        if assigns:
            text += "\n" + "".join(assigns)
    text += "\n__all__ = [\n    " + ",\n    ".join(f'"{n}"' for n in sorted(set(all_names))) + ",\n]\n" if all_names else "\n__all__: list[str] = []\n"
    return text


def render_package_init(pkg: str, children: list[str]) -> str:
    children = sorted(set(children))
    imports = "from . import (\n    " + ",\n    ".join(children) + ",\n)\n" if children else ""
    allist = ",\n    ".join(f'"{c}"' for c in children)
    text = H + f'"""n4m.{pkg.replace("/", ".")} role package."""\n'
    if children:
        text += "\n" + imports + f"\n__all__ = [\n    {allist},\n]\n"
    else:
        text += "\n__all__: list[str] = []\n"
    return text


def build() -> None:
    # Write all 49 leaf modules.
    for leaf in ALL_LEAVES:
        write(leaf + ".py", render_leaf(leaf))

    # Compute package tree (intermediate __init__.py files) from leaf paths.
    pkgs: dict[str, set[str]] = {}
    for leaf in ALL_LEAVES:
        parts = leaf.split("/")
        for depth in range(len(parts) - 1):
            pkg = "/".join(parts[: depth + 1])
            child = parts[depth + 1]
            pkgs.setdefault(pkg, set()).add(child)
    # top-level role packages and their direct children
    for pkg, children in pkgs.items():
        write(pkg + "/__init__.py", render_package_init(pkg, sorted(children)))


if __name__ == "__main__":
    build()
    print("role tree written under", SRC)
