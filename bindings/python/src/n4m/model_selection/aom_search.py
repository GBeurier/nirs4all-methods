# SPDX-License-Identifier: CECILL-2.1

from n4m._impl import (
    AOMCandidateSpec,
    AOMControlSelector,
    AOMDatasetMetadata,
    AOMPreprocessingChain,
    AOMRobustHPOCompact,
    AOMRobustHPORegressor,
    AOMRobustHPOWide,
    AOMStructuralPolicy,
    AOMStructuralPolicyWithP700BlockLocalAdmission,
    AOMStructuralPolicyWithP700ProtocolUnified,
    AOMStructuralPolicyWithPgt1200Admissions,
    build_aom_control_chain_bank,
)
from n4m._impl import NativeAOMChainRidgePLSRegressor as AOMChainRidgePLSRegressor
from n4m._impl import NativeAOMChainSweepRegressor as AOMChainSweepRegressor
from n4m._impl import NativeAOMFixedCandidateRegressor as AOMFixedCandidateRegressor
from n4m._impl import NativeAOMPLSRegressor as AOMPLSRegressor
from n4m._impl import NativeAOMRidgeGlobalRegressor as AOMRidgeGlobalRegressor
from n4m._impl import NativeAOMRobustHPORegressor as AOMRobustHPOSweepRegressor
from n4m._impl import NativeAOMScreenRefitRegressor as AOMScreenRefitRegressor
from n4m._impl import NativeAOMSweepRegressor as AOMSweepRegressor
from n4m._impl import NativePOPPLSRegressor as POPPLSRegressor
from n4m._impl import native as _native

aom_candidate_operator_summary = _native.aom_candidate_operator_summary
aom_candidate_preprocessing_impact = _native.aom_candidate_preprocessing_impact
aom_candidate_rank_diagnostics = _native.aom_candidate_rank_diagnostics
aom_candidate_report_records = _native.aom_candidate_report_records
aom_candidate_route_summary = _native.aom_candidate_route_summary
aom_candidate_table = _native.aom_candidate_table
aom_chain_fixed_fit_run = _native.aom_chain_fixed_fit_run
aom_chain_ridge_pls = _native.aom_chain_ridge_pls
aom_chain_sweep_run = _native.aom_chain_sweep_run
aom_evaluate_candidates = _native.aom_evaluate_candidates
aom_global_select = _native.aom_global_select
aom_load_candidate_report = _native.aom_load_candidate_report
aom_per_component_select = _native.aom_per_component_select
aom_pls = _native.aom_pls
aom_preprocess = _native.aom_preprocess
aom_refit_candidates = _native.aom_refit_candidates
aom_refit_execution_plan = _native.aom_refit_execution_plan
aom_ridge_global = _native.aom_ridge_global
aom_robust_hpo = _native.aom_robust_hpo
aom_save_candidate_report = _native.aom_save_candidate_report
aom_screen_refit_candidate_pool = _native.aom_screen_refit_candidate_pool
aom_sweep_run = _native.aom_sweep_run
build_aom_strict_chain_grid = _native.build_aom_strict_chain_grid
decode_aom_chains = _native.decode_aom_chains
iter_aom_strict_chain_grid = _native.iter_aom_strict_chain_grid
pop_pls = _native.pop_pls

__all__ = [
    "AOMCandidateSpec",
    "AOMChainRidgePLSRegressor",
    "AOMChainSweepRegressor",
    "AOMControlSelector",
    "AOMDatasetMetadata",
    "AOMFixedCandidateRegressor",
    "AOMPLSRegressor",
    "AOMPreprocessingChain",
    "AOMRidgeGlobalRegressor",
    "AOMRobustHPOCompact",
    "AOMRobustHPORegressor",
    "AOMRobustHPOSweepRegressor",
    "AOMRobustHPOWide",
    "AOMScreenRefitRegressor",
    "AOMStructuralPolicy",
    "AOMStructuralPolicyWithP700BlockLocalAdmission",
    "AOMStructuralPolicyWithP700ProtocolUnified",
    "AOMStructuralPolicyWithPgt1200Admissions",
    "AOMSweepRegressor",
    "POPPLSRegressor",
    "aom_candidate_operator_summary",
    "aom_candidate_preprocessing_impact",
    "aom_candidate_rank_diagnostics",
    "aom_candidate_report_records",
    "aom_candidate_route_summary",
    "aom_candidate_table",
    "aom_chain_fixed_fit_run",
    "aom_chain_ridge_pls",
    "aom_chain_sweep_run",
    "aom_evaluate_candidates",
    "aom_global_select",
    "aom_load_candidate_report",
    "aom_per_component_select",
    "aom_pls",
    "aom_preprocess",
    "aom_refit_candidates",
    "aom_refit_execution_plan",
    "aom_ridge_global",
    "aom_robust_hpo",
    "aom_save_candidate_report",
    "aom_screen_refit_candidate_pool",
    "aom_sweep_run",
    "build_aom_control_chain_bank",
    "build_aom_strict_chain_grid",
    "decode_aom_chains",
    "iter_aom_strict_chain_grid",
    "pop_pls",
]
