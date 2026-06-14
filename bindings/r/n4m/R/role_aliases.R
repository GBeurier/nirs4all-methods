# SPDX-License-Identifier: CECILL-2.1
#
# Locked ML-namespace role aliases (ABI 2.0 migration, TARGET_NAMESPACE_ML.md).
#
# Canonical `n4m_<role>_<method>()` public surface for the directly
# method-exposing entry points, per the locked examples
# (n4m_transform_snv(), n4m_feature_select_cars(),
# n4m_model_selection_kennard_stone()). Each is a thin forwarding wrapper
# around the existing idiomatic function; the legacy idiomatic names and the
# formula / S3 wrappers are kept. Internal .Call entry names are unchanged.
# Wrappers resolve their target at call time, so source/Collate order is
# irrelevant.


#' @rdname bipls_select
#' @export
n4m_feature_select_bipls <- function(...) bipls_select(...)

#' @rdname bve_select
#' @export
n4m_feature_select_bve <- function(...) bve_select(...)

#' @rdname cars_select
#' @export
n4m_feature_select_cars <- function(...) cars_select(...)

#' @rdname coefficient_select
#' @export
n4m_feature_select_coefficient <- function(...) coefficient_select(...)

#' @rdname emcuve_select
#' @export
n4m_feature_select_emcuve <- function(...) emcuve_select(...)

#' @rdname ga_select
#' @export
n4m_feature_select_ga <- function(...) ga_select(...)

#' @rdname interval_select
#' @export
n4m_feature_select_interval <- function(...) interval_select(...)

#' @rdname ipw_select
#' @export
n4m_feature_select_ipw <- function(...) ipw_select(...)

#' @rdname irf_select
#' @export
n4m_feature_select_irf <- function(...) irf_select(...)

#' @rdname iriv_select
#' @export
n4m_feature_select_iriv <- function(...) iriv_select(...)

#' @rdname pso_select
#' @export
n4m_feature_select_pso <- function(...) pso_select(...)

#' @rdname random_frog_select
#' @export
n4m_feature_select_random_frog <- function(...) random_frog_select(...)

#' @rdname randomization_select
#' @export
n4m_feature_select_randomization <- function(...) randomization_select(...)

#' @rdname rep_select
#' @export
n4m_feature_select_rep <- function(...) rep_select(...)

#' @rdname scars_select
#' @export
n4m_feature_select_scars <- function(...) scars_select(...)

#' @rdname selectivity_ratio_select
#' @export
n4m_feature_select_selectivity_ratio <- function(...) selectivity_ratio_select(...)

#' @rdname shaving_select
#' @export
n4m_feature_select_shaving <- function(...) shaving_select(...)

#' @rdname sipls_select
#' @export
n4m_feature_select_sipls <- function(...) sipls_select(...)

#' @rdname spa_select
#' @export
n4m_feature_select_spa <- function(...) spa_select(...)

#' @rdname st_select
#' @export
n4m_feature_select_st <- function(...) st_select(...)

#' @rdname stability_select
#' @export
n4m_feature_select_stability <- function(...) stability_select(...)

#' @rdname t2_select
#' @export
n4m_feature_select_t2 <- function(...) t2_select(...)

#' @rdname uve_select
#' @export
n4m_feature_select_uve <- function(...) uve_select(...)

#' @rdname vip_select
#' @export
n4m_feature_select_vip <- function(...) vip_select(...)

#' @rdname vip_spa_select
#' @export
n4m_feature_select_vip_spa <- function(...) vip_spa_select(...)

#' @rdname vissa_select
#' @export
n4m_feature_select_vissa <- function(...) vissa_select(...)

#' @rdname wvc_select
#' @export
n4m_feature_select_wvc <- function(...) wvc_select(...)

#' @rdname wvc_threshold_select
#' @export
n4m_feature_select_wvc_threshold <- function(...) wvc_threshold_select(...)

#' @rdname snv_transform
#' @export
n4m_transform_snv <- function(...) snv_transform(...)

#' @rdname savgol_transform
#' @export
n4m_transform_savgol <- function(...) savgol_transform(...)

#' @rdname kennard_stone_split
#' @export
n4m_model_selection_kennard_stone <- function(...) kennard_stone_split(...)

#' @rdname ridge_fit
#' @export
n4m_regression_ridge <- function(...) ridge_fit(...)
