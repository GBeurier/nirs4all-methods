# SPDX-License-Identifier: CECILL-2.1
#
# Unified low-level dispatcher wrapper. Calls the r_p4a_dispatch_fit C
# entry point which handles 33 MethodResult fits + 24 selectors + 4
# diagnostics by algorithm name + a named params list.
#
# Most users should reach for the dedicated tier-2 wrappers (sparse_pls,
# cppls, mb_pls, …); n4m_method is the escape hatch for new
# algorithms not yet wrapped or for advanced usage.

#' Low-level n4m method dispatcher.
#'
#' @param algo Character; algorithm name (see Details).
#' @param X Numeric matrix or NULL (for one_se_rule_compute).
#' @param Y Numeric matrix or vector. Pass an n x 1 placeholder for
#'   classifier-style fits where labels go in `params$y_labels`.
#' @param n_components Positive integer.
#' @param params Named list of algorithm-specific parameters
#'   (sparsity_lambda, sample_weights, block_sizes, X_target,
#'   y_labels, alpha_thresholds, ...).
#' @param center_x,scale_x,center_y,scale_y Boolean preprocessing flags
#'   (default centering, no scaling — matches the Python tier-1 defaults).
#' @return A named list with the MethodResult arrays + scalars. Index
#'   fields (`selected_indices`, `top_k_intervals`, ...) are 1-based.
#'
#' @details Supported algorithm names:
#'
#' MethodResult fits (33): "sparse_simpls" "cppls" "ecr" "di_pls"
#'   "weighted_pls" "robust_pls" "ridge_pls" "continuum_regression"
#'   "recursive_pls" "n_pls" "kernel_pls" "o2pls" "sparse_pls_da"
#'   "group_sparse_pls" "fused_sparse_pls" "so_pls" "on_pls" "rosa"
#'   "bagging_pls" "boosting_pls" "random_subspace_pls" "gpr_pls"
#'   "pls_glm" "pls_qda" "pls_cox" "pds" "ds" "mir_pls"
#'   "missing_aware_nipals" "mb_pls" "lw_pls" "pls_lda" "pls_logistic"
#'
#' Selectors (24): "spa_select" "cars_select" "interval_select"
#'   "stability_select" "uve_select" "random_frog_select" "scars_select"
#'   "ga_select" "pso_select" "vissa_select" "shaving_select"
#'   "bve_select" "t2_select" "wvc_select" "wvc_threshold_select"
#'   "emcuve_select" "randomization_select" "bipls_select"
#'   "sipls_select" "rep_select" "ipw_select" "st_select" "iriv_select"
#'   "irf_select" "vip_spa_select"
#'
#' Diagnostics (2 via dispatcher; the other 2 stay in r_methods.c):
#'   "approximate_press_compute" "one_se_rule_compute"
#'
#' @export
n4m_method <- function(algo, X, Y, n_components, params = list(),
                            center_x = TRUE, scale_x = FALSE,
                            center_y = TRUE, scale_y = FALSE) {
    if (!is.matrix(X)) X <- as.matrix(X)
    if (is.null(dim(Y))) Y <- matrix(as.numeric(Y), ncol = 1L)
    storage.mode(X) <- "double"
    storage.mode(Y) <- "double"
    .Call("r_p4a_dispatch_fit",
          as.character(algo), X, Y, as.integer(n_components), params,
          as.integer(as.logical(center_x)),
          as.integer(as.logical(scale_x)),
          as.integer(as.logical(center_y)),
          as.integer(as.logical(scale_y)),
          PACKAGE = "pls4all")
}

#' Back-compat alias for the unified dispatcher.
#'
#' The dispatcher was renamed `pls4all_method` -> `n4m_method` during the
#' p4a -> n4m token rename. The cross-binding benches and any downstream
#' caller that still uses the old name resolve through this alias.
#'
#' @export
pls4all_method <- n4m_method
