# SPDX-License-Identifier: CeCILL-2.1
#
# Variable selection wrappers (tier 1). All selectors return either:
#   - a list with `selected_indices` (1-based after the C-side conversion
#     done here), `scores` (when applicable), `best_rmse` (when applicable)

#' VIP (Variable Importance in Projection) ranker.
#'
#' Operates on an already-fitted pls4all model handle (from
#' `pls4all_fit`). Returns the top-`top_k` features by VIP score.
#'
#' @param model External pointer from `pls4all_fit()`.
#' @param X Numeric matrix used for the fit (re-passed for diagnostics).
#' @param top_k Integer; number of features to return.
#' @return A list with `scores` (length p VIP scores) and
#'   `selected_indices` (1-based, length top_k).
#' @export
vip_select <- function(model, X, top_k) {
  if (!is.matrix(X)) X <- as.matrix(X)
  storage.mode(X) <- "double"
  res <- .Call("r_p4a_variable_select_rank",
                model, X, 0L, as.integer(top_k),
                PACKAGE = "pls4all")
  # C side returns 0-based indices; expose 1-based to R idiom.
  if (!is.null(res$selected_indices))
    res$selected_indices <- as.integer(res$selected_indices) + 1L
  res
}

#' Coefficient-magnitude ranker.
#' @inheritParams vip_select
#' @return A list with `scores` (|coef| sums) and `selected_indices`.
#' @export
coefficient_select <- function(model, X, top_k) {
  if (!is.matrix(X)) X <- as.matrix(X)
  storage.mode(X) <- "double"
  res <- .Call("r_p4a_variable_select_rank",
                model, X, 1L, as.integer(top_k),
                PACKAGE = "pls4all")
  if (!is.null(res$selected_indices))
    res$selected_indices <- as.integer(res$selected_indices) + 1L
  res
}

#' Selectivity-ratio ranker.
#' @inheritParams vip_select
#' @return A list with `scores` and `selected_indices`.
#' @export
selectivity_ratio_select <- function(model, X, top_k) {
  if (!is.matrix(X)) X <- as.matrix(X)
  storage.mode(X) <- "double"
  res <- .Call("r_p4a_variable_select_rank",
                model, X, 2L, as.integer(top_k),
                PACKAGE = "pls4all")
  if (!is.null(res$selected_indices))
    res$selected_indices <- as.integer(res$selected_indices) + 1L
  res
}

#' SPA — Successive Projections Algorithm.
#'
#' Fit-data selector: pass (X, Y) and the desired number of features.
#'
#' @inheritParams sparse_simpls_fit
#' @param top_k Number of features to select.
#' @return A list with `selected_indices` (1-based) and `best_rmse`.
#' @export
spa_select <- function(X, Y, n_components, top_k) {
  xy <- .coerce_xy(X, Y)
  res <- .Call("r_p4a_spa_select",
                xy$X, xy$Y,
                as.integer(n_components),
                as.integer(top_k),
                PACKAGE = "pls4all")
  if (!is.null(res$selected_indices))
    res$selected_indices <- as.integer(res$selected_indices) + 1L
  res
}

#' CARS — Competitive Adaptive Reweighted Sampling.
#'
#' Fit-data selector with a built-in 5-fold validation plan (default
#' C-side fallback). For a custom plan, use the lower-level Python
#' binding or extend r_methods.c.
#'
#' @inheritParams sparse_simpls_fit
#' @param n_iterations Number of CARS iterations (typical 50-100).
#' @param min_features Lower bound on the final subset size.
#' @return A list with `selected_indices` (1-based) and `best_rmse`.
#' @export
cars_select <- function(X, Y, n_components, n_iterations = 50L,
                         min_features = 5L) {
    # Use the unified dispatcher (which builds a default 5-fold
    # ValidationPlan). The legacy r_p4a_cars_select path passed NULL
    # and the C side now rejects that with E_NULL_POINTER.
    pls4all_method("cars_select", X, Y, n_components,
                   params = list(n_iterations = as.integer(n_iterations),
                                  min_features = as.integer(min_features)))
}
