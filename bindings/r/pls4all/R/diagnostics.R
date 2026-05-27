# SPDX-License-Identifier: CECILL-2.1
#
# Diagnostic wrappers (tier 1). Compute PLS T²/Q/DModX scores from a
# fitted model, the approximate-PRESS curve, and the PLS monitoring
# (Phase 1/Phase 2 alarm) workflow.

#' PLS diagnostics: T², Q, DModX from a fitted model.
#'
#' @param model External pointer from `n4m_fit()`.
#' @param X Numeric matrix to score (typically the training matrix).
#' @return A list with `t2`, `q`, `dmodx` — each is a 1-row matrix of
#'   length nrow(X).
#' @export
pls_diagnostics <- function(model, X) {
  if (!is.matrix(X)) X <- as.matrix(X)
  storage.mode(X) <- "double"
  .Call("r_p4a_pls_diagnostics_compute",
        model, X,
        PACKAGE = "pls4all")
}

#' Approximate-PRESS component selection.
#'
#' For each component count k in 1..max_components, fits SIMPLS and
#' approximates PRESS via leverage-inflated in-sample residuals.
#'
#' @inheritParams sparse_simpls_fit
#' @param max_components Integer; maximum number of components to scan.
#' @return A list with `press_per_component`, `rmse_per_component`,
#'   `selected_n_components` (the argmin of PRESS, 1-based as an integer
#'   length 1).
#' @param X Numeric matrix of predictors (rows = samples, cols = features).
#' @param Y Numeric matrix or vector of responses, with one row per sample.
#' @export
approximate_press <- function(X, Y, max_components) {
  xy <- .coerce_xy(X, Y)
  .Call("r_p4a_approximate_press_compute",
        xy$X, xy$Y,
        as.integer(max_components),
        PACKAGE = "pls4all")
}

#' PLS process monitoring (Hotelling T² + Q with alarms).
#'
#' Computes phase-1 thresholds on `X_reference`, then evaluates
#' `X_monitor` against those thresholds and reports per-row alarms.
#'
#' @param model External pointer from `n4m_fit()`.
#' @param X_reference Phase-1 numeric matrix (used to set thresholds).
#' @param X_monitor Phase-2 numeric matrix (rows are evaluated).
#' @param alpha Confidence level (e.g. 0.95). Thresholds correspond to
#'   the (1 - alpha) quantile.
#' @return A list with `t2`, `q`, `t2_alarms`, `q_alarms`, `any_alarms`,
#'   `t2_threshold`, `q_threshold`.
#' @export
pls_monitoring <- function(model, X_reference, X_monitor, alpha = 0.95) {
  if (!is.matrix(X_reference)) X_reference <- as.matrix(X_reference)
  if (!is.matrix(X_monitor)) X_monitor <- as.matrix(X_monitor)
  storage.mode(X_reference) <- "double"
  storage.mode(X_monitor) <- "double"
  if (ncol(X_reference) != ncol(X_monitor))
    stop("X_reference and X_monitor must have the same ncol")
  .Call("r_p4a_pls_monitoring_run",
        model, X_reference, X_monitor, as.numeric(alpha),
        PACKAGE = "pls4all")
}
