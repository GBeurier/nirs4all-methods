# SPDX-License-Identifier: CECILL-2.1

#' Polynomial detrend per row.
#'
#' @param X numeric matrix.
#' @param polyorder integer(1). Default 1 (linear detrend).
#' @return numeric matrix with the same shape as `X`.
#' @export
detrend <- function(X, polyorder = 1L) {
  c4a_pp_detrend_transform(X, polyorder = polyorder)
}

#' Asymmetric Least Squares baseline correction (Eilers & Boelens 2005).
#'
#' @param X numeric matrix.
#' @param lam numeric(1). Smoothing penalty. Default 1e6.
#' @param p numeric(1). Asymmetry parameter, 0 < p < 1. Default 0.01.
#' @param max_iter integer(1). Default 50.
#' @param tol numeric(1). Relative weight tolerance. Default 1e-3.
#' @return numeric matrix.
#' @export
asls <- function(X, lam = 1e6, p = 0.01, max_iter = 50L, tol = 1e-3) {
  c4a_pp_asls_transform(X, lam = lam, p = p, max_iter = max_iter, tol = tol)
}

#' Adaptive iteratively reweighted PLS baseline (AirPLS, Zhang 2010).
#'
#' @param X numeric matrix.
#' @param lam numeric(1). Smoothing penalty. Default 1e6.
#' @param max_iter integer(1). Default 50.
#' @param tol numeric(1). Default 1e-3.
#' @return numeric matrix.
#' @export
airpls <- function(X, lam = 1e6, max_iter = 50L, tol = 1e-3) {
  c4a_pp_airpls_transform(X, lam = lam, max_iter = max_iter, tol = tol)
}
