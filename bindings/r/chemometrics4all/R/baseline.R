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

#' Asymmetrically reweighted penalized least-squares baseline (ArPLS).
#'
#' @param X numeric matrix.
#' @param lam numeric(1). Smoothing penalty. Default 1e5.
#' @param max_iter integer(1). Default 50.
#' @param tol numeric(1). Default 1e-3.
#' @return numeric matrix.
#' @export
arpls <- function(X, lam = 1e5, max_iter = 50L, tol = 1e-3) {
  c4a_pp_arpls_transform(X, lam = lam, max_iter = max_iter, tol = tol)
}

#' Modified polynomial baseline correction.
#'
#' @param X numeric matrix.
#' @param polyorder integer(1). Default 2.
#' @param max_iter integer(1). Default 250.
#' @param tol numeric(1). Default 1e-3.
#' @return numeric matrix.
#' @export
modpoly <- function(X, polyorder = 2L, max_iter = 250L, tol = 1e-3) {
  c4a_pp_modpoly_transform(X, polyorder = polyorder,
                           max_iter = max_iter, tol = tol)
}

#' Improved modified polynomial baseline correction.
#'
#' @param X numeric matrix.
#' @param polyorder integer(1). Default 2.
#' @param max_iter integer(1). Default 250.
#' @param tol numeric(1). Default 1e-3.
#' @return numeric matrix.
#' @export
imodpoly <- function(X, polyorder = 2L, max_iter = 250L, tol = 1e-3) {
  c4a_pp_imodpoly_transform(X, polyorder = polyorder,
                            max_iter = max_iter, tol = tol)
}

#' SNIP baseline correction.
#'
#' @param X numeric matrix.
#' @param max_half_window integer(1). Default 20.
#' @return numeric matrix.
#' @export
snip <- function(X, max_half_window = 20L) {
  c4a_pp_snip_transform(X, max_half_window = max_half_window)
}

#' Rolling-ball baseline correction.
#'
#' @param X numeric matrix.
#' @param half_window integer(1). Default 20.
#' @param smooth_half_window integer(1). Default 0.
#' @return numeric matrix.
#' @export
rolling_ball <- function(X, half_window = 20L, smooth_half_window = 0L) {
  c4a_pp_rolling_ball_transform(X, half_window = half_window,
                                smooth_half_window = smooth_half_window)
}

#' Improved asymmetric least-squares baseline correction.
#'
#' @param X numeric matrix.
#' @param lam numeric(1). Default 1e6.
#' @param p numeric(1). Default 0.01.
#' @param lam_1 numeric(1). Default 1e-4.
#' @param polyorder integer(1). Default 2.
#' @param diff_order integer(1). Default 2.
#' @param max_iter integer(1). Default 50.
#' @param tol numeric(1). Default 1e-3.
#' @return numeric matrix.
#' @export
iasls <- function(X, lam = 1e6, p = 0.01, lam_1 = 1e-4,
                  polyorder = 2L, diff_order = 2L,
                  max_iter = 50L, tol = 1e-3) {
  c4a_pp_iasls_transform(X, lam = lam, p = p, lam_1 = lam_1,
                         polyorder = polyorder, diff_order = diff_order,
                         max_iter = max_iter, tol = tol)
}

#' Baseline Estimation And Denoising with Sparsity (BEADS).
#'
#' @param X numeric matrix.
#' @param lam_0 numeric(1). Default 100.
#' @param lam_1 numeric(1). Default 0.5.
#' @param lam_2 numeric(1). Default 0.5.
#' @param max_iter integer(1). Default 50.
#' @param tol numeric(1). Default 1e-3.
#' @return numeric matrix.
#' @export
beads <- function(X, lam_0 = 100.0, lam_1 = 0.5, lam_2 = 0.5,
                  max_iter = 50L, tol = 1e-3) {
  c4a_pp_beads_transform(X, lam_0 = lam_0, lam_1 = lam_1,
                         lam_2 = lam_2, max_iter = max_iter, tol = tol)
}
