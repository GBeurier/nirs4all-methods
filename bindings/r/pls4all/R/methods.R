# SPDX-License-Identifier: CECILL-2.1
#
# R tier-1 wrappers for MethodResult-returning fits exposed by
# r_methods.c. These are thin: validate / coerce / .Call / repackage.
# Tier-2 idiomatic wrappers (formula+S3 / parsnip / mlr3) live in
# sklearn.R / parsnip.R / r6_learner.R and consume these.
#
# All fits accept:
#   X — numeric matrix (n × p)
#   Y — numeric matrix or vector (n × q or n)
#   n_components — integer >= 1
# Each returns a list with at least `coefficients`, `predictions`,
# `x_mean`, `y_mean`, `rmse` (the in-sample RMSE). Some expose more
# (weights, intercept, block weights …) — see per-fit docs.

# Internal helper: validate (X, Y) and coerce to double matrices.
.coerce_xy <- function(X, Y) {
  if (!is.numeric(X)) stop("X must be numeric")
  if (!is.matrix(X)) X <- as.matrix(X)
  if (is.null(dim(Y))) Y <- matrix(as.numeric(Y), ncol = 1L)
  storage.mode(X) <- "double"
  storage.mode(Y) <- "double"
  if (nrow(X) != nrow(Y)) {
    stop(sprintf("nrow(X) (%d) must equal nrow(Y) (%d)",
                 nrow(X), nrow(Y)))
  }
  list(X = X, Y = Y)
}

#' Sparse SIMPLS fit.
#'
#' @param X Numeric matrix.
#' @param Y Numeric matrix or vector.
#' @param n_components Integer >= 1.
#' @param sparsity_lambda Soft-threshold magnitude per component (>= 0).
#' @return A list with `coefficients`, `predictions`, `x_mean`, `y_mean`,
#'   `weights_w`, `rmse`.
#' @export
sparse_simpls_fit <- function(X, Y, n_components, sparsity_lambda = 0.05) {
  xy <- .coerce_xy(X, Y)
  .Call("r_p4a_sparse_simpls_fit",
        xy$X, xy$Y,
        as.integer(n_components),
        as.numeric(sparsity_lambda),
        PACKAGE = "pls4all")
}

#' Canonical Powered PLS fit (Indahl 2005).
#'
#' @param gamma Power exponent in [0, 1]. 0 recovers SIMPLS, 1 is fully
#'   power-rescaled.
#' @inheritParams sparse_simpls_fit
#' @return A list with `coefficients`, `predictions`, `x_mean`, `y_mean`, `rmse`.
#' @param n_components Integer. Number of latent components.
#' @param X Numeric matrix of predictors (rows = samples, cols = features).
#' @param Y Numeric matrix or vector of responses, with one row per sample.
#' @export
cppls_fit <- function(X, Y, n_components, gamma = 0.5) {
  xy <- .coerce_xy(X, Y)
  .Call("r_p4a_cppls_fit",
        xy$X, xy$Y,
        as.integer(n_components),
        as.numeric(gamma),
        PACKAGE = "pls4all")
}

#' Sample-weighted PLS (sqrt(w)-prescaled SIMPLS).
#'
#' @param sample_weights Numeric vector of length nrow(X), strictly
#'   positive finite weights.
#' @inheritParams sparse_simpls_fit
#' @return A list with `coefficients`, `predictions`, `x_mean`, `y_mean`, `rmse`.
#' @param n_components Integer. Number of latent components.
#' @param X Numeric matrix of predictors (rows = samples, cols = features).
#' @param Y Numeric matrix or vector of responses, with one row per sample.
#' @export
weighted_pls_fit <- function(X, Y, n_components, sample_weights) {
  xy <- .coerce_xy(X, Y)
  if (!is.numeric(sample_weights))
    stop("sample_weights must be numeric")
  if (length(sample_weights) != nrow(xy$X))
    stop(sprintf("length(sample_weights) (%d) must equal nrow(X) (%d)",
                 length(sample_weights), nrow(xy$X)))
  .Call("r_p4a_weighted_pls_fit",
        xy$X, xy$Y,
        as.integer(n_components),
        as.numeric(sample_weights),
        PACKAGE = "pls4all")
}

#' Multi-block PLS (block-weighted SIMPLS).
#'
#' @param block_sizes Integer vector; must sum to ncol(X).
#' @inheritParams sparse_simpls_fit
#' @return A list with `coefficients`, `predictions`, `x_mean`,
#'   `x_scale`, `intercept`, `block_weights`, `rmse`.
#' @param n_components Integer. Number of latent components.
#' @param X Numeric matrix of predictors (rows = samples, cols = features).
#' @param Y Numeric matrix or vector of responses, with one row per sample.
#' @export
mb_pls_fit <- function(X, Y, n_components, block_sizes) {
  xy <- .coerce_xy(X, Y)
  if (!is.numeric(block_sizes))
    stop("block_sizes must be numeric or integer")
  if (sum(block_sizes) != ncol(xy$X))
    stop(sprintf("sum(block_sizes)=%g must equal ncol(X)=%d",
                 sum(block_sizes), ncol(xy$X)))
  .Call("r_p4a_mb_pls_fit",
        xy$X, xy$Y,
        as.integer(n_components),
        as.integer(block_sizes),
        PACKAGE = "pls4all")
}

#' PLS-GLM — Gaussian (default) or Poisson IRLS.
#'
#' @param poisson Logical; TRUE selects the Poisson-link path.
#' @inheritParams sparse_simpls_fit
#' @return A list with `coefficients`, `intercept`, `predictions`,
#'   `x_mean`, `rmse`.
#' @param n_components Integer. Number of latent components.
#' @param X Numeric matrix of predictors (rows = samples, cols = features).
#' @param Y Numeric matrix or vector of responses, with one row per sample.
#' @export
pls_glm_fit <- function(X, Y, n_components, poisson = FALSE) {
  xy <- .coerce_xy(X, Y)
  .Call("r_p4a_pls_glm_fit",
        xy$X, xy$Y,
        as.integer(n_components),
        as.integer(as.logical(poisson)),
        PACKAGE = "pls4all")
}

#' MIR-PLS — Multivariate Inverse Regression PLS.
#'
#' @inheritParams sparse_simpls_fit
#' @return A list with `coefficients`, `predictions`, `x_mean`, `y_mean`, `rmse`.
#' @param n_components Integer. Number of latent components.
#' @param X Numeric matrix of predictors (rows = samples, cols = features).
#' @param Y Numeric matrix or vector of responses, with one row per sample.
#' @export
mir_pls_fit <- function(X, Y, n_components) {
  xy <- .coerce_xy(X, Y)
  .Call("r_p4a_mir_pls_fit",
        xy$X, xy$Y,
        as.integer(n_components),
        PACKAGE = "pls4all")
}
