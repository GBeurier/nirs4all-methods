#' Fit a PLS regression model via the libp4a C ABI.
#'
#' Accepts a numeric matrix X (n x p) and a numeric vector or matrix Y
#' (n x q). Both are coerced to double precision and row-major contiguous
#' before being passed across the C boundary.
#'
#' `algo` selects the solver. Recognized values:
#'   "pls_nipals", "pls_orthogonal_scores", "pls_simpls",
#'   "pls_kernel_algorithm", "pls_wide_kernel", "pls_svd",
#'   "pls_power", "pls_randomized_svd", "pcr_svd".
#'
#' @param X Numeric matrix, n x p.
#' @param Y Numeric matrix or vector.
#' @param algo Character. Solver name (see Details).
#' @param n_components Integer >= 1.
#'
#' @return An external pointer wrapping the fitted model handle. Pass it
#'   to [pls4all_predict()] to obtain predictions. The model is freed
#'   automatically when the external pointer is garbage-collected.
#' @export
pls4all_fit <- function(X, Y, algo, n_components, store_scores = FALSE) {
  if (!is.numeric(X)) stop("X must be numeric")
  if (!is.matrix(X)) X <- as.matrix(X)
  if (is.null(dim(Y))) Y <- matrix(as.numeric(Y), ncol = 1L)
  storage.mode(X) <- "double"
  storage.mode(Y) <- "double"
  if (nrow(X) != nrow(Y)) {
    stop(sprintf("nrow(X) (%d) must equal nrow(Y) (%d)",
                 nrow(X), nrow(Y)))
  }
  .Call("r_pls4all_fit",
        X, Y,
        as.character(algo),
        as.integer(n_components),
        as.logical(store_scores),
        PACKAGE = "pls4all")
}


#' Predict with a fitted pls4all model.
#'
#' @param model External pointer returned by [pls4all_fit()].
#' @param X Numeric matrix, n_new x p.
#' @return Numeric matrix, n_new x n_targets.
#' @export
pls4all_predict <- function(model, X) {
  if (!is.matrix(X)) X <- as.matrix(X)
  storage.mode(X) <- "double"
  .Call("r_pls4all_predict", model, X, PACKAGE = "pls4all")
}
