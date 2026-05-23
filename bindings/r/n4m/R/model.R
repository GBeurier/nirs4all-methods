#' Fit a PLS regression model via the libn4m C ABI.
#'
#' Accepts a numeric matrix X (n x p) and a numeric vector or matrix Y
#' (n x q). Both are coerced to double precision and row-major contiguous
#' before being passed across the C boundary.
#'
#' `algo` selects the solver. Recognized values:
#'   "pls_nipals", "pls_orthogonal_scores", "pls_simpls",
#'   "pls_kernel_algorithm", "pls_wide_kernel", "pls_svd",
#'   "pls_power", "pls_randomized_svd", "pcr_svd", "opls_nipals".
#'
#' @param X Numeric matrix, n x p.
#' @param Y Numeric matrix or vector.
#' @param algo Character. Solver name (see Details).
#' @param n_components Integer >= 1.
#'
#' @return An external pointer wrapping the fitted model handle. Pass it
#'   to [n4m_predict()] to obtain predictions. The model is freed
#'   automatically when the external pointer is garbage-collected.
#' @param store_scores Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
#' @param center_x Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
#' @param scale_x Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
#' @param center_y Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
#' @param scale_y Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
#' @export
n4m_fit <- function(X, Y, algo, n_components,
                         store_scores = FALSE,
                         center_x = TRUE, scale_x = TRUE,
                         center_y = TRUE, scale_y = TRUE) {
  if (!is.numeric(X)) stop("X must be numeric")
  if (!is.matrix(X)) X <- as.matrix(X)
  if (is.null(dim(Y))) Y <- matrix(as.numeric(Y), ncol = 1L)
  storage.mode(X) <- "double"
  storage.mode(Y) <- "double"
  if (nrow(X) != nrow(Y)) {
    stop(sprintf("nrow(X) (%d) must equal nrow(Y) (%d)",
                 nrow(X), nrow(Y)))
  }
  .Call("r_n4m_fit",
        X, Y,
        as.character(algo),
        as.integer(n_components),
        as.logical(store_scores),
        as.logical(center_x), as.logical(scale_x),
        as.logical(center_y), as.logical(scale_y),
        PACKAGE = "n4m")
}


#' Predict with a fitted n4m model.
#'
#' @param model External pointer returned by [n4m_fit()].
#' @param X Numeric matrix, n_new x p.
#' @return Numeric matrix, n_new x n_targets.
#' @export
n4m_predict <- function(model, X) {
  if (!is.matrix(X)) X <- as.matrix(X)
  storage.mode(X) <- "double"
  .Call("r_n4m_predict", model, X, PACKAGE = "n4m")
}
