# SPDX-License-Identifier: CECILL-2.1

#' Y-based outlier filter (fit + apply on same data).
#'
#' @param y numeric vector.
#' @param method character(1) or integer(1). One of "iqr", "zscore",
#'   "percentile", "mad".
#' @param threshold numeric(1). Method-specific threshold (default depends
#'   on method).
#' @param lower_pct numeric(1). Lower percentile (used by percentile method).
#' @param upper_pct numeric(1). Upper percentile.
#' @return list with:
#'   * `mask` logical vector: TRUE means keep, FALSE means exclude;
#'   * `n_samples`, `n_kept`, `n_excluded`: integer counts;
#'   * `exclusion_rate`: numeric scalar.
#' @export
y_outlier_filter <- function(y, method = "iqr", threshold = NULL,
                             lower_pct = 1.0, upper_pct = 99.0) {
  method_int <- if (is.character(method)) {
    switch(method,
           "iqr"        = 0L,
           "zscore"     = 1L,
           "percentile" = 2L,
           "mad"        = 3L,
           stop(sprintf("y_outlier_filter: unknown method '%s'", method),
                call. = FALSE))
  } else {
    as.integer(method)[1L]
  }
  if (is.null(threshold)) {
    threshold <- switch(method_int + 1L,
                        1.5,    # iqr
                        3.0,    # zscore
                        0.0,    # percentile (threshold ignored)
                        3.5)    # mad
  }
  c4a_filter_y_outlier_fit_apply(y, method = method_int,
                                 threshold = threshold,
                                 lower_pct = lower_pct,
                                 upper_pct = upper_pct)
}

#' X-based outlier filter (fit + apply on same data).
#'
#' @param X numeric matrix.
#' @param method character(1) or integer(1). One of "mahalanobis",
#'   "robust_mahalanobis", "pca_residual", "pca_leverage",
#'   "isolation_forest", "lof".
#' @param threshold numeric(1) or NULL. When NULL, contamination mode is used.
#' @param n_components integer(1).
#' @param contamination numeric(1).
#' @param seed numeric(1).
#' @param n_estimators integer(1).
#' @param max_samples numeric(1).
#' @return list with `mask`, `n_samples`, `n_kept`, `n_excluded`, and
#'   `exclusion_rate`.
#' @export
x_outlier_filter <- function(X, method = "mahalanobis", threshold = NULL,
                             n_components = min(5L, ncol(X)),
                             contamination = 0.1,
                             seed = 0,
                             n_estimators = 100L,
                             max_samples = 256) {
  method_int <- if (is.character(method)) {
    switch(method,
           "mahalanobis"        = 0L,
           "robust_mahalanobis" = 1L,
           "pca_residual"       = 2L,
           "pca_leverage"       = 3L,
           "isolation_forest"   = 4L,
           "lof"                = 5L,
           stop(sprintf("x_outlier_filter: unknown method '%s'", method),
                call. = FALSE))
  } else {
    as.integer(method)[1L]
  }
  use_threshold <- !is.null(threshold)
  if (is.null(threshold)) {
    threshold <- 0.0
  }
  c4a_filter_x_outlier_fit_apply(
    X,
    method = method_int,
    use_threshold = use_threshold,
    threshold = threshold,
    n_components = n_components,
    contamination = contamination,
    seed = seed,
    n_estimators = n_estimators,
    max_samples = max_samples)
}
