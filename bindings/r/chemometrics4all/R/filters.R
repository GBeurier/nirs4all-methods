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
