# SPDX-License-Identifier: CECILL-2.1

#' Direct Standardization.
#' @export
direct_standardization <- function(source, target, X = source,
                                   fit_intercept = TRUE, ridge = 0.0) {
  n4m_pp_direct_standardization_fit_transform(source, target, X,
                                              fit_intercept, ridge)
}

#' Robust Direct Standardization.
#' @export
robust_direct_standardization <- function(source, target, X = source,
                                          fit_intercept = TRUE, ridge = 0.0,
                                          trim_quantile = 0.1,
                                          max_iter = 5L) {
  n4m_pp_robust_direct_standardization_fit_transform(
    source, target, X, fit_intercept, ridge, trim_quantile, max_iter)
}

#' Piecewise Direct Standardization.
#' @export
piecewise_direct_standardization <- function(source, target, X = source,
                                             window_size = 5L,
                                             fit_intercept = TRUE,
                                             ridge = 0.0) {
  n4m_pp_piecewise_direct_standardization_fit_transform(
    source, target, X, window_size, fit_intercept, ridge)
}

#' Score-Augmented Projection Standardization.
#' @export
score_augmented_projection_standardization <- function(source, target, X = source,
                                                       n_components = 2L,
                                                       score_weight = 1.0,
                                                       fit_intercept = TRUE,
                                                       ridge = 1e-8) {
  n4m_pp_saps_fit_transform(source, target, X, n_components, score_weight,
                            fit_intercept, ridge)
}

#' Slope/Bias correction.
#' @export
slope_bias_correction <- function(source, target, y = source) {
  n4m_pp_slope_bias_fit_transform(source, target, y)
}

#' Local centering transfer correction.
#' @export
local_centering <- function(source, target, X = source) {
  n4m_pp_local_centering_fit_transform(source, target, X)
}

#' Weighted Standard Normal Variate.
#' @export
weighted_snv <- function(X, weights = NULL, ddof = 0L, eps = 1e-12) {
  n4m_pp_weighted_snv_fit_transform(X, weights, ddof, eps)
}

#' Variable Sorting Normalization.
#' @export
variable_sorting_normalization <- function(X, eps = 1e-12) {
  n4m_pp_vsn_fit_transform(X, eps)
}

#' Piecewise SNV.
#' @export
piecewise_snv <- function(X, window = 16L, ddof = 0L, eps = 1e-12) {
  n4m_pp_piecewise_snv_fit_transform(X, window, ddof, eps)
}

#' Piecewise MSC.
#' @export
piecewise_msc <- function(X, reference = NULL, window = 16L, eps = 1e-12) {
  n4m_pp_piecewise_msc_fit_transform(X, reference, window, eps)
}

#' Localized MSC.
#' @export
localized_msc <- function(X, reference = NULL, window = 16L, eps = 1e-12) {
  n4m_pp_localized_msc_fit_transform(X, reference, window, eps)
}

#' Cross-correlation alignment.
#' @export
cross_correlation_alignment <- function(X, reference = NULL,
                                        interval_size = 16L,
                                        max_shift = 3L) {
  n4m_pp_xcorr_align_fit_transform(X, reference, interval_size, max_shift)
}

#' Icoshift alignment.
#' @export
icoshift_alignment <- function(X, reference = NULL,
                               interval_size = 16L, max_shift = 3L) {
  n4m_pp_icoshift_align_fit_transform(X, reference, interval_size, max_shift)
}

#' Dynamic Time Warping alignment.
#' @export
dynamic_time_warping_alignment <- function(X, reference = NULL,
                                           interval_size = 16L,
                                           max_shift = 3L) {
  n4m_pp_dtw_align_fit_transform(X, reference, interval_size, max_shift)
}

#' Correlation Optimized Warping.
#' @export
correlation_optimized_warping <- function(X, reference = NULL,
                                          interval_size = 16L,
                                          max_shift = 3L) {
  n4m_pp_cow_align_fit_transform(X, reference, interval_size, max_shift)
}

#' Variance feature filter.
#' @export
variance_filter <- function(X, threshold = 0.0, top_k = 0L) {
  n4m_filter_variance_fit_transform(X, threshold, top_k)
}

#' Correlation feature filter.
#' @export
correlation_filter <- function(X, y, threshold = 0.0, top_k = 0L) {
  n4m_filter_correlation_fit_transform(X, y, threshold, top_k)
}

#' Interval feature generator.
#' @export
interval_generator <- function(X, interval_size = 16L, step = interval_size) {
  X <- .n4m_check_matrix(X, "X")
  interval_size <- max(1L, as.integer(interval_size)[1L])
  step <- max(1L, as.integer(step)[1L])
  out <- n4m_interval_generator_fit_transform(X, interval_size, step)
  starts <- seq.int(1L, ncol(X), by = step)
  blocks <- vector("list", length(starts))
  offset <- 1L
  for (idx in seq_along(starts)) {
    width <- min(interval_size, ncol(X) - starts[[idx]] + 1L)
    cols <- seq.int(offset, length.out = width)
    blocks[[idx]] <- out[, cols, drop = FALSE]
    offset <- offset + width
  }
  blocks
}
