# SPDX-License-Identifier: CECILL-2.1

.n4m_resampling_method <- function(method, caller) {
  if (is.character(method)) {
    switch(method,
           "linear"  = 0L,
           "cubic"   = 1L,
           "nearest" = 2L,
           stop(sprintf("%s: unknown method '%s'", caller, method),
                call. = FALSE))
  } else {
    as.integer(method)[1L]
  }
}

.n4m_kbins_strategy <- function(strategy, caller) {
  if (is.character(strategy)) {
    switch(strategy,
           "uniform"  = 0L,
           "quantile" = 1L,
           stop(sprintf("%s: unknown strategy '%s'", caller, strategy),
                call. = FALSE))
  } else {
    as.integer(strategy)[1L]
  }
}

#' Crop spectral columns.
#'
#' @param X numeric matrix.
#' @param start integer(1). Zero-based inclusive start column.
#' @param end integer(1). Zero-based exclusive end column.
#' @return numeric matrix with cropped columns.
#' @export
crop_transform <- function(X, start, end) {
  n4m_pp_crop_transform(X, start = start, end = end)
}

#' Resample spectra to a fixed number of columns.
#'
#' @param X numeric matrix.
#' @param num_samples integer(1).
#' @return numeric matrix with `num_samples` columns.
#' @export
resample_transform <- function(X, num_samples) {
  n4m_pp_resample_transform(X, num_samples = num_samples)
}

#' Wavelength-grid resampler.
#'
#' @param X numeric matrix.
#' @param source_wavelengths numeric vector.
#' @param target_wavelengths numeric vector.
#' @param method character(1) or integer(1). One of "linear", "cubic",
#'   "nearest".
#' @param crop_min,crop_max numeric crop bounds.
#' @param use_crop logical(1).
#' @param fill_value numeric(1).
#' @param bounds_error logical(1).
#' @param extrapolate logical(1).
#' @return numeric matrix on the target wavelength grid.
#' @export
resampler <- function(X, source_wavelengths, target_wavelengths,
                      method = "linear",
                      crop_min = 0.0,
                      crop_max = 0.0,
                      use_crop = FALSE,
                      fill_value = 0.0,
                      bounds_error = FALSE,
                      extrapolate = FALSE) {
  n4m_pp_resampler_fit_transform(
    X,
    source_wavelengths = source_wavelengths,
    target_wavelengths = target_wavelengths,
    method = .n4m_resampling_method(method, "resampler"),
    crop_min = crop_min,
    crop_max = crop_max,
    use_crop = use_crop,
    fill_value = fill_value,
    bounds_error = bounds_error,
    extrapolate = extrapolate)
}

#' Edge-based range discretizer.
#'
#' @param X numeric matrix.
#' @param edges numeric vector of monotonic bin edges.
#' @return integer matrix.
#' @export
range_discretizer <- function(X, edges) {
  n4m_pp_range_disc_transform(X, edges = edges)
}

#' Integer K-bins discretizer.
#'
#' @param X numeric matrix.
#' @param n_bins integer(1).
#' @param strategy character(1) or integer(1). One of "uniform", "quantile".
#' @return integer matrix.
#' @export
kbins_discretizer <- function(X, n_bins = 5L, strategy = "uniform") {
  n4m_pp_kbins_disc_fit_transform(
    X,
    n_bins = n_bins,
    strategy = .n4m_kbins_strategy(strategy, "kbins_discretizer"))
}
