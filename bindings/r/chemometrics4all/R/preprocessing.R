# SPDX-License-Identifier: CECILL-2.1
#
# Tier-2 idiomatic R API for preprocessing operators. Each function takes
# a numeric matrix and returns a numeric matrix of the same shape (or, for
# fit/transform operators, a separate test matrix).

#' Standard Normal Variate (SNV).
#'
#' Row-wise centring (and optional scaling) of a spectral matrix.
#'
#' @param X numeric matrix (n_samples × n_features).
#' @param with_mean logical(1). Subtract the row mean. Default TRUE.
#' @param with_std logical(1). Divide by the row standard deviation. Default TRUE.
#' @param ddof integer(1). Delta degrees of freedom for std (0 = population, 1 = sample).
#' @return numeric matrix with the same shape as `X`.
#' @examples
#' if (FALSE) {
#'   X <- matrix(rnorm(50 * 200), nrow = 50)
#'   Y <- snv(X)
#' }
#' @export
snv <- function(X, with_mean = TRUE, with_std = TRUE, ddof = 0L) {
  c4a_pp_snv_transform(X, with_mean = with_mean, with_std = with_std,
                       ddof = ddof)
}

#' Local Standard Normal Variate (LSNV).
#'
#' @param X numeric matrix.
#' @param window integer(1). Odd sliding-window size.
#' @param pad_mode character(1) or integer(1). One of "reflect", "edge",
#'   "constant".
#' @param constant_value numeric(1). Padding value for constant mode.
#' @return numeric matrix with the same shape as `X`.
#' @export
lsnv <- function(X, window = 11L, pad_mode = "reflect",
                 constant_value = 0.0) {
  pad_int <- if (is.character(pad_mode)) {
    switch(pad_mode,
           "reflect"  = 0L,
           "edge"     = 1L,
           "constant" = 2L,
           stop(sprintf("lsnv: unknown pad_mode '%s'", pad_mode),
                call. = FALSE))
  } else as.integer(pad_mode)[1L]
  c4a_pp_lsnv_transform(X, window = window, pad_mode = pad_int,
                        constant_value = constant_value)
}

#' Robust Normal Variate (RNV).
#'
#' @param X numeric matrix.
#' @param with_center logical(1).
#' @param with_scale logical(1).
#' @param k numeric(1). MAD consistency factor.
#' @return numeric matrix with the same shape as `X`.
#' @export
rnv <- function(X, with_center = TRUE, with_scale = TRUE, k = 1.4826) {
  c4a_pp_rnv_transform(X, with_center = with_center,
                       with_scale = with_scale, k = k)
}

#' Row-wise area normalization.
#'
#' @param X numeric matrix.
#' @param method character(1) or integer(1). One of "sum", "abs_sum", "trapz".
#' @return numeric matrix with the same shape as `X`.
#' @export
area_normalization <- function(X, method = "sum") {
  method_int <- if (is.character(method)) {
    switch(method,
           "sum"     = 0L,
           "abs_sum" = 1L,
           "trapz"   = 2L,
           stop(sprintf("area_normalization: unknown method '%s'", method),
                call. = FALSE))
  } else as.integer(method)[1L]
  c4a_pp_area_transform(X, method = method_int)
}

#' Column-wise normalization.
#'
#' @param X numeric matrix.
#' @param feature_min numeric(1).
#' @param feature_max numeric(1).
#' @return numeric matrix with the same shape as `X`.
#' @export
normalize <- function(X, feature_min = -1.0, feature_max = 1.0) {
  c4a_pp_normalize_transform(X, feature_min = feature_min,
                             feature_max = feature_max)
}

#' Column-wise min-max scaling to [0, 1].
#'
#' @param X numeric matrix.
#' @return numeric matrix with the same shape as `X`.
#' @export
simple_scale <- function(X) {
  c4a_pp_simple_scale_transform(X)
}

#' Multiplicative Scatter Correction (MSC).
#'
#' Fits per-column slope/intercept on `X_fit` and applies the correction.
#'
#' @param X numeric matrix to transform.
#' @param X_fit numeric matrix used for fitting. Defaults to `X` (in-sample fit).
#' @return numeric matrix with the same shape as `X`.
#' @export
msc <- function(X, X_fit = X) {
  c4a_pp_msc_fit_transform(X_fit, X)
}

#' Inverse Multiplicative Scatter Correction (MSC).
#'
#' Applies the inverse of an MSC fit learned from `X_fit`.
#'
#' @param X numeric matrix in MSC-corrected space.
#' @param X_fit numeric matrix used for fitting. Defaults to `X`.
#' @return numeric matrix with the same shape as `X`.
#' @export
msc_inverse_transform <- function(X, X_fit = X) {
  c4a_pp_msc_fit_inverse_transform(X_fit, X)
}

#' Extended Multiplicative Scatter Correction (EMSC).
#'
#' Per-sample scatter correction with polynomial wavelength terms.
#'
#' @param X numeric matrix to transform.
#' @param X_fit numeric matrix used for fitting. Defaults to `X`.
#' @param degree integer(1). Polynomial degree (>= 1). Default 1.
#' @return numeric matrix with the same shape as `X`.
#' @export
emsc <- function(X, X_fit = X, degree = 1L) {
  c4a_pp_emsc_fit_transform(X_fit, X, degree = degree)
}

#' Column-mean baseline centering.
#'
#' @param X numeric matrix to transform.
#' @param X_fit numeric matrix used for fitting. Defaults to `X`.
#' @return numeric matrix with the same shape as `X`.
#' @export
baseline_center <- function(X, X_fit = X) {
  c4a_pp_baseline_fit_transform(X_fit, X)
}

#' Inverse column-mean baseline centering.
#'
#' @param X numeric matrix in centered space.
#' @param X_fit numeric matrix used for fitting. Defaults to `X`.
#' @return numeric matrix with the same shape as `X`.
#' @export
baseline_center_inverse_transform <- function(X, X_fit = X) {
  c4a_pp_baseline_fit_inverse_transform(X_fit, X)
}

#' Savitzky-Golay filter (smoothing or derivative).
#'
#' @param X numeric matrix.
#' @param window_length integer(1). Must be odd and >= polyorder + 1.
#' @param polyorder integer(1).
#' @param deriv integer(1). Derivative order (0 = smoothing).
#' @param delta numeric(1). Sample spacing.
#' @param mode character(1) or integer(1). One of "mirror", "constant",
#'   "nearest", "wrap", "interp".
#' @param cval numeric(1). Fill value for mode = "constant".
#' @return numeric matrix with the same shape as `X`.
#' @export
savitzky_golay <- function(X, window_length, polyorder, deriv = 0L,
                           delta = 1.0, mode = "mirror", cval = 0.0) {
  mode_int <- if (is.character(mode)) {
    switch(mode,
           "mirror"   = 0L,
           "constant" = 1L,
           "nearest"  = 2L,
           "wrap"     = 3L,
           "interp"   = 4L,
           stop(sprintf("savitzky_golay: unknown mode '%s'", mode), call. = FALSE))
  } else {
    as.integer(mode)[1L]
  }
  c4a_pp_savgol_transform(X, window_length = window_length,
                          polyorder = polyorder, deriv = deriv,
                          delta = delta, mode = mode_int, cval = cval)
}

#' First derivative along the wavelength axis (np.gradient).
#'
#' @param X numeric matrix.
#' @param delta numeric(1). Sample spacing.
#' @param edge_order integer(1). 1 or 2.
#' @return numeric matrix with the same shape as `X`.
#' @export
first_derivative <- function(X, delta = 1.0, edge_order = 2L) {
  c4a_pp_first_derivative_transform(X, delta = delta, edge_order = edge_order)
}

#' Second derivative along the wavelength axis.
#'
#' @param X numeric matrix.
#' @param delta numeric(1).
#' @param edge_order integer(1).
#' @return numeric matrix with the same shape as `X`.
#' @export
second_derivative <- function(X, delta = 1.0, edge_order = 2L) {
  c4a_pp_second_derivative_transform(X, delta = delta, edge_order = edge_order)
}

#' Norris-Williams derivative.
#'
#' @param X numeric matrix.
#' @param segment integer(1).
#' @param gap integer(1).
#' @param derivative_order integer(1). 1 or 2.
#' @param delta numeric(1).
#' @return numeric matrix with the same shape as `X`.
#' @export
norris_williams <- function(X, segment = 5L, gap = 5L,
                            derivative_order = 1L, delta = 1.0) {
  c4a_pp_norris_williams_transform(X, segment = segment, gap = gap,
                                   derivative_order = derivative_order,
                                   delta = delta)
}

#' Gaussian filter along the wavelength axis.
#'
#' @param X numeric matrix.
#' @param sigma numeric(1).
#' @param order integer(1).
#' @param mode character(1) or integer(1). One of "reflect", "constant",
#'   "nearest", "mirror", "wrap".
#' @param cval numeric(1).
#' @param truncate numeric(1).
#' @return numeric matrix with the same shape as `X`.
#' @export
gaussian <- function(X, sigma = 1.0, order = 0L, mode = "reflect",
                     cval = 0.0, truncate = 4.0) {
  mode_int <- if (is.character(mode)) {
    switch(mode,
           "reflect"  = 0L,
           "constant" = 1L,
           "nearest"  = 2L,
           "mirror"   = 3L,
           "wrap"     = 4L,
           stop(sprintf("gaussian: unknown mode '%s'", mode),
                call. = FALSE))
  } else as.integer(mode)[1L]
  c4a_pp_gaussian_transform(X, sigma = sigma, order = order,
                            mode = mode_int, cval = cval,
                            truncate = truncate)
}

#' Convert reflectance to absorbance `-log10(R)`.
#'
#' @param X numeric matrix of reflectance values.
#' @param is_percent logical(1). TRUE for 0-100 scale, FALSE for 0-1.
#' @param epsilon numeric(1). Floor for clamping `R` before the log.
#' @param clip_negative logical(1). Clamp negatives to `epsilon`.
#' @return numeric matrix.
#' @export
to_absorbance <- function(X, is_percent = FALSE, epsilon = 1e-10,
                          clip_negative = TRUE) {
  c4a_pp_to_absorbance_transform(X, is_percent = is_percent,
                                 epsilon = epsilon,
                                 clip_negative = clip_negative)
}

#' Convert absorbance to reflectance.
#'
#' @param X numeric matrix of absorbance values.
#' @param is_percent logical(1). TRUE to return values on a 0-100 scale.
#' @return numeric matrix.
#' @export
from_absorbance <- function(X, is_percent = FALSE) {
  c4a_pp_from_absorbance_transform(X, is_percent = is_percent)
}

#' Convert percent values to fractions.
#'
#' @param X numeric matrix.
#' @return numeric matrix.
#' @export
percent_to_fraction <- function(X) {
  c4a_pp_pct_to_frac_transform(X)
}

#' Convert fraction values to percent.
#'
#' @param X numeric matrix.
#' @return numeric matrix.
#' @export
fraction_to_percent <- function(X) {
  c4a_pp_frac_to_pct_transform(X)
}

#' Kubelka-Munk transform `(1-R)^2 / (2*R)`.
#'
#' @param X numeric matrix.
#' @param is_percent logical(1).
#' @param epsilon numeric(1).
#' @return numeric matrix.
#' @export
kubelka_munk <- function(X, is_percent = FALSE, epsilon = 1e-10) {
  c4a_pp_kubelka_munk_transform(X, is_percent = is_percent,
                                epsilon = epsilon)
}
