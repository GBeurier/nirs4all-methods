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
