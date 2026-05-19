# SPDX-License-Identifier: CECILL-2.1
#
# Tier-1 .Call wrappers. Each function takes plain R vectors / matrices,
# performs minimal validation, and dispatches into the chemometrics4all C
# ABI through a single `r_*` symbol implemented in `src/r_methods.c`.
#
# Conventions:
#   - All numeric matrix inputs are coerced to `double` (storage.mode
#     "double"). R matrices are column-major; the C side reads the
#     storage layout via `c4a_matrix_view_init_colmajor` so no transpose
#     is necessary.
#   - Outputs are returned as newly-allocated R matrices in column-major
#     layout matching libc4a's row-major outputs row-for-row (see
#     `r_methods.c` for the copy logic).
#   - On error the C side calls `Rf_error()` with the libc4a error string,
#     so callers can `tryCatch` if needed.

.c4a_check_matrix <- function(X, name = "X") {
  if (!is.matrix(X) || !is.numeric(X)) {
    stop(sprintf("%s must be a numeric matrix", name), call. = FALSE)
  }
  storage.mode(X) <- "double"
  X
}

.c4a_check_numeric_vector <- function(x, name) {
  if (!is.numeric(x)) {
    stop(sprintf("%s must be numeric", name), call. = FALSE)
  }
  as.double(x)
}

#' Apply Standard Normal Variate (SNV).
#'
#' Stateless preprocessing — row-wise centring and scaling.
#'
#' @param X numeric matrix (n_samples × n_features).
#' @param with_mean logical(1). Subtract the row mean.
#' @param with_std logical(1). Divide by the row standard deviation.
#' @param ddof integer(1). Delta degrees of freedom for the std estimator.
#' @return numeric matrix with the same shape as `X`.
#' @export
c4a_pp_snv_transform <- function(X, with_mean = TRUE, with_std = TRUE,
                                 ddof = 0L) {
  X <- .c4a_check_matrix(X, "X")
  .Call(C_c4a_pp_snv_transform, X,
        as.logical(with_mean)[1L], as.logical(with_std)[1L],
        as.integer(ddof)[1L])
}

#' Apply Local SNV (sliding-window SNV).
#'
#' @param X numeric matrix.
#' @param window integer(1). Odd window size.
#' @param pad_mode integer(1). 0=reflect, 1=edge, 2=constant.
#' @param constant_value numeric(1). Padding value for constant mode.
#' @return numeric matrix with the same shape as `X`.
#' @export
c4a_pp_lsnv_transform <- function(X, window = 11L, pad_mode = 0L,
                                  constant_value = 0.0) {
  X <- .c4a_check_matrix(X, "X")
  .Call(C_c4a_pp_lsnv_transform, X,
        as.integer(window)[1L], as.integer(pad_mode)[1L],
        as.double(constant_value)[1L])
}

#' Apply Robust Normal Variate (RNV).
#'
#' @param X numeric matrix.
#' @param with_center logical(1).
#' @param with_scale logical(1).
#' @param k numeric(1). MAD consistency factor.
#' @return numeric matrix with the same shape as `X`.
#' @export
c4a_pp_rnv_transform <- function(X, with_center = TRUE, with_scale = TRUE,
                                 k = 1.4826) {
  X <- .c4a_check_matrix(X, "X")
  .Call(C_c4a_pp_rnv_transform, X,
        as.logical(with_center)[1L], as.logical(with_scale)[1L],
        as.double(k)[1L])
}

#' Apply row-wise area normalization.
#'
#' @param X numeric matrix.
#' @param method integer(1). 0=sum, 1=abs_sum, 2=trapz.
#' @return numeric matrix with the same shape as `X`.
#' @export
c4a_pp_area_transform <- function(X, method = 0L) {
  X <- .c4a_check_matrix(X, "X")
  .Call(C_c4a_pp_area_transform, X, as.integer(method)[1L])
}

#' Apply column-wise normalization.
#'
#' @param X numeric matrix.
#' @param feature_min numeric(1).
#' @param feature_max numeric(1).
#' @return numeric matrix with the same shape as `X`.
#' @export
c4a_pp_normalize_transform <- function(X, feature_min = -1.0,
                                       feature_max = 1.0) {
  X <- .c4a_check_matrix(X, "X")
  .Call(C_c4a_pp_normalize_transform, X,
        as.double(feature_min)[1L], as.double(feature_max)[1L])
}

#' Apply column-wise min-max scaling to [0, 1].
#'
#' @param X numeric matrix.
#' @return numeric matrix with the same shape as `X`.
#' @export
c4a_pp_simple_scale_transform <- function(X) {
  X <- .c4a_check_matrix(X, "X")
  .Call(C_c4a_pp_simple_scale_transform, X)
}

#' Apply Multiplicative Scatter Correction (MSC).
#'
#' Stateful — fit on training matrix `X_fit`, then transform `X` with the
#' learned per-column slope/intercept.
#'
#' @param X_fit numeric matrix used for fitting (training set).
#' @param X numeric matrix to transform (may equal `X_fit`).
#' @return numeric matrix with the same shape as `X`.
#' @export
c4a_pp_msc_fit_transform <- function(X_fit, X) {
  X_fit <- .c4a_check_matrix(X_fit, "X_fit")
  X <- .c4a_check_matrix(X, "X")
  if (ncol(X_fit) != ncol(X)) {
    stop("X_fit and X must have the same number of columns", call. = FALSE)
  }
  .Call(C_c4a_pp_msc_fit_transform, X_fit, X)
}

#' Fit MSC and apply inverse transform.
#'
#' @param X_fit numeric matrix used for fitting.
#' @param X numeric matrix in MSC-corrected space.
#' @return numeric matrix with the same shape as `X`.
#' @export
c4a_pp_msc_fit_inverse_transform <- function(X_fit, X) {
  X_fit <- .c4a_check_matrix(X_fit, "X_fit")
  X <- .c4a_check_matrix(X, "X")
  if (ncol(X_fit) != ncol(X)) {
    stop("X_fit and X must have the same number of columns", call. = FALSE)
  }
  .Call(C_c4a_pp_msc_fit_inverse_transform, X_fit, X)
}

#' Apply Extended Multiplicative Scatter Correction (EMSC).
#'
#' Stateful — fit on `X_fit`, then transform `X`.
#'
#' @param X_fit numeric matrix used for fitting.
#' @param X numeric matrix to transform.
#' @param degree integer(1). Polynomial degree (>= 1).
#' @return numeric matrix with the same shape as `X`.
#' @export
c4a_pp_emsc_fit_transform <- function(X_fit, X, degree = 1L) {
  X_fit <- .c4a_check_matrix(X_fit, "X_fit")
  X <- .c4a_check_matrix(X, "X")
  if (ncol(X_fit) != ncol(X)) {
    stop("X_fit and X must have the same number of columns", call. = FALSE)
  }
  .Call(C_c4a_pp_emsc_fit_transform, X_fit, X, as.integer(degree)[1L])
}

#' Fit baseline centering and apply transform.
#'
#' @param X_fit numeric matrix used for fitting.
#' @param X numeric matrix to transform.
#' @return numeric matrix with the same shape as `X`.
#' @export
c4a_pp_baseline_fit_transform <- function(X_fit, X) {
  X_fit <- .c4a_check_matrix(X_fit, "X_fit")
  X <- .c4a_check_matrix(X, "X")
  if (ncol(X_fit) != ncol(X)) {
    stop("X_fit and X must have the same number of columns", call. = FALSE)
  }
  .Call(C_c4a_pp_baseline_fit_transform, X_fit, X)
}

#' Fit baseline centering and apply inverse transform.
#'
#' @param X_fit numeric matrix used for fitting.
#' @param X numeric matrix in centered space.
#' @return numeric matrix with the same shape as `X`.
#' @export
c4a_pp_baseline_fit_inverse_transform <- function(X_fit, X) {
  X_fit <- .c4a_check_matrix(X_fit, "X_fit")
  X <- .c4a_check_matrix(X, "X")
  if (ncol(X_fit) != ncol(X)) {
    stop("X_fit and X must have the same number of columns", call. = FALSE)
  }
  .Call(C_c4a_pp_baseline_fit_inverse_transform, X_fit, X)
}

#' Apply Savitzky-Golay smoothing/derivative.
#'
#' @param X numeric matrix.
#' @param window_length integer(1). Must be odd, >= polyorder + 1.
#' @param polyorder integer(1).
#' @param deriv integer(1). Derivative order (0 = smoothing).
#' @param delta numeric(1). Sample spacing.
#' @param mode integer(1). Boundary mode: 0=mirror, 1=constant, 2=nearest,
#'   3=wrap, 4=interp.
#' @param cval numeric(1). Constant value for mode=constant.
#' @return numeric matrix with the same shape as `X`.
#' @export
c4a_pp_savgol_transform <- function(X, window_length, polyorder, deriv = 0L,
                                    delta = 1.0, mode = 0L, cval = 0.0) {
  X <- .c4a_check_matrix(X, "X")
  .Call(C_c4a_pp_savgol_transform, X,
        as.integer(window_length)[1L], as.integer(polyorder)[1L],
        as.integer(deriv)[1L], as.double(delta)[1L],
        as.integer(mode)[1L], as.double(cval)[1L])
}

#' Apply first derivative via np.gradient axis=1.
#'
#' @param X numeric matrix.
#' @param delta numeric(1). Sample spacing.
#' @param edge_order integer(1). 1 or 2.
#' @return numeric matrix with the same shape as `X`.
#' @export
c4a_pp_first_derivative_transform <- function(X, delta = 1.0, edge_order = 2L) {
  X <- .c4a_check_matrix(X, "X")
  .Call(C_c4a_pp_first_derivative_transform, X,
        as.double(delta)[1L], as.integer(edge_order)[1L])
}

#' Apply second derivative (two passes of `np.gradient`).
#'
#' @param X numeric matrix.
#' @param delta numeric(1). Sample spacing.
#' @param edge_order integer(1). 1 or 2.
#' @return numeric matrix with the same shape as `X`.
#' @export
c4a_pp_second_derivative_transform <- function(X, delta = 1.0, edge_order = 2L) {
  X <- .c4a_check_matrix(X, "X")
  .Call(C_c4a_pp_second_derivative_transform, X,
        as.double(delta)[1L], as.integer(edge_order)[1L])
}

#' Apply Norris-Williams derivative.
#'
#' @param X numeric matrix.
#' @param segment integer(1).
#' @param gap integer(1).
#' @param derivative_order integer(1). 1 or 2.
#' @param delta numeric(1).
#' @return numeric matrix with the same shape as `X`.
#' @export
c4a_pp_norris_williams_transform <- function(X, segment = 5L, gap = 5L,
                                             derivative_order = 1L,
                                             delta = 1.0) {
  X <- .c4a_check_matrix(X, "X")
  .Call(C_c4a_pp_norris_williams_transform, X,
        as.integer(segment)[1L], as.integer(gap)[1L],
        as.integer(derivative_order)[1L], as.double(delta)[1L])
}

#' Apply Gaussian 1-D filter along rows.
#'
#' @param X numeric matrix.
#' @param sigma numeric(1).
#' @param order integer(1).
#' @param mode integer(1). 0=reflect, 1=constant, 2=nearest, 3=mirror, 4=wrap.
#' @param cval numeric(1).
#' @param truncate numeric(1).
#' @return numeric matrix with the same shape as `X`.
#' @export
c4a_pp_gaussian_transform <- function(X, sigma = 1.0, order = 0L,
                                      mode = 0L, cval = 0.0,
                                      truncate = 4.0) {
  X <- .c4a_check_matrix(X, "X")
  .Call(C_c4a_pp_gaussian_transform, X,
        as.double(sigma)[1L], as.integer(order)[1L],
        as.integer(mode)[1L], as.double(cval)[1L],
        as.double(truncate)[1L])
}

#' Convert reflectance to absorbance `-log10(R)`.
#'
#' @param X numeric matrix.
#' @param is_percent logical(1). TRUE if `X` is in 0-100, FALSE for 0-1.
#' @param epsilon numeric(1). Floor for clamping `R` before taking log.
#' @param clip_negative logical(1). Clamp negative reflectance to epsilon.
#' @return numeric matrix with the same shape as `X`.
#' @export
c4a_pp_to_absorbance_transform <- function(X, is_percent = FALSE,
                                           epsilon = 1e-10,
                                           clip_negative = TRUE) {
  X <- .c4a_check_matrix(X, "X")
  .Call(C_c4a_pp_to_absorbance_transform, X,
        as.logical(is_percent)[1L], as.double(epsilon)[1L],
        as.logical(clip_negative)[1L])
}

#' Convert absorbance to reflectance.
#'
#' @param X numeric matrix.
#' @param is_percent logical(1). TRUE to return 0-100 values.
#' @return numeric matrix with the same shape as `X`.
#' @export
c4a_pp_from_absorbance_transform <- function(X, is_percent = FALSE) {
  X <- .c4a_check_matrix(X, "X")
  .Call(C_c4a_pp_from_absorbance_transform, X,
        as.logical(is_percent)[1L])
}

#' Convert percent values to fractions.
#'
#' @param X numeric matrix.
#' @return numeric matrix with the same shape as `X`.
#' @export
c4a_pp_pct_to_frac_transform <- function(X) {
  X <- .c4a_check_matrix(X, "X")
  .Call(C_c4a_pp_pct_to_frac_transform, X)
}

#' Convert fraction values to percent.
#'
#' @param X numeric matrix.
#' @return numeric matrix with the same shape as `X`.
#' @export
c4a_pp_frac_to_pct_transform <- function(X) {
  X <- .c4a_check_matrix(X, "X")
  .Call(C_c4a_pp_frac_to_pct_transform, X)
}

#' Apply Kubelka-Munk transform `(1-R)^2 / (2*R)`.
#'
#' @param X numeric matrix.
#' @param is_percent logical(1).
#' @param epsilon numeric(1). Floor for clamping `R`.
#' @return numeric matrix with the same shape as `X`.
#' @export
c4a_pp_kubelka_munk_transform <- function(X, is_percent = FALSE,
                                          epsilon = 1e-10) {
  X <- .c4a_check_matrix(X, "X")
  .Call(C_c4a_pp_kubelka_munk_transform, X,
        as.logical(is_percent)[1L], as.double(epsilon)[1L])
}

#' Polynomial detrend per row.
#'
#' @param X numeric matrix.
#' @param polyorder integer(1).
#' @return numeric matrix with the same shape as `X`.
#' @export
c4a_pp_detrend_transform <- function(X, polyorder = 1L) {
  X <- .c4a_check_matrix(X, "X")
  .Call(C_c4a_pp_detrend_transform, X, as.integer(polyorder)[1L])
}

#' Apply Asymmetric Least Squares baseline correction.
#'
#' @param X numeric matrix.
#' @param lam numeric(1). Smoothing penalty.
#' @param p numeric(1). Asymmetry (0 < p < 1).
#' @param max_iter integer(1).
#' @param tol numeric(1). Relative L2 weight tolerance.
#' @return numeric matrix with the same shape as `X`.
#' @export
c4a_pp_asls_transform <- function(X, lam = 1e6, p = 1e-2,
                                  max_iter = 50L, tol = 1e-3) {
  X <- .c4a_check_matrix(X, "X")
  .Call(C_c4a_pp_asls_transform, X,
        as.double(lam)[1L], as.double(p)[1L],
        as.integer(max_iter)[1L], as.double(tol)[1L])
}

#' Apply Adaptive iteratively reweighted PLS baseline (AirPLS).
#'
#' @param X numeric matrix.
#' @param lam numeric(1). Smoothing penalty.
#' @param max_iter integer(1).
#' @param tol numeric(1).
#' @return numeric matrix with the same shape as `X`.
#' @export
c4a_pp_airpls_transform <- function(X, lam = 1e6,
                                    max_iter = 50L, tol = 1e-3) {
  X <- .c4a_check_matrix(X, "X")
  .Call(C_c4a_pp_airpls_transform, X,
        as.double(lam)[1L], as.integer(max_iter)[1L], as.double(tol)[1L])
}

#' Apply ArPLS baseline correction.
#'
#' @param X numeric matrix.
#' @param lam numeric(1).
#' @param max_iter integer(1).
#' @param tol numeric(1).
#' @return numeric matrix with the same shape as `X`.
#' @export
c4a_pp_arpls_transform <- function(X, lam = 1e5,
                                   max_iter = 50L, tol = 1e-3) {
  X <- .c4a_check_matrix(X, "X")
  .Call(C_c4a_pp_arpls_transform, X,
        as.double(lam)[1L], as.integer(max_iter)[1L], as.double(tol)[1L])
}

#' Apply ModPoly baseline correction.
#'
#' @param X numeric matrix.
#' @param polyorder integer(1).
#' @param max_iter integer(1).
#' @param tol numeric(1).
#' @return numeric matrix with the same shape as `X`.
#' @export
c4a_pp_modpoly_transform <- function(X, polyorder = 2L,
                                     max_iter = 250L, tol = 1e-3) {
  X <- .c4a_check_matrix(X, "X")
  .Call(C_c4a_pp_modpoly_transform, X,
        as.integer(polyorder)[1L], as.integer(max_iter)[1L],
        as.double(tol)[1L])
}

#' Apply IModPoly baseline correction.
#'
#' @param X numeric matrix.
#' @param polyorder integer(1).
#' @param max_iter integer(1).
#' @param tol numeric(1).
#' @return numeric matrix with the same shape as `X`.
#' @export
c4a_pp_imodpoly_transform <- function(X, polyorder = 2L,
                                      max_iter = 250L, tol = 1e-3) {
  X <- .c4a_check_matrix(X, "X")
  .Call(C_c4a_pp_imodpoly_transform, X,
        as.integer(polyorder)[1L], as.integer(max_iter)[1L],
        as.double(tol)[1L])
}

#' Apply SNIP baseline correction.
#'
#' @param X numeric matrix.
#' @param max_half_window integer(1).
#' @return numeric matrix with the same shape as `X`.
#' @export
c4a_pp_snip_transform <- function(X, max_half_window = 20L) {
  X <- .c4a_check_matrix(X, "X")
  .Call(C_c4a_pp_snip_transform, X, as.integer(max_half_window)[1L])
}

#' Apply rolling-ball baseline correction.
#'
#' @param X numeric matrix.
#' @param half_window integer(1).
#' @param smooth_half_window integer(1).
#' @return numeric matrix with the same shape as `X`.
#' @export
c4a_pp_rolling_ball_transform <- function(X, half_window = 20L,
                                          smooth_half_window = 0L) {
  X <- .c4a_check_matrix(X, "X")
  .Call(C_c4a_pp_rolling_ball_transform, X,
        as.integer(half_window)[1L], as.integer(smooth_half_window)[1L])
}

#' Apply IAsLS baseline correction.
#'
#' @param X numeric matrix.
#' @param lam numeric(1).
#' @param p numeric(1).
#' @param lam_1 numeric(1).
#' @param polyorder integer(1).
#' @param diff_order integer(1).
#' @param max_iter integer(1).
#' @param tol numeric(1).
#' @return numeric matrix with the same shape as `X`.
#' @export
c4a_pp_iasls_transform <- function(X, lam = 1e6, p = 1e-2,
                                   lam_1 = 1e-4, polyorder = 2L,
                                   diff_order = 2L, max_iter = 50L,
                                   tol = 1e-3) {
  X <- .c4a_check_matrix(X, "X")
  .Call(C_c4a_pp_iasls_transform, X,
        as.double(lam)[1L], as.double(p)[1L], as.double(lam_1)[1L],
        as.integer(polyorder)[1L], as.integer(diff_order)[1L],
        as.integer(max_iter)[1L],
        as.double(tol)[1L])
}

#' Apply BEADS baseline correction.
#'
#' @param X numeric matrix.
#' @param lam_0 numeric(1).
#' @param lam_1 numeric(1).
#' @param lam_2 numeric(1).
#' @param max_iter integer(1).
#' @param tol numeric(1).
#' @return numeric matrix with the same shape as `X`.
#' @export
c4a_pp_beads_transform <- function(X, lam_0 = 100.0, lam_1 = 0.5,
                                   lam_2 = 0.5, max_iter = 50L,
                                   tol = 1e-3) {
  X <- .c4a_check_matrix(X, "X")
  .Call(C_c4a_pp_beads_transform, X,
        as.double(lam_0)[1L], as.double(lam_1)[1L],
        as.double(lam_2)[1L], as.integer(max_iter)[1L],
        as.double(tol)[1L])
}

#' Multi-level wavelet denoising.
#'
#' @param X numeric matrix.
#' @param family integer(1). 0=haar, 1=db4, 2=sym4, 3=coif1.
#' @param boundary integer(1). 0=periodization, 1=symmetric, 2=zero.
#' @param level integer(1). Decomposition level.
#' @param threshold_mode integer(1). 0=soft, 1=hard.
#' @param noise_estimator integer(1). 0=median, 1=std.
#' @return numeric matrix with the same shape as `X`.
#' @export
c4a_pp_wavelet_denoise_transform <- function(X, family, boundary, level,
                                             threshold_mode, noise_estimator) {
  X <- .c4a_check_matrix(X, "X")
  .Call(C_c4a_pp_wavelet_denoise_transform, X,
        as.integer(family)[1L], as.integer(boundary)[1L],
        as.integer(level)[1L], as.integer(threshold_mode)[1L],
        as.integer(noise_estimator)[1L])
}

#' Single-level wavelet transform.
#'
#' @param X numeric matrix.
#' @param family integer(1). 0=haar, 1=db4, 2=sym4, 3=coif1.
#' @param boundary integer(1). 0=periodization, 1=symmetric, 2=zero.
#' @return numeric matrix containing approximation/detail coefficients.
#' @export
c4a_pp_wavelet_transform <- function(X, family = 0L, boundary = 0L) {
  X <- .c4a_check_matrix(X, "X")
  .Call(C_c4a_pp_wavelet_transform, X,
        as.integer(family)[1L], as.integer(boundary)[1L])
}

#' Haar wavelet transform.
#'
#' @param X numeric matrix.
#' @return numeric matrix containing approximation/detail coefficients.
#' @export
c4a_pp_haar_transform <- function(X) {
  X <- .c4a_check_matrix(X, "X")
  .Call(C_c4a_pp_haar_transform, X)
}

#' Wavelet summary features.
#'
#' @param X numeric matrix.
#' @param family integer(1). 0=haar, 1=db4, 2=sym4, 3=coif1.
#' @param boundary integer(1). 0=periodization, 1=symmetric, 2=zero.
#' @param max_level integer(1).
#' @param entropy integer(1). 0=energy, 1=histogram.
#' @return numeric matrix of wavelet-domain features.
#' @export
c4a_pp_wavelet_features_transform <- function(X, family = 0L,
                                              boundary = 0L,
                                              max_level = 2L,
                                              entropy = 0L) {
  X <- .c4a_check_matrix(X, "X")
  .Call(C_c4a_pp_wavelet_features_transform, X,
        as.integer(family)[1L], as.integer(boundary)[1L],
        as.integer(max_level)[1L], as.integer(entropy)[1L])
}

#' Fit OSC and transform X.
#'
#' @param X numeric matrix.
#' @param y numeric response vector.
#' @param n_components integer(1).
#' @param scale logical(1).
#' @return numeric matrix with the same shape as `X`.
#' @export
c4a_pp_osc_fit_transform <- function(X, y, n_components = 1L,
                                     scale = TRUE) {
  X <- .c4a_check_matrix(X, "X")
  y <- .c4a_check_numeric_vector(y, "y")
  if (length(y) != nrow(X)) {
    stop("y length must match nrow(X)", call. = FALSE)
  }
  .Call(C_c4a_pp_osc_fit_transform, X, y,
        as.integer(n_components)[1L], as.logical(scale)[1L])
}

#' Fit EPO and transform X with the disturbance vector.
#'
#' @param X numeric matrix.
#' @param d numeric disturbance vector.
#' @param scale logical(1).
#' @return numeric matrix with the same shape as `X`.
#' @export
c4a_pp_epo_fit_transform <- function(X, d, scale = TRUE) {
  X <- .c4a_check_matrix(X, "X")
  d <- .c4a_check_numeric_vector(d, "d")
  if (length(d) != nrow(X)) {
    stop("d length must match nrow(X)", call. = FALSE)
  }
  .Call(C_c4a_pp_epo_fit_transform, X, d, as.logical(scale)[1L])
}

#' Fit EPO and apply inverse transform.
#'
#' @param X numeric matrix.
#' @param d numeric disturbance vector.
#' @param scale logical(1).
#' @return numeric matrix with the same shape as `X`.
#' @export
c4a_pp_epo_fit_inverse_transform <- function(X, d, scale = TRUE) {
  X <- .c4a_check_matrix(X, "X")
  d <- .c4a_check_numeric_vector(d, "d")
  if (length(d) != nrow(X)) {
    stop("d length must match nrow(X)", call. = FALSE)
  }
  .Call(C_c4a_pp_epo_fit_inverse_transform, X, d, as.logical(scale)[1L])
}

#' Fit FlexiblePCA and transform X.
#'
#' @param X numeric matrix.
#' @param n_components numeric(1).
#' @return numeric score matrix.
#' @export
c4a_pp_flex_pca_fit_transform <- function(X, n_components = 5.0) {
  X <- .c4a_check_matrix(X, "X")
  .Call(C_c4a_pp_flex_pca_fit_transform, X,
        as.double(n_components)[1L])
}

#' Fit FlexibleSVD and transform X.
#'
#' @param X numeric matrix.
#' @param n_components numeric(1).
#' @return numeric score matrix.
#' @export
c4a_pp_flex_svd_fit_transform <- function(X, n_components = 5.0) {
  X <- .c4a_check_matrix(X, "X")
  .Call(C_c4a_pp_flex_svd_fit_transform, X,
        as.double(n_components)[1L])
}

#' Apply static fractional convolutional kernels.
#'
#' @param X numeric matrix.
#' @param kernel_size integer(1).
#' @param filter_orders numeric vector.
#' @param filter_scales numeric vector.
#' @return numeric matrix with one convolved block per kernel.
#' @export
c4a_pp_fck_static_transform <- function(X, kernel_size = 5L,
                                        filter_orders = c(0.5, 1.0),
                                        filter_scales = c(1.0, 2.0)) {
  X <- .c4a_check_matrix(X, "X")
  filter_orders <- .c4a_check_numeric_vector(filter_orders, "filter_orders")
  filter_scales <- .c4a_check_numeric_vector(filter_scales, "filter_scales")
  .Call(C_c4a_pp_fck_static_transform, X, as.integer(kernel_size)[1L],
        filter_orders, filter_scales)
}

#' Crop spectral columns.
#'
#' @param X numeric matrix.
#' @param start integer(1). Zero-based inclusive start column.
#' @param end integer(1). Zero-based exclusive end column.
#' @return numeric matrix with cropped columns.
#' @export
c4a_pp_crop_transform <- function(X, start, end) {
  X <- .c4a_check_matrix(X, "X")
  .Call(C_c4a_pp_crop_transform, X,
        as.integer(start)[1L], as.integer(end)[1L])
}

#' Resample by column count.
#'
#' @param X numeric matrix.
#' @param num_samples integer(1).
#' @return numeric matrix with `num_samples` columns.
#' @export
c4a_pp_resample_transform <- function(X, num_samples) {
  X <- .c4a_check_matrix(X, "X")
  .Call(C_c4a_pp_resample_transform, X, as.integer(num_samples)[1L])
}

#' Fit wavelength resampler and transform X.
#'
#' @param X numeric matrix.
#' @param source_wavelengths numeric vector.
#' @param target_wavelengths numeric vector.
#' @param method integer(1). 0=linear, 1=cubic, 2=nearest.
#' @param crop_min,crop_max numeric crop bounds.
#' @param use_crop logical(1).
#' @param fill_value numeric(1).
#' @param bounds_error logical(1).
#' @param extrapolate logical(1).
#' @return numeric matrix on the target wavelength grid.
#' @export
c4a_pp_resampler_fit_transform <- function(X, source_wavelengths,
                                           target_wavelengths,
                                           method = 0L,
                                           crop_min = 0.0,
                                           crop_max = 0.0,
                                           use_crop = FALSE,
                                           fill_value = 0.0,
                                           bounds_error = FALSE,
                                           extrapolate = FALSE) {
  X <- .c4a_check_matrix(X, "X")
  source_wavelengths <- .c4a_check_numeric_vector(source_wavelengths,
                                                  "source_wavelengths")
  target_wavelengths <- .c4a_check_numeric_vector(target_wavelengths,
                                                  "target_wavelengths")
  .Call(C_c4a_pp_resampler_fit_transform, X,
        source_wavelengths, target_wavelengths,
        as.integer(method)[1L], as.double(crop_min)[1L],
        as.double(crop_max)[1L], as.logical(use_crop)[1L],
        as.double(fill_value)[1L], as.logical(bounds_error)[1L],
        as.logical(extrapolate)[1L])
}

#' Fit integer K-bins discretizer and transform X.
#'
#' @param X numeric matrix.
#' @param n_bins integer(1).
#' @param strategy integer(1). 0=uniform, 1=quantile.
#' @return integer matrix.
#' @export
c4a_pp_kbins_disc_fit_transform <- function(X, n_bins = 5L,
                                            strategy = 0L) {
  X <- .c4a_check_matrix(X, "X")
  .Call(C_c4a_pp_kbins_disc_fit_transform, X,
        as.integer(n_bins)[1L], as.integer(strategy)[1L])
}

#' Apply edge-based range discretizer.
#'
#' @param X numeric matrix.
#' @param edges numeric vector of monotonic bin edges.
#' @return integer matrix.
#' @export
c4a_pp_range_disc_transform <- function(X, edges) {
  X <- .c4a_check_matrix(X, "X")
  edges <- .c4a_check_numeric_vector(edges, "edges")
  .Call(C_c4a_pp_range_disc_transform, X, edges)
}

#' Kennard-Stone train/test split.
#'
#' @param X numeric matrix.
#' @param test_size numeric(1). Fraction of samples in the test set.
#' @return list with `train_idx` and `test_idx` integer vectors (1-based).
#' @export
c4a_split_kennard_stone_split <- function(X, test_size) {
  X <- .c4a_check_matrix(X, "X")
  .Call(C_c4a_split_kennard_stone_split, X, as.double(test_size)[1L])
}

#' SPXY (Sample set Partitioning based on X and Y) split.
#'
#' @param X numeric matrix.
#' @param Y numeric matrix or vector (will be reshaped to single column).
#' @param test_size numeric(1).
#' @return list with `train_idx` and `test_idx` integer vectors (1-based).
#' @export
c4a_split_spxy_split <- function(X, Y, test_size) {
  X <- .c4a_check_matrix(X, "X")
  if (!is.matrix(Y)) {
    Y <- matrix(as.double(Y), ncol = 1L)
  }
  Y <- .c4a_check_matrix(Y, "Y")
  if (nrow(X) != nrow(Y)) {
    stop("X and Y must have the same number of rows", call. = FALSE)
  }
  .Call(C_c4a_split_spxy_split, X, Y, as.double(test_size)[1L])
}

#' K-bins stratified train/test split.
#'
#' @param Y numeric response matrix or vector.
#' @param test_size numeric(1).
#' @param seed numeric(1).
#' @param n_bins integer(1).
#' @param strategy integer(1). 0=uniform, 1=quantile.
#' @return list with `train_idx` and `test_idx` integer vectors (1-based).
#' @export
c4a_split_kbins_stratified_split <- function(Y, test_size = 0.25,
                                             seed = 0,
                                             n_bins = 5L,
                                             strategy = 0L) {
  if (!is.matrix(Y)) {
    Y <- matrix(as.double(Y), ncol = 1L)
  }
  Y <- .c4a_check_matrix(Y, "Y")
  .Call(C_c4a_split_kbins_stratified_split, Y,
        as.double(test_size)[1L], as.double(seed)[1L],
        as.integer(n_bins)[1L], as.integer(strategy)[1L])
}

#' Fit + apply YOutlierFilter.
#'
#' @param y numeric vector.
#' @param method integer(1). 0=IQR, 1=ZSCORE, 2=PERCENTILE, 3=MAD.
#' @param threshold numeric(1).
#' @param lower_pct numeric(1).
#' @param upper_pct numeric(1).
#' @return list with `mask` (logical), `n_samples`, `n_kept`, `n_excluded`,
#'   and `exclusion_rate`.
#' @export
c4a_filter_y_outlier_fit_apply <- function(y, method, threshold,
                                           lower_pct = 1.0,
                                           upper_pct = 99.0) {
  if (!is.numeric(y)) {
    stop("y must be a numeric vector", call. = FALSE)
  }
  .Call(C_c4a_filter_y_outlier_fit_apply, as.double(y),
        as.integer(method)[1L], as.double(threshold)[1L],
        as.double(lower_pct)[1L], as.double(upper_pct)[1L])
}

#' Fit + apply XOutlierFilter.
#'
#' @param X numeric matrix.
#' @param method integer(1). 0=mahalanobis, 1=robust mahalanobis,
#'   2=PCA residual, 3=PCA leverage, 4=isolation forest, 5=LOF.
#' @param use_threshold logical(1). Use fixed threshold instead of contamination.
#' @param threshold numeric(1).
#' @param n_components integer(1).
#' @param contamination numeric(1).
#' @param seed numeric(1).
#' @param n_estimators integer(1).
#' @param max_samples numeric(1).
#' @return list with `mask`, `n_samples`, `n_kept`, `n_excluded`,
#'   and `exclusion_rate`.
#' @export
c4a_filter_x_outlier_fit_apply <- function(X, method = 0L,
                                           use_threshold = FALSE,
                                           threshold = 0.0,
                                           n_components = min(5L, ncol(X)),
                                           contamination = 0.1,
                                           seed = 0,
                                           n_estimators = 100L,
                                           max_samples = 256) {
  X <- .c4a_check_matrix(X, "X")
  .Call(C_c4a_filter_x_outlier_fit_apply, X,
        as.integer(method)[1L], as.logical(use_threshold)[1L],
        as.double(threshold)[1L], as.integer(n_components)[1L],
        as.double(contamination)[1L], as.double(seed)[1L],
        as.integer(n_estimators)[1L], as.double(max_samples)[1L])
}
