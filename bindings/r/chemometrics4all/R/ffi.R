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
