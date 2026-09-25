# SPDX-License-Identifier: CECILL-2.1

#' Standard Normal Variate transform.
#'
#' Applies libn4m's SNV operator row-wise to a numeric matrix.
#'
#' @param X Numeric matrix.
#' @param with_mean Logical; center each row before scaling.
#' @param with_std Logical; scale each row by its standard deviation.
#' @param ddof Integer degrees-of-freedom correction for the row standard
#'   deviation.
#' @return Numeric matrix with the same shape as `X`.
#' @export
snv_transform <- function(X, with_mean = TRUE, with_std = TRUE, ddof = 0L) {
    if (!is.matrix(X)) X <- as.matrix(X)
    storage.mode(X) <- "double"
    .Call("r_n4m_snv_transform",
          X,
          as.logical(with_mean),
          as.logical(with_std),
          as.integer(ddof),
          PACKAGE = "n4m")
}

#' Savitzky-Golay transform.
#'
#' Applies libn4m's Savitzky-Golay smoothing or derivative operator row-wise
#' to a numeric matrix.
#'
#' @param X Numeric matrix.
#' @param window_length Odd integer window length.
#' @param polyorder Polynomial order.
#' @param deriv Derivative order.
#' @param delta Sample spacing.
#' @param mode Boundary mode: `"mirror"`, `"constant"`, `"nearest"`, `"wrap"`,
#'   or `"interp"`.
#' @param cval Constant fill value used when `mode = "constant"`.
#' @return Numeric matrix with the same shape as `X`.
#' @export
savgol_transform <- function(X, window_length, polyorder = 3L, deriv = 0L,
                             delta = 1.0, mode = "mirror", cval = 0.0) {
    if (!is.matrix(X)) X <- as.matrix(X)
    storage.mode(X) <- "double"
    .Call("r_n4m_savgol_transform",
          X,
          as.integer(window_length),
          as.integer(polyorder),
          as.integer(deriv),
          as.numeric(delta),
          mode,
          as.numeric(cval),
          PACKAGE = "n4m")
}

#' Local Standard Normal Variate transform.
#' @param X Numeric matrix.
#' @param window Odd sliding window length.
#' @param pad_mode One of `"reflect"`, `"edge"`, or `"constant"`.
#' @param constant_value Padding value for constant mode.
#' @return Numeric matrix with the same shape as `X`.
#' @export
local_snv_transform <- function(X, window = 11L, pad_mode = "reflect",
                                constant_value = 0) {
    if (!is.matrix(X)) X <- as.matrix(X)
    storage.mode(X) <- "double"
    .Call("r_n4m_local_snv_transform", X, as.integer(window), pad_mode,
          as.numeric(constant_value), PACKAGE = "n4m")
}

#' Robust Standard Normal Variate transform.
#' @param X Numeric matrix.
#' @param with_center Center rows by their median.
#' @param with_scale Scale rows by robust dispersion.
#' @param k Robust scale factor, usually 1.4826.
#' @return Numeric matrix with the same shape as `X`.
#' @export
robust_snv_transform <- function(X, with_center = TRUE, with_scale = TRUE,
                                 k = 1.4826) {
    if (!is.matrix(X)) X <- as.matrix(X)
    storage.mode(X) <- "double"
    .Call("r_n4m_robust_snv_transform", X, as.logical(with_center),
          as.logical(with_scale), as.numeric(k), PACKAGE = "n4m")
}

#' Area normalization transform.
#' @param X Numeric matrix.
#' @param method One of `"sum"`, `"abs_sum"`, or `"trapz"`.
#' @return Numeric matrix with the same shape as `X`.
#' @export
area_normalization_transform <- function(X, method = "sum") {
    if (!is.matrix(X)) X <- as.matrix(X)
    storage.mode(X) <- "double"
    .Call("r_n4m_area_normalization_transform", X, method, PACKAGE = "n4m")
}

#' Polynomial detrend transform.
#' @param X Numeric matrix.
#' @param polyorder Non-negative baseline polynomial order.
#' @return Numeric matrix with the same shape as `X`.
#' @export
detrend_transform <- function(X, polyorder = 1L) {
    if (!is.matrix(X)) X <- as.matrix(X)
    storage.mode(X) <- "double"
    .Call("r_n4m_detrend_transform", X, as.integer(polyorder), PACKAGE = "n4m")
}

#' Fit Multiplicative Scatter Correction on training spectra.
#'
#' Returns only the native fitted reference spectrum. Save this vector and
#' pass it to [msc_transform()] for validation or future samples; do not fit
#' again on those samples.
#' @param X Finite numeric training matrix with at least two features.
#' @return Numeric reference spectrum, length `ncol(X)`.
#' @export
msc_fit <- function(X) {
    if (!is.matrix(X)) X <- as.matrix(X)
    storage.mode(X) <- "double"
    if (ncol(X) < 2L || anyNA(X) || any(!is.finite(X)))
        stop("X must be a finite matrix with at least two features", call. = FALSE)
    .Call("r_n4m_msc_fit", X, PACKAGE = "n4m")
}

#' Apply fitted Multiplicative Scatter Correction.
#' @param X Finite numeric matrix with the same feature order as training.
#' @param reference Reference spectrum from [msc_fit()].
#' @return Numeric matrix with the same shape as `X`.
#' @export
msc_transform <- function(X, reference) {
    if (!is.matrix(X)) X <- as.matrix(X)
    storage.mode(X) <- "double"
    if (!is.numeric(reference) || is.matrix(reference) ||
        length(reference) != ncol(X) || anyNA(reference) ||
        any(!is.finite(reference)) || anyNA(X) || any(!is.finite(X)))
        stop("X and fitted reference must be finite and feature-aligned", call. = FALSE)
    .Call("r_n4m_msc_transform", X, as.numeric(reference), PACKAGE = "n4m")
}

#' Kennard-Stone train/test split.
#'
#' Delegates to libn4m's Kennard-Stone splitter and returns train/test sample
#' indices. Indices are 1-based by default for idiomatic R usage; set
#' `zero_based = TRUE` for cross-language parity fixtures.
#'
#' @param X Numeric matrix.
#' @param test_size Fraction of samples assigned to the test set.
#' @param zero_based Logical; return 0-based indices when `TRUE`.
#' @return Named list with integer vectors `train` and `test`.
#' @export
kennard_stone_split <- function(X, test_size = 0.25, zero_based = FALSE) {
    if (!is.matrix(X)) X <- as.matrix(X)
    storage.mode(X) <- "double"
    .Call("r_n4m_kennard_stone_split",
          X,
          as.numeric(test_size),
          as.logical(zero_based),
          PACKAGE = "n4m")
}
