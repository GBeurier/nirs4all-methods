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
