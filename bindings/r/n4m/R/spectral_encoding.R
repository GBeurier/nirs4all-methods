# SPDX-License-Identifier: CECILL-2.1

#' Fit a native local variance subspace encoder
#' @param X Training spectra, samples by channels.
#' @param width Local window width.
#' @param rank Maximum local rank (GCU: global rank).
#' @param overlap Fraction of window overlap, in [0, 1).
#' @param standardize Scale channels using training standard deviations.
#' @param snv Apply row-wise SNV before learning local subspaces.
#' @return Fitted native encoder, consumed by spectral_transform.
#' @export
lvse_fit <- function(X, width = 64L, rank = 4L, overlap = 0,
                     standardize = TRUE, snv = FALSE) {
    X <- as.matrix(X)
    storage.mode(X) <- "double"
    .Call("r_n4m_spectral_fit", X,
          as.double(c(0, width, rank, overlap, standardize, snv, 60, 0.001)),
          PACKAGE = "n4m")
}

#' Fit a native global nonnegative spectral encoder
#' @inheritParams lvse_fit
#' @param max_iter Coordinate descent iteration limit.
#' @param tol Projected-gradient relative tolerance.
#' @export
gcu_fit <- function(X, rank = 16L, max_iter = 60L, tol = 0.001) {
    X <- as.matrix(X)
    storage.mode(X) <- "double"
    .Call("r_n4m_spectral_fit", X, as.double(c(1, 64, rank, 0, 1, 0, max_iter, tol)),
          PACKAGE = "n4m")
}

#' Apply a fitted spectral encoder without updating its basis
#' @param encoder Fitted native LVSE or GCU handle.
#' @param X Query spectra, samples by channels.
#' @export
spectral_transform <- function(encoder, X) {
    X <- as.matrix(X)
    storage.mode(X) <- "double"
    .Call("r_n4m_spectral_transform", encoder, X, PACKAGE = "n4m")
}

#' Export an affine LVSE map (requires snv=FALSE)
#' @param encoder Fitted LVSE handle.
#' @return List with operator (outputs by channels) and offset (one row).
#' @export
spectral_export_affine <- function(encoder) {
    .Call("r_n4m_spectral_affine", encoder, PACKAGE = "n4m")
}
