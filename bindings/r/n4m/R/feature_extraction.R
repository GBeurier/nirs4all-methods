# SPDX-License-Identifier: CECILL-2.1

#' Orthogonal Signal Correction (OSC).
#'
#' @param X numeric matrix.
#' @param y numeric response vector.
#' @param n_components integer(1).
#' @param scale logical(1).
#' @return numeric matrix with the same shape as `X`.
#' @export
osc <- function(X, y, n_components = 1L, scale = TRUE) {
  n4m_pp_osc_fit_transform(X, y = y, n_components = n_components,
                           scale = scale)
}

#' External Parameter Orthogonalisation (EPO).
#'
#' @param X numeric matrix.
#' @param d numeric disturbance vector.
#' @param scale logical(1).
#' @return numeric matrix with the same shape as `X`.
#' @export
epo <- function(X, d, scale = TRUE) {
  n4m_pp_epo_fit_transform(X, d = d, scale = scale)
}

#' Inverse EPO transform.
#'
#' @param X numeric matrix.
#' @param d numeric disturbance vector used to fit EPO.
#' @param scale logical(1).
#' @return numeric matrix with the same shape as `X`.
#' @export
epo_inverse_transform <- function(X, d, scale = TRUE) {
  n4m_pp_epo_fit_inverse_transform(X, d = d, scale = scale)
}

#' Flexible PCA.
#'
#' @param X numeric matrix.
#' @param n_components numeric(1). Component count or explained-variance ratio.
#' @return numeric score matrix.
#' @export
flexible_pca <- function(X, n_components = 5.0) {
  n4m_pp_flex_pca_fit_transform(X, n_components = n_components)
}

#' Flexible SVD.
#'
#' @param X numeric matrix.
#' @param n_components numeric(1). Component count.
#' @return numeric score matrix.
#' @export
flexible_svd <- function(X, n_components = 5.0) {
  n4m_pp_flex_svd_fit_transform(X, n_components = n_components)
}

#' Static fractional convolutional kernels.
#'
#' @param X numeric matrix.
#' @param kernel_size integer(1).
#' @param filter_orders numeric vector.
#' @param filter_scales numeric vector.
#' @return numeric matrix with one convolved block per kernel.
#' @export
fck_static <- function(X, kernel_size = 5L,
                       filter_orders = c(0.5, 1.0),
                       filter_scales = c(1.0, 2.0)) {
  n4m_pp_fck_static_transform(X, kernel_size = kernel_size,
                              filter_orders = filter_orders,
                              filter_scales = filter_scales)
}
