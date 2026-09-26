# SPDX-License-Identifier: CECILL-2.1

.n4m_sample_filter_kinds <- c(y_outlier = 0L, x_outlier = 1L,
    high_leverage = 2L, spectral_quality = 3L, composite = 4L)
.n4m_feature_filter_kinds <- c(variance = 0L, correlation = 1L)

.n4m_filter_kind <- function(kind, names) {
    if (!is.character(kind) || length(kind) != 1L || is.na(kind) ||
        !(kind %in% names(names))) stop("unknown native filter kind", call. = FALSE)
    unname(names[[kind]])
}

.n4m_filter_params <- function(integers, doubles, seed) {
    if (!is.numeric(integers) || anyNA(integers) ||
        any(!is.finite(integers)) || any(integers != floor(integers)) ||
        any(abs(integers) > 2^53 - 1))
        stop("filter integer parameters must be exact", call. = FALSE)
    if (!is.numeric(doubles) || anyNA(doubles) || any(!is.finite(doubles)))
        stop("filter double parameters must be finite", call. = FALSE)
    if (!is.numeric(seed) || length(seed) != 1L || is.na(seed) ||
        !is.finite(seed) || seed < 0 || seed > 2^53 - 1 || seed != floor(seed))
        stop("filter seed must be an exact nonnegative integer", call. = FALSE)
    list(as.double(integers), as.double(doubles), as.double(seed))
}

.n4m_filter_matrix <- function(X, label) {
    if (!is.matrix(X) || !is.numeric(X) || any(dim(X) < 1L))
        stop(sprintf("%s must be a nonempty numeric matrix", label), call. = FALSE)
    storage.mode(X) <- "double"
    X
}

#' Construct an opaque native row-mask filter
#'
#' The integer/double vectors follow the positional schema in `filter_role.h`.
#' `composite` accepts only owned `high_leverage` or `spectral_quality`
#' children through [n4m_sample_filter_add_child()]. This handle has no
#' portable fitted-state export.
#' @export
n4m_sample_filter_create <- function(kind, integers, doubles, seed = 0) {
    p <- .n4m_filter_params(integers, doubles, seed)
    .Call("r_n4m_sample_filter_create",
          .n4m_filter_kind(kind, .n4m_sample_filter_kinds),
          p[[1]], p[[2]], p[[3]], PACKAGE = "n4m")
}

#' Add an owned native child to a composite row filter
#' @export
n4m_sample_filter_add_child <- function(filter, kind, integers, doubles, seed = 0) {
    p <- .n4m_filter_params(integers, doubles, seed)
    .Call("r_n4m_sample_filter_add_child", filter,
          .n4m_filter_kind(kind, .n4m_sample_filter_kinds),
          p[[1]], p[[2]], p[[3]], PACKAGE = "n4m")
}

#' Fit an opaque native row-mask filter
#' @export
n4m_sample_filter_fit <- function(filter, X, Y = NULL) {
    .Call("r_n4m_sample_filter_fit", filter, .n4m_filter_matrix(X, "X"),
          if (is.null(Y)) NULL else .n4m_filter_matrix(Y, "Y"), PACKAGE = "n4m")
}

#' Apply a fitted native row-mask filter to aligned X/Y
#' @return List containing logical `mask` and numeric native `stats`.
#' @export
n4m_sample_filter_apply <- function(filter, X, Y = NULL) {
    .Call("r_n4m_sample_filter_apply", filter, .n4m_filter_matrix(X, "X"),
          if (is.null(Y)) NULL else .n4m_filter_matrix(Y, "Y"), PACKAGE = "n4m")
}

#' Construct an opaque native column selector
#' @export
n4m_feature_filter_create <- function(kind, threshold = 0, top_k = -1L) {
    if (!is.numeric(threshold) || length(threshold) != 1L || is.na(threshold) ||
        !is.finite(threshold)) stop("threshold must be finite", call. = FALSE)
    if (!is.numeric(top_k) || length(top_k) != 1L || is.na(top_k) ||
        !is.finite(top_k) || top_k != floor(top_k) ||
        top_k < -1 || top_k > .Machine$integer.max)
        stop("top_k must be an exact int32 >= -1", call. = FALSE)
    .Call("r_n4m_feature_filter_create",
          .n4m_filter_kind(kind, .n4m_feature_filter_kinds),
          as.double(threshold), as.integer(top_k), PACKAGE = "n4m")
}

#' Fit a native column selector; correlation requires one-column Y
#' @export
n4m_feature_filter_fit <- function(filter, X, Y = NULL) {
    .Call("r_n4m_feature_filter_fit", filter, .n4m_filter_matrix(X, "X"),
          if (is.null(Y)) NULL else .n4m_filter_matrix(Y, "Y"), PACKAGE = "n4m")
}

#' Return one-based selected native column indices in transform order
#' @export
n4m_feature_filter_indices <- function(filter) {
    .Call("r_n4m_feature_filter_indices", filter, PACKAGE = "n4m")
}

#' Transform held-out X with a fitted native column selector
#' @export
n4m_feature_filter_transform <- function(filter, X) {
    .Call("r_n4m_feature_filter_transform", filter, .n4m_filter_matrix(X, "X"),
          PACKAGE = "n4m")
}
