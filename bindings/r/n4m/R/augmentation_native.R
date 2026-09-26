# SPDX-License-Identifier: CECILL-2.1

.n4m_native_augmentation_kinds <- c(
    gaussian_noise = 0L, multiplicative_noise = 1L, spike_noise = 2L,
    hetero_noise = 3L, linear_drift = 4L, path_length = 5L,
    band_perturb = 6L, band_mask = 7L, channel_dropout = 8L,
    gauss_jitter = 9L, unsharp_mask = 10L, local_clip = 11L,
    rotate_translate = 12L, random_x_op = 13L, scatter_sim_msc = 14L,
    dead_band = 15L, batch_effect = 16L, spline_smoothing = 17L,
    spline_x_perturb = 18L, spline_y_perturb = 19L,
    spline_x_simplify = 20L, spline_curve_simplify = 21L)
.n4m_native_augmentation_counts <- c(
    1L, 1L, 4L, 2L, 4L, 2L, 7L, 5L, 2L, 3L, 4L,
    3L, 2L, 3L, 4L, 6L, 4L, 0L, 4L, 2L, 2L, 2L)

#' Apply a seeded native X-only augmentation to training spectra
#'
#' This is a one-shot, train-only transformation. The C++ kernel returns X
#' with the same rows and columns. It does not transform Y, expose random
#' mixing partners, or produce a portable fitted state. Thus Mixup and
#' LocalMixup are deliberately unavailable. `params` follows the positional
#' order of the corresponding libn4m create function.
#'
#' @param kind One of the 22 supported native augmentation names.
#' @param X Finite numeric matrix of training spectra.
#' @param params Finite numeric vector of method-specific positional parameters.
#' @param seed Exact nonnegative integer seed, at most `2^53-1`.
#' @return A new numeric matrix of the same shape as `X`.
#' @export
n4m_augmentation_apply <- function(kind, X, params = numeric(), seed = 0) {
    if (!is.character(kind) || length(kind) != 1L || is.na(kind) ||
        !(kind %in% names(.n4m_native_augmentation_kinds)))
        stop("unknown native augmentation kind", call. = FALSE)
    code <- unname(.n4m_native_augmentation_kinds[[kind]])
    if (!is.matrix(X) || !is.numeric(X) || any(dim(X) < 1L) ||
        anyNA(X) || any(!is.finite(X)))
        stop("X must be a nonempty finite numeric matrix", call. = FALSE)
    if (!is.numeric(params) || length(params) !=
        .n4m_native_augmentation_counts[code + 1L] ||
        anyNA(params) || any(!is.finite(params)))
        stop("invalid native augmentation parameter vector", call. = FALSE)
    if (!is.numeric(seed) || length(seed) != 1L || is.na(seed) ||
        !is.finite(seed) || seed < 0 || seed > 2^53 - 1 || seed != floor(seed))
        stop("seed must be an exact nonnegative integer", call. = FALSE)
    storage.mode(X) <- "double"
    .Call("r_n4m_augmentation_run", as.integer(code), X,
          as.double(params), as.double(seed), PACKAGE = "n4m")
}
