# SPDX-License-Identifier: CECILL-2.1

.c4a_wavelet_family <- function(family, caller) {
  if (is.character(family)) {
    switch(family,
           "haar"  = 0L,
           "db4"   = 1L,
           "sym4"  = 2L,
           "coif1" = 3L,
           stop(sprintf("%s: unknown family '%s'", caller, family),
                call. = FALSE))
  } else {
    as.integer(family)[1L]
  }
}

.c4a_wavelet_boundary <- function(boundary, caller) {
  if (is.character(boundary)) {
    switch(boundary,
           "periodization" = 0L,
           "symmetric"     = 1L,
           "zero"          = 2L,
           stop(sprintf("%s: unknown boundary '%s'", caller, boundary),
                call. = FALSE))
  } else {
    as.integer(boundary)[1L]
  }
}

.c4a_wavelet_threshold <- function(threshold_mode, caller) {
  if (is.character(threshold_mode)) {
    switch(threshold_mode,
           "soft" = 0L,
           "hard" = 1L,
           stop(sprintf("%s: unknown threshold_mode '%s'", caller,
                        threshold_mode), call. = FALSE))
  } else {
    as.integer(threshold_mode)[1L]
  }
}

.c4a_wavelet_noise <- function(noise_estimator, caller) {
  if (is.character(noise_estimator)) {
    switch(noise_estimator,
           "median" = 0L,
           "std"    = 1L,
           stop(sprintf("%s: unknown noise_estimator '%s'", caller,
                        noise_estimator), call. = FALSE))
  } else {
    as.integer(noise_estimator)[1L]
  }
}

.c4a_wavelet_entropy <- function(entropy, caller) {
  if (is.character(entropy)) {
    switch(entropy,
           "energy"    = 0L,
           "histogram" = 1L,
           stop(sprintf("%s: unknown entropy '%s'", caller, entropy),
                call. = FALSE))
  } else {
    as.integer(entropy)[1L]
  }
}

#' Single-level wavelet transform.
#'
#' @param X numeric matrix.
#' @param family character(1) or integer(1). One of "haar", "db4", "sym4", "coif1".
#' @param boundary character(1) or integer(1). One of "periodization",
#'   "symmetric", "zero".
#' @return numeric matrix containing approximation/detail coefficients.
#' @export
wavelet <- function(X, family = "haar", boundary = "periodization") {
  c4a_pp_wavelet_transform(
    X,
    family = .c4a_wavelet_family(family, "wavelet"),
    boundary = .c4a_wavelet_boundary(boundary, "wavelet"))
}

#' Haar wavelet transform.
#'
#' @param X numeric matrix.
#' @return numeric matrix containing approximation/detail coefficients.
#' @export
haar <- function(X) {
  c4a_pp_haar_transform(X)
}

#' Multi-level wavelet denoising (VisuShrink).
#'
#' @param X numeric matrix.
#' @param family character(1) or integer(1). One of "haar", "db4", "sym4", "coif1".
#' @param boundary character(1) or integer(1). One of "periodization",
#'   "symmetric", "zero".
#' @param level integer(1). Decomposition level.
#' @param threshold_mode character(1) or integer(1). "soft" or "hard".
#' @param noise_estimator character(1) or integer(1). "median" or "std".
#' @return numeric matrix with the same shape as `X`.
#' @export
wavelet_denoise <- function(X,
                            family = "db4",
                            boundary = "periodization",
                            level = 3L,
                            threshold_mode = "soft",
                            noise_estimator = "median") {
  c4a_pp_wavelet_denoise_transform(X,
                                   family = .c4a_wavelet_family(family, "wavelet_denoise"),
                                   boundary = .c4a_wavelet_boundary(boundary, "wavelet_denoise"),
                                   level = level,
                                   threshold_mode = .c4a_wavelet_threshold(threshold_mode, "wavelet_denoise"),
                                   noise_estimator = .c4a_wavelet_noise(noise_estimator, "wavelet_denoise"))
}

#' Wavelet-domain summary features.
#'
#' @param X numeric matrix.
#' @param family character(1) or integer(1). One of "haar", "db4", "sym4", "coif1".
#' @param boundary character(1) or integer(1). One of "periodization",
#'   "symmetric", "zero".
#' @param max_level integer(1).
#' @param entropy character(1) or integer(1). One of "energy", "histogram".
#' @return numeric matrix of wavelet summary features.
#' @export
wavelet_features <- function(X, family = "haar",
                             boundary = "periodization",
                             max_level = 2L,
                             entropy = "energy") {
  c4a_pp_wavelet_features_transform(
    X,
    family = .c4a_wavelet_family(family, "wavelet_features"),
    boundary = .c4a_wavelet_boundary(boundary, "wavelet_features"),
    max_level = max_level,
    entropy = .c4a_wavelet_entropy(entropy, "wavelet_features"))
}
