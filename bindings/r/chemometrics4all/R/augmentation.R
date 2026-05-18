# SPDX-License-Identifier: CECILL-2.1

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
  family_int <- if (is.character(family)) {
    switch(family,
           "haar"  = 0L,
           "db4"   = 1L,
           "sym4"  = 2L,
           "coif1" = 3L,
           stop(sprintf("wavelet_denoise: unknown family '%s'", family),
                call. = FALSE))
  } else as.integer(family)[1L]

  boundary_int <- if (is.character(boundary)) {
    switch(boundary,
           "periodization" = 0L,
           "symmetric"     = 1L,
           "zero"          = 2L,
           stop(sprintf("wavelet_denoise: unknown boundary '%s'", boundary),
                call. = FALSE))
  } else as.integer(boundary)[1L]

  thr_int <- if (is.character(threshold_mode)) {
    switch(threshold_mode,
           "soft" = 0L,
           "hard" = 1L,
           stop(sprintf("wavelet_denoise: unknown threshold_mode '%s'",
                        threshold_mode), call. = FALSE))
  } else as.integer(threshold_mode)[1L]

  noise_int <- if (is.character(noise_estimator)) {
    switch(noise_estimator,
           "median" = 0L,
           "std"    = 1L,
           stop(sprintf("wavelet_denoise: unknown noise_estimator '%s'",
                        noise_estimator), call. = FALSE))
  } else as.integer(noise_estimator)[1L]

  c4a_pp_wavelet_denoise_transform(X, family = family_int,
                                   boundary = boundary_int,
                                   level = level,
                                   threshold_mode = thr_int,
                                   noise_estimator = noise_int)
}
