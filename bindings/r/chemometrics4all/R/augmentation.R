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

.c4a_aug_wavelengths <- function(X, wavelengths) {
  if (is.null(wavelengths)) {
    return(seq(900.0, 1700.0, length.out = ncol(X)))
  }
  wavelengths <- .c4a_check_numeric_vector(wavelengths, "wavelengths")
  if (length(wavelengths) != ncol(X)) {
    stop("wavelengths must have length ncol(X)", call. = FALSE)
  }
  wavelengths
}

.c4a_aug_apply <- function(name, X, params = numeric(), seed = 17,
                           wavelengths = NULL) {
  X <- .c4a_check_matrix(X, "X")
  wl <- if (is.null(wavelengths)) NULL else .c4a_aug_wavelengths(X, wavelengths)
  .Call(C_c4a_aug_apply, as.character(name)[1L], X, wl,
        as.double(seed)[1L], as.double(params))
}

#' Gaussian additive noise augmenter.
#' @export
aug_gaussian_noise <- function(X, sigma = 0.01, seed = 17) {
  .c4a_aug_apply("gaussian_noise", X, c(sigma), seed)
}

#' Multiplicative noise augmenter.
#' @export
aug_multiplicative_noise <- function(X, sigma_gain = 0.01, seed = 17) {
  .c4a_aug_apply("multiplicative_noise", X, c(sigma_gain), seed)
}

#' Spike noise augmenter.
#' @export
aug_spike_noise <- function(X, n_spikes_min = 1L, n_spikes_max = 3L,
                            amplitude_min = -0.1, amplitude_max = 0.1,
                            seed = 17) {
  .c4a_aug_apply("spike_noise", X,
                 c(n_spikes_min, n_spikes_max, amplitude_min, amplitude_max),
                 seed)
}

#' Heteroscedastic noise augmenter.
#' @export
aug_hetero_noise <- function(X, noise_base = 0.001,
                             noise_signal_dep = 0.01, seed = 17) {
  .c4a_aug_apply("hetero_noise", X, c(noise_base, noise_signal_dep), seed)
}

#' Linear baseline drift augmenter.
#' @export
aug_linear_drift <- function(X, offset_min = -0.05, offset_max = 0.05,
                             slope_min = -0.01, slope_max = 0.01,
                             seed = 17) {
  .c4a_aug_apply("linear_drift", X,
                 c(offset_min, offset_max, slope_min, slope_max), seed)
}

#' Polynomial baseline drift augmenter.
#' @export
aug_poly_drift <- function(X, degree = 2L, coeff_min = -0.01,
                           coeff_max = 0.01, seed = 17) {
  .c4a_aug_apply("poly_drift", X, c(degree, coeff_min, coeff_max), seed)
}

#' Path-length augmenter.
#' @export
aug_path_length <- function(X, path_length_std = 0.05,
                            min_path_length = 0.1, seed = 17) {
  .c4a_aug_apply("path_length", X, c(path_length_std, min_path_length), seed)
}

#' Wavelength shift augmenter.
#' @export
aug_wavelength_shift <- function(X, wavelengths = NULL, shift_lo = -1.0,
                                 shift_hi = 1.0, seed = 17) {
  X <- .c4a_check_matrix(X, "X")
  .c4a_aug_apply("wavelength_shift", X, c(shift_lo, shift_hi), seed,
                 .c4a_aug_wavelengths(X, wavelengths))
}

#' Wavelength stretch augmenter.
#' @export
aug_wavelength_stretch <- function(X, wavelengths = NULL, stretch_lo = 0.99,
                                   stretch_hi = 1.01, seed = 17) {
  X <- .c4a_check_matrix(X, "X")
  .c4a_aug_apply("wavelength_stretch", X, c(stretch_lo, stretch_hi), seed,
                 .c4a_aug_wavelengths(X, wavelengths))
}

#' Local wavelength warp augmenter.
#' @export
aug_local_warp <- function(X, wavelengths = NULL, n_control_points = 3L,
                           max_shift = 1.0, seed = 17) {
  X <- .c4a_check_matrix(X, "X")
  .c4a_aug_apply("local_warp", X, c(n_control_points, max_shift), seed,
                 .c4a_aug_wavelengths(X, wavelengths))
}

#' Band perturbation augmenter.
#' @export
aug_band_perturb <- function(X, n_bands = 3L, bw_lo = 5L, bw_hi = 15L,
                             gain_lo = 0.9, gain_hi = 1.1,
                             offset_lo = -0.01, offset_hi = 0.01,
                             seed = 17) {
  .c4a_aug_apply("band_perturb", X,
                 c(n_bands, bw_lo, bw_hi, gain_lo, gain_hi,
                   offset_lo, offset_hi), seed)
}

#' Band masking augmenter.
#' @export
aug_band_mask <- function(X, n_bands_lo = 1L, n_bands_hi = 3L,
                          bw_lo = 5L, bw_hi = 15L, mode = 0L,
                          seed = 17) {
  .c4a_aug_apply("band_mask", X,
                 c(n_bands_lo, n_bands_hi, bw_lo, bw_hi, mode), seed)
}

#' Channel dropout augmenter.
#' @export
aug_channel_dropout <- function(X, dropout_prob = 0.05, mode = 0L,
                                seed = 17) {
  .c4a_aug_apply("channel_dropout", X, c(dropout_prob, mode), seed)
}

#' Gaussian smoothing jitter augmenter.
#' @export
aug_gauss_jitter <- function(X, sigma_lo = 0.5, sigma_hi = 1.5,
                             kernel_width = 9L, seed = 17) {
  .c4a_aug_apply("gauss_jitter", X,
                 c(sigma_lo, sigma_hi, kernel_width), seed)
}

#' Unsharp spectral mask augmenter.
#' @export
aug_unsharp_mask <- function(X, amount_lo = 0.1, amount_hi = 0.5,
                             sigma = 1.0, kernel_width = 11L,
                             seed = 17) {
  .c4a_aug_apply("unsharp_mask", X,
                 c(amount_lo, amount_hi, sigma, kernel_width), seed)
}

#' Smooth magnitude warp augmenter.
#' @export
aug_magnitude_warp <- function(X, wavelengths = NULL,
                               n_control_points = 3L,
                               gain_lo = 0.9, gain_hi = 1.1,
                               seed = 17) {
  X <- .c4a_check_matrix(X, "X")
  .c4a_aug_apply("magnitude_warp", X,
                 c(n_control_points, gain_lo, gain_hi), seed,
                 .c4a_aug_wavelengths(X, wavelengths))
}

#' Local clipping augmenter.
#' @export
aug_local_clip <- function(X, n_regions = 1L, width_lo = 5L,
                           width_hi = 15L, seed = 17) {
  .c4a_aug_apply("local_clip", X, c(n_regions, width_lo, width_hi), seed)
}

#' Wavelength/spectral augmenter bundle.
#' @export
aug_wavelength_spectral <- function(X, wavelengths = NULL, seed = 17) {
  cbind(
    aug_wavelength_shift(X, wavelengths = wavelengths, seed = seed),
    aug_wavelength_stretch(X, wavelengths = wavelengths, seed = seed),
    aug_local_warp(X, wavelengths = wavelengths, seed = seed),
    aug_band_perturb(X, seed = seed),
    aug_band_mask(X, seed = seed),
    aug_channel_dropout(X, seed = seed),
    aug_gauss_jitter(X, seed = seed),
    aug_unsharp_mask(X, seed = seed),
    aug_magnitude_warp(X, wavelengths = wavelengths, seed = seed),
    aug_local_clip(X, seed = seed))
}

#' Mixup augmenter.
#' @export
aug_mixup <- function(X, alpha = 1.0, seed = 17) {
  .c4a_aug_apply("mixup", X, c(alpha), seed)
}

#' Local mixup augmenter.
#' @export
aug_local_mixup <- function(X, alpha = 1.0, k_neighbors = 5L, seed = 17) {
  .c4a_aug_apply("local_mixup", X, c(alpha, k_neighbors), seed)
}

#' Scatter simulation MSC augmenter.
#' @export
aug_scatter_sim <- function(X, a_low = -0.05, a_high = 0.05,
                            b_low = 0.9, b_high = 1.1, seed = 17) {
  .c4a_aug_apply("scatter_sim", X, c(a_low, a_high, b_low, b_high), seed)
}

#' Particle size augmenter.
#' @export
aug_particle_size <- function(X, wavelengths = NULL, mean_size_um = 50.0,
                              size_variation_um = 15.0,
                              use_size_range = 0L,
                              size_range_low_um = 5.0,
                              size_range_high_um = 500.0,
                              reference_size_um = 50.0,
                              wavelength_exponent = 1.5,
                              size_effect_strength = 0.1,
                              include_path_length = 1L,
                              path_length_sensitivity = 0.5,
                              seed = 17) {
  X <- .c4a_check_matrix(X, "X")
  .c4a_aug_apply("particle_size", X,
                 c(mean_size_um, size_variation_um, use_size_range,
                   size_range_low_um, size_range_high_um,
                   reference_size_um, wavelength_exponent,
                   size_effect_strength, include_path_length,
                   path_length_sensitivity),
                 seed, .c4a_aug_wavelengths(X, wavelengths))
}

#' EMSC distortion augmenter.
#' @export
aug_emsc_distort <- function(X, wavelengths = NULL, mult_low = 0.9,
                             mult_high = 1.1, add_low = -0.05,
                             add_high = 0.05, polynomial_order = 2L,
                             polynomial_strength = 0.02,
                             correlation = 0.3, seed = 17) {
  X <- .c4a_check_matrix(X, "X")
  .c4a_aug_apply("emsc_distort", X,
                 c(mult_low, mult_high, add_low, add_high,
                   polynomial_order, polynomial_strength, correlation),
                 seed, .c4a_aug_wavelengths(X, wavelengths))
}

#' Batch effect augmenter.
#' @export
aug_batch_effect <- function(X, wavelengths = NULL, offset_std = 0.02,
                             slope_std = 0.01, gain_std = 0.03,
                             variation_scope = 0L, seed = 17) {
  X <- .c4a_check_matrix(X, "X")
  .c4a_aug_apply("batch_effect", X,
                 c(offset_std, slope_std, gain_std, variation_scope),
                 seed, .c4a_aug_wavelengths(X, wavelengths))
}

#' Instrumental broadening augmenter.
#' @export
aug_instrument_broaden <- function(X, wavelengths = NULL, fwhm = 3.0,
                                   use_fwhm_range = 0L, fwhm_low = 3.0,
                                   fwhm_high = 8.0,
                                   variation_scope = 0L, seed = 17) {
  X <- .c4a_check_matrix(X, "X")
  .c4a_aug_apply("instrument_broaden", X,
                 c(fwhm, use_fwhm_range, fwhm_low, fwhm_high,
                   variation_scope),
                 seed, .c4a_aug_wavelengths(X, wavelengths))
}

#' Dead-band augmenter.
#' @export
aug_dead_band <- function(X, n_bands = 1L, width_low = 5L,
                          width_high = 10L, noise_std = 0.05,
                          probability = 1.0, variation_scope = 0L,
                          seed = 17) {
  .c4a_aug_apply("dead_band", X,
                 c(n_bands, width_low, width_high, noise_std,
                   probability, variation_scope),
                 seed)
}

#' Temperature augmenter.
#' @export
aug_temperature <- function(X, wavelengths = NULL, temperature_delta = 5.0,
                            use_temp_range = 0L, temp_low = -5.0,
                            temp_high = 5.0, enable_shift = 1L,
                            enable_intensity = 1L, enable_broadening = 1L,
                            region_specific = 1L, seed = 17) {
  X <- .c4a_check_matrix(X, "X")
  .c4a_aug_apply("temperature", X,
                 c(temperature_delta, use_temp_range, temp_low, temp_high,
                   enable_shift, enable_intensity, enable_broadening,
                   region_specific),
                 seed, .c4a_aug_wavelengths(X, wavelengths))
}

#' Moisture augmenter.
#' @export
aug_moisture <- function(X, wavelengths = NULL,
                         water_activity_delta = 0.1,
                         use_aw_range = 0L, aw_low = 0.0, aw_high = 1.0,
                         reference_water_activity = 0.5,
                         free_water_fraction = 0.3,
                         bound_water_shift = 25.0,
                         moisture_content = 0.1,
                         enable_shift = 1L, enable_intensity = 1L,
                         seed = 17) {
  X <- .c4a_check_matrix(X, "X")
  .c4a_aug_apply("moisture", X,
                 c(water_activity_delta, use_aw_range, aw_low, aw_high,
                   reference_water_activity, free_water_fraction,
                   bound_water_shift, moisture_content, enable_shift,
                   enable_intensity),
                 seed, .c4a_aug_wavelengths(X, wavelengths))
}

#' Detector roll-off augmenter.
#' @export
aug_detector_rolloff <- function(X, wavelengths = NULL,
                                 detector_model = 4L,
                                 effect_strength = 1.0,
                                 noise_amplification = 0.02,
                                 include_baseline_distortion = 1L,
                                 seed = 17) {
  X <- .c4a_check_matrix(X, "X")
  .c4a_aug_apply("detector_rolloff", X,
                 c(detector_model, effect_strength, noise_amplification,
                   include_baseline_distortion),
                 seed, .c4a_aug_wavelengths(X, wavelengths))
}

#' Stray-light augmenter.
#' @export
aug_stray_light <- function(X, wavelengths = NULL,
                            stray_light_fraction = 0.001,
                            edge_enhancement = 2.0, edge_width = 0.1,
                            include_peak_truncation = 1L, seed = 17) {
  X <- .c4a_check_matrix(X, "X")
  .c4a_aug_apply("stray_light", X,
                 c(stray_light_fraction, edge_enhancement, edge_width,
                   include_peak_truncation),
                 seed, .c4a_aug_wavelengths(X, wavelengths))
}

#' Edge curvature augmenter.
#' @export
aug_edge_curve <- function(X, wavelengths = NULL,
                           curvature_strength = 0.02,
                           curvature_type = 0L,
                           asymmetry = 0.0, edge_focus = 0.7,
                           seed = 17) {
  X <- .c4a_check_matrix(X, "X")
  .c4a_aug_apply("edge_curve", X,
                 c(curvature_strength, curvature_type, asymmetry, edge_focus),
                 seed, .c4a_aug_wavelengths(X, wavelengths))
}

#' Truncated peak augmenter.
#' @export
aug_truncated_peak <- function(X, wavelengths = NULL,
                               peak_probability = 0.5,
                               amplitude_min = 0.01,
                               amplitude_max = 0.1,
                               width_min = 50.0, width_max = 200.0,
                               left_edge = 1L, right_edge = 1L,
                               seed = 17) {
  X <- .c4a_check_matrix(X, "X")
  .c4a_aug_apply("truncated_peak", X,
                 c(peak_probability, amplitude_min, amplitude_max,
                   width_min, width_max, left_edge, right_edge),
                 seed, .c4a_aug_wavelengths(X, wavelengths))
}

#' Combined edge artifact augmenter.
#' @export
aug_edge_artifacts <- function(X, wavelengths = NULL, enabled_flags = 15L,
                               overall_strength = 1.0,
                               detector_model = 4L, seed = 17) {
  X <- .c4a_check_matrix(X, "X")
  .c4a_aug_apply("edge_artifacts", X,
                 c(enabled_flags, overall_strength, detector_model),
                 seed, .c4a_aug_wavelengths(X, wavelengths))
}

#' Spline smoothing augmenter.
#' @export
aug_spline_smooth <- function(X, seed = 17) {
  .c4a_aug_apply("spline_smooth", X, numeric(), seed)
}

#' Spline X perturbation augmenter.
#' @export
aug_spline_x_perturb <- function(X, spline_degree = 3L,
                                 perturbation_density = 0.05,
                                 perturbation_range_min = -0.1,
                                 perturbation_range_max = 0.1,
                                 seed = 17) {
  .c4a_aug_apply("spline_x_perturb", X,
                 c(spline_degree, perturbation_density,
                   perturbation_range_min, perturbation_range_max),
                 seed)
}

#' Spline Y perturbation augmenter.
#' @export
aug_spline_y_perturb <- function(X, spline_points = -1L,
                                 perturbation_intensity = 0.005,
                                 seed = 17) {
  .c4a_aug_apply("spline_y_perturb", X,
                 c(spline_points, perturbation_intensity), seed)
}

#' Rotate/translate augmenter.
#' @export
aug_rotate_translate <- function(X, p_range = 2.0, y_factor = 3.0,
                                 seed = 17) {
  .c4a_aug_apply("rotate_translate", X, c(p_range, y_factor), seed)
}

#' Random X operation augmenter.
#' @export
aug_random_x_op <- function(X, op_kind = 0L, operator_range_min = 0.97,
                            operator_range_max = 1.03, seed = 17) {
  .c4a_aug_apply("random_x_op", X,
                 c(op_kind, operator_range_min, operator_range_max), seed)
}

#' Spectral quality filter mask.
#' @export
spectral_quality <- function(X, max_nan_ratio = 0.1,
                             max_zero_ratio = 0.5,
                             min_variance = 1e-8,
                             max_value = NULL, min_value = NULL,
                             check_inf = TRUE) {
  X <- .c4a_check_matrix(X, "X")
  .Call(C_c4a_filter_quality_apply, X, as.double(max_nan_ratio)[1L],
        as.double(max_zero_ratio)[1L], as.double(min_variance)[1L],
        !is.null(max_value), if (is.null(max_value)) 0.0 else as.double(max_value)[1L],
        !is.null(min_value), if (is.null(min_value)) 0.0 else as.double(min_value)[1L],
        as.logical(check_inf)[1L])
}
