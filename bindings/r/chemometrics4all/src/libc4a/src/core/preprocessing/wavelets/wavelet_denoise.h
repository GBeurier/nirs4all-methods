/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal engine for the WaveletDenoise operator (Phase 6).
 *
 * Algorithm (mirrors ``nirs4all.WaveletDenoise``):
 *
 *   1. Multi-level DWT (``c4a_wavelet_wavedec``).
 *   2. Estimate per-row noise from the finest detail coefficients
 *      using either the MAD estimator
 *      ``sigma = median(|cD_finest|) / 0.6745`` (default), or the
 *      sample standard deviation ``sigma = std(cD_finest)``.
 *   3. Universal Donoho threshold ``uthresh = sigma * sqrt(2 * log(L))``
 *      where ``L`` is the (original) signal length.
 *   4. Soft or hard thresholding on every detail level (cA is kept).
 *   5. Reconstruct via ``c4a_wavelet_waverec`` and truncate to the
 *      original signal length.
 *
 * The ``level`` parameter is automatically clamped to the maximum
 * allowable level reported by ``pywt.dwt_max_level``.
 */
#ifndef CHEMOMETRICS4ALL_CORE_PP_WAVELETS_WAVELET_DENOISE_H
#define CHEMOMETRICS4ALL_CORE_PP_WAVELETS_WAVELET_DENOISE_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"
#include "core/common/wavelet_kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum c4a_pp_wavelet_threshold_mode_t {
    C4A_PP_WD_THRESHOLD_SOFT = 0,
    C4A_PP_WD_THRESHOLD_HARD = 1
} c4a_pp_wavelet_threshold_mode_t;

typedef enum c4a_pp_wavelet_noise_estimator_t {
    C4A_PP_WD_NOISE_MEDIAN = 0,
    C4A_PP_WD_NOISE_STD    = 1
} c4a_pp_wavelet_noise_estimator_t;

typedef struct c4a_pp_wavelet_denoise_state_t c4a_pp_wavelet_denoise_state_t;

c4a_pp_wavelet_denoise_state_t* c4a_pp_wavelet_denoise_state_new(
    c4a_wavelet_family_t              family,
    c4a_wavelet_mode_t                mode,
    int32_t                            level,
    c4a_pp_wavelet_threshold_mode_t   threshold_mode,
    c4a_pp_wavelet_noise_estimator_t  noise_estimator);

void c4a_pp_wavelet_denoise_state_free(c4a_pp_wavelet_denoise_state_t* state);

c4a_status_t c4a_pp_wavelet_denoise_state_apply(
    const c4a_pp_wavelet_denoise_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_PP_WAVELETS_WAVELET_DENOISE_H */
