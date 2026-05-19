/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * GaussianSmoothingJitter augmenter — convolves each row with a normalized
 * 1-D Gaussian kernel whose sigma is sampled uniformly per sample. The
 * kernel width is a constant (`kernel_width`, must be odd).
 *
 * Algorithm (matches
 * nirs4all.operators.augmentation.spectral.GaussianSmoothingJitter):
 *
 *   sigmas = rng.uniform(sigma_lo, sigma_hi, size=n_samples)
 *   for each sample i:
 *     kernel = _get_gaussian_kernel(sigmas[i], kernel_width)
 *     out[i] = scipy.ndimage.convolve1d(X[i], kernel, mode='reflect')
 *
 * scipy.ndimage.convolve1d on a symmetric Gaussian collapses to a
 * reflect-padded correlation; we reuse the augmenter common helper for
 * both kernel construction and the convolution loop.
 *
 * The kernel is rebuilt for every row (since sigma varies), which trades a
 * tiny allocation per row for clarity. The width is bounded at the high
 * end by the engine (clamped at 1025 to avoid runaway), matching nirs4all's
 * documented configuration range.
 */
#ifndef CHEMOMETRICS4ALL_CORE_AUG_GAUSS_JITTER_H
#define CHEMOMETRICS4ALL_CORE_AUG_GAUSS_JITTER_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"
#include "core/common/rng_pcg64.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_aug_gauss_jitter_state_t c4a_aug_gauss_jitter_state_t;

c4a_aug_gauss_jitter_state_t* c4a_aug_gauss_jitter_state_new(
    c4a_rng_pcg64* rng,
    double sigma_lo, double sigma_hi,
    int32_t kernel_width);

void c4a_aug_gauss_jitter_state_free(c4a_aug_gauss_jitter_state_t* state);

c4a_status_t c4a_aug_gauss_jitter_state_apply(
    c4a_aug_gauss_jitter_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_AUG_GAUSS_JITTER_H */
