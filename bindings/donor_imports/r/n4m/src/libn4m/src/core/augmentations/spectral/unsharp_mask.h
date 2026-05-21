/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * UnsharpSpectralMask augmenter — sharpens each spectrum using a single
 * shared smoothing kernel and a per-sample amplification factor.
 *
 * Algorithm (matches
 * nirs4all.operators.augmentation.spectral.UnsharpSpectralMask):
 *
 *   kernel    = _get_gaussian_kernel(sigma, kernel_width)
 *   smoothed  = scipy.ndimage.convolve1d(X, kernel, axis=1, mode='reflect')
 *   amounts   = rng.uniform(amount_lo, amount_hi, size=(n_samples, 1))
 *   out       = X + amounts * (X - smoothed)
 *
 * The kernel is built once at `_new` time (sigma and width are constructor
 * parameters). At apply time the smoothing pass runs across all rows, then
 * a single `amounts` vector is drawn before the final fused multiply-add.
 */
#ifndef N4M_CORE_AUG_UNSHARP_MASK_H
#define N4M_CORE_AUG_UNSHARP_MASK_H

#include <stdint.h>

#include "n4m/n4m.h"
#include "core/common/rng_pcg64.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_aug_unsharp_mask_state_t n4m_aug_unsharp_mask_state_t;

n4m_aug_unsharp_mask_state_t* n4m_aug_unsharp_mask_state_new(
    n4m_rng_pcg64* rng,
    double amount_lo, double amount_hi,
    double sigma, int32_t kernel_width);

void n4m_aug_unsharp_mask_state_free(n4m_aug_unsharp_mask_state_t* state);

n4m_status_t n4m_aug_unsharp_mask_state_apply(
    n4m_aug_unsharp_mask_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_AUG_UNSHARP_MASK_H */
