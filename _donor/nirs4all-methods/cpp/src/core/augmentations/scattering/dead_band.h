/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for DeadBandAugmenter — zeroes random spectral regions
 * with added noise to simulate detector dead bands.
 *
 * Reference: nirs4all.operators.augmentation.synthesis.DeadBandAugmenter.
 *
 * Per-sample mode (variation_scope=0):
 *   for each row i:
 *     u ~ U(0, 1)
 *     if u < probability:
 *       for _ in range(n_bands):
 *         width ~ U_int[width_lo, width_hi]    (inclusive)
 *         start ~ U_int[0, max(1, cols - width))
 *         end   = min(start + width, cols)
 *         row[start:end] = N(0, noise_std)
 *
 * Batch mode (variation_scope=1):
 *   draw `u` once; if applied, the same dead bands wipe every row, but
 *   each row's noise sample is INDEPENDENT (size = (n_samples, end-start)).
 */
#ifndef N4M_CORE_AUG_DEAD_BAND_H
#define N4M_CORE_AUG_DEAD_BAND_H

#include "n4m/n4m.h"

#include "core/common/rng_pcg64.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_aug_dead_band_state_t n4m_aug_dead_band_state_t;

n4m_aug_dead_band_state_t* n4m_aug_dead_band_state_new(
    int32_t n_bands,
    int32_t width_low, int32_t width_high,
    double  noise_std, double probability,
    int32_t variation_scope);
void n4m_aug_dead_band_state_free(n4m_aug_dead_band_state_t* state);

n4m_status_t n4m_aug_dead_band_apply_impl(
    const n4m_aug_dead_band_state_t* state,
    n4m_rng_pcg64* rng,
    const double* X, int64_t rows, int64_t cols,
    double* out);

#ifdef __cplusplus
}
#endif

#endif /* N4M_CORE_AUG_DEAD_BAND_H */
