/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * BandMasking augmenter — masks out 1+ random spectral bands per sample,
 * either by zeroing or by linear interpolation between the band's endpoints.
 *
 * Algorithm (matches
 * nirs4all.operators.augmentation.spectral.BandMasking):
 *
 *   n_per_sample = rng.integers(n_bands_lo, n_bands_hi + 1, size=n_samples)
 *   centers      = rng.integers(0, n_features, size=(n_samples, max_bands))
 *   widths       = rng.integers(bw_lo, bw_hi, size=(n_samples, max_bands))
 *   for each sample i, for each band b in [0, n_per_sample[i]):
 *     start = max(0, centers[i, b] - widths[i, b] // 2)
 *     end   = min(n_features, centers[i, b] + widths[i, b] // 2)
 *     if mode == ZERO   : out[i, start:end] = 0
 *     if mode == INTERP : linear ramp between out[i, start-1] (or
 *                          out[i, start] when start == 0) and
 *                          out[i, end] (or out[i, end-1] when end == cols).
 *
 * Note `max_bands = n_bands_hi` (matches the Python reference's
 * `max_bands = self.n_bands_range[1]`). The full max_bands × n_samples
 * random blocks are pre-allocated even when individual samples ask for
 * fewer bands; only the first `n_per_sample[i]` rows are consumed.
 */
#ifndef N4M_CORE_AUG_BAND_MASK_H
#define N4M_CORE_AUG_BAND_MASK_H

#include <stdint.h>

#include "n4m/n4m.h"
#include "core/common/rng_pcg64.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum n4m_aug_band_mask_mode_t {
    N4M_AUG_BAND_MASK_ZERO   = 0,
    N4M_AUG_BAND_MASK_INTERP = 1
} n4m_aug_band_mask_mode_t;

typedef struct n4m_aug_band_mask_state_t n4m_aug_band_mask_state_t;

n4m_aug_band_mask_state_t* n4m_aug_band_mask_state_new(
    n4m_rng_pcg64* rng,
    int32_t n_bands_lo, int32_t n_bands_hi,
    int32_t bw_lo, int32_t bw_hi,
    n4m_aug_band_mask_mode_t mode);

void n4m_aug_band_mask_state_free(n4m_aug_band_mask_state_t* state);

n4m_status_t n4m_aug_band_mask_state_apply(
    n4m_aug_band_mask_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_AUG_BAND_MASK_H */
