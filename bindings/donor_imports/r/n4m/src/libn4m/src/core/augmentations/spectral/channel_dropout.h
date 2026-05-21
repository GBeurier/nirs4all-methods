/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * ChannelDropout augmenter — randomly drops individual wavelengths, with two
 * modes: ZERO (set to 0) or INTERP (replace via linear interpolation across
 * the surviving channels).
 *
 * Algorithm (matches
 * nirs4all.operators.augmentation.spectral.ChannelDropout):
 *
 *   mask[i, j] = rng.random((rows, cols)) < dropout_prob
 *   if mode == ZERO:   out[mask] = 0
 *   if mode == INTERP: for each row i, build kept_indices = where(~mask[i]),
 *                       dropped_indices = where(mask[i]);
 *                       out[i, dropped_indices] = np.interp(
 *                           dropped_indices, kept_indices,
 *                           X[i, kept_indices]).
 *
 * The random mask is drawn in row-major order via repeated `next_double()`
 * calls, matching NumPy's `rng.random(size=(rows, cols))`.
 */
#ifndef N4M_CORE_AUG_CHANNEL_DROPOUT_H
#define N4M_CORE_AUG_CHANNEL_DROPOUT_H

#include <stdint.h>

#include "n4m/n4m.h"
#include "core/common/rng_pcg64.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum n4m_aug_channel_dropout_mode_t {
    N4M_AUG_CHANNEL_DROPOUT_ZERO   = 0,
    N4M_AUG_CHANNEL_DROPOUT_INTERP = 1
} n4m_aug_channel_dropout_mode_t;

typedef struct n4m_aug_channel_dropout_state_t n4m_aug_channel_dropout_state_t;

n4m_aug_channel_dropout_state_t* n4m_aug_channel_dropout_state_new(
    n4m_rng_pcg64* rng,
    double dropout_prob,
    n4m_aug_channel_dropout_mode_t mode);

void n4m_aug_channel_dropout_state_free(n4m_aug_channel_dropout_state_t* state);

n4m_status_t n4m_aug_channel_dropout_state_apply(
    n4m_aug_channel_dropout_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_AUG_CHANNEL_DROPOUT_H */
