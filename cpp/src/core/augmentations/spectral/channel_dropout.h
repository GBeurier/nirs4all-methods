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
#ifndef CHEMOMETRICS4ALL_CORE_AUG_CHANNEL_DROPOUT_H
#define CHEMOMETRICS4ALL_CORE_AUG_CHANNEL_DROPOUT_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"
#include "core/common/rng_pcg64.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum c4a_aug_channel_dropout_mode_t {
    C4A_AUG_CHANNEL_DROPOUT_ZERO   = 0,
    C4A_AUG_CHANNEL_DROPOUT_INTERP = 1
} c4a_aug_channel_dropout_mode_t;

typedef struct c4a_aug_channel_dropout_state_t c4a_aug_channel_dropout_state_t;

c4a_aug_channel_dropout_state_t* c4a_aug_channel_dropout_state_new(
    c4a_rng_pcg64* rng,
    double dropout_prob,
    c4a_aug_channel_dropout_mode_t mode);

void c4a_aug_channel_dropout_state_free(c4a_aug_channel_dropout_state_t* state);

c4a_status_t c4a_aug_channel_dropout_state_apply(
    c4a_aug_channel_dropout_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_AUG_CHANNEL_DROPOUT_H */
