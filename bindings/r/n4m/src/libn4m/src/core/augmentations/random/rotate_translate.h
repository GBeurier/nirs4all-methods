/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Rotate_Translate augmenter — internal engine.
 *
 * Adds a piecewise-linear "rotation + translation" pattern (two slopes
 * meeting at a hinge point on the normalised x-axis) scaled by per-sample
 * std. Mirrors
 * `nirs4all.operators.augmentation.random.Rotate_Translate`.
 *
 * Pure C. INTERNAL.
 */
#ifndef N4M_CORE_AUGMENT_RANDOM_ROTATE_TRANSLATE_H
#define N4M_CORE_AUGMENT_RANDOM_ROTATE_TRANSLATE_H

#include <stddef.h>
#include <stdint.h>

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_aug_rotate_translate_state_t
    n4m_aug_rotate_translate_state_t;

n4m_aug_rotate_translate_state_t* n4m_aug_rotate_translate_state_new(
    double p_range, double y_factor);

void n4m_aug_rotate_translate_state_free(
    n4m_aug_rotate_translate_state_t* state);

n4m_status_t n4m_aug_rotate_translate_state_apply(
    const n4m_aug_rotate_translate_state_t* state,
    void* rng,
    const double* X, int64_t rows, int64_t cols,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_AUGMENT_RANDOM_ROTATE_TRANSLATE_H */
