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
#ifndef CHEMOMETRICS4ALL_CORE_AUGMENT_RANDOM_ROTATE_TRANSLATE_H
#define CHEMOMETRICS4ALL_CORE_AUGMENT_RANDOM_ROTATE_TRANSLATE_H

#include <stddef.h>
#include <stdint.h>

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_aug_rotate_translate_state_t
    c4a_aug_rotate_translate_state_t;

c4a_aug_rotate_translate_state_t* c4a_aug_rotate_translate_state_new(
    double p_range, double y_factor);

void c4a_aug_rotate_translate_state_free(
    c4a_aug_rotate_translate_state_t* state);

c4a_status_t c4a_aug_rotate_translate_state_apply(
    const c4a_aug_rotate_translate_state_t* state,
    void* rng,
    const double* X, int64_t rows, int64_t cols,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_AUGMENT_RANDOM_ROTATE_TRANSLATE_H */
