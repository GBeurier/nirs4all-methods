/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Spline_X_Simplification augmenter — internal engine.
 *
 * v2-deferred stub. The reference Python implementation builds a cubic
 * B-spline through a randomly-subsampled control set on the x-axis and
 * re-evaluates it on the dense grid. The behaviour is documented in
 * `nirs4all.operators.augmentation.splines.Spline_X_Simplification` but the
 * c4a engine returns C4A_ERR_NOT_IMPLEMENTED until v2 lands the random
 * sub-sampling primitive (rng.choice with replace=False) plus the splrep
 * surrogate. See DEFERRALS.md.
 *
 * Pure C. INTERNAL.
 */
#ifndef CHEMOMETRICS4ALL_CORE_AUGMENT_SPLINE_X_SIMPLIFY_H
#define CHEMOMETRICS4ALL_CORE_AUGMENT_SPLINE_X_SIMPLIFY_H

#include <stddef.h>
#include <stdint.h>

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_aug_spline_x_simplify_state_t
    c4a_aug_spline_x_simplify_state_t;

c4a_aug_spline_x_simplify_state_t* c4a_aug_spline_x_simplify_state_new(
    int32_t spline_points,
    int     uniform);

void c4a_aug_spline_x_simplify_state_free(
    c4a_aug_spline_x_simplify_state_t* state);

c4a_status_t c4a_aug_spline_x_simplify_state_apply(
    const c4a_aug_spline_x_simplify_state_t* state,
    void* rng,
    const double* X, int64_t rows, int64_t cols,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_AUGMENT_SPLINE_X_SIMPLIFY_H */
