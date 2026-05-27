/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Spline_X_Simplification augmenter — internal engine.
 *
 * Builds a cubic B-spline through a randomly-subsampled control set on the
 * x-axis and re-evaluates it on the dense grid — the reference behaviour of
 * `nirs4all.operators.augmentation.splines.Spline_X_Simplification`. The random
 * control subset uses the PCG64-backed numpy.choice(replace=False) primitive
 * (n4m_aug_rng_choice_no_replace); the cubic refit uses the splrep(s=0, k=3)
 * not-a-knot surrogate (bspline.h). Shared logic in spline_simplify_common.h.
 *
 * Pure C. INTERNAL.
 */
#ifndef N4M_CORE_AUGMENT_SPLINE_X_SIMPLIFY_H
#define N4M_CORE_AUGMENT_SPLINE_X_SIMPLIFY_H

#include <stddef.h>
#include <stdint.h>

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_aug_spline_x_simplify_state_t
    n4m_aug_spline_x_simplify_state_t;

n4m_aug_spline_x_simplify_state_t* n4m_aug_spline_x_simplify_state_new(
    int32_t spline_points,
    int     uniform);

void n4m_aug_spline_x_simplify_state_free(
    n4m_aug_spline_x_simplify_state_t* state);

n4m_status_t n4m_aug_spline_x_simplify_state_apply(
    const n4m_aug_spline_x_simplify_state_t* state,
    void* rng,
    const double* X, int64_t rows, int64_t cols,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_AUGMENT_SPLINE_X_SIMPLIFY_H */
