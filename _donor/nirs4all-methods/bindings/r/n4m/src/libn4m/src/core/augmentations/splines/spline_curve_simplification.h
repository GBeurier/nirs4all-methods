/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Spline_Curve_Simplification augmenter — internal engine.
 *
 * v2-deferred stub. Same rationale as Spline_X_Simplification (the engine
 * requires rng.choice(replace=False) over the wavelength index set, which
 * is gated to the next ABI minor). Returns N4M_ERR_NOT_IMPLEMENTED until
 * v2; see DEFERRALS.md.
 *
 * Pure C. INTERNAL.
 */
#ifndef N4M_CORE_AUGMENT_SPLINE_CURVE_SIMPLIFY_H
#define N4M_CORE_AUGMENT_SPLINE_CURVE_SIMPLIFY_H

#include <stddef.h>
#include <stdint.h>

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_aug_spline_curve_simplify_state_t
    n4m_aug_spline_curve_simplify_state_t;

n4m_aug_spline_curve_simplify_state_t*
n4m_aug_spline_curve_simplify_state_new(int32_t spline_points, int uniform);

void n4m_aug_spline_curve_simplify_state_free(
    n4m_aug_spline_curve_simplify_state_t* state);

n4m_status_t n4m_aug_spline_curve_simplify_state_apply(
    const n4m_aug_spline_curve_simplify_state_t* state,
    void* rng,
    const double* X, int64_t rows, int64_t cols,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_AUGMENT_SPLINE_CURVE_SIMPLIFY_H */
