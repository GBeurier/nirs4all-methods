/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Spline_Curve_Simplification augmenter — internal engine.
 *
 * v2-deferred stub. Same rationale as Spline_X_Simplification (the engine
 * requires rng.choice(replace=False) over the wavelength index set, which
 * is gated to the next ABI minor). Returns C4A_ERR_NOT_IMPLEMENTED until
 * v2; see DEFERRALS.md.
 *
 * Pure C. INTERNAL.
 */
#ifndef CHEMOMETRICS4ALL_CORE_AUGMENT_SPLINE_CURVE_SIMPLIFY_H
#define CHEMOMETRICS4ALL_CORE_AUGMENT_SPLINE_CURVE_SIMPLIFY_H

#include <stddef.h>
#include <stdint.h>

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_aug_spline_curve_simplify_state_t
    c4a_aug_spline_curve_simplify_state_t;

c4a_aug_spline_curve_simplify_state_t*
c4a_aug_spline_curve_simplify_state_new(int32_t spline_points, int uniform);

void c4a_aug_spline_curve_simplify_state_free(
    c4a_aug_spline_curve_simplify_state_t* state);

c4a_status_t c4a_aug_spline_curve_simplify_state_apply(
    const c4a_aug_spline_curve_simplify_state_t* state,
    void* rng,
    const double* X, int64_t rows, int64_t cols,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_AUGMENT_SPLINE_CURVE_SIMPLIFY_H */
