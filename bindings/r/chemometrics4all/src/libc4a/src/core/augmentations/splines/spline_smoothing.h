/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Spline_Smoothing augmenter — internal engine.
 *
 * Fits a smoothing spline (natural cubic) to each spectrum row, then
 * re-evaluates it on the same x-grid. Mirrors
 * `nirs4all.operators.augmentation.splines.Spline_Smoothing`.
 *
 * The Python reference uses scipy.interpolate.UnivariateSpline with
 * s = 1/n_features, which produces a soft smoothing. Our C version is
 * stateless and seeded — the engine consumes no random numbers.
 *
 * Pure C. INTERNAL.
 */
#ifndef CHEMOMETRICS4ALL_CORE_AUGMENT_SPLINE_SMOOTHING_H
#define CHEMOMETRICS4ALL_CORE_AUGMENT_SPLINE_SMOOTHING_H

#include <stddef.h>
#include <stdint.h>

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_aug_spline_smooth_state_t c4a_aug_spline_smooth_state_t;

c4a_aug_spline_smooth_state_t* c4a_aug_spline_smooth_state_new(void);
void c4a_aug_spline_smooth_state_free(c4a_aug_spline_smooth_state_t* state);

c4a_status_t c4a_aug_spline_smooth_state_apply(
    const c4a_aug_spline_smooth_state_t* state,
    void* rng,
    const double* X, int64_t rows, int64_t cols,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_AUGMENT_SPLINE_SMOOTHING_H */
