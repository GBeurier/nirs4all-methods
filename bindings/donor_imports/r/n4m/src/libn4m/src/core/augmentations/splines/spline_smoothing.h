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
#ifndef N4M_CORE_AUGMENT_SPLINE_SMOOTHING_H
#define N4M_CORE_AUGMENT_SPLINE_SMOOTHING_H

#include <stddef.h>
#include <stdint.h>

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_aug_spline_smooth_state_t n4m_aug_spline_smooth_state_t;

n4m_aug_spline_smooth_state_t* n4m_aug_spline_smooth_state_new(void);
void n4m_aug_spline_smooth_state_free(n4m_aug_spline_smooth_state_t* state);

n4m_status_t n4m_aug_spline_smooth_state_apply(
    const n4m_aug_spline_smooth_state_t* state,
    void* rng,
    const double* X, int64_t rows, int64_t cols,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_AUGMENT_SPLINE_SMOOTHING_H */
