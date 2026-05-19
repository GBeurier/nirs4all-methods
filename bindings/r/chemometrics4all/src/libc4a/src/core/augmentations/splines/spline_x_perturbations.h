/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Spline_X_Perturbations augmenter — internal engine.
 *
 * Perturbs the wavelength (x-axis) positions sampled by a cubic B-spline
 * fit to each spectrum, producing a smoothly-warped output. Mirrors
 * `nirs4all.operators.augmentation.splines.Spline_X_Perturbations`.
 *
 * The Python reference uses scipy.interpolate.splrep / BSpline. The C
 * version evaluates the natural cubic spline on a perturbed query grid.
 *
 * Pure C. INTERNAL.
 */
#ifndef CHEMOMETRICS4ALL_CORE_AUGMENT_SPLINE_X_PERTURB_H
#define CHEMOMETRICS4ALL_CORE_AUGMENT_SPLINE_X_PERTURB_H

#include <stddef.h>
#include <stdint.h>

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_aug_spline_x_perturb_state_t
    c4a_aug_spline_x_perturb_state_t;

c4a_aug_spline_x_perturb_state_t* c4a_aug_spline_x_perturb_state_new(
    int32_t spline_degree,
    double  perturbation_density,
    double  perturbation_range_min,
    double  perturbation_range_max);

void c4a_aug_spline_x_perturb_state_free(
    c4a_aug_spline_x_perturb_state_t* state);

c4a_status_t c4a_aug_spline_x_perturb_state_apply(
    const c4a_aug_spline_x_perturb_state_t* state,
    void* rng,
    const double* X, int64_t rows, int64_t cols,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_AUGMENT_SPLINE_X_PERTURB_H */
