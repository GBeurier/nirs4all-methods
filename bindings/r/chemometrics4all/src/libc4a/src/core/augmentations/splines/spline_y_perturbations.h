/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Spline_Y_Perturbations augmenter — internal engine.
 *
 * Generates a smooth y-axis distortion from a B-spline fit through randomly
 * drawn control points, then adds it to each spectrum. Mirrors
 * `nirs4all.operators.augmentation.splines.Spline_Y_Perturbations`.
 *
 * Pure C. INTERNAL.
 */
#ifndef CHEMOMETRICS4ALL_CORE_AUGMENT_SPLINE_Y_PERTURB_H
#define CHEMOMETRICS4ALL_CORE_AUGMENT_SPLINE_Y_PERTURB_H

#include <stddef.h>
#include <stdint.h>

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_aug_spline_y_perturb_state_t
    c4a_aug_spline_y_perturb_state_t;

/* spline_points <= 0 means "use cols / 2". */
c4a_aug_spline_y_perturb_state_t* c4a_aug_spline_y_perturb_state_new(
    int32_t spline_points,
    double  perturbation_intensity);

void c4a_aug_spline_y_perturb_state_free(
    c4a_aug_spline_y_perturb_state_t* state);

c4a_status_t c4a_aug_spline_y_perturb_state_apply(
    const c4a_aug_spline_y_perturb_state_t* state,
    void* rng,
    const double* X, int64_t rows, int64_t cols,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_AUGMENT_SPLINE_Y_PERTURB_H */
