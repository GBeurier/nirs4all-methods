/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * EdgeCurvature edge-artifact augmenter — internal engine.
 *
 * Adds a smooth curvature pattern (smile / frown / asymmetric) over the
 * normalised wavelength axis. Mirrors
 * `nirs4all.operators.augmentation.edge_artifacts.EdgeCurvatureAugmenter`.
 *
 * Curvature type encoding:
 *   0  random      (engine picks smile/frown/asym uniformly)
 *   1  smile       (upward at edges)
 *   2  frown       (downward at edges)
 *   3  asymmetric  (different curvature per side)
 *
 * Pure C. INTERNAL.
 */
#ifndef CHEMOMETRICS4ALL_CORE_AUGMENT_EDGE_CURVATURE_H
#define CHEMOMETRICS4ALL_CORE_AUGMENT_EDGE_CURVATURE_H

#include <stddef.h>
#include <stdint.h>

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum c4a_aug_edge_curvature_type_t {
    C4A_AUG_EDGE_CURVATURE_RANDOM     = 0,
    C4A_AUG_EDGE_CURVATURE_SMILE      = 1,
    C4A_AUG_EDGE_CURVATURE_FROWN      = 2,
    C4A_AUG_EDGE_CURVATURE_ASYMMETRIC = 3
} c4a_aug_edge_curvature_type_t;

typedef struct c4a_aug_edge_curvature_state_t c4a_aug_edge_curvature_state_t;

c4a_aug_edge_curvature_state_t* c4a_aug_edge_curvature_state_new(
    double curvature_strength,
    int32_t curvature_type,
    double asymmetry,
    double edge_focus);

void c4a_aug_edge_curvature_state_free(
    c4a_aug_edge_curvature_state_t* state);

c4a_status_t c4a_aug_edge_curvature_state_apply(
    const c4a_aug_edge_curvature_state_t* state,
    void* rng,
    const double* X, int64_t rows, int64_t cols,
    const double* wavelengths,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_AUGMENT_EDGE_CURVATURE_H */
