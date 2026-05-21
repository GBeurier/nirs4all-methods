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
#ifndef N4M_CORE_AUGMENT_EDGE_CURVATURE_H
#define N4M_CORE_AUGMENT_EDGE_CURVATURE_H

#include <stddef.h>
#include <stdint.h>

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum n4m_aug_edge_curvature_type_t {
    N4M_AUG_EDGE_CURVATURE_RANDOM     = 0,
    N4M_AUG_EDGE_CURVATURE_SMILE      = 1,
    N4M_AUG_EDGE_CURVATURE_FROWN      = 2,
    N4M_AUG_EDGE_CURVATURE_ASYMMETRIC = 3
} n4m_aug_edge_curvature_type_t;

typedef struct n4m_aug_edge_curvature_state_t n4m_aug_edge_curvature_state_t;

n4m_aug_edge_curvature_state_t* n4m_aug_edge_curvature_state_new(
    double curvature_strength,
    int32_t curvature_type,
    double asymmetry,
    double edge_focus);

void n4m_aug_edge_curvature_state_free(
    n4m_aug_edge_curvature_state_t* state);

n4m_status_t n4m_aug_edge_curvature_state_apply(
    const n4m_aug_edge_curvature_state_t* state,
    void* rng,
    const double* X, int64_t rows, int64_t cols,
    const double* wavelengths,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_AUGMENT_EDGE_CURVATURE_H */
