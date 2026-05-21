/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * StrayLight edge-artifact augmenter — internal engine.
 *
 * Models a wavelength-dependent stray-light profile (base value enhanced at
 * spectral edges via a sigmoid) and applies the observed-transmittance
 * stray-light equation
 *
 *     T_obs = (T_true + s) / (1 + s),       T_true = 10^(-A_true)
 *
 * before converting back to absorbance. Mirrors
 * `nirs4all.operators.augmentation.edge_artifacts.StrayLightAugmenter`.
 *
 * Pure C. INTERNAL.
 */
#ifndef N4M_CORE_AUGMENT_EDGE_STRAY_LIGHT_H
#define N4M_CORE_AUGMENT_EDGE_STRAY_LIGHT_H

#include <stddef.h>
#include <stdint.h>

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_aug_stray_light_state_t n4m_aug_stray_light_state_t;

n4m_aug_stray_light_state_t* n4m_aug_stray_light_state_new(
    double stray_light_fraction,
    double edge_enhancement,
    double edge_width,
    int    include_peak_truncation);

void n4m_aug_stray_light_state_free(n4m_aug_stray_light_state_t* state);

n4m_status_t n4m_aug_stray_light_state_apply(
    const n4m_aug_stray_light_state_t* state,
    void* rng,
    const double* X, int64_t rows, int64_t cols,
    const double* wavelengths,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_AUGMENT_EDGE_STRAY_LIGHT_H */
