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
#ifndef CHEMOMETRICS4ALL_CORE_AUGMENT_EDGE_STRAY_LIGHT_H
#define CHEMOMETRICS4ALL_CORE_AUGMENT_EDGE_STRAY_LIGHT_H

#include <stddef.h>
#include <stdint.h>

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_aug_stray_light_state_t c4a_aug_stray_light_state_t;

c4a_aug_stray_light_state_t* c4a_aug_stray_light_state_new(
    double stray_light_fraction,
    double edge_enhancement,
    double edge_width,
    int    include_peak_truncation);

void c4a_aug_stray_light_state_free(c4a_aug_stray_light_state_t* state);

c4a_status_t c4a_aug_stray_light_state_apply(
    const c4a_aug_stray_light_state_t* state,
    void* rng,
    const double* X, int64_t rows, int64_t cols,
    const double* wavelengths,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_AUGMENT_EDGE_STRAY_LIGHT_H */
