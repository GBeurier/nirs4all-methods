/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for ParticleSizeAugmenter — wavelength-aware particle
 * size scattering distortion.
 *
 * Reference: nirs4all.operators.augmentation.scattering.ParticleSizeAugmenter.
 *
 * Algorithm:
 *   1. Draw particle sizes:
 *        - if size_range provided: uniform[size_lo, size_hi]
 *        - else: clipped normal(mean_size, size_variation, [5.0, 500.0])
 *   2. For each sample:
 *        size_ratio = size / reference_size
 *        baseline   = strength * (size_ratio^(-0.5) - 1) * wl_factor
 *                     (centered to zero mean)
 *        wl_factor  = clip(wl/1500, 0.1, 10)^(-wavelength_exponent)
 *        result     = X[i] + baseline
 *        if include_path_length:
 *            result *= clip(1 + path_sensitivity * log(size_ratio), 0.7, 1.5)
 *        result += scatter_noise   (Gaussian smoothed normal noise)
 */
#ifndef CHEMOMETRICS4ALL_CORE_AUG_PARTICLE_SIZE_H
#define CHEMOMETRICS4ALL_CORE_AUG_PARTICLE_SIZE_H

#include "chemometrics4all/c4a.h"

#include "core/common/rng_pcg64.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_aug_particle_size_state_t c4a_aug_particle_size_state_t;

c4a_aug_particle_size_state_t* c4a_aug_particle_size_state_new(
    double mean_size_um, double size_variation_um,
    int   use_size_range, double size_range_low_um, double size_range_high_um,
    double reference_size_um, double wavelength_exponent,
    double size_effect_strength,
    int   include_path_length, double path_length_sensitivity);
void c4a_aug_particle_size_state_free(c4a_aug_particle_size_state_t* state);

c4a_status_t c4a_aug_particle_size_apply_impl(
    const c4a_aug_particle_size_state_t* state,
    c4a_rng_pcg64* rng,
    const double* X, int64_t rows, int64_t cols,
    const double* wavelengths,    /* length cols, must be non-NULL */
    double* out);

#ifdef __cplusplus
}
#endif

#endif /* CHEMOMETRICS4ALL_CORE_AUG_PARTICLE_SIZE_H */
