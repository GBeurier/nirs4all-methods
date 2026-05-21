/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * WavelengthStretch augmenter — stretches/compresses the wavelength axis
 * about its mean, then linear-interpolates each spectrum back onto the
 * original grid.
 *
 * Algorithm (matches nirs4all.operators.augmentation.spectral.WavelengthStretch
 * with `np.interp` linear interpolation):
 *
 *   center = mean(lambdas)
 *   for each sample i:
 *     f_i = rng.uniform(stretch_lo, stretch_hi)   (factor, typically near 1)
 *     query[j] = center + (lambdas[j] - center) / f_i
 *     out[i]   = np.interp(query, lambdas, X[i])
 *
 * When `wavelengths` is NULL the operator falls back to a unit-spaced grid
 * `[0, 1, ..., p-1]`.
 */
#ifndef N4M_CORE_AUG_WAVELENGTH_STRETCH_H
#define N4M_CORE_AUG_WAVELENGTH_STRETCH_H

#include <stdint.h>

#include "n4m/n4m.h"
#include "core/common/rng_pcg64.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_aug_wavelength_stretch_state_t n4m_aug_wavelength_stretch_state_t;

n4m_aug_wavelength_stretch_state_t* n4m_aug_wavelength_stretch_state_new(
    n4m_rng_pcg64* rng,
    double stretch_lo, double stretch_hi,
    const double* wavelengths, int64_t n_wavelengths);

void n4m_aug_wavelength_stretch_state_free(
    n4m_aug_wavelength_stretch_state_t* state);

n4m_status_t n4m_aug_wavelength_stretch_state_apply(
    n4m_aug_wavelength_stretch_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_AUG_WAVELENGTH_STRETCH_H */
