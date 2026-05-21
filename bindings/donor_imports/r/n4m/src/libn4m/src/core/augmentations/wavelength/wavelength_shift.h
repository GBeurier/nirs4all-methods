/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * WavelengthShift augmenter — shifts the wavelength axis by a random offset
 * sampled per sample, then linear-interpolates each spectrum onto the
 * original grid.
 *
 * Algorithm (matches nirs4all.operators.augmentation.spectral.WavelengthShift
 * with `np.interp` linear interpolation):
 *
 *   for each sample i in [0, n):
 *     s_i = rng.uniform(shift_lo, shift_hi)
 *     query[j] = lambdas[j] - s_i             for j in [0, p)
 *     out[i] = np.interp(query, lambdas, X[i])
 *
 * When `wavelengths` is NULL the operator falls back to a unit-spaced grid
 * `[0, 1, ..., p-1]`. The C engine pre-generates all shifts once before
 * walking the rows so the random-number stream order matches NumPy's
 * `size=n_samples` allocation.
 */
#ifndef N4M_CORE_AUG_WAVELENGTH_SHIFT_H
#define N4M_CORE_AUG_WAVELENGTH_SHIFT_H

#include <stdint.h>

#include "n4m/n4m.h"
#include "core/common/rng_pcg64.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_aug_wavelength_shift_state_t n4m_aug_wavelength_shift_state_t;

n4m_aug_wavelength_shift_state_t* n4m_aug_wavelength_shift_state_new(
    n4m_rng_pcg64* rng,
    double shift_lo, double shift_hi,
    const double* wavelengths, int64_t n_wavelengths);

void n4m_aug_wavelength_shift_state_free(
    n4m_aug_wavelength_shift_state_t* state);

n4m_status_t n4m_aug_wavelength_shift_state_apply(
    n4m_aug_wavelength_shift_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_AUG_WAVELENGTH_SHIFT_H */
