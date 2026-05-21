/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * LinearBaselineDrift — Phase 15 augmenter (n4m_aug_linear_drift_*).
 *
 * Adds a per-sample affine baseline (uniform in `[lo, hi]`):
 *
 *     offsets[i] = offset_lo + (offset_hi - offset_lo) * u_off_i
 *     slopes[i]  = slope_lo  + (slope_hi  - slope_lo ) * u_slope_i
 *     out[i, j]  = X[i, j] + offsets[i] + slopes[i] * (j - mean_j)
 *
 * where `u_off_i, u_slope_i ~ Uniform[0, 1)` and `mean_j` is the mean of the
 * implicit wavelength index `arange(n_features)`. Wavelengths default to
 * `arange(n_features)` (no explicit `wavelengths=` parameter on this ABI).
 *
 * Mirrors `nirs4all.operators.augmentation.spectral.LinearBaselineDrift`
 * with the implicit-wavelength branch (no wavelength array supplied).
 *
 * RNG draw order (locked):
 *   - rows offset draws
 *   - rows slope  draws
 */
#ifndef N4M_CORE_AUG_DRIFT_LINEAR_DRIFT_H
#define N4M_CORE_AUG_DRIFT_LINEAR_DRIFT_H

#include <stdint.h>

#include "n4m/n4m.h"
#include "core/common/rng_pcg64.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_aug_linear_drift_state_t n4m_aug_linear_drift_state_t;

n4m_aug_linear_drift_state_t* n4m_aug_linear_drift_state_new(
    n4m_rng_pcg64* rng,
    double offset_min, double offset_max,
    double slope_min,  double slope_max);

void n4m_aug_linear_drift_state_free(n4m_aug_linear_drift_state_t* state);

n4m_status_t n4m_aug_linear_drift_state_apply(
    n4m_aug_linear_drift_state_t* state,
    const double* X, int64_t rows, int64_t cols, double* out);

#ifdef __cplusplus
}
#endif

#endif /* N4M_CORE_AUG_DRIFT_LINEAR_DRIFT_H */
