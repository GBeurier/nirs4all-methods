/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for BatchEffectAugmenter — simulates instrument/session
 * variability via additive offset, wavelength-dependent slope, and
 * multiplicative gain.
 *
 * Reference: nirs4all.operators.augmentation.synthesis.BatchEffectAugmenter.
 *
 * Per-sample mode (variation_scope=0) — independent per-row effects:
 *   x_norm = (wl - mean(wl)) / (max(wl) - min(wl) + 1e-10)
 *   for each row i:
 *     offset_i ~ N(0, offset_std)
 *     slope_i  ~ N(0, slope_std)
 *     gain_i   ~ N(1, gain_std)
 *     out[i, j] = X[i, j] * gain_i + offset_i + slope_i * x_norm[j]
 *
 * Batch mode (variation_scope=1) — single (offset, slope, gain) tuple
 * applied to all rows:
 *   offset ~ N(0, offset_std)
 *   slope  ~ N(0, slope_std)
 *   gain   ~ N(1, gain_std)
 *   out[i, j] = X[i, j] * gain + offset + slope * x_norm[j]
 */
#ifndef N4M_CORE_AUG_BATCH_EFFECT_H
#define N4M_CORE_AUG_BATCH_EFFECT_H

#include "n4m/n4m.h"

#include "core/common/rng_pcg64.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_aug_batch_effect_state_t n4m_aug_batch_effect_state_t;

/* `variation_scope`: 0 = "sample" (per-row), 1 = "batch" (single tuple) */
n4m_aug_batch_effect_state_t* n4m_aug_batch_effect_state_new(
    double offset_std, double slope_std, double gain_std,
    int32_t variation_scope);
void n4m_aug_batch_effect_state_free(n4m_aug_batch_effect_state_t* state);

/* `wavelengths` may be NULL; if NULL, the centered/normalised x axis is
 * computed from the integer index 0..cols-1. */
n4m_status_t n4m_aug_batch_effect_apply_impl(
    const n4m_aug_batch_effect_state_t* state,
    n4m_rng_pcg64* rng,
    const double* X, int64_t rows, int64_t cols,
    const double* wavelengths,
    double* out);

#ifdef __cplusplus
}
#endif

#endif /* N4M_CORE_AUG_BATCH_EFFECT_H */
