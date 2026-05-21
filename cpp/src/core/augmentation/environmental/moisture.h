/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for MoistureAugmenter — water-band specific spectral
 * shifts and intensity enhancement.
 *
 * Reference: nirs4all.operators.augmentation.environmental.MoistureAugmenter.
 *
 * Per sample:
 *   aw_delta = uniform[low, high] (if range set) else fixed
 *   aw = clip(reference_water_activity + aw_delta, 0, 1)
 *   free_fraction = free_water_fraction * sigmoid(8 * (aw - 0.5))
 *   bound_fraction = 1 - free_fraction
 *   if enable_shift:
 *     row = shift_water_band(row, wl, center=1435, width=50,
 *                            shift=bound_water_shift * bound_fraction)
 *     row = shift_water_band(row, wl, center=1930, width=40,
 *                            shift=bound_water_shift * 0.8 * bound_fraction)
 *   if enable_intensity:
 *     wreg1 = gaussian(wl, 1435, 40)
 *     wreg2 = gaussian(wl, 1930, 35)
 *     enhance = moisture_content / 0.10 - 1
 *     row += enhance * 0.1 * (wreg1 + wreg2) * |mean(row)|
 */
#ifndef N4M_CORE_AUG_MOISTURE_H
#define N4M_CORE_AUG_MOISTURE_H

#include "n4m/n4m.h"

#include "core/common/rng_pcg64.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_aug_moisture_state_t n4m_aug_moisture_state_t;

n4m_aug_moisture_state_t* n4m_aug_moisture_state_new(
    double water_activity_delta,
    int    use_aw_range, double aw_low, double aw_high,
    double reference_water_activity,
    double free_water_fraction,
    double bound_water_shift,
    double moisture_content,
    int    enable_shift, int enable_intensity);
void n4m_aug_moisture_state_free(n4m_aug_moisture_state_t* state);

n4m_status_t n4m_aug_moisture_apply_impl(
    const n4m_aug_moisture_state_t* state,
    n4m_rng_pcg64* rng,
    const double* X, int64_t rows, int64_t cols,
    const double* wavelengths,    /* must be non-NULL */
    double* out);

#ifdef __cplusplus
}
#endif

#endif /* N4M_CORE_AUG_MOISTURE_H */
