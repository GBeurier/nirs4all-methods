/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for TemperatureAugmenter — region-specific NIR
 * temperature-induced spectral effects.
 *
 * Reference: nirs4all.operators.augmentation.environmental.TemperatureAugmenter.
 *
 * Six O-H / C-H / N-H / water regions each have a (shift, intensity,
 * broadening) coefficient. For each sample:
 *   delta_t = uniform[temperature_low, temperature_high] (if range set)
 *             else fixed temperature_delta
 *   if |delta_t| < 0.01: row unchanged
 *   else for each region:
 *     weights = sigmoid_window(wl, wl_min, wl_max, edge_width=20)
 *     if enable_shift:    row = np.interp(wl, wl + shift*weights, row)
 *     if enable_intensity: row = row * (1 + (factor - 1) * weights)
 *     if enable_broadening (factor > 1):
 *       broadened = gaussian_filter1d(row, sigma=(factor - 1) * 3)
 *       row = row * (1 - weights) + broadened * weights
 *
 * Mode `region_specific = 0` applies the average (shift, intensity,
 * broadening) coefficients uniformly across all wavelengths instead.
 */
#ifndef CHEMOMETRICS4ALL_CORE_AUG_TEMPERATURE_H
#define CHEMOMETRICS4ALL_CORE_AUG_TEMPERATURE_H

#include "chemometrics4all/c4a.h"

#include "core/common/rng_pcg64.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_aug_temperature_state_t c4a_aug_temperature_state_t;

c4a_aug_temperature_state_t* c4a_aug_temperature_state_new(
    double temperature_delta,
    int   use_temp_range, double temp_low, double temp_high,
    int   enable_shift, int enable_intensity, int enable_broadening,
    int   region_specific);
void c4a_aug_temperature_state_free(c4a_aug_temperature_state_t* state);

c4a_status_t c4a_aug_temperature_apply_impl(
    const c4a_aug_temperature_state_t* state,
    c4a_rng_pcg64* rng,
    const double* X, int64_t rows, int64_t cols,
    const double* wavelengths,    /* must be non-NULL */
    double* out);

#ifdef __cplusplus
}
#endif

#endif /* CHEMOMETRICS4ALL_CORE_AUG_TEMPERATURE_H */
