/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for EMSCDistortionAugmenter — wavelength-aware EMSC-
 * style scatter distortion.
 *
 * Reference: nirs4all.operators.augmentation.scattering.EMSCDistortionAugmenter.
 *
 * Algorithm (per sample):
 *   wl_norm[j] = 2 * (wl[j] - wl_min) / (wl_max - wl_min) - 1
 *   b ~ clipped_normal(b_mean, b_std)
 *   b_dev = (b - 1) / b_std  (if b_std > 0)
 *   a ~ clipped_normal(a_mean - corr * a_std * b_dev, a_std * sqrt(1 - corr^2))
 *   for order in 1..polynomial_order:
 *       c[order] ~ normal(0, polynomial_strength / sqrt(order))
 *   out[i, j] = a + b * X[i, j] + sum_{k=1..polynomial_order} c[k] * wl_norm[j]^k
 *
 * Means and stds for `a` and `b` are derived from the provided ranges:
 *   mean = (low + high) / 2,  std = (high - low) / 4.
 */
#ifndef CHEMOMETRICS4ALL_CORE_AUG_EMSC_DISTORT_H
#define CHEMOMETRICS4ALL_CORE_AUG_EMSC_DISTORT_H

#include "chemometrics4all/c4a.h"

#include "core/common/rng_pcg64.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_aug_emsc_distort_state_t c4a_aug_emsc_distort_state_t;

c4a_aug_emsc_distort_state_t* c4a_aug_emsc_distort_state_new(
    double mult_low, double mult_high,
    double add_low,  double add_high,
    int32_t polynomial_order,
    double  polynomial_strength,
    double  correlation);
void c4a_aug_emsc_distort_state_free(c4a_aug_emsc_distort_state_t* state);

c4a_status_t c4a_aug_emsc_distort_apply_impl(
    const c4a_aug_emsc_distort_state_t* state,
    c4a_rng_pcg64* rng,
    const double* X, int64_t rows, int64_t cols,
    const double* wavelengths,    /* length cols, must be non-NULL */
    double* out);

#ifdef __cplusplus
}
#endif

#endif /* CHEMOMETRICS4ALL_CORE_AUG_EMSC_DISTORT_H */
