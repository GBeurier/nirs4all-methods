/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for InstrumentalBroadeningAugmenter — Gaussian
 * convolution with FWHM-derived sigma per spectrum.
 *
 * Reference: nirs4all.operators.augmentation.synthesis.InstrumentalBroadeningAugmenter.
 *
 *   sigma_pts = (FWHM / (2 * sqrt(2 * ln 2))) / wl_step
 * with `wl_step = median(diff(wavelengths))` when wavelengths is supplied,
 * else 1.0.
 *
 * Fixed-FWHM mode draws nothing from the RNG; range mode draws one
 * uniform[fwhm_low, fwhm_high] per row (sample) or one per batch.
 */
#ifndef CHEMOMETRICS4ALL_CORE_AUG_INSTRUMENT_BROADEN_H
#define CHEMOMETRICS4ALL_CORE_AUG_INSTRUMENT_BROADEN_H

#include "chemometrics4all/c4a.h"

#include "core/common/rng_pcg64.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_aug_instrument_broaden_state_t
                c4a_aug_instrument_broaden_state_t;

/* `use_fwhm_range`: 0 = use fixed `fwhm`, 1 = sample from
 * [fwhm_low, fwhm_high]. `variation_scope`: 0 = per-sample, 1 = batch. */
c4a_aug_instrument_broaden_state_t* c4a_aug_instrument_broaden_state_new(
    double fwhm,
    int    use_fwhm_range, double fwhm_low, double fwhm_high,
    int32_t variation_scope);
void c4a_aug_instrument_broaden_state_free(
    c4a_aug_instrument_broaden_state_t* state);

c4a_status_t c4a_aug_instrument_broaden_apply_impl(
    const c4a_aug_instrument_broaden_state_t* state,
    c4a_rng_pcg64* rng,
    const double* X, int64_t rows, int64_t cols,
    const double* wavelengths,    /* may be NULL — wl_step then = 1.0 */
    double* out);

#ifdef __cplusplus
}
#endif

#endif /* CHEMOMETRICS4ALL_CORE_AUG_INSTRUMENT_BROADEN_H */
