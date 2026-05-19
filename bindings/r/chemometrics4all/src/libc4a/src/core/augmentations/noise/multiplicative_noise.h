/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * MultiplicativeNoise — Phase 15 augmenter (c4a_aug_multiplicative_noise_*).
 *
 * Applies per-sample multiplicative gain:
 *
 *     eps[i]      = sigma_gain * N_i        # N_i ~ N(0,1)
 *     out[i, j]   = X[i, j] * (1 + eps[i])
 *
 * Mirrors `nirs4all.operators.augmentation.spectral.MultiplicativeNoise`
 * with `per_wavelength=False, variation_scope="sample"` (default). Per-
 * wavelength gain (n_samples * n_features draws) and batch-wide gain
 * (single scalar) are deferred to a follow-up phase.
 *
 * Single bulk RNG call:
 *     `standard_normal_fill(rng, eps, n_samples)`.
 */
#ifndef CHEMOMETRICS4ALL_CORE_AUG_NOISE_MULTIPLICATIVE_NOISE_H
#define CHEMOMETRICS4ALL_CORE_AUG_NOISE_MULTIPLICATIVE_NOISE_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"
#include "core/common/rng_pcg64.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_aug_multiplicative_noise_state_t
    c4a_aug_multiplicative_noise_state_t;

c4a_aug_multiplicative_noise_state_t* c4a_aug_multiplicative_noise_state_new(
    c4a_rng_pcg64* rng, double sigma_gain);

void c4a_aug_multiplicative_noise_state_free(
    c4a_aug_multiplicative_noise_state_t* state);

c4a_status_t c4a_aug_multiplicative_noise_state_apply(
    c4a_aug_multiplicative_noise_state_t* state,
    const double* X, int64_t rows, int64_t cols, double* out);

#ifdef __cplusplus
}
#endif

#endif /* CHEMOMETRICS4ALL_CORE_AUG_NOISE_MULTIPLICATIVE_NOISE_H */
