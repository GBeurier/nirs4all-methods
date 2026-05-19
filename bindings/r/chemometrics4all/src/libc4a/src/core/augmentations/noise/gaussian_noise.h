/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * GaussianAdditiveNoise — Phase 15 augmenter (c4a_aug_gaussian_noise_*).
 *
 * Adds per-sample std-scaled Gaussian noise:
 *
 *     stds[i] = std(X[i, :])                # biased estimator, ddof=0
 *     noise[i, j] = N_ij * stds[i] * sigma  # N_ij ~ N(0,1)
 *     out  = X + noise
 *
 * The `N_ij` matrix is drawn from a single
 * `c4a_pcg64_engine_standard_normal_fill(n_samples * n_features)` call so the
 * draw order is deterministic and matches the Python reference exactly.
 *
 * Mirrors the `variation_scope="sample"` default of
 * `nirs4all.operators.augmentation.spectral.GaussianAdditiveNoise`.
 * `smoothing_kernel_width` is fixed at 1 (no kernel smoothing) — the kernel
 * variant is deferred to a follow-up phase.
 */
#ifndef CHEMOMETRICS4ALL_CORE_AUG_NOISE_GAUSSIAN_NOISE_H
#define CHEMOMETRICS4ALL_CORE_AUG_NOISE_GAUSSIAN_NOISE_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"
#include "core/common/rng_pcg64.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_aug_gaussian_noise_state_t c4a_aug_gaussian_noise_state_t;

c4a_aug_gaussian_noise_state_t* c4a_aug_gaussian_noise_state_new(
    c4a_rng_pcg64* rng, double sigma);

void c4a_aug_gaussian_noise_state_free(c4a_aug_gaussian_noise_state_t* state);

c4a_status_t c4a_aug_gaussian_noise_state_apply(
    c4a_aug_gaussian_noise_state_t* state,
    const double* X, int64_t rows, int64_t cols, double* out);

#ifdef __cplusplus
}
#endif

#endif /* CHEMOMETRICS4ALL_CORE_AUG_NOISE_GAUSSIAN_NOISE_H */
