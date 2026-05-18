/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * HeteroscedasticNoiseAugmenter — Phase 15 augmenter (c4a_aug_hetero_noise_*).
 *
 * Adds signal-dependent Gaussian noise:
 *
 *     sigma[i, j] = noise_base + noise_signal_dep * |X[i, j]|
 *     out[i, j]   = X[i, j] + N_ij * sigma[i, j]
 *
 * where `N_ij ~ N(0, 1)`. Mirrors `nirs4all.operators.augmentation.synthesis.
 * HeteroscedasticNoiseAugmenter` with `variation_scope="sample"` (default).
 *
 * Single bulk RNG call:
 *     `standard_normal_fill(rng, noise, n_samples * n_features)`.
 */
#ifndef CHEMOMETRICS4ALL_CORE_AUG_NOISE_HETERO_NOISE_H
#define CHEMOMETRICS4ALL_CORE_AUG_NOISE_HETERO_NOISE_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"
#include "core/common/rng_pcg64.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_aug_hetero_noise_state_t c4a_aug_hetero_noise_state_t;

c4a_aug_hetero_noise_state_t* c4a_aug_hetero_noise_state_new(
    c4a_rng_pcg64* rng,
    double noise_base,
    double noise_signal_dep);

void c4a_aug_hetero_noise_state_free(c4a_aug_hetero_noise_state_t* state);

c4a_status_t c4a_aug_hetero_noise_state_apply(
    c4a_aug_hetero_noise_state_t* state,
    const double* X, int64_t rows, int64_t cols, double* out);

#ifdef __cplusplus
}
#endif

#endif /* CHEMOMETRICS4ALL_CORE_AUG_NOISE_HETERO_NOISE_H */
