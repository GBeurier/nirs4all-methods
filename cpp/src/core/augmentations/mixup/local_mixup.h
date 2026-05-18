/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for the LocalMixupAugmenter — mixes each sample with
 * one of its k nearest neighbors (excluding self) drawn uniformly at
 * random, weighted by a Beta(alpha, alpha) variable.
 *
 * Reference: nirs4all.operators.augmentation.spectral.LocalMixupAugmenter.
 *
 * Algorithm:
 *   1. Compute the k+1 nearest neighbors of every row of X (Euclidean).
 *   2. For each row i:
 *        neighbor_idx = uniform_choice(neighbors_of_i excluding self)
 *        lam          = Beta(alpha, alpha)
 *        out[i]       = lam * X[i] + (1 - lam) * X[neighbor_idx]
 *
 * The neighbor search reference (sklearn NearestNeighbors with default
 * settings) uses an exact brute-force algorithm for small arrays, which
 * we replicate by computing pairwise squared distances and partial-sorting
 * the indices.
 */
#ifndef CHEMOMETRICS4ALL_CORE_AUG_LOCAL_MIXUP_H
#define CHEMOMETRICS4ALL_CORE_AUG_LOCAL_MIXUP_H

#include "chemometrics4all/c4a.h"

#include "core/common/rng_pcg64.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_aug_local_mixup_state_t c4a_aug_local_mixup_state_t;

c4a_aug_local_mixup_state_t* c4a_aug_local_mixup_state_new(double alpha,
                                                            int32_t k_neighbors);
void c4a_aug_local_mixup_state_free(c4a_aug_local_mixup_state_t* state);

c4a_status_t c4a_aug_local_mixup_apply_impl(
    const c4a_aug_local_mixup_state_t* state,
    c4a_rng_pcg64* rng,
    const double* X, int64_t rows, int64_t cols,
    double* out);

#ifdef __cplusplus
}
#endif

#endif /* CHEMOMETRICS4ALL_CORE_AUG_LOCAL_MIXUP_H */
