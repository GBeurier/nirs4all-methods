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
#ifndef N4M_CORE_AUG_LOCAL_MIXUP_H
#define N4M_CORE_AUG_LOCAL_MIXUP_H

#include "n4m/n4m.h"

#include "core/common/rng_pcg64.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_aug_local_mixup_state_t n4m_aug_local_mixup_state_t;

n4m_aug_local_mixup_state_t* n4m_aug_local_mixup_state_new(double alpha,
                                                            int32_t k_neighbors);
void n4m_aug_local_mixup_state_free(n4m_aug_local_mixup_state_t* state);

n4m_status_t n4m_aug_local_mixup_apply_impl(
    const n4m_aug_local_mixup_state_t* state,
    n4m_rng_pcg64* rng,
    const double* X, int64_t rows, int64_t cols,
    double* out);

#ifdef __cplusplus
}
#endif

#endif /* N4M_CORE_AUG_LOCAL_MIXUP_H */
