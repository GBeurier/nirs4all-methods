/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * PathLengthAugmenter — Phase 15 augmenter (n4m_aug_path_length_*).
 *
 * Applies a per-sample multiplicative path-length factor, clipped at a
 * lower bound to prevent sign inversion:
 *
 *     L[i]      = 1.0 + path_length_std * N_i      # N_i ~ N(0, 1)
 *     L[i]      = max(L[i], min_path_length)
 *     out[i, j] = L[i] * X[i, j]
 *
 * Mirrors `nirs4all.operators.augmentation.synthesis.PathLengthAugmenter`
 * with `variation_scope="sample"` (default). The batch scope is deferred to
 * a follow-up phase.
 *
 * Single bulk RNG call:
 *     `standard_normal_fill(rng, normals, n_samples)`.
 */
#ifndef N4M_CORE_AUG_DRIFT_PATH_LENGTH_H
#define N4M_CORE_AUG_DRIFT_PATH_LENGTH_H

#include <stdint.h>

#include "n4m/n4m.h"
#include "core/common/rng_pcg64.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_aug_path_length_state_t n4m_aug_path_length_state_t;

n4m_aug_path_length_state_t* n4m_aug_path_length_state_new(
    n4m_rng_pcg64* rng,
    double path_length_std,
    double min_path_length);

void n4m_aug_path_length_state_free(n4m_aug_path_length_state_t* state);

n4m_status_t n4m_aug_path_length_state_apply(
    n4m_aug_path_length_state_t* state,
    const double* X, int64_t rows, int64_t cols, double* out);

#ifdef __cplusplus
}
#endif

#endif /* N4M_CORE_AUG_DRIFT_PATH_LENGTH_H */
