/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for the MixupAugmenter — within-batch convex combination
 * of pairs of spectra weighted by a Beta(alpha, alpha) random variable.
 *
 * Reference: nirs4all.operators.augmentation.spectral.MixupAugmenter.
 *
 * Algorithm (matches numpy reference exactly when the same PCG64 stream is
 * supplied):
 *
 *   indices = rng.permutation(n_samples)           # shuffled row indices
 *   lam     = rng.beta(alpha, alpha, size=(n_samples, 1))
 *   out[i]  = lam[i] * X[i] + (1 - lam[i]) * X[indices[i]]
 *
 * MixupAugmenter combines TWO samples within the same batch — the output
 * has the same number of rows as the input, and every output row is a
 * convex combination of two source rows from X.
 *
 * The RNG handle is supplied by the caller; the augmenter does not own it.
 * RNG draws advance the supplied stream so consecutive _apply calls produce
 * independent shuffles and lam vectors.
 */
#ifndef CHEMOMETRICS4ALL_CORE_AUG_MIXUP_H
#define CHEMOMETRICS4ALL_CORE_AUG_MIXUP_H

#include "chemometrics4all/c4a.h"

#include "core/common/rng_pcg64.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_aug_mixup_state_t c4a_aug_mixup_state_t;

c4a_aug_mixup_state_t* c4a_aug_mixup_state_new(double alpha);
void                   c4a_aug_mixup_state_free(c4a_aug_mixup_state_t* state);

c4a_status_t c4a_aug_mixup_apply_impl(const c4a_aug_mixup_state_t* state,
                                      c4a_rng_pcg64* rng,
                                      const double* X, int64_t rows, int64_t cols,
                                      double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_AUG_MIXUP_H */
