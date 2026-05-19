/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * LocalClipping augmenter — clips each spectrum within `n_regions` random
 * bands to the band's 90th percentile, simulating detector saturation.
 *
 * Algorithm (matches nirs4all.operators.augmentation.spectral.LocalClipping):
 *
 *   centers = rng.integers(0, n_features, size=(n_samples, n_regions))
 *   widths  = rng.integers(width_lo, width_hi, size=(n_samples, n_regions))
 *   for each sample i, for each region r:
 *     start = max(0, centers[i, r] - widths[i, r] // 2)
 *     end   = min(n_features, centers[i, r] + widths[i, r] // 2)
 *     limit = np.percentile(X[i, start:end], 90)   (linear interpolation)
 *     X[i, start:end] = np.minimum(X[i, start:end], limit)
 *
 * NumPy's default `np.percentile(..., method='linear')` is used here — the
 * same `percentile_linear` helper that powers c4a_filter_y_outlier MAD /
 * percentile bounds.
 */
#ifndef CHEMOMETRICS4ALL_CORE_AUG_LOCAL_CLIP_H
#define CHEMOMETRICS4ALL_CORE_AUG_LOCAL_CLIP_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"
#include "core/common/rng_pcg64.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_aug_local_clip_state_t c4a_aug_local_clip_state_t;

c4a_aug_local_clip_state_t* c4a_aug_local_clip_state_new(
    c4a_rng_pcg64* rng,
    int32_t n_regions,
    int32_t width_lo, int32_t width_hi);

void c4a_aug_local_clip_state_free(c4a_aug_local_clip_state_t* state);

c4a_status_t c4a_aug_local_clip_state_apply(
    c4a_aug_local_clip_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_AUG_LOCAL_CLIP_H */
