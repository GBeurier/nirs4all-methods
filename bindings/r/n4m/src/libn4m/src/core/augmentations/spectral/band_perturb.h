/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * BandPerturbation augmenter — multiplies and offsets a fixed number of
 * random spectral bands per sample.
 *
 * Algorithm (matches nirs4all.operators.augmentation.spectral.BandPerturbation
 * with `variation_scope="sample"` semantics — the per-sample, per-band
 * variation that the C ABI exposes):
 *
 *   centers = rng.integers(0, n_features,             size=(n_samples, n_bands))
 *   widths  = rng.integers(bw_lo,    bw_hi,           size=(n_samples, n_bands))
 *   gains   = rng.uniform(gain_lo,   gain_hi,         size=(n_samples, n_bands))
 *   offsets = rng.uniform(offset_lo, offset_hi,       size=(n_samples, n_bands))
 *   for each (i, b):
 *     start = max(0, centers[i, b] - widths[i, b] // 2)
 *     end   = min(n_features, centers[i, b] + widths[i, b] // 2)
 *     out[i, start:end] = out[i, start:end] * gains[i, b] + offsets[i, b]
 *
 * Random-stream order: all centers first, then all widths, then all gains,
 * then all offsets (matches NumPy's separate `rng.integers / rng.uniform`
 * calls).
 */
#ifndef N4M_CORE_AUG_BAND_PERTURB_H
#define N4M_CORE_AUG_BAND_PERTURB_H

#include <stdint.h>

#include "n4m/n4m.h"
#include "core/common/rng_pcg64.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_aug_band_perturb_state_t n4m_aug_band_perturb_state_t;

n4m_aug_band_perturb_state_t* n4m_aug_band_perturb_state_new(
    n4m_rng_pcg64* rng,
    int32_t n_bands,
    int32_t bw_lo, int32_t bw_hi,
    double gain_lo, double gain_hi,
    double offset_lo, double offset_hi);

void n4m_aug_band_perturb_state_free(
    n4m_aug_band_perturb_state_t* state);

n4m_status_t n4m_aug_band_perturb_state_apply(
    n4m_aug_band_perturb_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_AUG_BAND_PERTURB_H */
