/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * TruncatedPeak edge-artifact augmenter — internal engine.
 *
 * Per-sample, with a configurable probability, adds a tail of a Gaussian
 * peak whose centre lies outside the measured wavelength range, mimicking
 * absorption bands clipped at the spectral boundary. Mirrors
 * `nirs4all.operators.augmentation.edge_artifacts.TruncatedPeakAugmenter`.
 *
 * Pure C. INTERNAL.
 */
#ifndef N4M_CORE_AUGMENT_EDGE_TRUNCATED_PEAK_H
#define N4M_CORE_AUGMENT_EDGE_TRUNCATED_PEAK_H

#include <stddef.h>
#include <stdint.h>

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_aug_truncated_peak_state_t
    n4m_aug_truncated_peak_state_t;

n4m_aug_truncated_peak_state_t* n4m_aug_truncated_peak_state_new(
    double peak_probability,
    double amplitude_min,
    double amplitude_max,
    double width_min,
    double width_max,
    int    left_edge,
    int    right_edge);

void n4m_aug_truncated_peak_state_free(
    n4m_aug_truncated_peak_state_t* state);

n4m_status_t n4m_aug_truncated_peak_state_apply(
    const n4m_aug_truncated_peak_state_t* state,
    void* rng,
    const double* X, int64_t rows, int64_t cols,
    const double* wavelengths,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_AUGMENT_EDGE_TRUNCATED_PEAK_H */
