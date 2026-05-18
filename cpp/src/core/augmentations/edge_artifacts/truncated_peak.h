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
#ifndef CHEMOMETRICS4ALL_CORE_AUGMENT_EDGE_TRUNCATED_PEAK_H
#define CHEMOMETRICS4ALL_CORE_AUGMENT_EDGE_TRUNCATED_PEAK_H

#include <stddef.h>
#include <stdint.h>

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_aug_truncated_peak_state_t
    c4a_aug_truncated_peak_state_t;

c4a_aug_truncated_peak_state_t* c4a_aug_truncated_peak_state_new(
    double peak_probability,
    double amplitude_min,
    double amplitude_max,
    double width_min,
    double width_max,
    int    left_edge,
    int    right_edge);

void c4a_aug_truncated_peak_state_free(
    c4a_aug_truncated_peak_state_t* state);

c4a_status_t c4a_aug_truncated_peak_state_apply(
    const c4a_aug_truncated_peak_state_t* state,
    void* rng,
    const double* X, int64_t rows, int64_t cols,
    const double* wavelengths,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_AUGMENT_EDGE_TRUNCATED_PEAK_H */
