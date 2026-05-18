/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * SpectralQuality sample filter — internal engine.
 *
 * Per-row multi-criteria quality check. A sample is KEPT iff every enabled
 * check passes (logical AND):
 *
 *   1. NaN ratio          <= max_nan_ratio
 *   2. Inf check          (when check_inf): no +/- inf in the row
 *   3. Zero ratio         <= max_zero_ratio  (NaN counted as non-zero)
 *   4. Variance           >= min_variance    (matches np.nanvar)
 *   5. Max value          <= max_value       (only when use_max != 0)
 *   6. Min value          >= min_value       (only when use_min != 0)
 *
 * The operator is stateless w.r.t. fit — the thresholds are constructor
 * parameters. We still expose an idempotent fit() entry point for API
 * symmetry with the rest of the filter category (composite holds onto
 * sub-filter handles; each must respond consistently to is_fitted).
 *
 * Pure C, no dependencies. INTERNAL: never exposed by the public ABI.
 */
#ifndef CHEMOMETRICS4ALL_CORE_FILTERS_SPECTRAL_QUALITY_H
#define CHEMOMETRICS4ALL_CORE_FILTERS_SPECTRAL_QUALITY_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_filter_quality_state_t c4a_filter_quality_state_t;

/* Allocate a SpectralQuality handle. Returns NULL when:
 *   - max_nan_ratio  not in [0, 1]
 *   - max_zero_ratio not in [0, 1]
 *   - min_variance   < 0
 *
 * ``use_max`` and ``use_min`` toggle the max/min value checks (when zero
 * the corresponding threshold is ignored). ``check_inf`` toggles the
 * infinity check. */
c4a_filter_quality_state_t* c4a_filter_quality_state_new(
    double max_nan_ratio,
    double max_zero_ratio,
    double min_variance,
    int    use_max,
    double max_value,
    int    use_min,
    double min_value,
    int    check_inf);

void c4a_filter_quality_state_free(c4a_filter_quality_state_t* state);

/* Compute the keep mask on a (rows, cols) row-major matrix. ``mask_out`` is
 * length rows (1 = keep, 0 = exclude). Optional output stats follow the
 * same triplet as HighLeverage. */
c4a_status_t c4a_filter_quality_state_apply(
    const c4a_filter_quality_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    uint8_t* mask_out,
    int64_t* out_n_samples,
    int64_t* out_n_kept,
    int64_t* out_n_excluded);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_FILTERS_SPECTRAL_QUALITY_H */
