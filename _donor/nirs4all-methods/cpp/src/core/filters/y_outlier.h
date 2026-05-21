/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * YOutlierFilter — y-array outlier detection with four selectable methods.
 *
 * Mirrors `nirs4all.operators.filters.YOutlierFilter` (Phase 12 — first member
 * of the new `n4m_filter_*` ABI category). The operator follows the classical
 * sklearn fit/apply contract:
 *   - `_state_new` constructs an unfitted state from the method + thresholds.
 *   - `_state_fit` learns the per-method bounds from the training y vector.
 *     Idempotent: replaces any previously fitted bounds.
 *   - `_state_apply` applies the previously learned bounds to a (possibly
 *     different) y vector, returning the keep-mask and statistics. Returns
 *     `N4M_ERR_NOT_FITTED` when called on an unfitted state.
 *   - `_state_is_fitted` reports whether `_state_fit` has been called at
 *     least once.
 *
 * Four methods:
 *   - IQR        : `[Q1 - t*IQR, Q3 + t*IQR]` (t default 1.5)
 *   - Z-score    : `[mu - t*sigma, mu + t*sigma]` (t default 3.0)
 *   - Percentile : `[pct(y, lower_pct), pct(y, upper_pct)]`
 *   - MAD        : `[median +/- t * MAD * 1.4826]` (t default 3.5)
 *
 * Quantiles use NumPy 1.26.4's default "linear" interpolation
 * (`np.percentile(..., method="linear")`).
 *
 * Output mask convention: `mask[i] = 1` keeps sample i, `mask[i] = 0` excludes.
 * NaN samples and degenerate (single-sample / zero-scale) inputs are treated
 * identically to the Python reference.
 */
#ifndef N4M_CORE_FILTERS_Y_OUTLIER_H
#define N4M_CORE_FILTERS_Y_OUTLIER_H

#include <stdint.h>

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Method enum mirrored from the public ABI header (`n4m_filters.h`). We
 * redeclare the integer codes here so the engine can be built without pulling
 * in the public filters header, matching the pattern used by the rest of the
 * core. */
typedef enum n4m_y_outlier_method_kind_t {
    N4M_CORE_Y_OUTLIER_IQR        = 0,
    N4M_CORE_Y_OUTLIER_ZSCORE     = 1,
    N4M_CORE_Y_OUTLIER_PERCENTILE = 2,
    N4M_CORE_Y_OUTLIER_MAD        = 3
} n4m_y_outlier_method_kind_t;

/* Filter statistics returned alongside the mask. Mirrors the public
 * `n4m_filter_stats_t` layout — same field order, same types. The
 * engine writes into a caller-supplied struct so the C ABI wrapper can
 * just forward the pointer. */
typedef struct n4m_core_filter_stats_t {
    int64_t n_samples;
    int64_t n_kept;
    int64_t n_excluded;
    double  exclusion_rate;
} n4m_core_filter_stats_t;

typedef struct n4m_filter_y_outlier_state_t n4m_filter_y_outlier_state_t;

n4m_filter_y_outlier_state_t* n4m_filter_y_outlier_state_new(
    int32_t method, double threshold,
    double lower_pct, double upper_pct);

void n4m_filter_y_outlier_state_free(n4m_filter_y_outlier_state_t* state);

/* Learn the bounds from the training y vector. Idempotent: subsequent
 * calls overwrite the previously learned bounds.
 *   y may be NULL only when n == 0.
 *   n < 0 is rejected with N4M_ERR_INVALID_ARGUMENT. */
n4m_status_t n4m_filter_y_outlier_state_fit(
    n4m_filter_y_outlier_state_t* state,
    const double* y, int64_t n);

/* Apply the fitted bounds to a (possibly different) y vector. Returns
 * N4M_ERR_INVALID_STATE when called on an unfitted state. */
n4m_status_t n4m_filter_y_outlier_state_apply(
    const n4m_filter_y_outlier_state_t* state,
    const double* y, int64_t n,
    uint8_t* mask_out,
    n4m_core_filter_stats_t* stats_out);

/* Report whether `_state_fit` has been called at least once. Writes 0/1
 * into *out. Returns N4M_ERR_NULL_POINTER if either argument is NULL. */
n4m_status_t n4m_filter_y_outlier_state_is_fitted(
    const n4m_filter_y_outlier_state_t* state, int* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_FILTERS_Y_OUTLIER_H */
