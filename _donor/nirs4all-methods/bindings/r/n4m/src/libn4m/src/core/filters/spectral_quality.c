/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * SpectralQuality reference implementation.
 *
 * For each row we apply a fixed set of threshold checks (see header).
 * Variance is computed with the np.nanvar convention:
 *   - skip NaN values
 *   - mean = sum(non-nan) / n_non_nan
 *   - var  = sum((x - mean)^2 over non-nan) / n_non_nan
 *   - an all-NaN row yields NaN variance → fails the check.
 *
 * Output is a row-wise boolean keep mask (1 = keep, 0 = exclude). The
 * thresholds are exact (no float tolerance) so the n4m output matches the
 * frozen NumPy reference bit-for-bit on mask values.
 */

#include "spectral_quality.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct n4m_filter_quality_state_t {
    double max_nan_ratio;
    double max_zero_ratio;
    double min_variance;
    int    use_max;
    double max_value;
    int    use_min;
    double min_value;
    int    check_inf;
};

n4m_filter_quality_state_t* n4m_filter_quality_state_new(
    double max_nan_ratio,
    double max_zero_ratio,
    double min_variance,
    int    use_max,
    double max_value,
    int    use_min,
    double min_value,
    int    check_inf) {
    if (!(max_nan_ratio  >= 0.0 && max_nan_ratio  <= 1.0)) return NULL;
    if (!(max_zero_ratio >= 0.0 && max_zero_ratio <= 1.0)) return NULL;
    if (min_variance < 0.0) return NULL;
    n4m_filter_quality_state_t* s =
        (n4m_filter_quality_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->max_nan_ratio  = max_nan_ratio;
    s->max_zero_ratio = max_zero_ratio;
    s->min_variance   = min_variance;
    s->use_max        = use_max ? 1 : 0;
    s->max_value      = max_value;
    s->use_min        = use_min ? 1 : 0;
    s->min_value      = min_value;
    s->check_inf      = check_inf ? 1 : 0;
    return s;
}

void n4m_filter_quality_state_free(n4m_filter_quality_state_t* state) {
    free(state);
}

/* Per-row evaluation. Returns 1 if the row passes every enabled check. */
static int row_passes(const n4m_filter_quality_state_t* state,
                       const double* row, int64_t cols) {
    /* 1. NaN ratio + Inf check + Zero ratio + value-range scan: do in a
     * single pass to limit cache traffic.
     *
     * Variance semantics follow np.nanvar exactly:
     *   - skip NaN
     *   - Inf participates in the mean and the squared deviation, producing
     *     NaN as soon as any element is Inf (matches the +inf - +inf = NaN
     *     arithmetic numpy actually executes).
     * That subtlety matters when check_inf is disabled: a row with Inf
     * still fails because the variance check sees NaN. */
    int64_t nan_count  = 0;
    int64_t zero_count = 0;
    int     saw_inf    = 0;
    double  vmin       =  HUGE_VAL;
    double  vmax       = -HUGE_VAL;
    double  sum_non_nan = 0.0;
    int64_t n_non_nan = 0;
    for (int64_t j = 0; j < cols; ++j) {
        const double v = row[j];
        if (isnan(v)) {
            ++nan_count;
            /* NaN counts as non-zero per the reference's
             *   np.nan_to_num(X, nan=1.0)
             * trick — replacement with 1.0 keeps the value non-zero. */
            continue;
        }
        if (isinf(v)) {
            saw_inf = 1;
            if (v > vmax) vmax = v;
            if (v < vmin) vmin = v;
            /* Inf is treated as non-zero (matches np.nan_to_num path
             * leaving Inf intact and the comparator x == 0 being false). */
        } else {
            if (v == 0.0) ++zero_count;
            if (v > vmax) vmax = v;
            if (v < vmin) vmin = v;
        }
        /* Both finite and Inf rows contribute to the variance sums to
         * reproduce the np.nanvar arithmetic. */
        sum_non_nan += v;
        ++n_non_nan;
    }
    /* Edge case: cols == 0. The reference rejects such rows because every
     * check becomes ill-defined. */
    if (cols == 0) return 0;

    const double nan_ratio  = (double)nan_count  / (double)cols;
    const double zero_ratio = (double)zero_count / (double)cols;

    if (nan_ratio > state->max_nan_ratio) return 0;
    if (state->check_inf && saw_inf)      return 0;
    if (zero_ratio > state->max_zero_ratio) return 0;

    /* Variance (np.nanvar semantics). When the row is all-NaN, the result is
     * NaN — fail. */
    double var;
    if (n_non_nan <= 0) {
        var = (double)NAN;
    } else {
        const double mean = sum_non_nan / (double)n_non_nan;
        double s = 0.0;
        for (int64_t j = 0; j < cols; ++j) {
            const double v = row[j];
            if (isnan(v)) continue;
            const double d = v - mean;
            s += d * d;
        }
        var = s / (double)n_non_nan;
    }
    if (!(var >= state->min_variance)) {
        /* NaN compares false in both directions → fail. */
        return 0;
    }

    if (state->use_max) {
        /* nanmax of all-NaN is NaN → fail. */
        if (!(vmax <= state->max_value)) return 0;
    }
    if (state->use_min) {
        if (!(vmin >= state->min_value)) return 0;
    }
    return 1;
}

n4m_status_t n4m_filter_quality_state_apply(
    const n4m_filter_quality_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    uint8_t* mask_out,
    int64_t* out_n_samples,
    int64_t* out_n_kept,
    int64_t* out_n_excluded) {
    if (state == NULL || mask_out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (rows > 0 && X == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    int64_t kept = 0;
    for (int64_t r = 0; r < rows; ++r) {
        const double* row = X + (size_t)r * (size_t)cols;
        const uint8_t keep = (uint8_t)row_passes(state, row, cols);
        mask_out[r] = keep;
        kept += keep;
    }
    if (out_n_samples)  *out_n_samples  = rows;
    if (out_n_kept)     *out_n_kept     = kept;
    if (out_n_excluded) *out_n_excluded = rows - kept;
    return N4M_OK;
}
