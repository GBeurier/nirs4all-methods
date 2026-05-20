/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * YOutlierFilter — implementation.
 *
 * The engine now splits the original single-call ``state_apply`` (a
 * fit+mask short-circuit) into two steps:
 *
 *   1. `state_fit(y, n)`: compute the lower/upper bounds from the training
 *      y vector using the configured method. The bounds are stored on the
 *      state along with a ``fitted = 1`` flag.
 *   2. `state_apply(y, n, mask, stats)`: sweep through the supplied y and
 *      write the keep-mask + stats using the previously learned bounds.
 *      Returns C4A_ERR_INVALID_STATE when called on an unfitted state.
 *
 * Algorithm sketch (fit step):
 *   1. Compute `n_valid`: count of finite (non-NaN) y values.
 *   2. If `n_valid == 0`: degenerate. Bounds left at +/-inf — the mask
 *      will be all-zero since all of y is NaN.
 *   3. If `n_valid == 1`: bounds collapse to `[v - 1e-10, v + 1e-10]`.
 *   4. Otherwise, dispatch to one of the four method-specific bound
 *      computations against the NaN-filtered slice.
 *
 * Apply step (pure sweep):
 *   mask[i] = (y[i] is finite) && (lower <= y[i] <= upper) ? 1 : 0.
 *
 * Sorting strategy (fit step): we copy the NaN-filtered values into a
 * working buffer and call qsort with a comparator that orders NaNs last
 * (NaNs are already filtered out, but the comparator is robust).
 * Percentile interpolation matches numpy's `method="linear"` (default in
 * NumPy 1.26.4) exactly:
 *     i_frac = q/100 * (n - 1)
 *     lo     = floor(i_frac)
 *     hi     = ceil(i_frac)
 *     return sorted[lo] + (i_frac - lo) * (sorted[hi] - sorted[lo])
 *
 * Z-score: `np.std` uses ddof=0 by default — biased standard deviation.
 * MAD scale factor: `1.4826` matches the Python reference (Gaussian-
 * consistent normalisation, sigma ~= 1.4826 * MAD on normal data).
 */

#include "y_outlier.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct c4a_filter_y_outlier_state_t {
    int32_t method;
    double  threshold;
    double  lower_pct;
    double  upper_pct;
    /* Fitted bounds (valid only when fitted != 0). */
    double  lower_bound;
    double  upper_bound;
    int     fitted;
};

c4a_filter_y_outlier_state_t* c4a_filter_y_outlier_state_new(
    int32_t method, double threshold,
    double lower_pct, double upper_pct) {
    if (method != C4A_CORE_Y_OUTLIER_IQR    &&
        method != C4A_CORE_Y_OUTLIER_ZSCORE &&
        method != C4A_CORE_Y_OUTLIER_PERCENTILE &&
        method != C4A_CORE_Y_OUTLIER_MAD) {
        return NULL;
    }
    if (threshold <= 0.0) {
        return NULL;
    }
    if (!(lower_pct >= 0.0 && upper_pct <= 100.0 && lower_pct < upper_pct)) {
        return NULL;
    }
    c4a_filter_y_outlier_state_t* s =
        (c4a_filter_y_outlier_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->method      = method;
    s->threshold   = threshold;
    s->lower_pct   = lower_pct;
    s->upper_pct   = upper_pct;
    s->lower_bound = -HUGE_VAL;
    s->upper_bound =  HUGE_VAL;
    s->fitted      = 0;
    return s;
}

void c4a_filter_y_outlier_state_free(c4a_filter_y_outlier_state_t* state) {
    free(state);
}

/* qsort comparator for ascending doubles (NaN-free input expected). */
static int compare_double_asc(const void* a, const void* b) {
    const double da = *(const double*)a;
    const double db = *(const double*)b;
    if (da < db) return -1;
    if (da > db) return  1;
    return 0;
}

/* NumPy linear-interpolation percentile on a pre-sorted array.
 *   pct in [0, 100]; n >= 1.
 * Matches `np.percentile(sorted, pct, method="linear")` element-wise. */
static double percentile_linear(const double* sorted, int64_t n, double pct) {
    if (n == 1) {
        return sorted[0];
    }
    /* Convert percentile to fractional index. */
    const double frac_idx = (pct / 100.0) * (double)(n - 1);
    int64_t lo = (int64_t)floor(frac_idx);
    int64_t hi = (int64_t)ceil(frac_idx);
    if (lo < 0)  lo = 0;
    if (hi >= n) hi = n - 1;
    if (lo == hi) {
        return sorted[lo];
    }
    const double weight = frac_idx - (double)lo;
    return sorted[lo] + weight * (sorted[hi] - sorted[lo]);
}

/* Compute bounds for the IQR method on a pre-sorted array. */
static void compute_bounds_iqr(const double* sorted, int64_t n,
                                double threshold,
                                double* lo_out, double* hi_out) {
    const double q1  = percentile_linear(sorted, n, 25.0);
    const double q3  = percentile_linear(sorted, n, 75.0);
    const double iqr = q3 - q1;
    *lo_out = q1 - threshold * iqr;
    *hi_out = q3 + threshold * iqr;
}

/* Compute bounds for the Z-score method on the NaN-filtered array. */
static void compute_bounds_zscore(const double* y_valid, int64_t n,
                                   double threshold,
                                   double* lo_out, double* hi_out) {
    /* mean. */
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) sum += y_valid[i];
    const double mean = sum / (double)n;
    /* std with ddof=0 (numpy default). */
    double sq = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double d = y_valid[i] - mean;
        sq += d * d;
    }
    const double std_ = sqrt(sq / (double)n);
    if (std_ == 0.0) {
        *lo_out = mean - 1e-10;
        *hi_out = mean + 1e-10;
    } else {
        *lo_out = mean - threshold * std_;
        *hi_out = mean + threshold * std_;
    }
}

/* Compute bounds for the percentile method. */
static void compute_bounds_percentile(const double* sorted, int64_t n,
                                       double lower_pct, double upper_pct,
                                       double* lo_out, double* hi_out) {
    *lo_out = percentile_linear(sorted, n, lower_pct);
    *hi_out = percentile_linear(sorted, n, upper_pct);
}

/* Compute bounds for the MAD method on the NaN-filtered array. Needs a
 * scratch buffer of size n for the |y - median| sort. */
static void compute_bounds_mad(const double* sorted_y, int64_t n,
                                double threshold,
                                double* lo_out, double* hi_out,
                                double* scratch) {
    /* Median of y. Reuse percentile_linear at 50. */
    const double median = percentile_linear(sorted_y, n, 50.0);
    /* MAD = median(|y - median|). Build into scratch and sort. */
    for (int64_t i = 0; i < n; ++i) {
        const double d = sorted_y[i] - median;
        scratch[i] = (d < 0.0) ? -d : d;
    }
    qsort(scratch, (size_t)n, sizeof(double), compare_double_asc);
    const double mad         = percentile_linear(scratch, n, 50.0);
    const double mad_scaled  = mad * 1.4826;
    if (mad_scaled == 0.0) {
        *lo_out = median - 1e-10;
        *hi_out = median + 1e-10;
    } else {
        *lo_out = median - threshold * mad_scaled;
        *hi_out = median + threshold * mad_scaled;
    }
}

c4a_status_t c4a_filter_y_outlier_state_fit(
    c4a_filter_y_outlier_state_t* state,
    const double* y, int64_t n) {
    if (state == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (n < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (y == NULL && n > 0) {
        return C4A_ERR_NULL_POINTER;
    }

    double lo = -HUGE_VAL;
    double hi =  HUGE_VAL;

    if (n == 0) {
        /* Empty input: bounds stay at +/-inf. Still mark as fitted. */
        state->lower_bound = lo;
        state->upper_bound = hi;
        state->fitted      = 1;
        return C4A_OK;
    }

    /* Pass 1: count finite values and copy them into a working buffer. */
    double* y_valid = (double*)malloc((size_t)n * sizeof(double));
    if (y_valid == NULL) {
        return C4A_ERR_OUT_OF_MEMORY;
    }
    int64_t n_valid = 0;
    for (int64_t i = 0; i < n; ++i) {
        const double v = y[i];
        if (!isnan(v)) {
            y_valid[n_valid++] = v;
        }
    }

    if (n_valid == 0) {
        /* Mirror the Python reference: empty-valid input keeps everything
         * within +/- inf bounds. */
        /* lo, hi already at +/-inf. */
    } else if (n_valid == 1) {
        lo = y_valid[0] - 1e-10;
        hi = y_valid[0] + 1e-10;
    } else {
        switch (state->method) {
            case C4A_CORE_Y_OUTLIER_IQR: {
                qsort(y_valid, (size_t)n_valid, sizeof(double), compare_double_asc);
                compute_bounds_iqr(y_valid, n_valid, state->threshold, &lo, &hi);
                break;
            }
            case C4A_CORE_Y_OUTLIER_ZSCORE: {
                compute_bounds_zscore(y_valid, n_valid, state->threshold, &lo, &hi);
                break;
            }
            case C4A_CORE_Y_OUTLIER_PERCENTILE: {
                qsort(y_valid, (size_t)n_valid, sizeof(double), compare_double_asc);
                compute_bounds_percentile(y_valid, n_valid,
                                           state->lower_pct, state->upper_pct,
                                           &lo, &hi);
                break;
            }
            case C4A_CORE_Y_OUTLIER_MAD: {
                qsort(y_valid, (size_t)n_valid, sizeof(double), compare_double_asc);
                double* scratch = (double*)malloc((size_t)n_valid * sizeof(double));
                if (scratch == NULL) {
                    free(y_valid);
                    return C4A_ERR_OUT_OF_MEMORY;
                }
                compute_bounds_mad(y_valid, n_valid, state->threshold,
                                    &lo, &hi, scratch);
                free(scratch);
                break;
            }
            default:
                free(y_valid);
                return C4A_ERR_INVALID_ARGUMENT;
        }
    }

    free(y_valid);
    state->lower_bound = lo;
    state->upper_bound = hi;
    state->fitted      = 1;
    return C4A_OK;
}

c4a_status_t c4a_filter_y_outlier_state_apply(
    const c4a_filter_y_outlier_state_t* state,
    const double* y, int64_t n,
    uint8_t* mask_out,
    c4a_core_filter_stats_t* stats_out) {
    if (state == NULL || mask_out == NULL || stats_out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (n < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (y == NULL && n > 0) {
        return C4A_ERR_NULL_POINTER;
    }
    if (state->fitted == 0) {
        return C4A_ERR_NOT_FITTED;
    }

    stats_out->n_samples      = n;
    stats_out->n_kept         = 0;
    stats_out->n_excluded     = 0;
    stats_out->exclusion_rate = 0.0;

    if (n == 0) {
        /* Empty input: nothing to mask, exclusion rate 0. */
        return C4A_OK;
    }

    const double lo = state->lower_bound;
    const double hi = state->upper_bound;

    /* Single-pass mask write. */
    int64_t n_kept = 0;
    for (int64_t i = 0; i < n; ++i) {
        const double v = y[i];
        if (isnan(v)) {
            mask_out[i] = 0;
        } else if (v >= lo && v <= hi) {
            mask_out[i] = 1;
            ++n_kept;
        } else {
            mask_out[i] = 0;
        }
    }

    stats_out->n_kept         = n_kept;
    stats_out->n_excluded     = n - n_kept;
    stats_out->exclusion_rate = (double)stats_out->n_excluded / (double)n;
    return C4A_OK;
}

c4a_status_t c4a_filter_y_outlier_state_is_fitted(
    const c4a_filter_y_outlier_state_t* state, int* out) {
    if (state == NULL || out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = state->fitted ? 1 : 0;
    return C4A_OK;
}
