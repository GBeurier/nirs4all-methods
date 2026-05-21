/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * NIRS regression metrics. Mirror of nirs4all.core.metrics._eval_single.
 *
 * All accumulators are double precision. The reference Python implementation
 * uses NumPy reductions; we keep the same evaluation order (i.e. mean and
 * std are computed in one forward pass with two accumulators) so the floating
 * point shape matches.
 *
 * "std" is the population (ddof=0) standard deviation, matching np.std(...).
 */

#include "nirs_metrics.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

static n4m_status_t check_inputs(const double* y_true, const double* y_pred,
                                  int64_t n, const double* out) {
    if (y_true == NULL || y_pred == NULL || out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (n <= 0) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    return N4M_OK;
}

n4m_status_t n4m_metric_rmse_impl(const double* y_true, const double* y_pred,
                                   int64_t n, double* out) {
    const n4m_status_t s = check_inputs(y_true, y_pred, n, out);
    if (s != N4M_OK) return s;
    double sse = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double d = y_true[i] - y_pred[i];
        sse += d * d;
    }
    *out = sqrt(sse / (double)n);
    return N4M_OK;
}

n4m_status_t n4m_metric_mae_impl(const double* y_true, const double* y_pred,
                                  int64_t n, double* out) {
    const n4m_status_t s = check_inputs(y_true, y_pred, n, out);
    if (s != N4M_OK) return s;
    double sae = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        sae += fabs(y_true[i] - y_pred[i]);
    }
    *out = sae / (double)n;
    return N4M_OK;
}

n4m_status_t n4m_metric_bias_impl(const double* y_true, const double* y_pred,
                                   int64_t n, double* out) {
    const n4m_status_t s = check_inputs(y_true, y_pred, n, out);
    if (s != N4M_OK) return s;
    double sum_dev = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        sum_dev += y_pred[i] - y_true[i];
    }
    *out = sum_dev / (double)n;
    return N4M_OK;
}

static double compute_std(const double* y, int64_t n) {
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) sum += y[i];
    const double mean = sum / (double)n;
    double sq_dev = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double d = y[i] - mean;
        sq_dev += d * d;
    }
    return sqrt(sq_dev / (double)n);
}

n4m_status_t n4m_metric_sep_impl(const double* y_true, const double* y_pred,
                                  int64_t n, double* out) {
    const n4m_status_t s = check_inputs(y_true, y_pred, n, out);
    if (s != N4M_OK) return s;
    /* SEP = std(y_pred - y_true), ddof=0. */
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) sum += y_pred[i] - y_true[i];
    const double mean = sum / (double)n;
    double sq_dev = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double r = (y_pred[i] - y_true[i]) - mean;
        sq_dev += r * r;
    }
    *out = sqrt(sq_dev / (double)n);
    return N4M_OK;
}

n4m_status_t n4m_metric_rpd_impl(const double* y_true, const double* y_pred,
                                  int64_t n, double* out) {
    const n4m_status_t s = check_inputs(y_true, y_pred, n, out);
    if (s != N4M_OK) return s;
    /* SEP = std(y_pred - y_true). */
    double sum_res = 0.0;
    for (int64_t i = 0; i < n; ++i) sum_res += y_pred[i] - y_true[i];
    const double mean_res = sum_res / (double)n;
    double sq_res = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double r = (y_pred[i] - y_true[i]) - mean_res;
        sq_res += r * r;
    }
    const double sep = sqrt(sq_res / (double)n);
    /* SD = std(y_true). */
    const double sd = compute_std(y_true, n);
    if (sep == 0.0) {
        *out = HUGE_VAL;
    } else {
        *out = sd / sep;
    }
    return N4M_OK;
}

/* Helper: NumPy linear-interpolation percentile on a sorted array. */
static double linear_quantile(const double* sorted, int64_t n, double q) {
    /* q in [0, 1].  pos = q * (n - 1). */
    const double pos = q * (double)(n - 1);
    const int64_t lo = (int64_t)floor(pos);
    const int64_t hi = (int64_t)ceil(pos);
    if (lo == hi) return sorted[lo];
    const double frac = pos - (double)lo;
    return sorted[lo] + frac * (sorted[hi] - sorted[lo]);
}

static int dbl_cmp(const void* a, const void* b) {
    const double da = *(const double*)a;
    const double db = *(const double*)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

n4m_status_t n4m_metric_rpiq_impl(const double* y_true, const double* y_pred,
                                   int64_t n, double* out) {
    const n4m_status_t s = check_inputs(y_true, y_pred, n, out);
    if (s != N4M_OK) return s;
    /* RMSE. */
    double sse = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double d = y_true[i] - y_pred[i];
        sse += d * d;
    }
    const double rmse = sqrt(sse / (double)n);
    /* IQR = q75 - q25 via NumPy linear interpolation. */
    double* sorted = (double*)malloc((size_t)n * sizeof(double));
    if (sorted == NULL) return N4M_ERR_OUT_OF_MEMORY;
    memcpy(sorted, y_true, (size_t)n * sizeof(double));
    qsort(sorted, (size_t)n, sizeof(double), dbl_cmp);
    const double q25 = linear_quantile(sorted, n, 0.25);
    const double q75 = linear_quantile(sorted, n, 0.75);
    const double iqr = q75 - q25;
    free(sorted);
    if (rmse == 0.0) {
        *out = HUGE_VAL;
    } else {
        *out = iqr / rmse;
    }
    return N4M_OK;
}

n4m_status_t n4m_metric_r2_impl(const double* y_true, const double* y_pred,
                                 int64_t n, double* out) {
    const n4m_status_t s = check_inputs(y_true, y_pred, n, out);
    if (s != N4M_OK) return s;
    double sum_y = 0.0;
    for (int64_t i = 0; i < n; ++i) sum_y += y_true[i];
    const double mean_y = sum_y / (double)n;
    double sse = 0.0;
    double sst = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double e = y_true[i] - y_pred[i];
        const double v = y_true[i] - mean_y;
        sse += e * e;
        sst += v * v;
    }
    if (sst == 0.0) {
        /* sklearn convention: constant y -> R² = 1 if predictions are perfect
         * (SSE == 0), else 0.0. */
        *out = (sse == 0.0) ? 1.0 : 0.0;
        return N4M_OK;
    }
    *out = 1.0 - (sse / sst);
    return N4M_OK;
}

n4m_status_t n4m_metric_nrmse_impl(const double* y_true, const double* y_pred,
                                    int64_t n, double* out) {
    const n4m_status_t s = check_inputs(y_true, y_pred, n, out);
    if (s != N4M_OK) return s;
    double sse = 0.0;
    double sum_y = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double d = y_true[i] - y_pred[i];
        sse += d * d;
        sum_y += y_true[i];
    }
    const double rmse   = sqrt(sse / (double)n);
    const double mean_y = sum_y / (double)n;
    if (mean_y == 0.0) {
        *out = HUGE_VAL;
    } else {
        *out = rmse / mean_y;
    }
    return N4M_OK;
}
