/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * SNIP — Ryan 1988 / Morháč 1997 statistics-sensitive non-linear iterative
 * peak-clipping baseline.
 *
 * Internal parity fixture:
 * parity/python_generator/src/n4m_parity_pybaselines_ref/snip.py
 *
 * Algorithm: see snip.h. The benchmark comparator is pybaselines.smooth.snip
 * with its default raw-data filter_order=2 contract, including linear
 * extrapolation padding at both edges.
 */

#include "snip.h"

#include <stdlib.h>

struct n4m_pp_snip_state_t {
    int32_t max_half_window;
};

n4m_pp_snip_state_t* n4m_pp_snip_state_new(int32_t max_half_window) {
    if (max_half_window < 1) {
        return NULL;
    }
    n4m_pp_snip_state_t* s =
        (n4m_pp_snip_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->max_half_window = max_half_window;
    return s;
}

void n4m_pp_snip_state_free(n4m_pp_snip_state_t* state) {
    free(state);
}

static void fit_line_and_predict_left(const double* y, int64_t n,
                                      int32_t pad, double* out) {
    int64_t win = (int64_t)pad;
    if (win > n) win = n;
    if (win <= 1) {
        for (int32_t i = 0; i < pad; ++i) out[i] = y[0];
        return;
    }

    double sum_x = 0.0;
    double sum_y = 0.0;
    for (int64_t k = 0; k < win; ++k) {
        sum_x += (double)pad + (double)k;
        sum_y += y[k];
    }
    const double mean_x = sum_x / (double)win;
    const double mean_y = sum_y / (double)win;
    double num = 0.0;
    double den = 0.0;
    for (int64_t k = 0; k < win; ++k) {
        const double dx = (double)pad + (double)k - mean_x;
        num += dx * (y[k] - mean_y);
        den += dx * dx;
    }
    const double slope = den > 0.0 ? num / den : 0.0;
    const double intercept = mean_y - slope * mean_x;
    for (int32_t i = 0; i < pad; ++i) {
        out[i] = intercept + slope * (double)i;
    }
}

static void fit_line_and_predict_right(const double* y, int64_t n,
                                       int32_t pad, double* out) {
    int64_t win = (int64_t)pad;
    if (win > n) win = n;
    if (win <= 1) {
        for (int32_t i = 0; i < pad; ++i) out[i] = y[n - 1];
        return;
    }

    const int64_t start = n - win;
    double sum_x = 0.0;
    double sum_y = 0.0;
    for (int64_t k = 0; k < win; ++k) {
        sum_x += (double)pad + (double)start + (double)k;
        sum_y += y[start + k];
    }
    const double mean_x = sum_x / (double)win;
    const double mean_y = sum_y / (double)win;
    double num = 0.0;
    double den = 0.0;
    for (int64_t k = 0; k < win; ++k) {
        const double dx = (double)pad + (double)start + (double)k - mean_x;
        num += dx * (y[start + k] - mean_y);
        den += dx * dx;
    }
    const double slope = den > 0.0 ? num / den : 0.0;
    const double intercept = mean_y - slope * mean_x;
    for (int32_t i = 0; i < pad; ++i) {
        const double x = (double)pad + (double)n + (double)i;
        out[i] = intercept + slope * x;
    }
}

n4m_status_t n4m_pp_snip_state_apply(const n4m_pp_snip_state_t* state,
                                      const double* X,
                                      int64_t rows, int64_t cols,
                                      double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0) {
        return N4M_OK;
    }
    int32_t maxw = state->max_half_window;
    const int32_t max_effective = (int32_t)((cols - 1) / 2);
    if (maxw > max_effective) maxw = max_effective;
    if (maxw <= 0) {
        for (int64_t i = 0; i < rows * cols; ++i) out[i] = 0.0;
        return N4M_OK;
    }

    const int64_t padded_cols = cols + 2 * (int64_t)maxw;
    double* baseline = (double*)malloc((size_t)padded_cols * sizeof(double));
    double* filters = (double*)malloc((size_t)padded_cols * sizeof(double));
    if (baseline == NULL || filters == NULL) {
        free(baseline);
        free(filters);
        return N4M_ERR_OUT_OF_MEMORY;
    }

    for (int64_t r = 0; r < rows; ++r) {
        const double* y = X + (size_t)r * (size_t)cols;
        fit_line_and_predict_left(y, cols, maxw, baseline);
        for (int64_t i = 0; i < cols; ++i) {
            baseline[(int64_t)maxw + i] = y[i];
        }
        fit_line_and_predict_right(y, cols, maxw,
                                   baseline + (int64_t)maxw + cols);

        for (int32_t w = 1; w <= maxw; ++w) {
            const int64_t lo = (int64_t)w;
            const int64_t hi = padded_cols - (int64_t)w;
            for (int64_t i = lo; i < hi; ++i) {
                filters[i] = 0.5 * (baseline[i - w] + baseline[i + w]);
            }
            for (int64_t i = lo; i < hi; ++i) {
                if (filters[i] < baseline[i]) baseline[i] = filters[i];
            }
        }

        double* orow = out + (size_t)r * (size_t)cols;
        for (int64_t i = 0; i < cols; ++i) {
            orow[i] = y[i] - baseline[(int64_t)maxw + i];
        }
    }

    free(filters);
    free(baseline);
    return N4M_OK;
}
