/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * RollingBall — Kneen & Annegarn 1996 morphological baseline.
 *
 * Internal parity fixture: parity/python_generator/src/c4a_parity_pybaselines_ref/rolling_ball.py
 *
 * Implementation: O(n * w) scan with centred windows clipped at the array
 * boundaries. The pybaselines variant uses scipy.ndimage.{minimum,maximum}_filter
 * with `mode='nearest'`; for the parity reference we emit the equivalent
 * value-clipped-to-boundary behaviour (centred half-window R, window indices
 * [max(0, i-R), min(n-1, i+R)]).
 *
 * The optional final moving-average smoothing uses the same centred clipped
 * window pattern. `smooth_half_window == 0` disables smoothing — the output
 * is z directly.
 */

#include "rolling_ball.h"

#include <stdlib.h>

struct c4a_pp_rolling_ball_state_t {
    int32_t half_window;
    int32_t smooth_half_window;
};

c4a_pp_rolling_ball_state_t* c4a_pp_rolling_ball_state_new(int32_t half_window,
                                                             int32_t smooth_half_window) {
    if (half_window < 1 || smooth_half_window < 0) {
        return NULL;
    }
    c4a_pp_rolling_ball_state_t* s =
        (c4a_pp_rolling_ball_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->half_window        = half_window;
    s->smooth_half_window = smooth_half_window;
    return s;
}

void c4a_pp_rolling_ball_state_free(c4a_pp_rolling_ball_state_t* state) {
    free(state);
}

static void min_filter(const double* src, double* dst, int64_t n, int32_t w) {
    for (int64_t i = 0; i < n; ++i) {
        const int64_t lo = (i - w >= 0) ? (i - w) : 0;
        const int64_t hi = (i + w < n)  ? (i + w) : (n - 1);
        double m = src[lo];
        for (int64_t j = lo + 1; j <= hi; ++j) {
            if (src[j] < m) m = src[j];
        }
        dst[i] = m;
    }
}

static void max_filter(const double* src, double* dst, int64_t n, int32_t w) {
    for (int64_t i = 0; i < n; ++i) {
        const int64_t lo = (i - w >= 0) ? (i - w) : 0;
        const int64_t hi = (i + w < n)  ? (i + w) : (n - 1);
        double m = src[lo];
        for (int64_t j = lo + 1; j <= hi; ++j) {
            if (src[j] > m) m = src[j];
        }
        dst[i] = m;
    }
}

static void moving_avg(const double* src, double* dst, int64_t n, int32_t w) {
    for (int64_t i = 0; i < n; ++i) {
        const int64_t lo = (i - w >= 0) ? (i - w) : 0;
        const int64_t hi = (i + w < n)  ? (i + w) : (n - 1);
        double s = 0.0;
        for (int64_t j = lo; j <= hi; ++j) s += src[j];
        dst[i] = s / (double)(hi - lo + 1);
    }
}

c4a_status_t c4a_pp_rolling_ball_state_apply(
    const c4a_pp_rolling_ball_state_t* state,
    const double* X,
    int64_t rows, int64_t cols,
    double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0) {
        return C4A_OK;
    }

    const int32_t W = state->half_window;
    const int32_t S = state->smooth_half_window;

    double* buf_a = (double*)malloc((size_t)cols * sizeof(double));
    double* buf_b = (double*)malloc((size_t)cols * sizeof(double));
    if (!buf_a || !buf_b) {
        free(buf_a); free(buf_b);
        return C4A_ERR_OUT_OF_MEMORY;
    }

    for (int64_t r = 0; r < rows; ++r) {
        const double* y = X + (size_t)r * (size_t)cols;
        /* erosion */
        min_filter(y, buf_a, cols, W);
        /* dilation */
        max_filter(buf_a, buf_b, cols, W);
        const double* z;
        if (S > 0) {
            moving_avg(buf_b, buf_a, cols, S);
            z = buf_a;
        } else {
            z = buf_b;
        }
        double* orow = out + (size_t)r * (size_t)cols;
        for (int64_t i = 0; i < cols; ++i) {
            orow[i] = y[i] - z[i];
        }
    }

    free(buf_a); free(buf_b);
    return C4A_OK;
}
