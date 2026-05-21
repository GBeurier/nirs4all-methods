/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * ResampleTransformer — linear resample of every row to a fixed `num_samples`
 * target column count, sharing the same [0, 1] normalised axis as scipy
 * interp1d.
 *
 * Reference: nirs4all.operators.transforms.features.ResampleTransformer.
 *
 *   src = np.linspace(0, 1, len(x))
 *   dst = np.linspace(0, 1, num_samples)
 *   f   = scipy.interpolate.interp1d(src, x, kind='linear')
 *   out = f(dst)
 *
 * scipy's _call_linear computes (essentially):
 *   idx   = np.searchsorted(src, dst, side='left').clip(1, n - 1)
 *   slope = (y[idx] - y[idx-1]) / (src[idx] - src[idx-1])
 *   out   = slope * (dst - src[idx-1]) + y[idx-1]
 *
 * Critical numerical detail: the source / target axes use np.linspace which
 * fills `start + step * i` with `step = (stop - start) / (N - 1)` and then
 * pins the last element to `stop`. We replicate exactly so the equality
 * checks and the slope rounding both match scipy's IEEE 754 ordering.
 */

#include "resample_transformer.h"

#include <stdlib.h>
#include <string.h>

struct n4m_pp_resample_state_t {
    int64_t num_samples;   /* -1 == identity */
};

n4m_pp_resample_state_t* n4m_pp_resample_state_new(int64_t num_samples) {
    if (num_samples != -1 && num_samples < 2) {
        return NULL;
    }
    n4m_pp_resample_state_t* s = (n4m_pp_resample_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->num_samples = num_samples;
    return s;
}

void n4m_pp_resample_state_free(n4m_pp_resample_state_t* state) {
    free(state);
}

int64_t n4m_pp_resample_output_cols_helper(const n4m_pp_resample_state_t* state,
                                            int64_t input_cols) {
    if (state == NULL) {
        return 0;
    }
    if (state->num_samples == -1) {
        return input_cols;
    }
    return state->num_samples;
}

/* Build a length-n linspace(0, 1, n) buffer matching numpy.linspace order:
 *   step = (stop - start) / (n - 1)
 *   value[i] = start + step * i        (i in [0, n-1))
 *   value[n - 1] = stop                (numpy pins the endpoint explicitly)
 */
static void fill_linspace01(double* buf, int64_t n) {
    if (n <= 0) return;
    if (n == 1) { buf[0] = 0.0; return; }
    const double step = 1.0 / (double)(n - 1);
    buf[0] = 0.0;
    for (int64_t i = 1; i < n - 1; ++i) {
        buf[i] = (double)i * step;
    }
    buf[n - 1] = 1.0;
}

/* Standard numpy.searchsorted(src, q, side='left'): leftmost insertion index
 * such that src[idx-1] < q <= src[idx]. src must be sorted ascending. */
static int64_t searchsorted_left(const double* src, int64_t n, double q) {
    int64_t lo = 0;
    int64_t hi = n;
    while (lo < hi) {
        const int64_t mid = (lo + hi) >> 1;
        if (src[mid] < q) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

n4m_status_t n4m_pp_resample_apply(const n4m_pp_resample_state_t* state,
                                    const double* X,
                                    int64_t rows, int64_t cols,
                                    int64_t out_cols,
                                    double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0 || out_cols < 0) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const int64_t expected = (state->num_samples == -1) ? cols : state->num_samples;
    if (out_cols != expected) {
        return N4M_ERR_SHAPE_MISMATCH;
    }
    if (rows == 0 || out_cols == 0) {
        return N4M_OK;
    }
    /* Identity passthrough. */
    if (state->num_samples == -1) {
        if (out != X) {
            for (int64_t i = 0; i < rows; ++i) {
                memcpy(out + (size_t)i * (size_t)cols,
                       X   + (size_t)i * (size_t)cols,
                       (size_t)cols * sizeof(double));
            }
        }
        return N4M_OK;
    }
    /* nirs4all's ResampleTransformer copies the row verbatim when
     * `len(x) == num_samples`. That branch never goes through interp1d, so
     * we mirror it to keep parity bit-exact. */
    if (cols == out_cols) {
        if (out != X) {
            for (int64_t i = 0; i < rows; ++i) {
                memcpy(out + (size_t)i * (size_t)cols,
                       X   + (size_t)i * (size_t)cols,
                       (size_t)cols * sizeof(double));
            }
        }
        return N4M_OK;
    }
    if (cols < 2) {
        /* Linear interp needs >= 2 anchors. */
        return N4M_ERR_INVALID_ARGUMENT;
    }
    /* Build the source / target axes once. */
    double* src = (double*)malloc((size_t)cols * sizeof(double));
    if (src == NULL) {
        return N4M_ERR_OUT_OF_MEMORY;
    }
    double* dst = (double*)malloc((size_t)out_cols * sizeof(double));
    if (dst == NULL) {
        free(src);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    fill_linspace01(src, cols);
    fill_linspace01(dst, out_cols);

    /* Pre-compute the bracketing indices for every target query (independent
     * of the row data) so per-row work is dominated by the slope evaluation. */
    int64_t* idx_hi = (int64_t*)malloc((size_t)out_cols * sizeof(int64_t));
    if (idx_hi == NULL) {
        free(src);
        free(dst);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    for (int64_t k = 0; k < out_cols; ++k) {
        int64_t hi = searchsorted_left(src, cols, dst[k]);
        if (hi < 1) hi = 1;
        if (hi > cols - 1) hi = cols - 1;
        idx_hi[k] = hi;
    }

    for (int64_t i = 0; i < rows; ++i) {
        const double* row = X + (size_t)i * (size_t)cols;
        double* out_row = out + (size_t)i * (size_t)out_cols;
        for (int64_t k = 0; k < out_cols; ++k) {
            const int64_t hi = idx_hi[k];
            const int64_t lo = hi - 1;
            const double y_lo = row[lo];
            const double y_hi = row[hi];
            const double slope = (y_hi - y_lo) / (src[hi] - src[lo]);
            out_row[k] = slope * (dst[k] - src[lo]) + y_lo;
        }
    }

    free(idx_hi);
    free(dst);
    free(src);
    return N4M_OK;
}
