/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * NorrisWilliams reference implementation.  See norris_williams.h for the
 * algorithm summary.  Boundary handling uses edge replication (clamp to the
 * first / last sample) on both the smoothing and the gap-difference passes,
 * matching `np.pad(..., mode="edge")` used by the Python reference.
 */

#include "norris_williams.h"

#include <stdlib.h>
#include <string.h>

#include "core/common/padding.h"

struct n4m_pp_norris_williams_state_t {
    int32_t segment;
    int32_t gap;
    int32_t derivative_order;
    double  delta;
};

n4m_pp_norris_williams_state_t*
n4m_pp_norris_williams_state_new(int32_t segment, int32_t gap,
                                  int32_t derivative_order, double delta) {
    if (segment < 1 || (segment % 2) == 0) {
        return NULL;
    }
    if (gap < 1) {
        return NULL;
    }
    if (derivative_order != 1 && derivative_order != 2) {
        return NULL;
    }
    if (delta == 0.0) {
        return NULL;
    }
    n4m_pp_norris_williams_state_t* s =
        (n4m_pp_norris_williams_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->segment          = segment;
    s->gap              = gap;
    s->derivative_order = derivative_order;
    s->delta            = delta;
    return s;
}

void n4m_pp_norris_williams_state_free(
    n4m_pp_norris_williams_state_t* state) {
    free(state);
}

/* Centred moving average (segment size, edge-padded) on a single row.
 * `dst` and `src` may not alias. */
static void nw_segment_smooth(const double* src, double* dst,
                              int64_t cols, int32_t segment) {
    if (segment <= 1) {
        memcpy(dst, src, (size_t)cols * sizeof(double));
        return;
    }
    const int32_t half = segment / 2;
    const double seg_d = (double)segment;
    for (int64_t j = 0; j < cols; ++j) {
        double acc = 0.0;
        /* Window is [j - half, j - half + segment) on the padded row. */
        for (int32_t k = 0; k < segment; ++k) {
            const int64_t idx = j - (int64_t)half + (int64_t)k;
            int64_t resolved;
            if (idx < 0)            resolved = 0;
            else if (idx >= cols)   resolved = cols - 1;
            else                    resolved = idx;
            acc += src[resolved];
        }
        dst[j] = acc / seg_d;
    }
}

/* Gap derivative with edge padding on a single row.
 *   out[j] = (padded[j + 2*gap] - padded[j]) / (2*gap*delta)
 * Equivalent index in the unpadded row: `idx_right = j + gap` and
 * `idx_left = j - gap` (mapping back from the padding to the original
 * coordinate system).  Both clamp to [0, cols - 1]. */
static void nw_gap_derivative(const double* src, double* dst,
                              int64_t cols, int32_t gap, double delta) {
    const double denom = 2.0 * (double)gap * delta;
    for (int64_t j = 0; j < cols; ++j) {
        const int64_t r_idx = j + gap;
        const int64_t l_idx = j - gap;
        const double r =
            (r_idx >= cols) ? src[cols - 1] : src[r_idx];
        const double l = (l_idx < 0) ? src[0] : src[l_idx];
        dst[j] = (r - l) / denom;
    }
}

n4m_status_t n4m_pp_norris_williams_state_apply(
    const n4m_pp_norris_williams_state_t* state,
    const double* X, int64_t rows, int64_t cols, double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0) {
        return N4M_OK;
    }
    /* Two row scratch buffers to ping-pong between smoothing and gap diff.
     * For derivative_order == 2 we apply the smooth + gap pipeline twice. */
    double* buf_a = (double*)malloc((size_t)cols * sizeof(double));
    double* buf_b = (double*)malloc((size_t)cols * sizeof(double));
    if (buf_a == NULL || buf_b == NULL) {
        free(buf_a); free(buf_b);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    for (int64_t r = 0; r < rows; ++r) {
        const double* row_in = X + (size_t)r * (size_t)cols;
        double*       row_out = out + (size_t)r * (size_t)cols;
        /* First smooth+diff pass uses row_in as the source. */
        nw_segment_smooth(row_in, buf_a, cols, state->segment);
        nw_gap_derivative(buf_a, buf_b, cols, state->gap, state->delta);
        if (state->derivative_order == 2) {
            nw_segment_smooth(buf_b, buf_a, cols, state->segment);
            nw_gap_derivative(buf_a, buf_b, cols, state->gap, state->delta);
        }
        memcpy(row_out, buf_b, (size_t)cols * sizeof(double));
    }
    free(buf_a);
    free(buf_b);
    return N4M_OK;
}
