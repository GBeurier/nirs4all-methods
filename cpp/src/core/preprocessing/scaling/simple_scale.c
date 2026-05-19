/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * SimpleScale — per-column min-max scaling to [0, 1].
 *
 * Matches nirs4all's `X = (X - min_) / (max_ - min_)` element-wise.
 * Division is performed inline (NOT replaced with reciprocal multiply) to
 * preserve numpy's exact rounding.
 */

#include "simple_scale.h"

#include <stdlib.h>

#define C4A_PP_SIMPLE_SCALE_STACK_COLS 4096

#if defined(__GNUC__) || defined(__clang__)
#define C4A_RESTRICT __restrict__
#else
#define C4A_RESTRICT
#endif

struct c4a_pp_simple_scale_state_t {
    int _reserved;
};

c4a_pp_simple_scale_state_t* c4a_pp_simple_scale_state_new(void) {
    static c4a_pp_simple_scale_state_t state = {0};
    return &state;
}

void c4a_pp_simple_scale_state_free(c4a_pp_simple_scale_state_t* state) {
    (void)state;
}

c4a_status_t c4a_pp_simple_scale_apply(const c4a_pp_simple_scale_state_t* state,
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

    const size_t n_cols = (size_t)cols;
    double min_stack[C4A_PP_SIMPLE_SCALE_STACK_COLS];
    double range_stack[C4A_PP_SIMPLE_SCALE_STACK_COLS];
    double* col_min = min_stack;
    double* col_range = range_stack;
    int heap = 0;

    if (n_cols > C4A_PP_SIMPLE_SCALE_STACK_COLS) {
        col_min = (double*)malloc(n_cols * sizeof(*col_min));
        col_range = (double*)malloc(n_cols * sizeof(*col_range));
        if (col_min == NULL || col_range == NULL) {
            free(col_min);
            free(col_range);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        heap = 1;
    }

    for (size_t j = 0; j < n_cols; ++j) {
        col_min[j] = X[j];
        col_range[j] = X[j];
    }
    /* Preserve the per-column left-to-right min/max order while reading the
     * matrix row-major. */
    for (int64_t i = 1; i < rows; ++i) {
        const double* C4A_RESTRICT row = X + (size_t)i * n_cols;
        for (size_t j = 0; j < n_cols; ++j) {
            const double v = row[j];
            if (v < col_min[j]) col_min[j] = v;
            if (v > col_range[j]) col_range[j] = v;
        }
    }
    for (size_t j = 0; j < n_cols; ++j) {
        col_range[j] = 1.0 / (col_range[j] - col_min[j]);
    }
    for (int64_t i = 0; i < rows; ++i) {
        const double* C4A_RESTRICT row_in = X + (size_t)i * n_cols;
        double* C4A_RESTRICT row_out = out + (size_t)i * n_cols;
        for (size_t j = 0; j < n_cols; ++j) {
            row_out[j] = (row_in[j] - col_min[j]) * col_range[j];
        }
    }
    if (heap) {
        free(col_min);
        free(col_range);
    }
    return C4A_OK;
}
