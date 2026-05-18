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

struct c4a_pp_simple_scale_state_t {
    int _reserved;
};

c4a_pp_simple_scale_state_t* c4a_pp_simple_scale_state_new(void) {
    c4a_pp_simple_scale_state_t* s =
        (c4a_pp_simple_scale_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->_reserved = 0;
    return s;
}

void c4a_pp_simple_scale_state_free(c4a_pp_simple_scale_state_t* state) {
    free(state);
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

    for (int64_t j = 0; j < cols; ++j) {
        double col_min = X[(size_t)j];
        double col_max = col_min;
        for (int64_t i = 1; i < rows; ++i) {
            const double v = X[(size_t)i * (size_t)cols + (size_t)j];
            if (v < col_min) col_min = v;
            if (v > col_max) col_max = v;
        }
        const double range = col_max - col_min;
        for (int64_t i = 0; i < rows; ++i) {
            const size_t idx = (size_t)i * (size_t)cols + (size_t)j;
            out[idx] = (X[idx] - col_min) / range;
        }
    }
    return C4A_OK;
}
