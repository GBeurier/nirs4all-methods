/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * CropTransformer — column-slice [start, end) of X.
 *
 * Reference (nirs4all.operators.transforms.features.CropTransformer):
 *   end = X.shape[1] if self.end is None or self.end > X.shape[1] else self.end
 *   return X[:, self.start:end]
 *
 * The state stores only `start` and `end`; the actual clamp on `end` happens
 * inside `_apply` so that the same crop op can be reused across inputs with
 * different column counts.
 */

#include "crop.h"

#include <stdlib.h>
#include <string.h>

struct c4a_pp_crop_state_t {
    int64_t start;
    int64_t end;   /* -1 means "to end of row" */
};

c4a_pp_crop_state_t* c4a_pp_crop_state_new(int64_t start, int64_t end) {
    if (start < 0) {
        return NULL;
    }
    if (end != -1 && end <= start) {
        return NULL;
    }
    c4a_pp_crop_state_t* s = (c4a_pp_crop_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->start = start;
    s->end   = end;
    return s;
}

void c4a_pp_crop_state_free(c4a_pp_crop_state_t* state) {
    free(state);
}

int64_t c4a_pp_crop_output_cols_helper(const c4a_pp_crop_state_t* state,
                                        int64_t input_cols) {
    if (state == NULL || input_cols <= 0) {
        return 0;
    }
    if (state->start >= input_cols) {
        return 0;
    }
    int64_t end_eff = state->end;
    if (end_eff == -1 || end_eff > input_cols) {
        end_eff = input_cols;
    }
    if (end_eff <= state->start) {
        return 0;
    }
    return end_eff - state->start;
}

c4a_status_t c4a_pp_crop_apply(const c4a_pp_crop_state_t* state,
                                const double* X,
                                int64_t rows, int64_t cols,
                                int64_t out_cols,
                                double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0 || out_cols < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (state->start > cols) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    int64_t end_eff = state->end;
    if (end_eff == -1 || end_eff > cols) {
        end_eff = cols;
    }
    const int64_t expected_out = (end_eff > state->start) ? end_eff - state->start : 0;
    if (out_cols != expected_out) {
        return C4A_ERR_SHAPE_MISMATCH;
    }
    if (rows == 0 || out_cols == 0) {
        return C4A_OK;
    }
    for (int64_t i = 0; i < rows; ++i) {
        const double* src = X + (size_t)i * (size_t)cols + (size_t)state->start;
        double* dst = out + (size_t)i * (size_t)out_cols;
        memcpy(dst, src, (size_t)out_cols * sizeof(double));
    }
    return C4A_OK;
}
