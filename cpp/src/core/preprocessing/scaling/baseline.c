/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Baseline — per-column mean centering with cached mean vector.
 *
 *   fit(X):                  mean[j] = sum_i(X[i, j]) / rows
 *   transform(X):            out[i, j] = X[i, j] - mean[j]
 *   inverse_transform(X):    out[i, j] = X[i, j] + mean[j]
 *
 * Accumulation order matches numpy.mean(X, axis=0): sum down each column
 * left-to-right by row index, then divide by rows.
 */

#include "baseline.h"

#include <stdlib.h>
#include <string.h>

struct c4a_pp_baseline_state_t {
    int      fitted;
    int64_t  cols;
    double*  mean;     /* per-column mean, length cols */
};

c4a_pp_baseline_state_t* c4a_pp_baseline_state_new(void) {
    c4a_pp_baseline_state_t* s = (c4a_pp_baseline_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->fitted = 0;
    s->cols   = 0;
    s->mean   = NULL;
    return s;
}

void c4a_pp_baseline_state_free(c4a_pp_baseline_state_t* state) {
    if (state == NULL) return;
    free(state->mean);
    free(state);
}

int c4a_pp_baseline_state_is_fitted(const c4a_pp_baseline_state_t* state) {
    return (state != NULL && state->fitted) ? 1 : 0;
}

c4a_status_t c4a_pp_baseline_state_fit(c4a_pp_baseline_state_t* state,
                                        const double* X,
                                        int64_t rows, int64_t cols) {
    if (state == NULL || X == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 1 || cols < 1) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    double* mean = (double*)malloc((size_t)cols * sizeof(double));
    if (mean == NULL) {
        return C4A_ERR_OUT_OF_MEMORY;
    }
    const double rows_d = (double)rows;
    for (int64_t j = 0; j < cols; ++j) {
        double acc = 0.0;
        for (int64_t i = 0; i < rows; ++i) {
            acc += X[(size_t)i * (size_t)cols + (size_t)j];
        }
        mean[j] = acc / rows_d;
    }
    free(state->mean);
    state->cols   = cols;
    state->mean   = mean;
    state->fitted = 1;
    return C4A_OK;
}

c4a_status_t c4a_pp_baseline_state_apply(const c4a_pp_baseline_state_t* state,
                                          const double* X,
                                          int64_t rows, int64_t cols,
                                          double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (!state->fitted) {
        return C4A_ERR_NOT_FITTED;
    }
    if (rows < 0 || cols < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (cols != state->cols) {
        return C4A_ERR_SHAPE_MISMATCH;
    }
    if (rows == 0 || cols == 0) {
        return C4A_OK;
    }
    for (int64_t i = 0; i < rows; ++i) {
        const size_t base = (size_t)i * (size_t)cols;
        for (int64_t j = 0; j < cols; ++j) {
            out[base + (size_t)j] = X[base + (size_t)j] - state->mean[j];
        }
    }
    return C4A_OK;
}

c4a_status_t c4a_pp_baseline_state_inverse_apply(
    const c4a_pp_baseline_state_t* state,
    const double* X,
    int64_t rows, int64_t cols,
    double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (!state->fitted) {
        return C4A_ERR_NOT_FITTED;
    }
    if (rows < 0 || cols < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (cols != state->cols) {
        return C4A_ERR_SHAPE_MISMATCH;
    }
    if (rows == 0 || cols == 0) {
        return C4A_OK;
    }
    for (int64_t i = 0; i < rows; ++i) {
        const size_t base = (size_t)i * (size_t)cols;
        for (int64_t j = 0; j < cols; ++j) {
            out[base + (size_t)j] = X[base + (size_t)j] + state->mean[j];
        }
    }
    return C4A_OK;
}
