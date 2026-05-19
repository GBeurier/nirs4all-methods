/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Normalize (column-wise) reference implementation.
 *
 * Linalg-norm mode replicates np.linalg.norm with axis=0 (Frobenius / L2
 * column norm) by accumulating x*x left-to-right per column and taking
 * sqrt at the end — same order as numpy's reduction.
 *
 * User-defined-range mode computes per-column min/max via standard
 * left-to-right scan; matches numpy.min/max on the same memory order.
 */

#include "normalize.h"

#include <math.h>
#include <stdlib.h>

#define C4A_PP_NORMALIZE_STACK_COLS 4096

struct c4a_pp_normalize_state_t {
    double feature_min;
    double feature_max;
    int    user_defined;
};

c4a_pp_normalize_state_t* c4a_pp_normalize_state_new(double feature_min,
                                                     double feature_max) {
    c4a_pp_normalize_state_t* s = (c4a_pp_normalize_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->feature_min  = feature_min;
    s->feature_max  = feature_max;
    /* nirs4all: `user_defined = feature_range[0] != -1 or feature_range[1] != 1` */
    s->user_defined = (feature_min != -1.0) || (feature_max != 1.0);
    return s;
}

void c4a_pp_normalize_state_free(c4a_pp_normalize_state_t* state) {
    free(state);
}

c4a_status_t c4a_pp_normalize_apply(const c4a_pp_normalize_state_t* state,
                                    const double* X, int64_t rows, int64_t cols,
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

    if (state->user_defined) {
        const double imin = state->feature_min;
        const double imax = state->feature_max;
        const double span = imax - imin;
        double min_stack[C4A_PP_NORMALIZE_STACK_COLS];
        double max_stack[C4A_PP_NORMALIZE_STACK_COLS];
        double* col_min = min_stack;
        double* col_max = max_stack;
        int heap = 0;

        if (n_cols > C4A_PP_NORMALIZE_STACK_COLS) {
            col_min = (double*)malloc(n_cols * sizeof(*col_min));
            col_max = (double*)malloc(n_cols * sizeof(*col_max));
            if (col_min == NULL || col_max == NULL) {
                free(col_min);
                free(col_max);
                return C4A_ERR_OUT_OF_MEMORY;
            }
            heap = 1;
        }

        for (size_t j = 0; j < n_cols; ++j) {
            col_min[j] = X[j];
            col_max[j] = X[j];
        }
        /* numpy.min / numpy.max scan left-to-right within each column. This
         * row-major traversal preserves that per-column order while avoiding
         * repeated long-stride passes over X. */
        for (int64_t i = 1; i < rows; ++i) {
            const double* row = X + (size_t)i * n_cols;
            for (size_t j = 0; j < n_cols; ++j) {
                const double v = row[j];
                if (v < col_min[j]) col_min[j] = v;
                if (v > col_max[j]) col_max[j] = v;
            }
        }
        for (size_t j = 0; j < n_cols; ++j) {
            col_max[j] = span / (col_max[j] - col_min[j]);
        }

        for (int64_t i = 0; i < rows; ++i) {
            const double* row_in = X + (size_t)i * n_cols;
            double* row_out = out + (size_t)i * n_cols;
            for (size_t j = 0; j < n_cols; ++j) {
                /* nirs4all: `f = (imax - imin) / (max - min)` then
                 *           `X = imin + f * (X - min_)`. We replicate this
                 * exact arithmetic tree (single division per column, single
                 * multiplication per element) so the rounding sequence
                 * matches. */
                row_out[j] = imin + col_max[j] * (row_in[j] - col_min[j]);
            }
        }
        if (heap) {
            free(col_min);
            free(col_max);
        }
    } else {
        double norm_stack[C4A_PP_NORMALIZE_STACK_COLS];
        double* norm = norm_stack;
        int heap = 0;

        if (n_cols > C4A_PP_NORMALIZE_STACK_COLS) {
            norm = (double*)malloc(n_cols * sizeof(*norm));
            if (norm == NULL) {
                return C4A_ERR_OUT_OF_MEMORY;
            }
            heap = 1;
        }

        for (size_t j = 0; j < n_cols; ++j) {
            norm[j] = 0.0;
        }
        /* np.linalg.norm(X, axis=0) = sqrt(sum(X[:, j]^2)). For every column
         * the additions still happen in row order, matching the previous
         * parity-preserving reduction order. */
        for (int64_t i = 0; i < rows; ++i) {
            const double* row = X + (size_t)i * n_cols;
            for (size_t j = 0; j < n_cols; ++j) {
                const double v = row[j];
                norm[j] += v * v;
            }
        }
        for (size_t j = 0; j < n_cols; ++j) {
            norm[j] = sqrt(norm[j]);
        }
        for (int64_t i = 0; i < rows; ++i) {
            const double* row_in = X + (size_t)i * n_cols;
            double* row_out = out + (size_t)i * n_cols;
            /* nirs4all evaluates `X / self.linalg_norm_` — a true element-wise
             * division, NOT a multiplication by a precomputed reciprocal.
             * IEEE 754 ordering matters; the parity test below would fail
             * with a `mul by 1/norm` path on selected fixtures. */
            for (size_t j = 0; j < n_cols; ++j) {
                row_out[j] = row_in[j] / norm[j];
            }
        }
        if (heap) {
            free(norm);
        }
    }
    return C4A_OK;
}
