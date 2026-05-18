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

    if (state->user_defined) {
        const double imin = state->feature_min;
        const double imax = state->feature_max;
        const double span = imax - imin;

        for (int64_t j = 0; j < cols; ++j) {
            /* numpy.min / numpy.max scan left-to-right. */
            double col_min = X[(size_t)j];
            double col_max = col_min;
            for (int64_t i = 1; i < rows; ++i) {
                const double v = X[(size_t)i * (size_t)cols + (size_t)j];
                if (v < col_min) col_min = v;
                if (v > col_max) col_max = v;
            }
            /* nirs4all: `f = (imax - imin) / (max - min)` then
             *           `X = imin + f * (X - min_)`. We replicate this
             * exact arithmetic tree (single division per column, single
             * multiplication per element) so the rounding sequence matches. */
            const double f = span / (col_max - col_min);
            for (int64_t i = 0; i < rows; ++i) {
                const size_t idx = (size_t)i * (size_t)cols + (size_t)j;
                out[idx] = imin + f * (X[idx] - col_min);
            }
        }
    } else {
        for (int64_t j = 0; j < cols; ++j) {
            /* np.linalg.norm(X, axis=0) = sqrt(sum(X[:, j]^2)). */
            double sq = 0.0;
            for (int64_t i = 0; i < rows; ++i) {
                const double v = X[(size_t)i * (size_t)cols + (size_t)j];
                sq += v * v;
            }
            const double norm = sqrt(sq);
            /* nirs4all evaluates `X / self.linalg_norm_` — a true element-wise
             * division, NOT a multiplication by a precomputed reciprocal.
             * IEEE 754 ordering matters; the parity test below would fail
             * with a `mul by 1/norm` path on selected fixtures. */
            for (int64_t i = 0; i < rows; ++i) {
                const size_t idx = (size_t)i * (size_t)cols + (size_t)j;
                out[idx] = X[idx] / norm;
            }
        }
    }
    return C4A_OK;
}
