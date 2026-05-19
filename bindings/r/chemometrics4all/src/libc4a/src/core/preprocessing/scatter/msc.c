/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Multiplicative Scatter Correction — per-column linear regression against
 * the per-row mean of the training matrix.
 *
 * fit():
 *   ref[i]  = mean(X[i, :])                # per-row mean, length n_samples
 *   ref_mean = mean(ref)
 *   for j in 0..p-1:
 *       num  = sum_i (ref[i] - ref_mean) * (X[i, j] - col_mean[j])
 *       den  = sum_i (ref[i] - ref_mean)^2
 *       a[j] = num / den                  # slope (matches np.polyfit deg=1)
 *       b[j] = col_mean[j] - a[j] * ref_mean   # intercept
 *
 * The slope/intercept formulae match np.polyfit(ref, col, 1) which returns
 * (slope, intercept) for a deg-1 polynomial. We compute col_mean[j] inside
 * the j-loop to keep memory bounded at O(rows + cols), but accumulate `num`
 * via `(ref[i] - ref_mean) * X[i, j]` since the cross term
 * `(ref[i] - ref_mean) * col_mean[j]` sums to zero. Equivalently:
 *   num = sum_i (ref[i] - ref_mean) * X[i, j]
 * This shaves one pass over each column and preserves NumPy's accumulation
 * order (left-to-right, single-precision double).
 */

#include "msc.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct c4a_pp_msc_state_t {
    int      fitted;
    int64_t  cols;
    double*  a;    /* slope per column, length cols */
    double*  b;    /* intercept per column, length cols */
};

c4a_pp_msc_state_t* c4a_pp_msc_state_new(void) {
    c4a_pp_msc_state_t* s = (c4a_pp_msc_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->fitted = 0;
    s->cols   = 0;
    s->a      = NULL;
    s->b      = NULL;
    return s;
}

void c4a_pp_msc_state_free(c4a_pp_msc_state_t* state) {
    if (state == NULL) return;
    free(state->a);
    free(state->b);
    free(state);
}

int c4a_pp_msc_state_is_fitted(const c4a_pp_msc_state_t* state) {
    return (state != NULL && state->fitted) ? 1 : 0;
}

c4a_status_t c4a_pp_msc_state_fit(c4a_pp_msc_state_t* state,
                                   const double* X,
                                   int64_t rows, int64_t cols) {
    if (state == NULL || X == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 2 || cols < 1) {
        return C4A_ERR_INVALID_ARGUMENT;
    }

    /* Compute per-row means (ref) and their grand mean. */
    double* ref = (double*)malloc((size_t)rows * sizeof(double));
    if (ref == NULL) {
        return C4A_ERR_OUT_OF_MEMORY;
    }
    const double cols_d = (double)cols;
    for (int64_t i = 0; i < rows; ++i) {
        const double* row = X + (size_t)i * (size_t)cols;
        double acc = 0.0;
        for (int64_t j = 0; j < cols; ++j) {
            acc += row[j];
        }
        ref[i] = acc / cols_d;
    }
    double ref_mean = 0.0;
    for (int64_t i = 0; i < rows; ++i) {
        ref_mean += ref[i];
    }
    ref_mean /= (double)rows;

    /* Denominator: sum of squared deviations of ref. Constant across cols. */
    double den = 0.0;
    for (int64_t i = 0; i < rows; ++i) {
        const double d = ref[i] - ref_mean;
        den += d * d;
    }
    if (den == 0.0) {
        free(ref);
        return C4A_ERR_NUMERICAL_FAILURE;
    }

    /* Allocate / reallocate slope and intercept arrays. */
    double* a = (double*)malloc((size_t)cols * sizeof(double));
    double* b = (double*)malloc((size_t)cols * sizeof(double));
    if (a == NULL || b == NULL) {
        free(a);
        free(b);
        free(ref);
        return C4A_ERR_OUT_OF_MEMORY;
    }

    const double rows_d = (double)rows;
    for (int64_t j = 0; j < cols; ++j) {
        /* col mean and num = sum_i (ref[i] - ref_mean) * X[i, j].
         * Accumulate left-to-right in the same order as np.polyfit's
         * internal numpy sum. */
        double col_sum = 0.0;
        double num     = 0.0;
        for (int64_t i = 0; i < rows; ++i) {
            const double x = X[(size_t)i * (size_t)cols + (size_t)j];
            col_sum += x;
            num     += (ref[i] - ref_mean) * x;
        }
        const double col_mean = col_sum / rows_d;
        const double slope     = num / den;
        a[j] = slope;
        b[j] = col_mean - slope * ref_mean;
    }

    /* Commit: release the previous fit if any, then move new arrays in. */
    free(state->a);
    free(state->b);
    state->cols   = cols;
    state->a      = a;
    state->b      = b;
    state->fitted = 1;
    free(ref);
    return C4A_OK;
}

c4a_status_t c4a_pp_msc_state_apply(const c4a_pp_msc_state_t* state,
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
            const double a = state->a[j];
            const double b = state->b[j];
            out[base + (size_t)j] = (X[base + (size_t)j] - b) / a;
        }
    }
    return C4A_OK;
}

c4a_status_t c4a_pp_msc_state_inverse_apply(const c4a_pp_msc_state_t* state,
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
            const double a = state->a[j];
            const double b = state->b[j];
            out[base + (size_t)j] = X[base + (size_t)j] * a + b;
        }
    }
    return C4A_OK;
}
