/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Derivate — n-th order finite-difference derivative along axis=1.
 *
 *   out = np.diff(X, n=order, axis=1) / delta ** order
 *
 * np.diff(X, n=k, axis=1) is equivalent to applying first-order differences
 * k times. We allocate a single (cols)-element scratch buffer per row and
 * unroll the k passes in place — pass 1 reads `cols` elements and writes
 * `cols - 1`, pass 2 writes `cols - 2`, etc. After k passes the buffer
 * holds the k-th order differences in the first `cols - k` slots.
 *
 * Division by `delta**order` is performed once per element at the end (a
 * single multiplication by the precomputed reciprocal scalar). The order
 * matches NumPy exactly when delta=1 since the divisor reduces to 1.
 */

#include "derivate.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct c4a_pp_derivate_state_t {
    int32_t  order;
    double   delta;
    int      fitted;
    int64_t  cols;
};

c4a_pp_derivate_state_t* c4a_pp_derivate_state_new(int32_t order, double delta) {
    if (order < 1 || delta == 0.0) {
        return NULL;
    }
    c4a_pp_derivate_state_t* s = (c4a_pp_derivate_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->order  = order;
    s->delta  = delta;
    s->fitted = 0;
    s->cols   = 0;
    return s;
}

void c4a_pp_derivate_state_free(c4a_pp_derivate_state_t* state) {
    free(state);
}

int c4a_pp_derivate_state_is_fitted(const c4a_pp_derivate_state_t* state) {
    return (state != NULL && state->fitted) ? 1 : 0;
}

c4a_status_t c4a_pp_derivate_state_fit(c4a_pp_derivate_state_t* state,
                                        const double* X,
                                        int64_t rows, int64_t cols) {
    (void)X;
    (void)rows;
    if (state == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (cols <= (int64_t)state->order) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    state->cols   = cols;
    state->fitted = 1;
    return C4A_OK;
}

int64_t c4a_pp_derivate_output_cols_helper(int32_t order, int64_t input_cols) {
    if (order < 0 || input_cols <= 0 || (int64_t)order >= input_cols) {
        return 0;
    }
    return input_cols - (int64_t)order;
}

c4a_status_t c4a_pp_derivate_state_apply(const c4a_pp_derivate_state_t* state,
                                          const double* X,
                                          int64_t rows, int64_t cols,
                                          int64_t out_cols,
                                          double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (!state->fitted) {
        return C4A_ERR_NOT_FITTED;
    }
    if (rows < 0 || cols < 0 || out_cols < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (cols != state->cols) {
        return C4A_ERR_SHAPE_MISMATCH;
    }
    const int64_t expected = cols - (int64_t)state->order;
    if (out_cols != expected) {
        return C4A_ERR_SHAPE_MISMATCH;
    }
    if (rows == 0 || cols == 0 || out_cols == 0) {
        return C4A_OK;
    }

    /* Precompute the divisor delta^order using repeated multiplication
     * (matches Python's int / float `**` which is repeated multiply for
     * integer exponents — np.diff(X) / (delta**order) on the Python side
     * uses the same scalar value). */
    double divisor = 1.0;
    for (int32_t k = 0; k < state->order; ++k) {
        divisor *= state->delta;
    }
    /* True element-wise division (not multiply by reciprocal) so the
     * rounding matches numpy's `array / scalar` byte-for-byte. */

    /* Scratch buffer to hold the working row across the `order` passes.
     * Length cols is enough since each pass shrinks the active prefix. */
    double* scratch = (double*)malloc((size_t)cols * sizeof(double));
    if (scratch == NULL) {
        return C4A_ERR_OUT_OF_MEMORY;
    }

    for (int64_t i = 0; i < rows; ++i) {
        const double* in_row = X + (size_t)i * (size_t)cols;
        /* Copy this row into the scratch buffer. */
        memcpy(scratch, in_row, (size_t)cols * sizeof(double));

        /* Apply first-difference `order` times in place. After pass `k`
         * (1-indexed) the first `cols - k` slots hold the k-th differences.
         * np.diff scans left-to-right and writes scratch[j] = scratch[j+1] -
         * scratch[j], so we cannot overwrite scratch[j] before computing
         * scratch[j+1]'s diff. Since we only need scratch[j] one more time
         * (to compute the diff at j-1), a simple two-pointer scheme works:
         * we overwrite scratch[j] with scratch[j+1] - scratch[j], because
         * scratch[j-1]'s diff was already written before this iteration. */
        int64_t active = cols;
        for (int32_t k = 0; k < state->order; ++k) {
            for (int64_t j = 0; j < active - 1; ++j) {
                scratch[j] = scratch[j + 1] - scratch[j];
            }
            --active;
        }

        /* Now scratch[0..out_cols) holds the n-th differences. Apply the
         * divisor and write to the output. */
        double* out_row = out + (size_t)i * (size_t)out_cols;
        for (int64_t j = 0; j < out_cols; ++j) {
            out_row[j] = scratch[j] / divisor;
        }
    }

    free(scratch);
    return C4A_OK;
}
