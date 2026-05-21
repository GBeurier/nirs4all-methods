/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * First-derivative reference implementation mirroring
 * `np.gradient(X, delta, axis=1, edge_order=...)`.
 *
 * Interior (i ∈ [1, cols - 2]):
 *     out[i] = (X[i + 1] - X[i - 1]) / (2 * delta)
 *
 * edge_order == 1 (one-sided two-point):
 *     out[0]      = (X[1] - X[0]) / delta
 *     out[N - 1]  = (X[N - 1] - X[N - 2]) / delta
 *
 * edge_order == 2 (one-sided three-point):
 *     out[0]      = (-3 X[0] + 4 X[1] - X[2]) / (2 * delta)
 *     out[N - 1]  = ( 3 X[N - 1] - 4 X[N - 2] + X[N - 3]) / (2 * delta)
 *
 * Divisions use the exact denominators (no precomputed reciprocal) so the
 * IEEE-754 rounding sequence matches NumPy's `array / scalar` byte-for-byte.
 * In-place is supported by computing the new edges first into temporaries
 * (so a single row scratch buffer is enough — we stage the result and copy
 * back at the end).
 */

#include "first_derivative.h"

#include <stdlib.h>
#include <string.h>

struct n4m_pp_first_derivative_state_t {
    double  delta;
    int32_t edge_order;
};

n4m_pp_first_derivative_state_t*
n4m_pp_first_derivative_state_new(double delta, int32_t edge_order) {
    if (delta == 0.0) {
        return NULL;
    }
    if (edge_order != 1 && edge_order != 2) {
        return NULL;
    }
    n4m_pp_first_derivative_state_t* s =
        (n4m_pp_first_derivative_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->delta      = delta;
    s->edge_order = edge_order;
    return s;
}

void n4m_pp_first_derivative_state_free(
    n4m_pp_first_derivative_state_t* state) {
    free(state);
}

n4m_status_t n4m_pp_first_derivative_state_apply(
    const n4m_pp_first_derivative_state_t* state,
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
    if (state->edge_order == 2 && cols < 3) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (state->edge_order == 1 && cols < 2) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const double delta = state->delta;
    const double two_delta = 2.0 * delta;
    double* scratch = (double*)malloc((size_t)cols * sizeof(double));
    if (scratch == NULL) {
        return N4M_ERR_OUT_OF_MEMORY;
    }
    for (int64_t r = 0; r < rows; ++r) {
        const double* row_in  = X + (size_t)r * (size_t)cols;
        double*       row_out = out + (size_t)r * (size_t)cols;
        /* Interior. */
        for (int64_t j = 1; j + 1 < cols; ++j) {
            scratch[j] = (row_in[j + 1] - row_in[j - 1]) / two_delta;
        }
        if (state->edge_order == 1) {
            scratch[0]        = (row_in[1] - row_in[0]) / delta;
            scratch[cols - 1] = (row_in[cols - 1] - row_in[cols - 2]) / delta;
        } else {
            scratch[0] =
                (-3.0 * row_in[0] + 4.0 * row_in[1] - row_in[2]) / two_delta;
            scratch[cols - 1] =
                (3.0 * row_in[cols - 1] - 4.0 * row_in[cols - 2] +
                  row_in[cols - 3]) / two_delta;
        }
        memcpy(row_out, scratch, (size_t)cols * sizeof(double));
    }
    free(scratch);
    return N4M_OK;
}
