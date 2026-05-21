/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * SecondDerivative reference implementation: apply `np.gradient` twice
 * along axis=1.  The intermediate first derivative is materialised in a
 * full-width scratch matrix (rows x cols) because the second pass needs
 * the whole row at once.
 */

#include "second_derivative.h"

#include "first_derivative.h"

#include <stdlib.h>

struct n4m_pp_second_derivative_state_t {
    double  delta;
    int32_t edge_order;
};

n4m_pp_second_derivative_state_t*
n4m_pp_second_derivative_state_new(double delta, int32_t edge_order) {
    if (delta == 0.0) {
        return NULL;
    }
    if (edge_order != 1 && edge_order != 2) {
        return NULL;
    }
    n4m_pp_second_derivative_state_t* s =
        (n4m_pp_second_derivative_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->delta      = delta;
    s->edge_order = edge_order;
    return s;
}

void n4m_pp_second_derivative_state_free(
    n4m_pp_second_derivative_state_t* state) {
    free(state);
}

n4m_status_t n4m_pp_second_derivative_state_apply(
    const n4m_pp_second_derivative_state_t* state,
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
    /* First pass: write the intermediate into a fresh buffer so we don't
     * stomp on `X` (which the user may want to keep intact for in-place
     * semantics where X and out alias).  The Phase 3 contract permits
     * `X == out`. */
    double* mid = (double*)malloc((size_t)rows * (size_t)cols * sizeof(double));
    if (mid == NULL) {
        return N4M_ERR_OUT_OF_MEMORY;
    }
    n4m_pp_first_derivative_state_t* first =
        n4m_pp_first_derivative_state_new(state->delta, state->edge_order);
    if (first == NULL) {
        free(mid);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    n4m_status_t st =
        n4m_pp_first_derivative_state_apply(first, X, rows, cols, mid);
    if (st != N4M_OK) {
        n4m_pp_first_derivative_state_free(first);
        free(mid);
        return st;
    }
    st = n4m_pp_first_derivative_state_apply(first, mid, rows, cols, out);
    n4m_pp_first_derivative_state_free(first);
    free(mid);
    return st;
}
