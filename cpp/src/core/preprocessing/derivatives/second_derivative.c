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

struct c4a_pp_second_derivative_state_t {
    double  delta;
    int32_t edge_order;
};

c4a_pp_second_derivative_state_t*
c4a_pp_second_derivative_state_new(double delta, int32_t edge_order) {
    if (delta == 0.0) {
        return NULL;
    }
    if (edge_order != 1 && edge_order != 2) {
        return NULL;
    }
    c4a_pp_second_derivative_state_t* s =
        (c4a_pp_second_derivative_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->delta      = delta;
    s->edge_order = edge_order;
    return s;
}

void c4a_pp_second_derivative_state_free(
    c4a_pp_second_derivative_state_t* state) {
    free(state);
}

c4a_status_t c4a_pp_second_derivative_state_apply(
    const c4a_pp_second_derivative_state_t* state,
    const double* X, int64_t rows, int64_t cols, double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0) {
        return C4A_OK;
    }
    if (state->edge_order == 2 && cols < 3) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (state->edge_order == 1 && cols < 2) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    /* First pass: write the intermediate into a fresh buffer so we don't
     * stomp on `X` (which the user may want to keep intact for in-place
     * semantics where X and out alias).  The Phase 3 contract permits
     * `X == out`. */
    double* mid = (double*)malloc((size_t)rows * (size_t)cols * sizeof(double));
    if (mid == NULL) {
        return C4A_ERR_OUT_OF_MEMORY;
    }
    c4a_pp_first_derivative_state_t* first =
        c4a_pp_first_derivative_state_new(state->delta, state->edge_order);
    if (first == NULL) {
        free(mid);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    c4a_status_t st =
        c4a_pp_first_derivative_state_apply(first, X, rows, cols, mid);
    if (st != C4A_OK) {
        c4a_pp_first_derivative_state_free(first);
        free(mid);
        return st;
    }
    st = c4a_pp_first_derivative_state_apply(first, mid, rows, cols, out);
    c4a_pp_first_derivative_state_free(first);
    free(mid);
    return st;
}
