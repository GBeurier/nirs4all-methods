/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * FractionToPercent — element-wise X * 100.0.
 *
 * 100.0 is exactly representable in IEEE 754 binary64, so multiplying by it
 * is bit-exact and equivalent to dividing by 0.01; we use the multiply form
 * to match nirs4all's `X * 100.0` expression literally.
 */

#include "fraction_to_percent.h"

#include <stdlib.h>

struct c4a_pp_frac_to_pct_state_t {
    int _reserved;
};

c4a_pp_frac_to_pct_state_t* c4a_pp_frac_to_pct_state_new(void) {
    c4a_pp_frac_to_pct_state_t* s =
        (c4a_pp_frac_to_pct_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->_reserved = 0;
    return s;
}

void c4a_pp_frac_to_pct_state_free(c4a_pp_frac_to_pct_state_t* state) {
    free(state);
}

c4a_status_t c4a_pp_frac_to_pct_apply(
    const c4a_pp_frac_to_pct_state_t* state,
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

    const size_t total = (size_t)rows * (size_t)cols;
    for (size_t i = 0; i < total; ++i) {
        out[i] = X[i] * 100.0;
    }
    return C4A_OK;
}
