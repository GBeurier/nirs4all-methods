/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * PercentToFraction — element-wise X / 100.0.
 *
 * The division is performed inline (NOT replaced with a multiply by 0.01)
 * because 0.01 has no exact binary representation; multiplying by a
 * precomputed reciprocal would introduce a different rounding sequence
 * than numpy's element-wise true-divide.
 */

#include "percent_to_fraction.h"

struct c4a_pp_pct_to_frac_state_t {
    int _reserved;
};

c4a_pp_pct_to_frac_state_t* c4a_pp_pct_to_frac_state_new(void) {
    static c4a_pp_pct_to_frac_state_t state = {0};
    return &state;
}

void c4a_pp_pct_to_frac_state_free(c4a_pp_pct_to_frac_state_t* state) {
    (void)state;
}

c4a_status_t c4a_pp_pct_to_frac_apply(
    const c4a_pp_pct_to_frac_state_t* state,
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
        out[i] = X[i] / 100.0;
    }
    return C4A_OK;
}
