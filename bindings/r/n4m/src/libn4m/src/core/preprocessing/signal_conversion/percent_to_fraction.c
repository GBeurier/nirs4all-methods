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

struct n4m_pp_pct_to_frac_state_t {
    int _reserved;
};

n4m_pp_pct_to_frac_state_t* n4m_pp_pct_to_frac_state_new(void) {
    static n4m_pp_pct_to_frac_state_t state = {0};
    return &state;
}

void n4m_pp_pct_to_frac_state_free(n4m_pp_pct_to_frac_state_t* state) {
    (void)state;
}

n4m_status_t n4m_pp_pct_to_frac_apply(
    const n4m_pp_pct_to_frac_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0) {
        return N4M_OK;
    }

    const size_t total = (size_t)rows * (size_t)cols;
    for (size_t i = 0; i < total; ++i) {
        out[i] = X[i] / 100.0;
    }
    return N4M_OK;
}
