/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * FromAbsorbance — R/T = 10^(-A), optionally rescaled by 100 for percent output.
 *
 * Numerical notes:
 *
 *   - We compute the power as `pow(10.0, -X[i])` to match
 *     `np.power(10.0, -X)` exactly. The alternative `exp(-X[i] * ln10)` would
 *     introduce one extra multiply rounding step before the exp call,
 *     producing a different last-bit pattern even though the magnitude
 *     matches to >15 digits.
 *
 *   - The percent multiply uses `* 100.0`, performed as a separate kernel
 *     pass / inline op to mirror `result = result * 100.0` in nirs4all
 *     (rather than folding the constant into the exponent).
 */

#include "from_absorbance.h"

#include <math.h>
#include <stdlib.h>

struct c4a_pp_from_absorbance_state_t {
    int is_percent;
};

c4a_pp_from_absorbance_state_t* c4a_pp_from_absorbance_state_new(int is_percent) {
    c4a_pp_from_absorbance_state_t* s =
        (c4a_pp_from_absorbance_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->is_percent = is_percent ? 1 : 0;
    return s;
}

void c4a_pp_from_absorbance_state_free(c4a_pp_from_absorbance_state_t* state) {
    free(state);
}

c4a_status_t c4a_pp_from_absorbance_apply(
    const c4a_pp_from_absorbance_state_t* state,
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

    if (state->is_percent) {
        for (size_t i = 0; i < total; ++i) {
            out[i] = pow(10.0, -X[i]) * 100.0;
        }
    } else {
        for (size_t i = 0; i < total; ++i) {
            out[i] = pow(10.0, -X[i]);
        }
    }
    return C4A_OK;
}
