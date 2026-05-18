/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * ToAbsorbance — A = -log10(X) with optional percent-to-fraction conversion
 * and a one-sided positive clamp via `epsilon`.
 *
 * Numerical notes for byte-for-byte parity with nirs4all/numpy:
 *
 *   - The percent-to-fraction division uses `X[i] / 100.0`, not
 *     `X[i] * 0.01`. The two are NOT bit-equal because 0.01 has no exact
 *     binary representation, so multiplying by a precomputed reciprocal
 *     introduces a different rounding sequence than numpy's element-wise
 *     true-divide.
 *
 *   - The clamp is one-sided (`X = max(X, epsilon)`). nirs4all picks between
 *     `np.clip(X, epsilon, None)` and `np.maximum(X, epsilon)` based on the
 *     `clip_negative` flag, but both produce identical output for a
 *     one-sided lower bound. The flag is preserved in the state for ABI
 *     symmetry only.
 *
 *   - `-log10(x)` is computed as the literal IEEE 754 `-log10(x)` call to
 *     match numpy's vectorised `-np.log10(X)` rounding.
 */

#include "to_absorbance.h"

#include <math.h>
#include <stdlib.h>

struct c4a_pp_to_absorbance_state_t {
    int    is_percent;
    double epsilon;
    int    clip_negative;
};

c4a_pp_to_absorbance_state_t* c4a_pp_to_absorbance_state_new(
    int is_percent, double epsilon, int clip_negative) {
    if (!(epsilon > 0.0)) {
        return NULL;
    }
    c4a_pp_to_absorbance_state_t* s =
        (c4a_pp_to_absorbance_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->is_percent    = is_percent ? 1 : 0;
    s->epsilon       = epsilon;
    s->clip_negative = clip_negative ? 1 : 0;
    return s;
}

void c4a_pp_to_absorbance_state_free(c4a_pp_to_absorbance_state_t* state) {
    free(state);
}

c4a_status_t c4a_pp_to_absorbance_apply(
    const c4a_pp_to_absorbance_state_t* state,
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
    const double eps   = state->epsilon;

    if (state->is_percent) {
        for (size_t i = 0; i < total; ++i) {
            double v = X[i] / 100.0;
            if (v < eps) v = eps;
            out[i] = -log10(v);
        }
    } else {
        for (size_t i = 0; i < total; ++i) {
            double v = X[i];
            if (v < eps) v = eps;
            out[i] = -log10(v);
        }
    }
    return C4A_OK;
}
