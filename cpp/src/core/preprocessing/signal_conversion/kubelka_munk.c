/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * KubelkaMunk — F(R) = (1 - R)^2 / (2R) with a symmetric clamp on R.
 *
 * Numerical notes for byte-for-byte parity with nirs4all/numpy:
 *
 *   - The clamp uses `np.clip(X, epsilon, 1 - epsilon)` — a two-sided
 *     symmetric saturation. The `1 - epsilon` upper bound is computed once
 *     (not per element) and then compared via the standard
 *     `if v > upper, v = upper` branch, which is bit-equivalent to
 *     `min(v, upper)`.
 *
 *   - The percent-to-fraction division uses `X[i] / 100.0`, not
 *     `X[i] * 0.01` (0.01 has no exact binary representation).
 *
 *   - The squared term is `(1 - R) * (1 - R)` — a single multiply of the
 *     subtraction result, matching numpy's `np.square` ufunc which evaluates
 *     as `x * x` element-wise.
 *
 *   - The division `square / (2.0 * R)` is performed as a true element-wise
 *     divide; the denominator `2.0 * R` is computed inline rather than via
 *     `R * 2.0` to match the operator precedence of nirs4all's literal
 *     `(1.0 - R)**2 / (2.0 * R)` expression.
 */

#include "kubelka_munk.h"

#include <stdlib.h>

struct c4a_pp_kubelka_munk_state_t {
    int    is_percent;
    double epsilon;
};

c4a_pp_kubelka_munk_state_t* c4a_pp_kubelka_munk_state_new(
    int is_percent, double epsilon) {
    if (!(epsilon > 0.0) || !(epsilon < 0.5)) {
        return NULL;
    }
    c4a_pp_kubelka_munk_state_t* s =
        (c4a_pp_kubelka_munk_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->is_percent = is_percent ? 1 : 0;
    s->epsilon    = epsilon;
    return s;
}

void c4a_pp_kubelka_munk_state_free(c4a_pp_kubelka_munk_state_t* state) {
    free(state);
}

c4a_status_t c4a_pp_kubelka_munk_apply(
    const c4a_pp_kubelka_munk_state_t* state,
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
    const double lo    = state->epsilon;
    const double hi    = 1.0 - state->epsilon;

    if (state->is_percent) {
        for (size_t i = 0; i < total; ++i) {
            double R = X[i] / 100.0;
            if (R < lo) R = lo;
            else if (R > hi) R = hi;
            const double one_minus_R = 1.0 - R;
            out[i] = (one_minus_R * one_minus_R) / (2.0 * R);
        }
    } else {
        for (size_t i = 0; i < total; ++i) {
            double R = X[i];
            if (R < lo) R = lo;
            else if (R > hi) R = hi;
            const double one_minus_R = 1.0 - R;
            out[i] = (one_minus_R * one_minus_R) / (2.0 * R);
        }
    }
    return C4A_OK;
}
