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
 *   - The scalar fallback keeps the literal reference expression. The SSE2
 *     hot loop uses the equivalent `0.5 / R - 1.0 + 0.5 * R` form to reduce
 *     arithmetic while staying within the public parity tolerance.
 */

#include "kubelka_munk.h"

#include <stdlib.h>

#if defined(__SSE2__)
#include <emmintrin.h>
#endif

static inline double c4a_kubelka_munk_value(double R) {
    const double one_minus_R = 1.0 - R;
    return (one_minus_R * one_minus_R) / (2.0 * R);
}

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
        size_t i = 0;
        for (; i + 3 < total; i += 4) {
            double R0 = X[i] / 100.0;
            double R1 = X[i + 1] / 100.0;
            double R2 = X[i + 2] / 100.0;
            double R3 = X[i + 3] / 100.0;
            if (R0 < lo) R0 = lo;
            else if (R0 > hi) R0 = hi;
            if (R1 < lo) R1 = lo;
            else if (R1 > hi) R1 = hi;
            if (R2 < lo) R2 = lo;
            else if (R2 > hi) R2 = hi;
            if (R3 < lo) R3 = lo;
            else if (R3 > hi) R3 = hi;
            out[i]     = c4a_kubelka_munk_value(R0);
            out[i + 1] = c4a_kubelka_munk_value(R1);
            out[i + 2] = c4a_kubelka_munk_value(R2);
            out[i + 3] = c4a_kubelka_munk_value(R3);
        }
        for (; i < total; ++i) {
            double R = X[i] / 100.0;
            if (R < lo) R = lo;
            else if (R > hi) R = hi;
            out[i] = c4a_kubelka_munk_value(R);
        }
    } else {
        size_t i = 0;
#if defined(__SSE2__)
        const __m128d vlo  = _mm_set1_pd(lo);
        const __m128d vhi  = _mm_set1_pd(hi);
        const __m128d vhalf = _mm_set1_pd(0.5);
        const __m128d vone = _mm_set1_pd(1.0);
        for (; i + 1 < total; i += 2) {
            __m128d R = _mm_loadu_pd(X + i);
            R = _mm_max_pd(R, vlo);
            R = _mm_min_pd(R, vhi);
            __m128d y = _mm_div_pd(vhalf, R);
            y = _mm_sub_pd(y, vone);
            y = _mm_add_pd(y, _mm_mul_pd(vhalf, R));
            _mm_storeu_pd(out + i, y);
        }
#else
        for (; i + 3 < total; i += 4) {
            double R0 = X[i];
            double R1 = X[i + 1];
            double R2 = X[i + 2];
            double R3 = X[i + 3];
            if (R0 < lo) R0 = lo;
            else if (R0 > hi) R0 = hi;
            if (R1 < lo) R1 = lo;
            else if (R1 > hi) R1 = hi;
            if (R2 < lo) R2 = lo;
            else if (R2 > hi) R2 = hi;
            if (R3 < lo) R3 = lo;
            else if (R3 > hi) R3 = hi;
            out[i]     = c4a_kubelka_munk_value(R0);
            out[i + 1] = c4a_kubelka_munk_value(R1);
            out[i + 2] = c4a_kubelka_munk_value(R2);
            out[i + 3] = c4a_kubelka_munk_value(R3);
        }
#endif
        for (; i < total; ++i) {
            double R = X[i];
            if (R < lo) R = lo;
            else if (R > hi) R = hi;
            out[i] = c4a_kubelka_munk_value(R);
        }
    }
    return C4A_OK;
}
