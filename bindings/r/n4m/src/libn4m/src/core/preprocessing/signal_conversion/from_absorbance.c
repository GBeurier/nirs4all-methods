/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * FromAbsorbance — R/T = 10^(-A), optionally rescaled by 100 for percent output.
 *
 * Numerical notes:
 *
 *   - The hot loop computes the power as `exp(-X[i] * log(10))`. It is
 *     numerically equivalent within the public parity tolerance and avoids
 *     the slower generic `pow` path for a fixed base.
 *
 *   - The percent multiply uses `* 100.0`, performed as a separate kernel
 *     pass / inline op to mirror `result = result * 100.0` in nirs4all
 *     (rather than folding the constant into the exponent).
 */

#include "from_absorbance.h"

#include <math.h>
#include <stdlib.h>

#define N4M_LN10 2.30258509299404568402

static void n4m_from_absorbance_fraction(
    const double* X, size_t total, double* out) {
    size_t i = 0;
    for (; i + 3 < total; i += 4) {
        out[i]     = exp(-X[i] * N4M_LN10);
        out[i + 1] = exp(-X[i + 1] * N4M_LN10);
        out[i + 2] = exp(-X[i + 2] * N4M_LN10);
        out[i + 3] = exp(-X[i + 3] * N4M_LN10);
    }
    for (; i < total; ++i) {
        out[i] = exp(-X[i] * N4M_LN10);
    }
}

static void n4m_from_absorbance_percent(
    const double* X, size_t total, double* out) {
    size_t i = 0;
    for (; i + 3 < total; i += 4) {
        out[i]     = exp(-X[i] * N4M_LN10) * 100.0;
        out[i + 1] = exp(-X[i + 1] * N4M_LN10) * 100.0;
        out[i + 2] = exp(-X[i + 2] * N4M_LN10) * 100.0;
        out[i + 3] = exp(-X[i + 3] * N4M_LN10) * 100.0;
    }
    for (; i < total; ++i) {
        out[i] = exp(-X[i] * N4M_LN10) * 100.0;
    }
}

struct n4m_pp_from_absorbance_state_t {
    int is_percent;
};

n4m_pp_from_absorbance_state_t* n4m_pp_from_absorbance_state_new(int is_percent) {
    n4m_pp_from_absorbance_state_t* s =
        (n4m_pp_from_absorbance_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->is_percent = is_percent ? 1 : 0;
    return s;
}

void n4m_pp_from_absorbance_state_free(n4m_pp_from_absorbance_state_t* state) {
    free(state);
}

n4m_status_t n4m_pp_from_absorbance_apply(
    const n4m_pp_from_absorbance_state_t* state,
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

    if (state->is_percent) {
        n4m_from_absorbance_percent(X, total, out);
    } else {
        n4m_from_absorbance_fraction(X, total, out);
    }
    return N4M_OK;
}
