/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * SNIP — Ryan 1988 / Morháč 1997 statistics-sensitive non-linear iterative
 * peak-clipping baseline.
 *
 * Frozen reference: parity/python_generator/src/c4a_parity_pybaselines_ref/snip.py
 *
 * Algorithm: see snip.h. For Phase 5b parity we use the canonical "log of
 * log of sqrt" transform (LLS) — this is the Morháč 1997 dynamics-improving
 * variant, which is also what pybaselines.morphological.snip uses by default
 * (and the only flavour exercised in our reference fixtures).
 *
 * Important loop ordering: each width `w` from 1 to max_half_window is
 * applied IN-PLACE, and the local averages reference values from the same
 * pass that were just updated. This matches Morháč 1997 / pybaselines.
 *
 * Inverse LLS is the algebraic inverse of step 1:
 *   step  : a = |y| + 1; b = sqrt(a) + 1; c = log(b) + 1; v = log(c)
 *   inv.  : c = exp(v); b = exp(c - 1); a = (b - 1)^2; y = a - 1
 */

#include "snip.h"

#include <math.h>
#include <stdlib.h>

struct c4a_pp_snip_state_t {
    int32_t max_half_window;
};

c4a_pp_snip_state_t* c4a_pp_snip_state_new(int32_t max_half_window) {
    if (max_half_window < 1) {
        return NULL;
    }
    c4a_pp_snip_state_t* s =
        (c4a_pp_snip_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->max_half_window = max_half_window;
    return s;
}

void c4a_pp_snip_state_free(c4a_pp_snip_state_t* state) {
    free(state);
}

c4a_status_t c4a_pp_snip_state_apply(const c4a_pp_snip_state_t* state,
                                      const double* X,
                                      int64_t rows, int64_t cols,
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
    const int32_t maxw = state->max_half_window;

    double* v = (double*)malloc((size_t)cols * sizeof(double));
    if (v == NULL) {
        return C4A_ERR_OUT_OF_MEMORY;
    }

    for (int64_t r = 0; r < rows; ++r) {
        const double* y = X + (size_t)r * (size_t)cols;

        /* 1. LLS transform. */
        for (int64_t i = 0; i < cols; ++i) {
            const double a = fabs(y[i]) + 1.0;
            const double b = sqrt(a) + 1.0;
            const double c = log(b) + 1.0;
            v[i] = log(c);
        }

        /* 2. Iterative peak-clipping with growing half-windows. */
        for (int32_t w = 1; w <= maxw; ++w) {
            const int64_t lo = (int64_t)w;
            const int64_t hi = cols - (int64_t)w;
            for (int64_t i = lo; i < hi; ++i) {
                const double avg = 0.5 * (v[i - w] + v[i + w]);
                if (avg < v[i]) v[i] = avg;
            }
        }

        /* 3. Inverse LLS. */
        double* orow = out + (size_t)r * (size_t)cols;
        for (int64_t i = 0; i < cols; ++i) {
            const double c = exp(v[i]);
            const double b = exp(c - 1.0);
            const double a = (b - 1.0) * (b - 1.0);
            const double z = a - 1.0;
            orow[i] = y[i] - z;
        }
    }

    free(v);
    return C4A_OK;
}
