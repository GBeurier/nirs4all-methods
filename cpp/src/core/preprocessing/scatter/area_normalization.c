/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Area Normalization — per-row division by sum/abs_sum/trapezoidal area.
 *
 * Numerical details:
 *   - All accumulations are computed in the same left-to-right order as
 *     numpy's ufunc reductions, so the rounding sequence matches bit-for-bit.
 *   - trapz with unit spacing equals scipy.integrate.trapezoid(x) =
 *       sum(x) - (x[0] + x[-1]) / 2
 *     which is equivalent to (x[0] + x[-1]) / 2 + sum(x[1:-1]). We use the
 *     latter to match nirs4all's exact tree of additions when the row length
 *     equals 1 or 2 (degenerate trapz cases) — see comments below.
 */

#include "area_normalization.h"

#include <math.h>
#include <stdlib.h>

struct c4a_pp_area_state_t {
    c4a_pp_area_method_t method;
};

c4a_pp_area_state_t* c4a_pp_area_state_new(c4a_pp_area_method_t method) {
    if (method != C4A_PP_AREA_SUM &&
        method != C4A_PP_AREA_ABS_SUM &&
        method != C4A_PP_AREA_TRAPZ) {
        return NULL;
    }
    c4a_pp_area_state_t* s = (c4a_pp_area_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->method = method;
    return s;
}

void c4a_pp_area_state_free(c4a_pp_area_state_t* state) {
    free(state);
}

c4a_status_t c4a_pp_area_apply(const c4a_pp_area_state_t* state,
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

    for (int64_t i = 0; i < rows; ++i) {
        const double* row_in  = X + (size_t)i * (size_t)cols;
        double*       row_out = out + (size_t)i * (size_t)cols;

        double area = 0.0;
        switch (state->method) {
            case C4A_PP_AREA_SUM:
                for (int64_t j = 0; j < cols; ++j) {
                    area += row_in[j];
                }
                break;
            case C4A_PP_AREA_ABS_SUM:
                for (int64_t j = 0; j < cols; ++j) {
                    area += fabs(row_in[j]);
                }
                break;
            case C4A_PP_AREA_TRAPZ:
            default:
                /* scipy.integrate.trapezoid / numpy.trapz with default unit
                 * spacing computes 0.5 * sum(x[:-1] + x[1:]) — that is, the
                 * sum of adjacent-pair means. We match this accumulation
                 * order bit-for-bit by iterating once over pair sums.
                 * Degenerate cases: cols == 0 → area = 0 (handled above);
                 * cols == 1 → no pairs, area = 0. */
                if (cols == 1) {
                    area = 0.0;
                } else {
                    double s = 0.0;
                    for (int64_t j = 0; j < cols - 1; ++j) {
                        s += row_in[j] + row_in[j + 1];
                    }
                    area = 0.5 * s;
                }
                break;
        }

        if (fabs(area) < 1e-10) {
            area = 1.0;
        }
        /* nirs4all writes `X_transformed[i] = X_transformed[i] / area` as
         * element-wise true division. Multiplying by a precomputed
         * reciprocal would yield different rounding. */
        for (int64_t j = 0; j < cols; ++j) {
            row_out[j] = row_in[j] / area;
        }
    }
    return C4A_OK;
}
