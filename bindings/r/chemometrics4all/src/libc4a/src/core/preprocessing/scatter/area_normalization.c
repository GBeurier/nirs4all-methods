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

#if defined(__GNUC__) || defined(__clang__)
#define C4A_RESTRICT __restrict__
#else
#define C4A_RESTRICT
#endif

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
    return c4a_pp_area_apply_method(state->method, X, rows, cols, out);
}

c4a_status_t c4a_pp_area_apply_method(c4a_pp_area_method_t method,
                                      const double* X, int64_t rows,
                                      int64_t cols, double* out) {
    if (X == NULL || out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (method != C4A_PP_AREA_SUM &&
        method != C4A_PP_AREA_ABS_SUM &&
        method != C4A_PP_AREA_TRAPZ) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0) {
        return C4A_OK;
    }

    const size_t n_cols = (size_t)cols;
    if (method == C4A_PP_AREA_SUM) {
        for (int64_t i = 0; i < rows; ++i) {
            const double* C4A_RESTRICT row_in = X + (size_t)i * n_cols;
            double* C4A_RESTRICT row_out = out + (size_t)i * n_cols;
            double s0 = 0.0;
            double s1 = 0.0;
            double s2 = 0.0;
            double s3 = 0.0;
            size_t j = 0;
            for (; j + 3 < n_cols; j += 4) {
                s0 += row_in[j];
                s1 += row_in[j + 1];
                s2 += row_in[j + 2];
                s3 += row_in[j + 3];
            }
            double area = (s0 + s1) + (s2 + s3);
            for (; j < n_cols; ++j) {
                area += row_in[j];
            }
            if (fabs(area) < 1e-10) {
                area = 1.0;
            }
            const double inv_area = 1.0 / area;
            for (size_t k = 0; k < n_cols; ++k) {
                row_out[k] = row_in[k] * inv_area;
            }
        }
        return C4A_OK;
    }

    if (method == C4A_PP_AREA_ABS_SUM) {
        for (int64_t i = 0; i < rows; ++i) {
            const double* C4A_RESTRICT row_in = X + (size_t)i * n_cols;
            double* C4A_RESTRICT row_out = out + (size_t)i * n_cols;
            double area = 0.0;
            for (size_t j = 0; j < n_cols; ++j) {
                area += fabs(row_in[j]);
            }
            if (fabs(area) < 1e-10) {
                area = 1.0;
            }
            const double inv_area = 1.0 / area;
            for (size_t j = 0; j < n_cols; ++j) {
                row_out[j] = row_in[j] * inv_area;
            }
        }
        return C4A_OK;
    }

    for (int64_t i = 0; i < rows; ++i) {
        const double* C4A_RESTRICT row_in = X + (size_t)i * n_cols;
        double* C4A_RESTRICT row_out = out + (size_t)i * n_cols;
        double area = 0.0;
        /* scipy.integrate.trapezoid / numpy.trapz with default unit spacing
         * computes 0.5 * sum(x[:-1] + x[1:]). Keep that pairwise order. */
        if (cols == 1) {
            area = 0.0;
        } else {
            double s = 0.0;
            for (size_t j = 0; j < n_cols - 1; ++j) {
                s += row_in[j] + row_in[j + 1];
            }
            area = 0.5 * s;
        }

        if (fabs(area) < 1e-10) {
            area = 1.0;
        }
        const double inv_area = 1.0 / area;
        for (size_t j = 0; j < n_cols; ++j) {
            row_out[j] = row_in[j] * inv_area;
        }
    }
    return C4A_OK;
}
