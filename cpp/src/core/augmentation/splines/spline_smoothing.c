/* SPDX-License-Identifier: CECILL-2.1 */
#include "spline_smoothing.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "core/common/_vendored/fitpack/fitpack.h"

struct n4m_aug_spline_smooth_state_t {
    int dummy;  /* parameterless operator */
};

n4m_aug_spline_smooth_state_t* n4m_aug_spline_smooth_state_new(void) {
    n4m_aug_spline_smooth_state_t* s = (n4m_aug_spline_smooth_state_t*)
        calloc(1, sizeof(*s));
    return s;
}

void n4m_aug_spline_smooth_state_free(
    n4m_aug_spline_smooth_state_t* state) {
    free(state);
}

n4m_status_t n4m_aug_spline_smooth_state_apply(
    const n4m_aug_spline_smooth_state_t* state,
    void* rng_void,
    const double* X, int64_t rows, int64_t cols,
    double* out) {
    (void)state;
    (void)rng_void;
    if (X == NULL || out == NULL) return N4M_ERR_NULL_POINTER;
    if (rows < 0 || cols < 0) return N4M_ERR_INVALID_ARGUMENT;

    /* Total-size guard. rows/cols are API-accepted int64_t; their product (the
     * element count) and its byte size feed both the passthrough memcpy and
     * the per-row pointer arithmetic X + i*cols. Reject dims whose product
     * would overflow int64_t, or whose byte size would overflow size_t, BEFORE
     * the multiplication is performed (signed overflow is UB) and before any
     * pointer is formed — otherwise an absurd-but-legal (rows, cols) could wrap
     * the index/size and produce a wild pointer. The division tests never
     * overflow because both operands are non-negative here. */
    if (rows != 0 && cols > INT64_MAX / rows) return N4M_ERR_INVALID_ARGUMENT;
    const int64_t total = rows * cols;
    if ((uint64_t)total > (uint64_t)(SIZE_MAX / sizeof(double)))
        return N4M_ERR_INVALID_ARGUMENT;
    const size_t total_bytes = (size_t)total * sizeof(double);

    if (rows == 0 || cols == 0) {
        if (out != X) memcpy(out, X, total_bytes);
        return N4M_OK;
    }
    if (cols < 4) {
        if (out != X) memcpy(out, X, total_bytes);
        return N4M_OK;
    }
    /* The FITPACK workspace (20*cols + 64 doubles) must be indexable by int
     * and its byte size must fit in size_t. */
    if (cols > (INT_MAX - 64) / 20 ||
        (uint64_t)(20 * cols + 64) > (uint64_t)(SIZE_MAX / sizeof(double)))
        return N4M_ERR_INVALID_ARGUMENT;

    /* scipy.interpolate.UnivariateSpline(x, y, k=3, s=1/cols) on x = 0..cols-1
     * with unit weights: curfit with SciPy's default nest, retried with the
     * maximal nest when that is too small (ier == 1), then splev. */
    const int k = 3;
    const int k1 = k + 1;
    const int m = (int)cols;
    const int max_nest = m + k1;
    int default_nest = m / 2;
    if (default_nest < 2 * k1) default_nest = 2 * k1;
    if (default_nest > max_nest) default_nest = max_nest;
    const int lwrk = m * k1 + max_nest * (7 + 3 * k);
    const double s = 1.0 / (double)cols;

    double* x = (double*)malloc((size_t)m * sizeof(double));
    double* w = (double*)malloc((size_t)m * sizeof(double));
    double* knots = (double*)malloc((size_t)max_nest * sizeof(double));
    double* coef = (double*)malloc((size_t)max_nest * sizeof(double));
    double* wrk = (double*)malloc((size_t)lwrk * sizeof(double));
    int* iwrk = (int*)malloc((size_t)max_nest * sizeof(int));
    if (x == NULL || w == NULL || knots == NULL || coef == NULL ||
        wrk == NULL || iwrk == NULL) {
        free(x); free(w); free(knots); free(coef); free(wrk); free(iwrk);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    for (int j = 0; j < m; ++j) {
        x[j] = (double)j;
        w[j] = 1.0;
    }

    for (int64_t i = 0; i < rows; ++i) {
        const double* xrow = X + i * cols;
        double* orow = out + i * cols;
        memset(knots, 0, (size_t)max_nest * sizeof(double));
        memset(coef, 0, (size_t)max_nest * sizeof(double));
        memset(wrk, 0, (size_t)lwrk * sizeof(double));
        memset(iwrk, 0, (size_t)max_nest * sizeof(int));

        int n = 0;
        double fp = 0.0;
        if (n4m_fitpack_curfit(0, m, x, xrow, w, 0.0, (double)(m - 1), k, s,
                               default_nest, &n, knots, coef, &fp, wrk,
                               iwrk) == 1) {
            n4m_fitpack_curfit(1, m, x, xrow, w, 0.0, (double)(m - 1), k, s,
                               max_nest, &n, knots, coef, &fp, wrk, iwrk);
        }
        n4m_fitpack_splev(knots, n, coef, k, x, orow, m);
    }
    free(x); free(w); free(knots); free(coef); free(wrk); free(iwrk);
    return N4M_OK;
}
