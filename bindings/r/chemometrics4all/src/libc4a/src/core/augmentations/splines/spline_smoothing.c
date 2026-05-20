/* SPDX-License-Identifier: CECILL-2.1 */
#include "spline_smoothing.h"

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef C4A_HAVE_FITPACK
#define C4A_HAVE_FITPACK 0
#endif

#if C4A_HAVE_FITPACK
#if defined(__cplusplus)
extern "C" {
#endif
void curfit_(int* iopt, int* m, double* x, double* y, double* w,
             double* xb, double* xe, int* k, double* s, int* nest,
             int* n, double* t, double* c, double* fp, double* wrk,
             int* lwrk, int* iwrk, int* ier);
void splev_(double* t, int* n, double* c, int* nc, int* k, double* x,
            double* y, int* m, int* e, int* ier);
#if defined(__cplusplus)
}
#endif
#endif

struct c4a_aug_spline_smooth_state_t {
    int dummy;  /* parameterless operator */
};

c4a_aug_spline_smooth_state_t* c4a_aug_spline_smooth_state_new(void) {
    c4a_aug_spline_smooth_state_t* s = (c4a_aug_spline_smooth_state_t*)
        calloc(1, sizeof(*s));
    return s;
}

void c4a_aug_spline_smooth_state_free(
    c4a_aug_spline_smooth_state_t* state) {
    free(state);
}

c4a_status_t c4a_aug_spline_smooth_state_apply(
    const c4a_aug_spline_smooth_state_t* state,
    void* rng_void,
    const double* X, int64_t rows, int64_t cols,
    double* out) {
    (void)state;
    (void)rng_void;
    if (X == NULL || out == NULL) return C4A_ERR_NULL_POINTER;
    if (rows < 0 || cols < 0) return C4A_ERR_INVALID_ARGUMENT;
    if (rows == 0 || cols == 0) {
        if (out != X) memcpy(out, X, (size_t)(rows * cols) * sizeof(double));
        return C4A_OK;
    }
    if (cols < 4) {
        if (out != X) memcpy(out, X, (size_t)(rows * cols) * sizeof(double));
        return C4A_OK;
    }
#if !C4A_HAVE_FITPACK
    if (out != X) memcpy(out, X, (size_t)(rows * cols) * sizeof(double));
    return C4A_OK;
#else
    if (cols > (int64_t)(INT_MAX - 4)) return C4A_ERR_INVALID_ARGUMENT;

    const int k_value = 3;
    const int k1 = k_value + 1;
    const int m_value = (int)cols;
    const int max_nest_value = m_value + k1;
    int default_nest_value = m_value / 2;
    const int min_nest_value = 2 * k1;
    if (default_nest_value < min_nest_value) default_nest_value = min_nest_value;
    if (default_nest_value > max_nest_value) default_nest_value = max_nest_value;
    const int lwrk_value = m_value * k1 + max_nest_value * (7 + 3 * k_value);

    double* x = (double*)malloc((size_t)m_value * sizeof(double));
    double* y = (double*)malloc((size_t)m_value * sizeof(double));
    double* w = (double*)malloc((size_t)m_value * sizeof(double));
    double* knots = (double*)malloc((size_t)max_nest_value * sizeof(double));
    double* coef = (double*)malloc((size_t)max_nest_value * sizeof(double));
    double* wrk = (double*)malloc((size_t)lwrk_value * sizeof(double));
    int* iwrk = (int*)malloc((size_t)max_nest_value * sizeof(int));
    if (x == NULL || y == NULL || w == NULL || knots == NULL || coef == NULL ||
        wrk == NULL || iwrk == NULL) {
        free(x); free(y); free(w); free(knots); free(coef); free(wrk); free(iwrk);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    for (int64_t j = 0; j < cols; ++j) {
        x[j] = (double)j;
        w[j] = 1.0;
    }

    for (int64_t i = 0; i < rows; ++i) {
        const double* xrow = X + i * cols;
        double* orow = out + i * cols;
        memcpy(y, xrow, (size_t)m_value * sizeof(double));
        memset(knots, 0, (size_t)max_nest_value * sizeof(double));
        memset(coef, 0, (size_t)max_nest_value * sizeof(double));
        memset(wrk, 0, (size_t)lwrk_value * sizeof(double));
        memset(iwrk, 0, (size_t)max_nest_value * sizeof(int));

        int iopt = 0;
        int m = m_value;
        int k = k_value;
        int nest = default_nest_value;
        int n = 0;
        int lwrk = lwrk_value;
        int ier = 0;
        double xb = 0.0;
        double xe = (double)(cols - 1);
        double s = 1.0 / (double)cols;
        double fp = 0.0;
        curfit_(&iopt, &m, x, y, w, &xb, &xe, &k, &s, &nest, &n, knots, coef,
                &fp, wrk, &lwrk, iwrk, &ier);

        if (ier == 1 && default_nest_value < max_nest_value) {
            iopt = 1;
            nest = max_nest_value;
            curfit_(&iopt, &m, x, y, w, &xb, &xe, &k, &s, &nest, &n, knots,
                    coef, &fp, wrk, &lwrk, iwrk, &ier);
        }

        if (ier == 10 || n <= k + 1) {
            memcpy(orow, xrow, (size_t)cols * sizeof(double));
            continue;
        }

        int nc = max_nest_value;
        int e = 0;
        int eval_ier = 0;
        splev_(knots, &n, coef, &nc, &k, x, orow, &m, &e, &eval_ier);
        if (eval_ier != 0) memcpy(orow, xrow, (size_t)cols * sizeof(double));
    }
    free(x); free(y); free(w); free(knots); free(coef); free(wrk); free(iwrk);
    return C4A_OK;
#endif
}
