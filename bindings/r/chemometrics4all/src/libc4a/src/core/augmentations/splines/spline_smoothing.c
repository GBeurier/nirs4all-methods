/* SPDX-License-Identifier: CECILL-2.1 */
#include "spline_smoothing.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../../common/bspline.h"

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
        /* Natural cubic spline needs at least 4 points; pass through. */
        if (out != X) memcpy(out, X, (size_t)(rows * cols) * sizeof(double));
        return C4A_OK;
    }
    /* Build x[i] = (double)i. */
    double* x = (double*)malloc((size_t)cols * sizeof(double));
    double* knots = (double*)malloc((size_t)(cols + 6) * sizeof(double));
    double* coef  = (double*)malloc((size_t)(cols + 2) * sizeof(double));
    if (x == NULL || knots == NULL || coef == NULL) {
        free(x); free(knots); free(coef);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    for (int64_t j = 0; j < cols; ++j) x[j] = (double)j;
    for (int64_t i = 0; i < rows; ++i) {
        const double* xrow = X + i * cols;
        double* orow = out + i * cols;
        const int rc = c4a_bspline_build_natural(x, xrow, (int32_t)cols,
                                                  knots, coef);
        if (rc != 0) {
            /* Fallback: copy through. */
            memcpy(orow, xrow, (size_t)cols * sizeof(double));
            continue;
        }
        c4a_bspline_eval_array(knots, coef, (int32_t)cols, x, (int32_t)cols,
                                /*extrapolate=*/1, orow);
    }
    free(x); free(knots); free(coef);
    return C4A_OK;
}
