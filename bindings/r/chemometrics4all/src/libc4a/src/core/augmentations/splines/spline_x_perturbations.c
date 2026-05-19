/* SPDX-License-Identifier: CECILL-2.1 */
#include "spline_x_perturbations.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../../common/bspline.h"
#include "../../common/rng_pcg64.h"

struct c4a_aug_spline_x_perturb_state_t {
    int32_t spline_degree;
    double  perturbation_density;
    double  perturbation_range_min;
    double  perturbation_range_max;
};

c4a_aug_spline_x_perturb_state_t* c4a_aug_spline_x_perturb_state_new(
    int32_t spline_degree, double perturbation_density,
    double perturbation_range_min, double perturbation_range_max) {
    if (spline_degree != 3) {
        /* Only cubic is supported in this vendored bspline. */
        return NULL;
    }
    if (perturbation_density < 0.0 || perturbation_density > 1.0) return NULL;
    if (perturbation_range_min > perturbation_range_max) return NULL;
    c4a_aug_spline_x_perturb_state_t* s =
        (c4a_aug_spline_x_perturb_state_t*)calloc(1, sizeof(*s));
    if (s == NULL) return NULL;
    s->spline_degree          = spline_degree;
    s->perturbation_density   = perturbation_density;
    s->perturbation_range_min = perturbation_range_min;
    s->perturbation_range_max = perturbation_range_max;
    return s;
}

void c4a_aug_spline_x_perturb_state_free(
    c4a_aug_spline_x_perturb_state_t* state) {
    free(state);
}

static double uniform(c4a_rng_pcg64* rng, double lo, double hi) {
    return lo + (hi - lo) * c4a_pcg64_engine_next_double(rng);
}

c4a_status_t c4a_aug_spline_x_perturb_state_apply(
    const c4a_aug_spline_x_perturb_state_t* state,
    void* rng_void,
    const double* X, int64_t rows, int64_t cols,
    double* out) {
    if (state == NULL || X == NULL || out == NULL || rng_void == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0) return C4A_ERR_INVALID_ARGUMENT;
    if (rows == 0 || cols == 0) {
        if (out != X) memcpy(out, X, (size_t)(rows * cols) * sizeof(double));
        return C4A_OK;
    }
    if (cols < 4) {
        if (out != X) memcpy(out, X, (size_t)(rows * cols) * sizeof(double));
        return C4A_OK;
    }
    c4a_rng_pcg64* rng = (c4a_rng_pcg64*)rng_void;
    /* Sizing: mimic Python — len(t) ~ cols + 4 (degree+1 trailing knots
     * in scipy's natural-cubic representation). We use cols as the rough
     * proxy. */
    const int64_t knots_proxy = cols + 4;
    int64_t delta_size = (int64_t)round((double)knots_proxy *
                                          state->perturbation_density);
    if (delta_size < 2) delta_size = 2;
    /* delta_x are anchor positions for the perturbation field, on
     * [0, cols-1]. */
    double* delta_x = (double*)malloc((size_t)delta_size * sizeof(double));
    if (delta_x == NULL) return C4A_ERR_OUT_OF_MEMORY;
    for (int64_t k = 0; k < delta_size; ++k) {
        delta_x[k] = (delta_size <= 1) ? 0.0 :
                     (double)(cols - 1) * (double)k /
                       (double)(delta_size - 1);
    }
    /* Build x[i] = (double)i. */
    double* x = (double*)malloc((size_t)cols * sizeof(double));
    double* knots = (double*)malloc((size_t)(cols + 6) * sizeof(double));
    double* coef  = (double*)malloc((size_t)(cols + 2) * sizeof(double));
    double* xq    = (double*)malloc((size_t)cols * sizeof(double));
    double* delta_y = (double*)malloc((size_t)delta_size * sizeof(double));
    if (x == NULL || knots == NULL || coef == NULL || xq == NULL ||
        delta_y == NULL) {
        free(x); free(knots); free(coef); free(xq); free(delta_y);
        free(delta_x);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    for (int64_t j = 0; j < cols; ++j) x[j] = (double)j;
    for (int64_t i = 0; i < rows; ++i) {
        const double* xrow = X + i * cols;
        double* orow = out + i * cols;
        /* Draw delta_y samples per the perturbation_range. */
        for (int64_t k = 0; k < delta_size; ++k) {
            delta_y[k] = uniform(rng, state->perturbation_range_min,
                                  state->perturbation_range_max);
        }
        /* Build a natural cubic spline through (x, xrow). */
        const int rc = c4a_bspline_build_natural(x, xrow, (int32_t)cols,
                                                  knots, coef);
        if (rc != 0) {
            memcpy(orow, xrow, (size_t)cols * sizeof(double));
            continue;
        }
        /* xq[j] = x[j] + linear-interp of delta_y over delta_x at x[j]. */
        for (int64_t j = 0; j < cols; ++j) {
            const double xj = x[j];
            /* Find the segment in delta_x containing xj. */
            double dy = 0.0;
            if (delta_size == 1) {
                dy = delta_y[0];
            } else if (xj <= delta_x[0]) {
                dy = delta_y[0];
            } else if (xj >= delta_x[delta_size - 1]) {
                dy = delta_y[delta_size - 1];
            } else {
                int64_t lo = 0;
                int64_t hi = delta_size - 1;
                while (hi - lo > 1) {
                    const int64_t mid = (lo + hi) >> 1;
                    if (delta_x[mid] <= xj) lo = mid;
                    else hi = mid;
                }
                const double dxs = delta_x[lo + 1] - delta_x[lo];
                const double w = (dxs > 0.0) ? ((xj - delta_x[lo]) / dxs)
                                                 : 0.0;
                dy = delta_y[lo] + w * (delta_y[lo + 1] - delta_y[lo]);
            }
            xq[j] = xj + dy;
        }
        c4a_bspline_eval_array(knots, coef, (int32_t)cols, xq, (int32_t)cols,
                                /*extrapolate=*/1, orow);
    }
    free(x); free(knots); free(coef); free(xq); free(delta_y);
    free(delta_x);
    return C4A_OK;
}
