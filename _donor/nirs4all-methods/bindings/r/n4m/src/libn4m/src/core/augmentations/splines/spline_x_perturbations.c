/* SPDX-License-Identifier: CECILL-2.1 */
#include "spline_x_perturbations.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../../common/bspline.h"
#include "../../common/rng_pcg64.h"

struct n4m_aug_spline_x_perturb_state_t {
    int32_t spline_degree;
    double  perturbation_density;
    double  perturbation_range_min;
    double  perturbation_range_max;
};

n4m_aug_spline_x_perturb_state_t* n4m_aug_spline_x_perturb_state_new(
    int32_t spline_degree, double perturbation_density,
    double perturbation_range_min, double perturbation_range_max) {
    if (spline_degree != 3) {
        /* Only cubic is supported in this vendored bspline. */
        return NULL;
    }
    if (perturbation_density < 0.0 || perturbation_density > 1.0) return NULL;
    if (perturbation_range_min > perturbation_range_max) return NULL;
    n4m_aug_spline_x_perturb_state_t* s =
        (n4m_aug_spline_x_perturb_state_t*)calloc(1, sizeof(*s));
    if (s == NULL) return NULL;
    s->spline_degree          = spline_degree;
    s->perturbation_density   = perturbation_density;
    s->perturbation_range_min = perturbation_range_min;
    s->perturbation_range_max = perturbation_range_max;
    return s;
}

void n4m_aug_spline_x_perturb_state_free(
    n4m_aug_spline_x_perturb_state_t* state) {
    free(state);
}

static double uniform(n4m_rng_pcg64* rng, double lo, double hi) {
    return lo + (hi - lo) * n4m_pcg64_engine_next_double(rng);
}

static double interp_linear(const double* xp, const double* fp, int64_t n,
                            double x) {
    if (n <= 1) return (n == 1) ? fp[0] : 0.0;
    if (x <= xp[0]) return fp[0];
    if (x >= xp[n - 1]) return fp[n - 1];
    int64_t lo = 0;
    int64_t hi = n - 1;
    while (hi - lo > 1) {
        const int64_t mid = (lo + hi) >> 1;
        if (xp[mid] <= x) lo = mid;
        else hi = mid;
    }
    const double dx = xp[lo + 1] - xp[lo];
    const double w = (dx > 0.0) ? ((x - xp[lo]) / dx) : 0.0;
    return fp[lo] + w * (fp[lo + 1] - fp[lo]);
}

n4m_status_t n4m_aug_spline_x_perturb_state_apply(
    const n4m_aug_spline_x_perturb_state_t* state,
    void* rng_void,
    const double* X, int64_t rows, int64_t cols,
    double* out) {
    if (state == NULL || X == NULL || out == NULL || rng_void == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0) return N4M_ERR_INVALID_ARGUMENT;
    if (rows == 0 || cols == 0) {
        if (out != X) memcpy(out, X, (size_t)(rows * cols) * sizeof(double));
        return N4M_OK;
    }
    if (cols < 4) {
        if (out != X) memcpy(out, X, (size_t)(rows * cols) * sizeof(double));
        return N4M_OK;
    }
    n4m_rng_pcg64* rng = (n4m_rng_pcg64*)rng_void;
    /* Sizing: mimic Python splrep(..., s=0, k=3), whose knot vector length is
     * cols + 4. nearbyint follows the default half-even mode used by
     * numpy.around. */
    const int64_t knots_proxy = cols + 4;
    int64_t delta_size = (int64_t)nearbyint((double)knots_proxy *
                                            state->perturbation_density);
    if (delta_size < 2) delta_size = 2;
    /* delta_x are anchor positions for the perturbation field, on
     * [0, cols-1]. */
    double* delta_x = (double*)malloc((size_t)delta_size * sizeof(double));
    if (delta_x == NULL) return N4M_ERR_OUT_OF_MEMORY;
    for (int64_t k = 0; k < delta_size; ++k) {
        delta_x[k] = (delta_size <= 1) ? 0.0 :
                     (double)(cols - 1) * (double)k /
                       (double)(delta_size - 1);
    }
    /* Build x[i] = (double)i. */
    double* x = (double*)malloc((size_t)cols * sizeof(double));
    double* knots = (double*)malloc((size_t)(cols + 4) * sizeof(double));
    double* coef  = (double*)malloc((size_t)(cols + 4) * sizeof(double));
    double* t_perturbed = (double*)malloc((size_t)(cols + 4) * sizeof(double));
    double* delta_y = (double*)malloc((size_t)delta_size * sizeof(double));
    if (x == NULL || knots == NULL || coef == NULL || t_perturbed == NULL ||
        delta_y == NULL) {
        free(x); free(knots); free(coef); free(t_perturbed); free(delta_y);
        free(delta_x);
        return N4M_ERR_OUT_OF_MEMORY;
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
        const int rc = n4m_bspline_build_not_a_knot_cubic(
            x, xrow, (int32_t)cols, knots, coef);
        if (rc != 0) {
            memcpy(orow, xrow, (size_t)cols * sizeof(double));
            continue;
        }
        for (int64_t j = 0; j < cols + 4; ++j) {
            t_perturbed[j] = knots[j] +
                interp_linear(delta_x, delta_y, delta_size, knots[j]);
        }
        n4m_bspline_deboor_eval_array(
            t_perturbed, coef, (int32_t)cols, /*degree=*/3,
            x, (int32_t)cols, /*extrapolate=*/1, orow);
    }
    free(x); free(knots); free(coef); free(t_perturbed); free(delta_y);
    free(delta_x);
    return N4M_OK;
}
