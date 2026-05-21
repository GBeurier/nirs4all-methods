/* SPDX-License-Identifier: CECILL-2.1 */
#include "spline_y_perturbations.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../../common/bspline.h"
#include "../../common/rng_pcg64.h"

struct n4m_aug_spline_y_perturb_state_t {
    int32_t spline_points;
    double  perturbation_intensity;
};

n4m_aug_spline_y_perturb_state_t* n4m_aug_spline_y_perturb_state_new(
    int32_t spline_points, double perturbation_intensity) {
    n4m_aug_spline_y_perturb_state_t* s =
        (n4m_aug_spline_y_perturb_state_t*)calloc(1, sizeof(*s));
    if (s == NULL) return NULL;
    s->spline_points          = spline_points;
    s->perturbation_intensity = perturbation_intensity;
    return s;
}

void n4m_aug_spline_y_perturb_state_free(
    n4m_aug_spline_y_perturb_state_t* state) {
    free(state);
}

static double uniform(n4m_rng_pcg64* rng, double lo, double hi) {
    return lo + (hi - lo) * n4m_pcg64_engine_next_double(rng);
}

n4m_status_t n4m_aug_spline_y_perturb_state_apply(
    const n4m_aug_spline_y_perturb_state_t* state,
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
    n4m_rng_pcg64* rng = (n4m_rng_pcg64*)rng_void;
    int32_t nb_pts = (state->spline_points > 0) ? state->spline_points
                                                  : (int32_t)(cols / 2);
    if (nb_pts < 4) nb_pts = 4;
    /* Compute X global max for variation scaling — Python uses np.max(X). */
    double xmax = X[0];
    const int64_t total = rows * cols;
    for (int64_t k = 1; k < total; ++k) {
        if (X[k] > xmax) xmax = X[k];
    }
    const double variation = xmax * state->perturbation_intensity;
    /* baseline draws U(-variation, variation) ONCE — the Python reference
     * does this outside the per-sample loop. */
    const double baseline      = uniform(rng, -variation, variation);
    const double interval_min  = -variation + baseline;
    const double interval_max  =  variation + baseline;
    /* Anchor x positions on [0, cols]. */
    double* x_pts = (double*)malloc((size_t)nb_pts * sizeof(double));
    double* y_pts = (double*)malloc((size_t)nb_pts * sizeof(double));
    double* knots = (double*)malloc((size_t)(nb_pts + 4) * sizeof(double));
    double* coef  = (double*)malloc((size_t)(nb_pts + 4) * sizeof(double));
    double* xq    = (double*)malloc((size_t)cols * sizeof(double));
    if (x_pts == NULL || y_pts == NULL || knots == NULL || coef == NULL ||
        xq == NULL) {
        free(x_pts); free(y_pts); free(knots); free(coef); free(xq);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    for (int32_t k = 0; k < nb_pts; ++k) {
        x_pts[k] = (nb_pts <= 1) ? 0.0 :
                    (double)cols * (double)k / (double)(nb_pts - 1);
    }
    for (int64_t j = 0; j < cols; ++j) xq[j] = (double)j;
    for (int64_t i = 0; i < rows; ++i) {
        const double* xrow = X + i * cols;
        double* orow = out + i * cols;
        for (int32_t k = 0; k < nb_pts; ++k) {
            y_pts[k] = uniform(rng, interval_min, interval_max);
        }
        const int rc = n4m_bspline_build_not_a_knot_cubic(
            x_pts, y_pts, nb_pts, knots, coef);
        if (rc != 0) {
            memcpy(orow, xrow, (size_t)cols * sizeof(double));
            continue;
        }
        /* distor[j] = spline(xq[j]) with extrapolate=False (=> 0 outside). */
        double distor;
        for (int64_t j = 0; j < cols; ++j) {
            distor = n4m_bspline_deboor_eval(
                knots, coef, nb_pts, /*degree=*/3, xq[j],
                /*extrapolate=*/0);
            orow[j] = xrow[j] + distor;
        }
    }
    free(x_pts); free(y_pts); free(knots); free(coef); free(xq);
    return N4M_OK;
}
