/* SPDX-License-Identifier: CECILL-2.1 */
#include "spline_simplify_common.h"

#include <stdlib.h>
#include <string.h>

#include "../../common/bspline.h"
#include "../../common/rng_pcg64.h"
#include "../aug_rng_utils.h"

static int cmp_int64(const void* a, const void* b) {
    const int64_t ia = *(const int64_t*)a;
    const int64_t ib = *(const int64_t*)b;
    return (ia > ib) - (ia < ib);
}

/* Sort `buf[0..n)` ascending and remove duplicates in place; returns the
 * number of distinct entries (np.unique semantics for integer indices). */
static int64_t sort_unique(int64_t* buf, int64_t n) {
    if (n <= 1) return n;
    qsort(buf, (size_t)n, sizeof(int64_t), cmp_int64);
    int64_t w = 1;
    for (int64_t r = 1; r < n; ++r) {
        if (buf[r] != buf[w - 1]) buf[w++] = buf[r];
    }
    return w;
}

n4m_status_t n4m_spline_simplify_apply_impl(void* rng_void, const double* X,
                                            int64_t rows, int64_t cols,
                                            int32_t spline_points, int uniform,
                                            int unique_in_uniform, double* out) {
    if (X == NULL || out == NULL) return N4M_ERR_NULL_POINTER;
    if (!uniform && rng_void == NULL) return N4M_ERR_NULL_POINTER;
    if (rows < 0 || cols < 0) return N4M_ERR_INVALID_ARGUMENT;
    if (rows == 0 || cols == 0) {
        if (out != X) memcpy(out, X, (size_t)(rows * cols) * sizeof(double));
        return N4M_OK;
    }

    /* nb_points: explicit when > 0, else the reference default int(cols / 4). */
    int64_t nb_points = (spline_points > 0) ? (int64_t)spline_points : cols / 4;
    if (nb_points < 0) nb_points = 0;
    /* The non-uniform path draws nb_points distinct indices without replacement,
     * so it needs nb_points <= cols (the reference raises otherwise). Uniform
     * spacing has no such limit. Reject rather than silently clamp. */
    if (!uniform && nb_points > cols) return N4M_ERR_INVALID_ARGUMENT;

    n4m_rng_pcg64* rng = (n4m_rng_pcg64*)rng_void;

    /* x grid 0..cols-1 (shared across rows). */
    double* x = (double*)malloc((size_t)cols * sizeof(double));
    /* ctrl index scratch: at most nb_points + 2 (endpoints) for non-uniform,
     * nb_points for uniform. */
    int64_t* ctrl = (int64_t*)malloc((size_t)(nb_points + 2) * sizeof(int64_t));
    double* xs = (double*)malloc((size_t)(nb_points + 2) * sizeof(double));
    double* ys = (double*)malloc((size_t)(nb_points + 2) * sizeof(double));
    /* not-a-knot cubic buffers: knots/coef length n_ctrl + 4. */
    double* knots = (double*)malloc((size_t)(nb_points + 2 + 4) * sizeof(double));
    double* coef = (double*)malloc((size_t)(nb_points + 2 + 4) * sizeof(double));
    if (x == NULL || ctrl == NULL || xs == NULL || ys == NULL ||
        knots == NULL || coef == NULL) {
        free(x); free(ctrl); free(xs); free(ys); free(knots); free(coef);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    for (int64_t j = 0; j < cols; ++j) x[j] = (double)j;

    for (int64_t i = 0; i < rows; ++i) {
        const double* xrow = X + i * cols;
        double* orow = out + i * cols;

        int64_t n_ctrl;
        if (uniform) {
            /* np.linspace(0, cols-1, nb_points).astype(int): val[k] = k*step,
             * last forced to cols-1, truncated toward zero. */
            if (nb_points <= 0) {
                n_ctrl = 0;
            } else if (nb_points == 1) {
                ctrl[0] = 0;
                n_ctrl = 1;
            } else {
                const double step = (double)(cols - 1) / (double)(nb_points - 1);
                for (int64_t k = 0; k < nb_points - 1; ++k) {
                    ctrl[k] = (int64_t)((double)k * step);
                }
                ctrl[nb_points - 1] = cols - 1;
                n_ctrl = nb_points;
            }
            if (unique_in_uniform) n_ctrl = sort_unique(ctrl, n_ctrl);
        } else {
            /* np.unique(concat([0], choice(cols, nb_points, replace=False),
             *                   [cols-1])). */
            ctrl[0] = 0;
            if (nb_points > 0) {
                if (n4m_aug_rng_choice_no_replace(rng, cols, nb_points,
                                                  ctrl + 1) != 0) {
                    free(x); free(ctrl); free(xs); free(ys);
                    free(knots); free(coef);
                    return N4M_ERR_INTERNAL;
                }
            }
            ctrl[nb_points + 1] = cols - 1;
            n_ctrl = sort_unique(ctrl, nb_points + 2);
        }

        /* C-ABI policy: any spectrum whose control set cannot yield a cubic
         * interpolant passes through unchanged rather than erroring the whole
         * batch (matches the cols<4 pass-through already shipped in
         * spline_x_perturbations). scipy raises in these cases; this is a
         * documented divergence. Trigger 1: fewer than k+1 = 4 strictly
         * increasing knots after np.unique. */
        if (n_ctrl < 4) {
            memcpy(orow, xrow, (size_t)cols * sizeof(double));
            continue;
        }

        for (int64_t k = 0; k < n_ctrl; ++k) {
            xs[k] = (double)ctrl[k];
            ys[k] = xrow[ctrl[k]];
        }
        /* Trigger 2: the not-a-knot cubic build fails (e.g. non-monotonic x from
         * duplicate uniform indices in the X variant) — same pass-through. */
        if (n4m_bspline_build_not_a_knot_cubic(xs, ys, (int32_t)n_ctrl, knots,
                                               coef) != 0) {
            memcpy(orow, xrow, (size_t)cols * sizeof(double));
            continue;
        }
        /* BSpline(extrapolate=False) evaluated on x in [0, cols-1]; every grid
         * point lies within [ctrl[0], ctrl[-1]] = [0, cols-1], so no point is
         * extrapolated. */
        n4m_bspline_deboor_eval_array(knots, coef, (int32_t)n_ctrl, /*degree=*/3,
                                      x, (int32_t)cols, /*extrapolate=*/0, orow);
    }

    free(x); free(ctrl); free(xs); free(ys); free(knots); free(coef);
    return N4M_OK;
}
