/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * ModPoly — Lieber & Mahadevan-Jansen 2003 modified polynomial baseline.
 *
 * Frozen reference: parity/python_generator/src/c4a_parity_pybaselines_ref/modpoly.py
 *
 * For each row y of length n:
 *
 *   z := polyfit(positions, y, polyorder) evaluated at positions
 *   for iter in 0..max_iter:
 *       y := min(y, z)              # clip peaks to current baseline
 *       z_new := polyfit(positions, y, polyorder) evaluated at positions
 *       if rel_l2_diff(z, z_new) < tol: break
 *       z := z_new
 *   out := original_y - z
 *
 * The polynomial is fit in the standard (positions = 0..n-1, descending
 * powers via np.vander) basis used by detrend.c — identical algebra to
 * pybaselines.modpoly modulo the [-1, 1] domain mapping, which only changes
 * the conditioning of the QR but not the recovered baseline.
 */

#include "modpoly.h"

#include <stdlib.h>
#include <string.h>

#include "core/common/banded_solver.h"
#include "core/common/linalg.h"

struct c4a_pp_modpoly_state_t {
    int32_t polyorder;
    int32_t max_iter;
    double  tol;
};

c4a_pp_modpoly_state_t* c4a_pp_modpoly_state_new(int32_t polyorder,
                                                   int32_t max_iter,
                                                   double tol) {
    if (polyorder < 0 || max_iter < 0 || !(tol >= 0.0)) {
        return NULL;
    }
    c4a_pp_modpoly_state_t* s =
        (c4a_pp_modpoly_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->polyorder = polyorder;
    s->max_iter  = max_iter;
    s->tol       = tol;
    return s;
}

void c4a_pp_modpoly_state_free(c4a_pp_modpoly_state_t* state) {
    free(state);
}

c4a_status_t c4a_pp_modpoly_state_apply(const c4a_pp_modpoly_state_t* state,
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
    const int32_t deg = state->polyorder;
    const int64_t p   = (int64_t)(deg + 1);
    if (cols < p) {
        return C4A_ERR_INVALID_ARGUMENT;
    }

    /* Build Vandermonde V (cols, p) in descending powers; factor once. */
    double* V_orig = (double*)malloc((size_t)cols * (size_t)p * sizeof(double));
    double* V_qr   = (double*)malloc((size_t)cols * (size_t)p * sizeof(double));
    double* tau    = (double*)malloc((size_t)p * sizeof(double));
    double* qt_y   = (double*)malloc((size_t)cols * sizeof(double));
    double* coefs  = (double*)malloc((size_t)p * sizeof(double));
    double* y_buf  = (double*)malloc((size_t)cols * sizeof(double));
    double* z      = (double*)malloc((size_t)cols * sizeof(double));
    double* z_new  = (double*)malloc((size_t)cols * sizeof(double));
    if (!V_orig || !V_qr || !tau || !qt_y || !coefs || !y_buf || !z || !z_new) {
        free(V_orig); free(V_qr); free(tau); free(qt_y); free(coefs);
        free(y_buf); free(z); free(z_new);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    for (int64_t i = 0; i < cols; ++i) {
        const double xi = (double)i;
        double power = 1.0;
        for (int32_t j = (int32_t)p - 1; j >= 0; --j) {
            V_orig[(size_t)i * (size_t)p + (size_t)j] = power;
            power *= xi;
        }
    }
    memcpy(V_qr, V_orig, (size_t)cols * (size_t)p * sizeof(double));
    const c4a_status_t qrst = c4a_householder_qr(V_qr, cols, p, tau);
    if (qrst != C4A_OK) {
        free(V_orig); free(V_qr); free(tau); free(qt_y); free(coefs);
        free(y_buf); free(z); free(z_new);
        return qrst;
    }

    for (int64_t r = 0; r < rows; ++r) {
        const double* y0 = X + (size_t)r * (size_t)cols;
        memcpy(y_buf, y0, (size_t)cols * sizeof(double));

        /* Initial baseline fit on the unclipped row. */
        memcpy(qt_y, y_buf, (size_t)cols * sizeof(double));
        c4a_status_t st = c4a_apply_qt(V_qr, cols, p, tau, qt_y);
        if (st != C4A_OK) { goto fail; }
        st = c4a_back_solve_R(V_qr, cols, p, qt_y, coefs);
        if (st != C4A_OK) { goto fail; }
        for (int64_t i = 0; i < cols; ++i) {
            double bval = 0.0;
            for (int32_t j = 0; j < (int32_t)p; ++j) {
                bval += coefs[j] * V_orig[(size_t)i * (size_t)p + (size_t)j];
            }
            z[i] = bval;
        }

        for (int32_t it = 0; it < state->max_iter; ++it) {
            /* Clip current y to current baseline: y[i] = min(y[i], z[i]). */
            for (int64_t i = 0; i < cols; ++i) {
                if (z[i] < y_buf[i]) y_buf[i] = z[i];
            }
            /* Re-fit. */
            memcpy(qt_y, y_buf, (size_t)cols * sizeof(double));
            st = c4a_apply_qt(V_qr, cols, p, tau, qt_y);
            if (st != C4A_OK) { goto fail; }
            st = c4a_back_solve_R(V_qr, cols, p, qt_y, coefs);
            if (st != C4A_OK) { goto fail; }
            for (int64_t i = 0; i < cols; ++i) {
                double bval = 0.0;
                for (int32_t j = 0; j < (int32_t)p; ++j) {
                    bval += coefs[j] * V_orig[(size_t)i * (size_t)p + (size_t)j];
                }
                z_new[i] = bval;
            }
            /* Convergence on the baseline relative L2 change. Matches
             * pybaselines.modpoly which uses relative_difference(baseline_old,
             * baseline). */
            const double rdiff = c4a_relative_l2_diff(z, z_new, cols);
            /* Swap z and z_new: keep latest baseline in z, recycle z_new. */
            double* tmp = z; z = z_new; z_new = tmp;
            if (rdiff < state->tol) {
                break;
            }
        }

        double* orow = out + (size_t)r * (size_t)cols;
        for (int64_t i = 0; i < cols; ++i) {
            orow[i] = y0[i] - z[i];
        }
        continue;
    fail:
        free(V_orig); free(V_qr); free(tau); free(qt_y); free(coefs);
        free(y_buf); free(z); free(z_new);
        return st;
    }

    free(V_orig); free(V_qr); free(tau); free(qt_y); free(coefs);
    free(y_buf); free(z); free(z_new);
    return C4A_OK;
}
