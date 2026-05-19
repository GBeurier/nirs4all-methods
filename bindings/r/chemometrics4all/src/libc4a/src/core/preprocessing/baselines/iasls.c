/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * IAsLS — Improved Asymmetric Least Squares (He et al. 2014).
 *
 * Frozen reference: parity/python_generator/src/c4a_parity_pybaselines_ref/iasls.py
 *
 * Algorithm (for each row y):
 *
 *   z0 := polyfit(positions, y, polyorder)             # initial baseline
 *   w[i] := p if y[i] > z0[i] else (1 - p)              # AsLS-style init
 *   d1_y := lam_1 * (D_1^T D_1) y
 *   for iter in 0..max_iter:
 *       A := diag(w^2) + lam * D_2^T D_2 + lam_1 * D_1^T D_1
 *       z := A^{-1} (w^2 * y + d1_y)
 *       w_new[i] := p if y[i] > z[i] else (1 - p)
 *       if rel_l2_diff(w, w_new) < tol: break
 *       w := w_new
 *   out := y - z
 *
 * This follows pybaselines.whittaker.Baseline.iasls for diff_order=2. Higher
 * differential orders need a wider band solver and are rejected for now.
 */

#include "iasls.h"

#include <stdlib.h>
#include <string.h>

#include "core/common/banded_solver.h"
#include "core/common/linalg.h"

struct c4a_pp_iasls_state_t {
    double  lam;
    double  p;
    double  lam_1;
    int32_t polyorder;
    int32_t diff_order;
    int32_t max_iter;
    double  tol;
};

c4a_pp_iasls_state_t* c4a_pp_iasls_state_new(double lam, double p,
                                              int32_t polyorder,
                                              int32_t max_iter, double tol) {
    return c4a_pp_iasls_state_new_ex(lam, p, 1e-4, polyorder, 2,
                                      max_iter, tol);
}

c4a_pp_iasls_state_t* c4a_pp_iasls_state_new_ex(double lam, double p,
                                                 double lam_1,
                                                 int32_t polyorder,
                                                 int32_t diff_order,
                                                 int32_t max_iter,
                                                 double tol) {
    if (!(lam > 0.0) || !(p > 0.0 && p < 1.0) || !(lam_1 > 0.0) ||
        polyorder < 0 || diff_order != 2 || max_iter < 0 ||
        !(tol >= 0.0)) {
        return NULL;
    }
    c4a_pp_iasls_state_t* s =
        (c4a_pp_iasls_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->lam        = lam;
    s->p          = p;
    s->lam_1      = lam_1;
    s->polyorder  = polyorder;
    s->diff_order = diff_order;
    s->max_iter   = max_iter;
    s->tol        = tol;
    return s;
}

void c4a_pp_iasls_state_free(c4a_pp_iasls_state_t* state) {
    free(state);
}

c4a_status_t c4a_pp_iasls_state_apply(const c4a_pp_iasls_state_t* state,
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
    if (cols < 3) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    const double  lam        = state->lam;
    const double  p          = state->p;
    const double  lam_1      = state->lam_1;
    const int32_t polyorder  = state->polyorder;
    const int32_t diff_order = state->diff_order;
    const int32_t max_iter   = state->max_iter;
    const double  tol        = state->tol;
    const int64_t n          = cols;
    const int64_t pcoefs     = (int64_t)(polyorder + 1);
    if (n < pcoefs) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (diff_order != 2) {
        return C4A_ERR_INVALID_ARGUMENT;
    }

    /* Penalty diagonals. */
    double* pen_main = (double*)malloc((size_t)n * sizeof(double));
    double* pen_sup1 = (double*)malloc((size_t)n * sizeof(double));
    double* pen_sup2 = (double*)malloc((size_t)n * sizeof(double));
    double* main_d   = (double*)malloc((size_t)n * sizeof(double));
    double* w        = (double*)malloc((size_t)n * sizeof(double));
    double* w_new    = (double*)malloc((size_t)n * sizeof(double));
    double* z        = (double*)malloc((size_t)n * sizeof(double));
    double* rhs      = (double*)malloc((size_t)n * sizeof(double));
    double* d1_y     = (double*)malloc((size_t)n * sizeof(double));
    double* L_buf    = (double*)malloc((size_t)n * (size_t)2 * sizeof(double));
    double* D_buf    = (double*)malloc((size_t)n * sizeof(double));
    /* Polynomial scratch. */
    double* V_orig = (double*)malloc((size_t)n * (size_t)pcoefs * sizeof(double));
    double* V_qr   = (double*)malloc((size_t)n * (size_t)pcoefs * sizeof(double));
    double* tau    = (double*)malloc((size_t)pcoefs * sizeof(double));
    double* qt_y   = (double*)malloc((size_t)n * sizeof(double));
    double* coefs  = (double*)malloc((size_t)pcoefs * sizeof(double));
    if (!pen_main || !pen_sup1 || !pen_sup2 || !main_d || !w || !w_new ||
        !z || !rhs || !d1_y || !L_buf || !D_buf || !V_orig || !V_qr || !tau ||
        !qt_y || !coefs) {
        free(pen_main); free(pen_sup1); free(pen_sup2);
        free(main_d); free(w); free(w_new); free(z); free(rhs);
        free(d1_y);
        free(L_buf); free(D_buf);
        free(V_orig); free(V_qr); free(tau); free(qt_y); free(coefs);
        return C4A_ERR_OUT_OF_MEMORY;
    }

    c4a_second_diff_penalty_pent5(n, lam, pen_main, pen_sup1, pen_sup2);
    for (int64_t i = 0; i < n; ++i) {
        pen_main[i] += lam_1 * ((i == 0 || i == n - 1) ? 1.0 : 2.0);
        if (i < n - 1) {
            pen_sup1[i] -= lam_1;
        }
    }

    /* Build and factor Vandermonde once (positions 0..n-1, descending). */
    for (int64_t i = 0; i < n; ++i) {
        const double xi = (double)i;
        double power = 1.0;
        for (int32_t j = (int32_t)pcoefs - 1; j >= 0; --j) {
            V_orig[(size_t)i * (size_t)pcoefs + (size_t)j] = power;
            power *= xi;
        }
    }
    memcpy(V_qr, V_orig, (size_t)n * (size_t)pcoefs * sizeof(double));
    const c4a_status_t qrst = c4a_householder_qr(V_qr, n, pcoefs, tau);
    if (qrst != C4A_OK) {
        free(pen_main); free(pen_sup1); free(pen_sup2);
        free(main_d); free(w); free(w_new); free(z); free(rhs);
        free(d1_y);
        free(L_buf); free(D_buf);
        free(V_orig); free(V_qr); free(tau); free(qt_y); free(coefs);
        return qrst;
    }

    for (int64_t r = 0; r < rows; ++r) {
        const double* y = X + (size_t)r * (size_t)cols;

        /* Polynomial pre-fit z0. */
        memcpy(qt_y, y, (size_t)n * sizeof(double));
        c4a_status_t st = c4a_apply_qt(V_qr, n, pcoefs, tau, qt_y);
        if (st != C4A_OK) { goto fail; }
        st = c4a_back_solve_R(V_qr, n, pcoefs, qt_y, coefs);
        if (st != C4A_OK) { goto fail; }
        /* Re-use `z` as z0 buffer for initial weight init, then it'll be
         * overwritten with the iterative baseline. */
        for (int64_t i = 0; i < n; ++i) {
            double bval = 0.0;
            for (int32_t j = 0; j < (int32_t)pcoefs; ++j) {
                bval += coefs[j] * V_orig[(size_t)i * (size_t)pcoefs + (size_t)j];
            }
            z[i] = bval;
        }

        /* Initial weights from the polynomial baseline. */
        for (int64_t i = 0; i < n; ++i) {
            w[i] = (y[i] > z[i]) ? p : (1.0 - p);
        }

        d1_y[0] = lam_1 * (y[0] - y[1]);
        for (int64_t i = 1; i < n - 1; ++i) {
            d1_y[i] = lam_1 * (2.0 * y[i] - y[i - 1] - y[i + 1]);
        }
        d1_y[n - 1] = lam_1 * (y[n - 1] - y[n - 2]);

        /* pybaselines-compatible IAsLS banded reweighting loop. */
        for (int32_t it = 0; it <= max_iter; ++it) {
            for (int64_t i = 0; i < n; ++i) {
                const double w2 = w[i] * w[i];
                main_d[i] = pen_main[i] + w2;
                rhs[i]    = w2 * y[i] + d1_y[i];
            }
            st = c4a_banded5_factor_into(L_buf, D_buf, main_d,
                                          pen_sup1, pen_sup2, n);
            if (st != C4A_OK) { goto fail; }
            st = c4a_banded5_solve_into(L_buf, D_buf, n, rhs, z);
            if (st != C4A_OK) { goto fail; }
            for (int64_t i = 0; i < n; ++i) {
                w_new[i] = (y[i] > z[i]) ? p : (1.0 - p);
            }
            const double rdiff = c4a_relative_l2_diff(w, w_new, n);
            if (rdiff < tol) {
                break;
            }
            memcpy(w, w_new, (size_t)n * sizeof(double));
        }

        double* orow = out + (size_t)r * (size_t)cols;
        for (int64_t i = 0; i < n; ++i) {
            orow[i] = y[i] - z[i];
        }
        continue;
    fail:
        free(pen_main); free(pen_sup1); free(pen_sup2);
        free(main_d); free(w); free(w_new); free(z); free(rhs);
        free(d1_y);
        free(L_buf); free(D_buf);
        free(V_orig); free(V_qr); free(tau); free(qt_y); free(coefs);
        return st;
    }

    free(pen_main); free(pen_sup1); free(pen_sup2);
    free(main_d); free(w); free(w_new); free(z); free(rhs);
    free(d1_y);
    free(L_buf); free(D_buf);
    free(V_orig); free(V_qr); free(tau); free(qt_y); free(coefs);
    return C4A_OK;
}
