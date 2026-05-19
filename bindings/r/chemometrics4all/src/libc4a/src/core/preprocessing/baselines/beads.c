/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * BEADS (simplified) — Ning & Selesnick 2014.
 *
 * Frozen reference: parity/python_generator/src/c4a_parity_pybaselines_ref/beads.py
 *
 * Simplified pentadiagonal variant. The full Ning/Selesnick algorithm uses
 * a Chebyshev approximation of |.| over a 7-diagonal system that combines
 * 1st- and 2nd-difference penalties. For Phase 5b we approximate with a
 * reweighted-L2 sparsity surrogate over the pentadiagonal D_2^T D_2 system
 * scaled by (lam_1 + lam_2), and a data weight lam_0.
 *
 *   For each row y of length n:
 *     w := ones(n)
 *     z := zeros(n)
 *     for iter in 0..max_iter:
 *         A := diag(lam_0 * w) + (lam_1 + lam_2) * D_2^T D_2
 *         rhs := lam_0 * w * y
 *         z_new := A^{-1} rhs
 *         d := y - z_new
 *         w[i] := 1 / (|d[i]| + eps)
 *         normalise w so that mean(w) == 1
 *         if rel_l2_diff(z, z_new) < tol: break
 *         z := z_new
 *     out := y - z
 *
 * The eps regulariser is 1e-3 to avoid division by zero on perfectly
 * matched residuals; this matches the docs/algorithms/beads.md description.
 */

#include "beads.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "core/common/banded_solver.h"

#define C4A_BEADS_EPS 1.0e-3

struct c4a_pp_beads_state_t {
    double  lam_0;
    double  lam_1;
    double  lam_2;
    int32_t max_iter;
    double  tol;
};

c4a_pp_beads_state_t* c4a_pp_beads_state_new(double lam_0, double lam_1,
                                              double lam_2,
                                              int32_t max_iter, double tol) {
    if (!(lam_0 > 0.0) || !(lam_1 > 0.0) || !(lam_2 > 0.0) ||
        max_iter < 0 || !(tol >= 0.0)) {
        return NULL;
    }
    c4a_pp_beads_state_t* s =
        (c4a_pp_beads_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->lam_0    = lam_0;
    s->lam_1    = lam_1;
    s->lam_2    = lam_2;
    s->max_iter = max_iter;
    s->tol      = tol;
    return s;
}

void c4a_pp_beads_state_free(c4a_pp_beads_state_t* state) {
    free(state);
}

c4a_status_t c4a_pp_beads_state_apply(const c4a_pp_beads_state_t* state,
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

    const double  lam_0    = state->lam_0;
    const double  lam_12   = state->lam_1 + state->lam_2;
    const int32_t max_iter = state->max_iter;
    const double  tol      = state->tol;
    const int64_t n        = cols;

    double* pen_main = (double*)malloc((size_t)n * sizeof(double));
    double* pen_sup1 = (double*)malloc((size_t)n * sizeof(double));
    double* pen_sup2 = (double*)malloc((size_t)n * sizeof(double));
    double* main_d   = (double*)malloc((size_t)n * sizeof(double));
    double* w        = (double*)malloc((size_t)n * sizeof(double));
    double* z        = (double*)malloc((size_t)n * sizeof(double));
    double* z_new    = (double*)malloc((size_t)n * sizeof(double));
    double* rhs      = (double*)malloc((size_t)n * sizeof(double));
    double* L_buf    = (double*)malloc((size_t)n * (size_t)2 * sizeof(double));
    double* D_buf    = (double*)malloc((size_t)n * sizeof(double));
    if (!pen_main || !pen_sup1 || !pen_sup2 || !main_d ||
        !w || !z || !z_new || !rhs || !L_buf || !D_buf) {
        free(pen_main); free(pen_sup1); free(pen_sup2);
        free(main_d); free(w); free(z); free(z_new); free(rhs);
        free(L_buf); free(D_buf);
        return C4A_ERR_OUT_OF_MEMORY;
    }

    /* (lam_1 + lam_2) * D_2^T D_2 — constant across rows / iterations. */
    c4a_second_diff_penalty_pent5(n, lam_12, pen_main, pen_sup1, pen_sup2);

    for (int64_t r = 0; r < rows; ++r) {
        const double* y = X + (size_t)r * (size_t)cols;
        for (int64_t i = 0; i < n; ++i) {
            w[i] = 1.0;
            z[i] = 0.0;
        }

        for (int32_t it = 0; it <= max_iter; ++it) {
            for (int64_t i = 0; i < n; ++i) {
                main_d[i] = pen_main[i] + lam_0 * w[i];
                rhs[i]    = lam_0 * w[i] * y[i];
            }
            c4a_status_t st = c4a_banded5_factor_into(
                L_buf, D_buf, main_d, pen_sup1, pen_sup2, n);
            if (st != C4A_OK) {
                free(pen_main); free(pen_sup1); free(pen_sup2);
                free(main_d); free(w); free(z); free(z_new); free(rhs);
                free(L_buf); free(D_buf);
                return st;
            }
            st = c4a_banded5_solve_into(L_buf, D_buf, n, rhs, z_new);
            if (st != C4A_OK) {
                free(pen_main); free(pen_sup1); free(pen_sup2);
                free(main_d); free(w); free(z); free(z_new); free(rhs);
                free(L_buf); free(D_buf);
                return st;
            }
            /* Reweighted-L2 sparsity surrogate. Renormalise to mean 1 so
             * the effective data weight lam_0 stays comparable across
             * iterations. */
            double w_sum = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                const double d = y[i] - z_new[i];
                w[i] = 1.0 / (fabs(d) + C4A_BEADS_EPS);
                w_sum += w[i];
            }
            const double inv_mean = (w_sum > DBL_MIN)
                ? ((double)n / w_sum) : 1.0;
            for (int64_t i = 0; i < n; ++i) {
                w[i] *= inv_mean;
            }
            /* Convergence on the baseline change. */
            const double rdiff = c4a_relative_l2_diff(z, z_new, n);
            memcpy(z, z_new, (size_t)n * sizeof(double));
            if (rdiff < tol) {
                break;
            }
        }

        double* orow = out + (size_t)r * (size_t)cols;
        for (int64_t i = 0; i < n; ++i) {
            orow[i] = y[i] - z[i];
        }
    }

    free(pen_main); free(pen_sup1); free(pen_sup2);
    free(main_d); free(w); free(z); free(z_new); free(rhs);
    free(L_buf); free(D_buf);
    return C4A_OK;
}
