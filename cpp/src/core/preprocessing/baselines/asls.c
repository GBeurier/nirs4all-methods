/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * AsLS — Asymmetric Least Squares (Eilers & Boelens 2005).
 *
 * Frozen reference: parity/python_generator/src/c4a_parity_pybaselines_ref/asls.py
 *
 * For each row y of length N:
 *
 *   w := ones(N)
 *   penalty := lam * D2^T D2          (n, 3) symmetric pentadiagonal, constant
 *   for iter in 0..max_iter:
 *       (main_diag, super1, super2) := (w + diag(penalty), penalty_sup1, penalty_sup2)
 *       z := banded5_solve(...)
 *       new_w[i] := p if y[i] > z[i] else 1-p
 *       if rel_l2_diff(w, new_w) < tol: break
 *       w := new_w
 *   out[i] := y[i] - z[i]
 *
 * Note: we iterate `for k in 0..max_iter inclusive` which is max_iter + 1
 * solves — same loop pattern as pybaselines.whittaker.asls.
 */

#include "asls.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "core/common/banded_solver.h"

struct c4a_pp_asls_state_t {
    double  lam;
    double  p;
    int32_t max_iter;
    double  tol;
};

c4a_pp_asls_state_t* c4a_pp_asls_state_new(double lam, double p,
                                            int32_t max_iter, double tol) {
    if (!(lam > 0.0) || !(p > 0.0 && p < 1.0) || max_iter < 0 || !(tol >= 0.0)) {
        return NULL;
    }
    c4a_pp_asls_state_t* s = (c4a_pp_asls_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->lam      = lam;
    s->p        = p;
    s->max_iter = max_iter;
    s->tol      = tol;
    return s;
}

void c4a_pp_asls_state_free(c4a_pp_asls_state_t* state) {
    free(state);
}

/* Build the 2nd-order difference penalty diagonals lam * (D2^T D2) for n >= 3.
 * Stores into pen_main / pen_sup1 / pen_sup2 (each length n; the last 1-2
 * elements of pen_sup1 and pen_sup2 are filled with 0 for completeness). */
static void build_penalty_diagonals(int64_t n, double lam,
                                     double* pen_main,
                                     double* pen_sup1,
                                     double* pen_sup2) {
    /* (D^T D) values: rows 0 and n-1 contribute 1 to the main; rows 1 and
     * n-2 contribute 5; interior rows contribute 6. super1 is -2 at the
     * boundaries (k=0, k=n-2) and -4 elsewhere. super2 is 1 for k in
     * [0, n-3] and 0 otherwise. */
    for (int64_t k = 0; k < n; ++k) {
        double m;
        if (k == 0 || k == n - 1)             m = 1.0;
        else if (k == 1 || k == n - 2)        m = 5.0;
        else                                  m = 6.0;
        pen_main[k] = lam * m;
    }
    for (int64_t k = 0; k < n; ++k) {
        double v;
        if (k >= n - 1)                       v = 0.0;
        else if (k == 0 || k == n - 2)        v = -2.0;
        else                                  v = -4.0;
        pen_sup1[k] = lam * v;
    }
    for (int64_t k = 0; k < n; ++k) {
        const double v = (k <= n - 3) ? 1.0 : 0.0;
        pen_sup2[k] = lam * v;
    }
}

/* Relative L2 difference: ||new - old||_2 / max(||old||_2, DBL_MIN).
 * Matches pybaselines.utils.relative_difference. */
static double relative_l2_diff(const double* old, const double* new_,
                                int64_t n) {
    double num_sq = 0.0;
    double den_sq = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double d = new_[i] - old[i];
        num_sq += d * d;
        den_sq += old[i] * old[i];
    }
    const double num = sqrt(num_sq);
    double den = sqrt(den_sq);
    if (den < DBL_MIN) den = DBL_MIN;
    return num / den;
}

c4a_status_t c4a_pp_asls_state_apply(const c4a_pp_asls_state_t* state,
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

    const double lam      = state->lam;
    const double p        = state->p;
    const int32_t max_iter = state->max_iter;
    const double tol      = state->tol;
    const int64_t n       = cols;

    double* pen_main = (double*)malloc((size_t)n * sizeof(double));
    double* pen_sup1 = (double*)malloc((size_t)n * sizeof(double));
    double* pen_sup2 = (double*)malloc((size_t)n * sizeof(double));
    double* main_d   = (double*)malloc((size_t)n * sizeof(double));
    double* w        = (double*)malloc((size_t)n * sizeof(double));
    double* w_new    = (double*)malloc((size_t)n * sizeof(double));
    double* z        = (double*)malloc((size_t)n * sizeof(double));
    double* rhs      = (double*)malloc((size_t)n * sizeof(double));
    if (!pen_main || !pen_sup1 || !pen_sup2 || !main_d ||
        !w || !w_new || !z || !rhs) {
        free(pen_main); free(pen_sup1); free(pen_sup2);
        free(main_d); free(w); free(w_new); free(z); free(rhs);
        return C4A_ERR_OUT_OF_MEMORY;
    }

    build_penalty_diagonals(n, lam, pen_main, pen_sup1, pen_sup2);

    for (int64_t r = 0; r < rows; ++r) {
        const double* y = X + (size_t)r * (size_t)cols;
        for (int64_t i = 0; i < n; ++i) {
            w[i] = 1.0;
        }
        for (int32_t it = 0; it <= max_iter; ++it) {
            /* Build A = diag(w) + penalty; only the diagonal changes per
             * iteration. The super1/super2 stay constant. */
            for (int64_t i = 0; i < n; ++i) {
                main_d[i] = pen_main[i] + w[i];
                rhs[i]    = w[i] * y[i];
            }
            c4a_banded5_t fact = {0, NULL, NULL};
            const c4a_status_t fst = c4a_banded5_factor(main_d, pen_sup1,
                                                        pen_sup2, n, &fact);
            if (fst != C4A_OK) {
                free(pen_main); free(pen_sup1); free(pen_sup2);
                free(main_d); free(w); free(w_new); free(z); free(rhs);
                return fst;
            }
            const c4a_status_t sst = c4a_banded5_solve(&fact, rhs, z);
            c4a_banded5_free(&fact);
            if (sst != C4A_OK) {
                free(pen_main); free(pen_sup1); free(pen_sup2);
                free(main_d); free(w); free(w_new); free(z); free(rhs);
                return sst;
            }
            for (int64_t i = 0; i < n; ++i) {
                w_new[i] = (y[i] > z[i]) ? p : (1.0 - p);
            }
            const double rdiff = relative_l2_diff(w, w_new, n);
            if (rdiff < tol) {
                break;
            }
            memcpy(w, w_new, (size_t)n * sizeof(double));
        }
        double* orow = out + (size_t)r * (size_t)cols;
        for (int64_t i = 0; i < n; ++i) {
            orow[i] = y[i] - z[i];
        }
    }

    free(pen_main); free(pen_sup1); free(pen_sup2);
    free(main_d); free(w); free(w_new); free(z); free(rhs);
    return C4A_OK;
}
