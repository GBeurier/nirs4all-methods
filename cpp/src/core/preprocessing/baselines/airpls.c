/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * AirPLS — Zhang 2010.
 *
 * Internal parity fixture: parity/python_generator/src/c4a_parity_pybaselines_ref/airpls.py
 *
 * Iterative loop:
 *
 *   y_l1 := sum |y|
 *   w    := ones(N)
 *   for i = 1 .. max_iter+1:
 *       z := banded5_solve((diag(w) + lam D2^T D2), w * y)
 *       d := y - z
 *       neg_mask := d < 0
 *       neg_residual := d[neg_mask]
 *       if size(neg_residual) < 2: break               // exit_early
 *       l1neg := sum(neg_residual)                     // signed (<= 0)
 *       log_max := log(DBL_MAX)
 *       inner[neg_mask] := clip( (min(i, 50) / l1neg) * neg_residual,
 *                                0, log_max - spacing(log_max) )
 *       new_w := zeros(N)
 *       new_w[neg_mask] := exp(inner[neg_mask])
 *       if |l1neg| / y_l1 < tol: break
 *       w := new_w
 *   out[i] := y[i] - z[i]
 *
 * Note: NaN's never appear because if (i / l1neg) * neg_residual is +0
 * (when l1neg is non-zero) we always exponentiate a non-negative number.
 * If l1neg == 0 we already have exit_early. Edge case: if y_l1 == 0 the
 * convergence test is undefined; we skip it (the data is all zero and the
 * baseline is zero too).
 *
 * Phase 5b refactor: shared penalty builder + in-place factor; the L/D
 * scratch buffers are lifted to row scope.
 */

#include "airpls.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>

#include "core/common/banded_solver.h"

struct c4a_pp_airpls_state_t {
    double  lam;
    int32_t max_iter;
    double  tol;
};

c4a_pp_airpls_state_t* c4a_pp_airpls_state_new(double lam, int32_t max_iter,
                                                 double tol) {
    if (!(lam > 0.0) || max_iter < 0 || !(tol >= 0.0)) {
        return NULL;
    }
    c4a_pp_airpls_state_t* s = (c4a_pp_airpls_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->lam      = lam;
    s->max_iter = max_iter;
    s->tol      = tol;
    return s;
}

void c4a_pp_airpls_state_free(c4a_pp_airpls_state_t* state) {
    free(state);
}

c4a_status_t c4a_pp_airpls_state_apply(const c4a_pp_airpls_state_t* state,
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

    const double lam       = state->lam;
    const int32_t max_iter = state->max_iter;
    const double tol       = state->tol;
    const int64_t n        = cols;

    double* pen_main = (double*)malloc((size_t)n * sizeof(double));
    double* pen_sup1 = (double*)malloc((size_t)n * sizeof(double));
    double* pen_sup2 = (double*)malloc((size_t)n * sizeof(double));
    double* main_d   = (double*)malloc((size_t)n * sizeof(double));
    double* w        = (double*)malloc((size_t)n * sizeof(double));
    double* z        = (double*)malloc((size_t)n * sizeof(double));
    double* rhs      = (double*)malloc((size_t)n * sizeof(double));
    double* L_buf    = (double*)malloc((size_t)n * (size_t)2 * sizeof(double));
    double* D_buf    = (double*)malloc((size_t)n * sizeof(double));
    if (!pen_main || !pen_sup1 || !pen_sup2 || !main_d ||
        !w || !z || !rhs || !L_buf || !D_buf) {
        free(pen_main); free(pen_sup1); free(pen_sup2);
        free(main_d); free(w); free(z); free(rhs);
        free(L_buf); free(D_buf);
        return C4A_ERR_OUT_OF_MEMORY;
    }

    c4a_second_diff_penalty_pent5(n, lam, pen_main, pen_sup1, pen_sup2);
    /* Match pybaselines' clip upper bound: log(DBL_MAX) - spacing(log(DBL_MAX)).
     * spacing(x) here is the gap to the next representable double. */
    const double log_max         = log(DBL_MAX);
    const double spacing_log_max = nextafter(log_max, HUGE_VAL) - log_max;
    const double clip_hi         = log_max - spacing_log_max;

    for (int64_t r = 0; r < rows; ++r) {
        const double* y = X + (size_t)r * (size_t)cols;
        /* Compute y_l1. */
        double y_l1 = 0.0;
        for (int64_t i = 0; i < n; ++i) y_l1 += fabs(y[i]);
        for (int64_t i = 0; i < n; ++i) w[i] = 1.0;

        for (int32_t i_iter = 1; i_iter <= max_iter + 1; ++i_iter) {
            for (int64_t i = 0; i < n; ++i) {
                main_d[i] = pen_main[i] + w[i];
                rhs[i]    = w[i] * y[i];
            }
            const c4a_status_t fst = c4a_banded5_factor_into(
                L_buf, D_buf, main_d, pen_sup1, pen_sup2, n);
            if (fst != C4A_OK) {
                free(pen_main); free(pen_sup1); free(pen_sup2);
                free(main_d); free(w); free(z); free(rhs);
                free(L_buf); free(D_buf);
                return fst;
            }
            const c4a_status_t sst = c4a_banded5_solve_into(
                L_buf, D_buf, n, rhs, z);
            if (sst != C4A_OK) {
                free(pen_main); free(pen_sup1); free(pen_sup2);
                free(main_d); free(w); free(z); free(rhs);
                free(L_buf); free(D_buf);
                return sst;
            }
            /* Compute residual d, count negatives, sum negatives. */
            int64_t neg_count = 0;
            double signed_neg_sum = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                const double d = y[i] - z[i];
                if (d < 0.0) {
                    ++neg_count;
                    signed_neg_sum += d;
                }
            }
            if (neg_count < 2) {
                /* exit_early — pybaselines silently breaks and returns the
                 * current baseline; we do the same. */
                break;
            }
            /* Update weights. Scaling factor = min(iter, 50) / signed_neg_sum.
             * Sign is negative / negative = positive. */
            const int32_t capped = (i_iter < 50) ? i_iter : 50;
            const double scale = (double)capped / signed_neg_sum;
            /* For convergence check we need |signed_neg_sum| / y_l1 < tol;
             * we have signed_neg_sum < 0, so abs is -signed_neg_sum. */
            const double conv_metric = (y_l1 > 0.0)
                ? (-signed_neg_sum / y_l1)
                : 0.0;
            if (conv_metric < tol) {
                /* Don't update weights this iteration — we're done. */
                break;
            }
            /* Update w in place. */
            for (int64_t i = 0; i < n; ++i) {
                const double d = y[i] - z[i];
                if (d < 0.0) {
                    double inner = scale * d;
                    if (inner < 0.0)             inner = 0.0;
                    else if (inner > clip_hi)    inner = clip_hi;
                    w[i] = exp(inner);
                } else {
                    w[i] = 0.0;
                }
            }
        }
        double* orow = out + (size_t)r * (size_t)cols;
        for (int64_t i = 0; i < n; ++i) {
            orow[i] = y[i] - z[i];
        }
    }

    free(pen_main); free(pen_sup1); free(pen_sup2);
    free(main_d); free(w); free(z); free(rhs);
    free(L_buf); free(D_buf);
    return C4A_OK;
}
