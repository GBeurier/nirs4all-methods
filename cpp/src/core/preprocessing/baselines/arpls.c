/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * ArPLS — Baek 2015.
 *
 * Frozen reference: parity/python_generator/src/c4a_parity_pybaselines_ref/arpls.py
 *
 * Iterative loop:
 *
 *   w := ones(N)
 *   for iter in 0..max_iter:
 *       z := banded5_solve((diag(w) + lam D2^T D2), w * y)
 *       d := y - z
 *       neg := d[d < 0]
 *       if size(neg) < 2: break                          // exit_early
 *       std_neg := std(neg, ddof=1)                      // clamped to DBL_MIN
 *       mean_neg := mean(neg)
 *       new_w[i] := 1 / (1 + exp((2/std) * (d[i] - (2*std - mean))))
 *       if rel_l2_diff(w, new_w) < tol: break
 *       w := new_w
 *   out[i] := y[i] - z[i]
 */

#include "arpls.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "core/common/banded_solver.h"

struct c4a_pp_arpls_state_t {
    double  lam;
    int32_t max_iter;
    double  tol;
};

c4a_pp_arpls_state_t* c4a_pp_arpls_state_new(double lam, int32_t max_iter,
                                              double tol) {
    if (!(lam > 0.0) || max_iter < 0 || !(tol >= 0.0)) {
        return NULL;
    }
    c4a_pp_arpls_state_t* s = (c4a_pp_arpls_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->lam      = lam;
    s->max_iter = max_iter;
    s->tol      = tol;
    return s;
}

void c4a_pp_arpls_state_free(c4a_pp_arpls_state_t* state) {
    free(state);
}

static void arpls_build_penalty(int64_t n, double lam,
                                 double* pen_main,
                                 double* pen_sup1,
                                 double* pen_sup2) {
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

static double arpls_rel_l2_diff(const double* old, const double* new_, int64_t n) {
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

c4a_status_t c4a_pp_arpls_state_apply(const c4a_pp_arpls_state_t* state,
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
    double* w_new    = (double*)malloc((size_t)n * sizeof(double));
    double* z        = (double*)malloc((size_t)n * sizeof(double));
    double* rhs      = (double*)malloc((size_t)n * sizeof(double));
    if (!pen_main || !pen_sup1 || !pen_sup2 || !main_d ||
        !w || !w_new || !z || !rhs) {
        free(pen_main); free(pen_sup1); free(pen_sup2);
        free(main_d); free(w); free(w_new); free(z); free(rhs);
        return C4A_ERR_OUT_OF_MEMORY;
    }

    arpls_build_penalty(n, lam, pen_main, pen_sup1, pen_sup2);

    for (int64_t r = 0; r < rows; ++r) {
        const double* y = X + (size_t)r * (size_t)cols;
        for (int64_t i = 0; i < n; ++i) w[i] = 1.0;

        for (int32_t it = 0; it <= max_iter; ++it) {
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
            /* Collect negative residuals, compute mean and std (ddof=1). */
            int64_t neg_count = 0;
            double sum_neg = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                const double d = y[i] - z[i];
                if (d < 0.0) {
                    ++neg_count;
                    sum_neg += d;
                }
            }
            if (neg_count < 2) {
                /* exit_early — match pybaselines silent return. */
                break;
            }
            const double mean_neg = sum_neg / (double)neg_count;
            double ss = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                const double d = y[i] - z[i];
                if (d < 0.0) {
                    const double diff = d - mean_neg;
                    ss += diff * diff;
                }
            }
            double std_neg = sqrt(ss / (double)(neg_count - 1));
            if (!(std_neg > 0.0)) {
                std_neg = DBL_MIN;
            }
            /* new_w = 1 / (1 + exp((2/std) * (d - (2*std - mean_neg)))) */
            const double inv2 = 2.0 / std_neg;
            const double shift = 2.0 * std_neg - mean_neg;
            for (int64_t i = 0; i < n; ++i) {
                const double d = y[i] - z[i];
                const double arg = inv2 * (d - shift);
                /* Logistic 1 / (1 + exp(arg)) — clip arg to avoid overflow. */
                double e;
                if (arg > 700.0) {
                    /* exp(arg) -> inf, so logistic -> 0 */
                    e = 0.0;
                } else if (arg < -700.0) {
                    /* exp(arg) -> 0, so logistic -> 1 */
                    e = 1.0;
                } else {
                    e = 1.0 / (1.0 + exp(arg));
                }
                w_new[i] = e;
            }
            const double rdiff = arpls_rel_l2_diff(w, w_new, n);
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
