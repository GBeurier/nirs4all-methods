/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * AsLS — Asymmetric Least Squares (Eilers & Boelens 2005).
 *
 * Internal parity fixture: parity/python_generator/src/c4a_parity_pybaselines_ref/asls.py
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
 *
 * Hot path notes:
 *   - initial w=1 factorization is shared across rows;
 *   - weighted LDLT is specialized for the constant D2^T D2 stencil;
 *   - weighted factorization also performs the forward substitution, leaving
 *     only diagonal/back substitution for the solve step.
 */

#include "asls.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>

#if defined(__GNUC__) || defined(__clang__)
#define C4A_ASLS_RESTRICT __restrict__
#else
#define C4A_ASLS_RESTRICT
#endif

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

static inline double asls_penalty_main(int64_t k, int64_t n, double lam) {
    double m;
    if (k == 0 || k == n - 1)             m = 1.0;
    else if (k == 1 || k == n - 2)        m = 5.0;
    else                                  m = 6.0;
    return lam * m;
}

static inline double asls_penalty_super1(int64_t k, int64_t n, double lam) {
    const double v = (k == 0 || k == n - 2) ? -2.0 : -4.0;
    return lam * v;
}

static c4a_status_t asls_factor_into(double* C4A_ASLS_RESTRICT L,
                                      double* C4A_ASLS_RESTRICT D,
                                      const double* C4A_ASLS_RESTRICT w,
                                      int64_t n, double lam) {
    const int use_weights = (w != NULL);

    D[0] = asls_penalty_main(0, n, lam) + (use_weights ? w[0] : 1.0);
    if (D[0] == 0.0) {
        return C4A_ERR_NUMERICAL_FAILURE;
    }
    if (n >= 2) {
        L[0 * 2 + 0] = asls_penalty_super1(0, n, lam) / D[0];
    }
    if (n >= 3) {
        L[0 * 2 + 1] = lam / D[0];
    }

    if (n >= 2) {
        const double L10 = L[0 * 2 + 0];
        D[1] = asls_penalty_main(1, n, lam) + (use_weights ? w[1] : 1.0)
             - L10 * L10 * D[0];
        if (D[1] == 0.0) {
            return C4A_ERR_NUMERICAL_FAILURE;
        }
        if (n >= 3) {
            const double L20 = L[0 * 2 + 1];
            L[1 * 2 + 0] = (asls_penalty_super1(1, n, lam)
                          - L20 * D[0] * L10) / D[1];
        }
        if (n >= 4) {
            L[1 * 2 + 1] = lam / D[1];
        }
    }

    for (int64_t k = 2; k < n; ++k) {
        const double Lk_k1 = L[(k - 1) * 2 + 0];
        const double Lk_k2 = L[(k - 2) * 2 + 1];
        D[k] = asls_penalty_main(k, n, lam) + (use_weights ? w[k] : 1.0)
             - Lk_k1 * Lk_k1 * D[k - 1]
             - Lk_k2 * Lk_k2 * D[k - 2];
        if (D[k] == 0.0) {
            return C4A_ERR_NUMERICAL_FAILURE;
        }
        if (k + 1 < n) {
            const double Lkp1_km1 = L[(k - 1) * 2 + 1];
            L[k * 2 + 0] = (asls_penalty_super1(k, n, lam)
                          - Lkp1_km1 * D[k - 1] * Lk_k1) / D[k];
        }
        if (k + 2 < n) {
            L[k * 2 + 1] = lam / D[k];
        }
    }
    return C4A_OK;
}

static c4a_status_t asls_factor_weighted_into(double* C4A_ASLS_RESTRICT L,
                                               double* C4A_ASLS_RESTRICT D,
                                               const double* C4A_ASLS_RESTRICT w,
                                               const double* C4A_ASLS_RESTRICT y,
                                               double* C4A_ASLS_RESTRICT z,
                                               int64_t n, double lam) {
    const double main_edge = lam;
    const double main_near = 5.0 * lam;
    const double main_mid  = 6.0 * lam;
    const double sup_edge  = -2.0 * lam;
    const double sup_mid   = -4.0 * lam;

    const double w0 = w[0];
    D[0] = main_edge + w0;
    if (D[0] == 0.0) {
        return C4A_ERR_NUMERICAL_FAILURE;
    }
    L[0 * 2 + 0] = sup_edge / D[0];
    L[0 * 2 + 1] = lam / D[0];
    z[0] = w0 * y[0];

    const double L10 = L[0 * 2 + 0];
    const double w1 = w[1];
    D[1] = main_near + w1 - L10 * L10 * D[0];
    if (D[1] == 0.0) {
        return C4A_ERR_NUMERICAL_FAILURE;
    }
    const double L20 = L[0 * 2 + 1];
    L[1 * 2 + 0] = (((n == 3) ? sup_edge : sup_mid)
                  - L20 * D[0] * L10) / D[1];
    if (n >= 4) {
        L[1 * 2 + 1] = lam / D[1];
    }
    z[1] = w1 * y[1] - L10 * z[0];

    if (n == 3) {
        const double w2 = w[2];
        const double L2_1 = L[1 * 2 + 0];
        const double L2_0 = L[0 * 2 + 1];
        D[2] = main_edge + w2
             - L2_1 * L2_1 * D[1]
             - L2_0 * L2_0 * D[0];
        if (D[2] == 0.0) {
            return C4A_ERR_NUMERICAL_FAILURE;
        }
        z[2] = w2 * y[2]
             - L2_1 * z[1]
             - L2_0 * z[0];
        return C4A_OK;
    }

    for (int64_t k = 2; k < n - 2; ++k) {
        const double wk = w[k];
        const double Lk_k1 = L[(k - 1) * 2 + 0];
        const double Lk_k2 = L[(k - 2) * 2 + 1];
        D[k] = main_mid + wk
             - Lk_k1 * Lk_k1 * D[k - 1]
             - Lk_k2 * Lk_k2 * D[k - 2];
        if (D[k] == 0.0) {
            return C4A_ERR_NUMERICAL_FAILURE;
        }
        const double Lkp1_km1 = L[(k - 1) * 2 + 1];
        L[k * 2 + 0] = (sup_mid - Lkp1_km1 * D[k - 1] * Lk_k1) / D[k];
        L[k * 2 + 1] = lam / D[k];
        z[k] = wk * y[k]
             - Lk_k1 * z[k - 1]
             - Lk_k2 * z[k - 2];
    }

    {
        const int64_t k = n - 2;
        const double wk = w[k];
        const double Lk_k1 = L[(k - 1) * 2 + 0];
        const double Lk_k2 = L[(k - 2) * 2 + 1];
        D[k] = main_near + wk
             - Lk_k1 * Lk_k1 * D[k - 1]
             - Lk_k2 * Lk_k2 * D[k - 2];
        if (D[k] == 0.0) {
            return C4A_ERR_NUMERICAL_FAILURE;
        }
        const double Lkp1_km1 = L[(k - 1) * 2 + 1];
        L[k * 2 + 0] = (sup_edge - Lkp1_km1 * D[k - 1] * Lk_k1) / D[k];
        z[k] = wk * y[k]
             - Lk_k1 * z[k - 1]
             - Lk_k2 * z[k - 2];
    }

    {
        const int64_t k = n - 1;
        const double wk = w[k];
        const double Lk_k1 = L[(k - 1) * 2 + 0];
        const double Lk_k2 = L[(k - 2) * 2 + 1];
        D[k] = main_edge + wk
             - Lk_k1 * Lk_k1 * D[k - 1]
             - Lk_k2 * Lk_k2 * D[k - 2];
        if (D[k] == 0.0) {
            return C4A_ERR_NUMERICAL_FAILURE;
        }
        z[k] = wk * y[k]
             - Lk_k1 * z[k - 1]
             - Lk_k2 * z[k - 2];
    }
    return C4A_OK;
}

static void asls_invert_diag(double* D, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        D[i] = 1.0 / D[i];
    }
}

static void asls_solve_y_invdiag(const double* C4A_ASLS_RESTRICT L,
                                 const double* C4A_ASLS_RESTRICT invD,
                                 int64_t n,
                                 const double* C4A_ASLS_RESTRICT y,
                                 double* C4A_ASLS_RESTRICT z) {
    z[0] = y[0];
    z[1] = y[1] - L[0 * 2 + 0] * z[0];
    for (int64_t i = 2; i < n; ++i) {
        z[i] = y[i]
             - L[(i - 1) * 2 + 0] * z[i - 1]
             - L[(i - 2) * 2 + 1] * z[i - 2];
    }
    z[n - 1] = z[n - 1] * invD[n - 1];
    z[n - 2] = z[n - 2] * invD[n - 2]
             - L[(n - 2) * 2 + 0] * z[n - 1];
    for (int64_t i = n - 3; i >= 0; --i) {
        z[i] = z[i] * invD[i]
             - L[i * 2 + 0] * z[i + 1]
             - L[i * 2 + 1] * z[i + 2];
    }
}

static void asls_backsolve_weighted_y(const double* C4A_ASLS_RESTRICT L,
                                      const double* C4A_ASLS_RESTRICT D,
                                      int64_t n,
                                      double* C4A_ASLS_RESTRICT z) {
    z[n - 1] = z[n - 1] / D[n - 1];
    z[n - 2] = z[n - 2] / D[n - 2]
             - L[(n - 2) * 2 + 0] * z[n - 1];
    for (int64_t i = n - 3; i >= 0; --i) {
        z[i] = z[i] / D[i]
             - L[i * 2 + 0] * z[i + 1]
             - L[i * 2 + 1] * z[i + 2];
    }
}

static double asls_update_initial_weights(double* C4A_ASLS_RESTRICT w,
                                          const double* C4A_ASLS_RESTRICT y,
                                          const double* C4A_ASLS_RESTRICT z,
                                          int64_t n,
                                          double p, double q) {
    double num_sq = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double new_w = (y[i] > z[i]) ? p : q;
        const double d = new_w - 1.0;
        num_sq += d * d;
        w[i] = new_w;
    }
    if (num_sq == 0.0) {
        return 0.0;
    }
    return sqrt(num_sq) / sqrt((double)n);
}

static double asls_update_weights(double* C4A_ASLS_RESTRICT w,
                                  const double* C4A_ASLS_RESTRICT y,
                                  const double* C4A_ASLS_RESTRICT z,
                                  int64_t n, double p, double q) {
    double num_sq = 0.0;
    double den_sq = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double old_w = w[i];
        const double new_w = (y[i] > z[i]) ? p : q;
        const double d = new_w - old_w;
        num_sq += d * d;
        den_sq += old_w * old_w;
        w[i] = new_w;
    }
    if (num_sq == 0.0) {
        return 0.0;
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
    const double q        = 1.0 - p;

    const size_t nsz = (size_t)n;
    const size_t scratch_count = nsz * 8u;
    double* scratch = (double*)malloc(scratch_count * sizeof(double));
    if (scratch == NULL) {
        return C4A_ERR_OUT_OF_MEMORY;
    }
    size_t offset = 0;
    double* w = scratch + offset; offset += nsz;
    double* z = scratch + offset; offset += nsz;
    double* L_buf = scratch + offset; offset += nsz * 2u;
    double* D_buf = scratch + offset; offset += nsz;
    double* L_init = scratch + offset; offset += nsz * 2u;
    double* D_init = scratch + offset; offset += nsz;
    (void)offset;

    c4a_status_t st = asls_factor_into(L_init, D_init, NULL, n, lam);
    if (st != C4A_OK) {
        free(scratch);
        return st;
    }
    asls_invert_diag(D_init, n);

    for (int64_t r = 0; r < rows; ++r) {
        const double* y = X + (size_t)r * (size_t)cols;
        double* orow = out + (size_t)r * (size_t)cols;
        asls_solve_y_invdiag(L_init, D_init, n, y, z);
        double rdiff = asls_update_initial_weights(w, y, z, n, p, q);
        if (!(rdiff < tol)) {
            for (int32_t it = 1; it <= max_iter; ++it) {
                st = asls_factor_weighted_into(L_buf, D_buf, w, y, z, n, lam);
                if (st != C4A_OK) {
                    free(scratch);
                    return st;
                }
                asls_backsolve_weighted_y(L_buf, D_buf, n, z);
                rdiff = asls_update_weights(w, y, z, n, p, q);
                if (rdiff < tol) {
                    break;
                }
            }
        }
        for (int64_t i = 0; i < n; ++i) {
            orow[i] = y[i] - z[i];
        }
    }

    free(scratch);
    return C4A_OK;
}

#undef C4A_ASLS_RESTRICT
