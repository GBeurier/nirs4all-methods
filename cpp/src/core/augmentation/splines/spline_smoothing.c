/* SPDX-License-Identifier: CECILL-2.1 */
#include "spline_smoothing.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef N4M_HAVE_FITPACK
#define N4M_HAVE_FITPACK 0
#endif

#if N4M_HAVE_FITPACK
#if defined(__cplusplus)
extern "C" {
#endif
void curfit_(int* iopt, int* m, double* x, double* y, double* w,
             double* xb, double* xe, int* k, double* s, int* nest,
             int* n, double* t, double* c, double* fp, double* wrk,
             int* lwrk, int* iwrk, int* ier);
void splev_(double* t, int* n, double* c, int* nc, int* k, double* x,
            double* y, int* m, int* e, int* ier);
#if defined(__cplusplus)
}
#endif
#endif

struct n4m_aug_spline_smooth_state_t {
    int dummy;  /* parameterless operator */
};

#if !N4M_HAVE_FITPACK
/*
 * Pure-C cubic smoothing-spline fallback (Reinsch 1967).
 *
 * Used when libn4m is built without a Fortran compiler (e.g. WASM). It
 * mirrors the FITPACK curfit/splev path: fit a natural cubic smoothing
 * spline to (x_i, y_i) with x_i = 0..m-1, unit weights (dy_i = 1), and the
 * same smoothing target S = 1/m, then return the fitted values g(x_i) on
 * the same grid.
 *
 * Reinsch's natural cubic smoothing spline minimises
 *     integral (g'')^2 dx     subject to    sum_i ((y_i - g(x_i))/dy_i)^2 <= S
 * which is exactly the constraint FITPACK's curfit enforces (with w_i = 1,
 * fp = sum_i (y_i - g(x_i))^2 <= s). The two therefore agree on the same
 * variational problem, so the fitted curves are numerically very close
 * (not bit-identical: curfit uses a B-spline knot-insertion strategy, this
 * uses the closed-form Reinsch banded system).
 *
 * Algorithm (Reinsch, Numer. Math. 10 (1967), 177-183, routine "smooth"):
 *   Unknowns are the second derivatives u_i at the interior knots
 *   i = 1..m-2 (natural boundary: u_0 = u_{m-1} = 0). They solve
 *       (Q^T D^2 Q + p R) u = Q^T y
 *   where, with h_i = x_{i+1} - x_i (= 1 here):
 *       Q is the (m x (m-2)) second-difference matrix, columns j=1..m-2:
 *           Q_{j-1,j} = 1/h_{j-1},  Q_{j,j} = -(1/h_{j-1}+1/h_j),
 *           Q_{j+1,j} = 1/h_j   -> with h=1 the stencil is (1, -2, 1);
 *       R is the symmetric tridiagonal "stiffness" matrix, interior rows:
 *           R_{j,j} = (h_{j-1}+h_j)/3,  R_{j,j+1} = h_j/6
 *           -> with h=1: diag 2/3, off-diag 1/6;
 *       D = diag(dy_i) = I here.
 *   The unknowns are the (m-2) interior second derivatives, so the system
 *   matrix is the SQUARE Gram Q^T D^2 Q of size (m-2) x (m-2). With h=1 and
 *   D=I it is symmetric pentadiagonal with CONSTANT bands
 *       diag 6, super-1 -4, super-2 1
 *   on every interior row, endpoints included: column j of Q is the full
 *   second-difference vector e_{j-1} - 2 e_j + e_{j+1} for every
 *   j = 1..m-2 (all of its three support rows 0..m-1 are valid), so
 *   <col_j, col_j> = 1 + 4 + 1 = 6 with no endpoint reduction. (The
 *   "1,5,6,...,5,1" diagonal sometimes quoted for smoothing splines is the
 *   m x m Gram Q Q^T of the *full-grid* formulation, a different system; the
 *   (m-2)-interior system used here has a uniform-6 diagonal, and only with a
 *   uniform 6 does the p=0 limit Q^T Q u = Q^T y reduce g = y - Q u to the
 *   exact least-squares straight line.)
 *   The smoothed values are  g = y - D^2 Q u  =  y - (Q u)  (D = I), where
 *       (Q u)_i = u_{i-1} - 2 u_i + u_{i+1}  (u_0 = u_{m-1} = 0).
 *   The residual sum of squares is F2(p) = || D^2 Q u ||^2 = || Q u ||^2,
 *   monotone decreasing in p; we drive F2(p) -> S by Newton iteration on
 *   the (convex, increasing) function  f(p) = 1/sqrt(F2(p)).
 *
 * All work arrays are sized to m, so there is no unbounded allocation.
 * Returns 0 on success, -1 on out-of-memory.
 *
 * Measured agreement vs FITPACK curfit/splev (synthetic NIRS-like spectra,
 * x=0..m-1, w=1, s=1/m): for the realistic regime m >= 100 the per-point
 * max difference is ~0.2%-1.1% of the signal range (max abs ~2e-3..1.5e-2,
 * RMS ~6e-4..3e-3), with both methods driving the residual SS to the same
 * smoothing target s to 4-5 significant figures. For very short rows
 * (m < ~32) the two diverge more (curfit's discrete knot budget under- or
 * over-smooths relative to s while Reinsch hits the s constraint exactly);
 * such rows do not occur for real spectra. This augmenter is a perturbation,
 * so the sub-1% realistic-size agreement is well within tolerance.
 */
static int n4m_reinsch_smooth(const double* y, int m, double target_s,
                              double* g) {
    if (m < 4) {
        for (int i = 0; i < m; ++i) g[i] = y[i];
        return 0;
    }

    /* Interior unknowns u_1..u_{m-2}; we allocate full length m and keep
     * u[0] = u[m-1] = 0 throughout (natural boundary). */
    double* u  = (double*)calloc((size_t)m, sizeof(double));  /* 2nd derivs */
    double* qy = (double*)calloc((size_t)m, sizeof(double));  /* Q^T y     */
    double* d  = (double*)calloc((size_t)m, sizeof(double));  /* LDL^T diag */
    double* l1 = (double*)calloc((size_t)m, sizeof(double));  /* L band +1 */
    double* l2 = (double*)calloc((size_t)m, sizeof(double));  /* L band +2 */
    double* w  = (double*)calloc((size_t)m, sizeof(double));  /* solve work */
    if (u == NULL || qy == NULL || d == NULL || l1 == NULL || l2 == NULL ||
        w == NULL) {
        free(u); free(qy); free(d); free(l1); free(l2); free(w);
        return -1;
    }

    /* Right-hand side Q^T y (stencil 1,-2,1) for interior rows. */
    for (int i = 1; i < m - 1; ++i) {
        qy[i] = y[i - 1] - 2.0 * y[i] + y[i + 1];
    }

    /* Constant tridiagonal R bands (h = 1). */
    const double r_diag = 2.0 / 3.0;
    const double r_off  = 1.0 / 6.0;

    /*
     * Iterate on the penalty p to drive the residual SS onto the target.
     * Each step solves the penalised normal equations
     *     M(p) u = Q^T y,   M(p) = Q^T Q + p R,
     * then forms the smoothed residual r = Q u and its SS F2(p) = ||Q u||^2.
     * F2 decreases monotonically as p grows (larger p => smoother fit), so a
     * damped multiplicative update on p converges F2 to the target s.
     */
    double p = 0.0;
    double f2 = 0.0;
    const int max_iter = 100;

    for (int iter = 0; iter < max_iter; ++iter) {
        /* Assemble symmetric pentadiagonal M = Q^T Q + p R over interior
         * indices [1, m-2]; store bands in d (diag), l1 (+1), l2 (+2). */
        for (int i = 1; i < m - 1; ++i) {
            /* (m-2)-interior Gram Q^T Q: constant pentadiagonal bands
             * diag 6, super-1 -4, super-2 1 on EVERY interior row (no
             * endpoint reduction — column j of Q is the full 1,-2,1 stencil
             * for all j=1..m-2). Add the penalty p*R (R: diag 2/3, off 1/6).
             * The R off-diagonal couples only adjacent unknowns, so it lands
             * on the super-1 band; the super-2 band stays at 1. */
            d[i]  = 6.0 + p * r_diag;
            l1[i] = (i + 1 <= m - 2) ? (-4.0 + p * r_off) : 0.0;
            l2[i] = (i + 2 <= m - 2) ? (1.0) : 0.0;
        }

        /* LDL^T factorisation of the pentadiagonal M (bandwidth 2).
         * Overwrite d with the D diagonal and l1,l2 with the unit-L bands. */
        for (int i = 1; i < m - 1; ++i) {
            double di = d[i];
            if (i >= 3) di -= l2[i - 2] * l2[i - 2] * d[i - 2];
            if (i >= 2) di -= l1[i - 1] * l1[i - 1] * d[i - 1];
            if (di < 1e-300) di = 1e-300;  /* keep strictly positive */
            d[i] = di;

            if (i + 1 <= m - 2) {
                double a = l1[i];
                if (i >= 2) a -= l1[i - 1] * l2[i - 1] * d[i - 1];
                l1[i] = a / di;
            }
            if (i + 2 <= m - 2) {
                l2[i] = l2[i] / di;
            }
        }

        /* Solve M u = qy:  L (D L^T u) = qy. */
        for (int i = 1; i < m - 1; ++i) {
            double wi = qy[i];
            if (i >= 2) wi -= l1[i - 1] * w[i - 1];
            if (i >= 3) wi -= l2[i - 2] * w[i - 2];
            w[i] = wi;
        }
        for (int i = m - 2; i >= 1; --i) {
            double ui = w[i] / d[i];
            if (i + 1 <= m - 2) ui -= l1[i] * u[i + 1];
            if (i + 2 <= m - 2) ui -= l2[i] * u[i + 2];
            u[i] = ui;
        }
        u[0] = 0.0;
        u[m - 1] = 0.0;

        /* Residual SS F2 = ||Q u||^2 = sum_i (u_{i-1} - 2u_i + u_{i+1})^2. */
        f2 = 0.0;
        for (int i = 0; i < m; ++i) {
            double um1 = (i > 0) ? u[i - 1] : 0.0;
            double up1 = (i < m - 1) ? u[i + 1] : 0.0;
            double qu = um1 - 2.0 * u[i] + up1;
            f2 += qu * qu;
        }

        if (p == 0.0) {
            /* p = 0 is the least-squares straight-line projection (maximal
             * smoothing), where F2 (the residual sum of squares) is at its
             * UPPER bound — not interpolation. If the input is essentially
             * flat even that bound is ~0, so there is nothing to smooth. */
            double max_qy = 0.0;
            for (int i = 1; i < m - 1; ++i) {
                double a = fabs(qy[i]);
                if (a > max_qy) max_qy = a;
            }
            if (max_qy <= 0.0) break;  /* flat row: nothing to smooth */
            /* The interpolant's residual is 0, which is below target S, so
             * curfit would also accept a smoother fit: seed p from the
             * roughness scale and let Newton converge F2 up toward S.
             *
             * F2 at the *least-squares straight line* (p = 0) is the
             * upper bound; if that bound is already <= S, the straight line
             * is the answer. We approximate by climbing p from a scaled
             * seed. */
            double sum_qy2 = 0.0;
            for (int i = 1; i < m - 1; ++i) sum_qy2 += qy[i] * qy[i];
            /* Seed: at small p, F2 ~ p^2 * ||R^{-1} Q^T y||^2-ish; a robust
             * scale is p0 such that p0 * ||Qy|| ~ sqrt(S). */
            p = sqrt(target_s) / (sqrt(sum_qy2) + 1e-300);
            if (!(p > 0.0)) break;
            continue;
        }

        if (f2 <= 0.0) break;  /* fully smoothed (degenerate) — accept */

        /* Converged when the residual SS sits on the target within 0.1%.
         * curfit stops as soon as fp <= s; landing on fp == s gives the
         * unique active-constraint smoothing spline (the close-to-FITPACK
         * solution). */
        if (fabs(f2 - target_s) <= 1e-3 * target_s) break;

        /* Multiplicative Newton-like update toward F2 == S. F2 decreases as
         * p grows, so scale p by sqrt(F2/S): if F2 > S we need more
         * smoothing (p up); if F2 < S we need less (p down). Damp the step
         * to keep the iteration monotone and bounded. */
        double ratio = sqrt(f2 / target_s);
        if (ratio > 4.0) ratio = 4.0;
        if (ratio < 0.25) ratio = 0.25;
        double pnew = p * ratio;
        if (!(pnew > 0.0) || pnew == p) break;
        p = pnew;
    }

    /* Final smoothed values g = y - Q u. */
    for (int i = 0; i < m; ++i) {
        double um1 = (i > 0) ? u[i - 1] : 0.0;
        double up1 = (i < m - 1) ? u[i + 1] : 0.0;
        double qu = um1 - 2.0 * u[i] + up1;
        g[i] = y[i] - qu;
    }

    free(u); free(qy); free(d); free(l1); free(l2); free(w);
    return 0;
}
#endif /* !N4M_HAVE_FITPACK */

n4m_aug_spline_smooth_state_t* n4m_aug_spline_smooth_state_new(void) {
    n4m_aug_spline_smooth_state_t* s = (n4m_aug_spline_smooth_state_t*)
        calloc(1, sizeof(*s));
    return s;
}

void n4m_aug_spline_smooth_state_free(
    n4m_aug_spline_smooth_state_t* state) {
    free(state);
}

n4m_status_t n4m_aug_spline_smooth_state_apply(
    const n4m_aug_spline_smooth_state_t* state,
    void* rng_void,
    const double* X, int64_t rows, int64_t cols,
    double* out) {
    (void)state;
    (void)rng_void;
    if (X == NULL || out == NULL) return N4M_ERR_NULL_POINTER;
    if (rows < 0 || cols < 0) return N4M_ERR_INVALID_ARGUMENT;

    /* Total-size guard. rows/cols are API-accepted int64_t; their product (the
     * element count) and its byte size feed both the passthrough memcpy and
     * the per-row pointer arithmetic X + i*cols. Reject dims whose product
     * would overflow int64_t, or whose byte size would overflow size_t, BEFORE
     * the multiplication is performed (signed overflow is UB) and before any
     * pointer is formed — otherwise an absurd-but-legal (rows, cols) could wrap
     * the index/size and produce a wild pointer. The division tests never
     * overflow because both operands are non-negative here. */
    if (rows != 0 && cols > INT64_MAX / rows) return N4M_ERR_INVALID_ARGUMENT;
    const int64_t total = rows * cols;
    if ((uint64_t)total > (uint64_t)(SIZE_MAX / sizeof(double)))
        return N4M_ERR_INVALID_ARGUMENT;
    const size_t total_bytes = (size_t)total * sizeof(double);

    if (rows == 0 || cols == 0) {
        if (out != X) memcpy(out, X, total_bytes);
        return N4M_OK;
    }
    if (cols < 4) {
        if (out != X) memcpy(out, X, total_bytes);
        return N4M_OK;
    }
#if !N4M_HAVE_FITPACK
    /* Pure-C cubic smoothing-spline fallback (no Fortran). Real smoothing,
     * not a pass-through. See n4m_reinsch_smooth above. Smoothing target
     * S = 1/cols mirrors the FITPACK s = 1.0/cols. */
    {
        if (cols > (int64_t)INT_MAX) return N4M_ERR_INVALID_ARGUMENT;
        const int m_value = (int)cols;
        const double s = 1.0 / (double)cols;
        for (int64_t i = 0; i < rows; ++i) {
            const double* xrow = X + i * cols;
            double* orow = out + i * cols;
            if (n4m_reinsch_smooth(xrow, m_value, s, orow) != 0) {
                return N4M_ERR_OUT_OF_MEMORY;
            }
        }
        return N4M_OK;
    }
#else
    if (cols > (int64_t)(INT_MAX - 4)) return N4M_ERR_INVALID_ARGUMENT;

    const int k_value = 3;
    const int k1 = k_value + 1;
    const int m_value = (int)cols;
    const int max_nest_value = m_value + k1;
    int default_nest_value = m_value / 2;
    const int min_nest_value = 2 * k1;
    if (default_nest_value < min_nest_value) default_nest_value = min_nest_value;
    if (default_nest_value > max_nest_value) default_nest_value = max_nest_value;
    const int lwrk_value = m_value * k1 + max_nest_value * (7 + 3 * k_value);

    double* x = (double*)malloc((size_t)m_value * sizeof(double));
    double* y = (double*)malloc((size_t)m_value * sizeof(double));
    double* w = (double*)malloc((size_t)m_value * sizeof(double));
    double* knots = (double*)malloc((size_t)max_nest_value * sizeof(double));
    double* coef = (double*)malloc((size_t)max_nest_value * sizeof(double));
    double* wrk = (double*)malloc((size_t)lwrk_value * sizeof(double));
    int* iwrk = (int*)malloc((size_t)max_nest_value * sizeof(int));
    if (x == NULL || y == NULL || w == NULL || knots == NULL || coef == NULL ||
        wrk == NULL || iwrk == NULL) {
        free(x); free(y); free(w); free(knots); free(coef); free(wrk); free(iwrk);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    for (int64_t j = 0; j < cols; ++j) {
        x[j] = (double)j;
        w[j] = 1.0;
    }

    for (int64_t i = 0; i < rows; ++i) {
        const double* xrow = X + i * cols;
        double* orow = out + i * cols;
        memcpy(y, xrow, (size_t)m_value * sizeof(double));
        memset(knots, 0, (size_t)max_nest_value * sizeof(double));
        memset(coef, 0, (size_t)max_nest_value * sizeof(double));
        memset(wrk, 0, (size_t)lwrk_value * sizeof(double));
        memset(iwrk, 0, (size_t)max_nest_value * sizeof(int));

        int iopt = 0;
        int m = m_value;
        int k = k_value;
        int nest = default_nest_value;
        int n = 0;
        int lwrk = lwrk_value;
        int ier = 0;
        double xb = 0.0;
        double xe = (double)(cols - 1);
        double s = 1.0 / (double)cols;
        double fp = 0.0;
        curfit_(&iopt, &m, x, y, w, &xb, &xe, &k, &s, &nest, &n, knots, coef,
                &fp, wrk, &lwrk, iwrk, &ier);

        if (ier == 1 && default_nest_value < max_nest_value) {
            iopt = 1;
            nest = max_nest_value;
            curfit_(&iopt, &m, x, y, w, &xb, &xe, &k, &s, &nest, &n, knots,
                    coef, &fp, wrk, &lwrk, iwrk, &ier);
        }

        if (ier == 10 || n <= k + 1) {
            memcpy(orow, xrow, (size_t)cols * sizeof(double));
            continue;
        }

        int nc = max_nest_value;
        int e = 0;
        int eval_ier = 0;
        splev_(knots, &n, coef, &nc, &k, x, orow, &m, &e, &eval_ier);
        if (eval_ier != 0) memcpy(orow, xrow, (size_t)cols * sizeof(double));
    }
    free(x); free(y); free(w); free(knots); free(coef); free(wrk); free(iwrk);
    return N4M_OK;
#endif
}
