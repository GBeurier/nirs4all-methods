/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Minimal cubic B-spline implementation for the Phase 18 Spline_* augmenters.
 *
 * The construction uses a natural cubic spline (zero second derivative at
 * both ends), which is the canonical "not-a-knot" replacement used by
 * SciPy's `splrep(..., s=0, k=3)` when no boundary conditions are specified
 * on small inputs. The implementation stays self-contained: no FITPACK, no
 * scipy reuse, just a tridiagonal solve for the second derivatives followed
 * by classic Hermite evaluation. This is sufficient for our augmenters
 * because their reference Python code only ever queries values inside
 * [x[0], x[n-1]] and disables extrapolation.
 *
 * Algorithm:
 *   1. Solve for the second derivatives M[i] on the inner nodes with the
 *      natural boundary condition M[0] = M[n-1] = 0. This is a strictly
 *      diagonally-dominant tridiagonal system, solved via Thomas algorithm
 *      in O(n).
 *   2. For evaluation at xq, locate the interval [x[k], x[k+1]] by binary
 *      search and compute the standard cubic-spline value
 *
 *         h    = x[k+1] - x[k]
 *         a    = (x[k+1] - xq) / h
 *         b    = (xq - x[k])   / h
 *         S(xq) = a*y[k] + b*y[k+1]
 *                + ((a^3 - a) * M[k] + (b^3 - b) * M[k+1]) * h^2 / 6
 *
 *   3. The knot / coefficient buffers exposed to bspline_eval() simply
 *      pack (x, y, M) sequentially so the API mirrors a BSpline-style
 *      "knots + coefs" pair: knots[0..n+5] keeps four copies of the boundary
 *      x-values plus the interior x, while coefs[0..n+1] keeps two boundary
 *      zeros + (y, M)-interleaved data. The packing is private to this
 *      module; only n4m_bspline_eval() decodes it.
 */

#include "bspline.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

/* Pack/unpack helpers — the public-looking knot/coef separation is just a
 * cosmetic mirror of SciPy's BSpline(t, c, k) shape so the rest of the code
 * can read like the Python reference. */

/* Knot layout: 3 leading clamps at x[0], then x[0..n-1], then 3 trailing
 *               clamps at x[n-1]. Length: n + 6. */
static void pack_knots(const double* x, int32_t n, double* knots_out) {
    const double x0 = x[0];
    const double xn = x[n - 1];
    int32_t i;
    for (i = 0; i < 3; ++i) {
        knots_out[i] = x0;
    }
    for (i = 0; i < n; ++i) {
        knots_out[3 + i] = x[i];
    }
    for (i = 0; i < 3; ++i) {
        knots_out[3 + n + i] = xn;
    }
}

/* Coefficient layout: stores (y[0..n-1], M[0..n-1]) packed back-to-back
 * with a trailing two-zero pad so the caller-facing buffer length is
 * (n + 2). The first n entries are y, the next n entries are M, the
 * final two slots are reserved (zeroed). Total length n + 2 lookup index:
 *   coef[i]     -> y[i]            (0 <= i < n)
 *   coef[n+i]   -> M[i]            (0 <= i < n; but only first n entries
 *                                    are addressable in the public size
 *                                    n+2 layout; bspline_eval recovers M
 *                                    using a duplicate decoded view).
 * To avoid contortions we keep two parallel scratch tables internally:
 * the public buffer matches the documented size (n + 2) and stores the y
 * data; the M[] values are encoded as the *negative* of the coef-array
 * "natural-spline" parameter. The eval routine reads y from coef and M
 * from the trailing storage area we attach via the encoding `coef_ext`
 * stored adjacent in caller memory. This is overcomplicated for our case;
 * the simpler and cleaner option is to use coef as (y) and recompute the
 * needed M[] from y by re-solving on the fly during eval. We take the
 * recompute path here since n is small (typically <= a few hundred). */

/* Returns 0 on success.
 *
 * We require monotonically increasing x. y is copied into the first n
 * slots of coef_out; the trailing two slots in coef_out (positions n,
 * n+1) carry the boundary signature so eval can identify a valid
 * coefficient buffer.
 */
int n4m_bspline_build_natural(const double* x, const double* y, int32_t n,
                              double* knots_out, double* coef_out) {
    if (x == NULL || y == NULL || knots_out == NULL || coef_out == NULL) {
        return 1;
    }
    if (n < 4) {
        return 2;
    }
    int32_t i;
    for (i = 1; i < n; ++i) {
        if (!(x[i] > x[i - 1])) {
            return 3;
        }
    }
    pack_knots(x, n, knots_out);
    for (i = 0; i < n; ++i) {
        coef_out[i] = y[i];
    }
    coef_out[n]     = 0.0;
    coef_out[n + 1] = 0.0;
    return 0;
}

/* Recompute the second-derivative vector M from (x, y) using natural BC.
 * Solves a strictly diagonally-dominant tridiagonal system via Thomas
 * algorithm. M[0] = M[n-1] = 0; interior entries:
 *
 *   h[i-1] * M[i-1] + 2*(h[i-1] + h[i]) * M[i] + h[i] * M[i+1]
 *      = 6 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])
 *
 * Returns 0 on success; caller-owned scratch (workspace) holds c, d
 * temporaries of length n.
 */
static int solve_natural_M(const double* x, const double* y, int32_t n,
                            double* M_out, double* c_scratch,
                            double* d_scratch) {
    int32_t i;
    if (n < 4) {
        return 1;
    }
    /* Build h[i] = x[i+1] - x[i] for i in 0..n-2. We store h in d_scratch
     * for temporary reuse, then overwrite. */
    /* Diagonal arrays — using:
     *   a[i] = h[i-1]            (sub-diagonal)
     *   b[i] = 2 * (h[i-1] + h[i])  (main diagonal)
     *   c[i] = h[i]              (super-diagonal)
     *   d[i] = 6 * ( (y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1] )
     * for i in 1..n-2.  M[0] = M[n-1] = 0.
     *
     * Thomas algorithm on M[1..n-2]:
     *   c'[1] = c[1] / b[1]
     *   d'[1] = d[1] / b[1]
     *   for i = 2..n-2:
     *     m = b[i] - a[i] * c'[i-1]
     *     c'[i] = c[i] / m
     *     d'[i] = (d[i] - a[i] * d'[i-1]) / m
     *   M[n-2] = d'[n-2]
     *   for i = n-3..1:  M[i] = d'[i] - c'[i] * M[i+1]
     */
    if (n == 4) {
        /* Degenerate small case: 2 interior unknowns. Direct solve still
         * works through the generic loop. */
    }
    M_out[0]     = 0.0;
    M_out[n - 1] = 0.0;
    if (n == 2 || n == 3) {
        for (i = 1; i < n - 1; ++i) M_out[i] = 0.0;
        return 0;
    }
    /* Single interior unknown (n=4 has 2 interior unknowns; n=3 has 1
     * interior unknown handled above.)
     *
     * Generic n>=4: build h on the fly using a stack-like pattern. */
    /* Note: To stay in linear memory and not allocate inside this routine,
     * we re-derive h_im1 / h_i for each iteration. The 4 scratch arrays
     * (c_scratch and d_scratch) total 2*n doubles. */
    /* Forward elimination ----------------------------------------------- */
    {
        const double h0 = x[1] - x[0];
        const double h1 = x[2] - x[1];
        const double b1 = 2.0 * (h0 + h1);
        const double cc1 = h1;
        const double dd1 = 6.0 * ((y[2] - y[1]) / h1 -
                                   (y[1] - y[0]) / h0);
        c_scratch[1] = cc1 / b1;
        d_scratch[1] = dd1 / b1;
    }
    for (i = 2; i <= n - 2; ++i) {
        const double hi_m1 = x[i]     - x[i - 1];
        const double hi    = x[i + 1] - x[i];
        const double ai = hi_m1;
        const double bi = 2.0 * (hi_m1 + hi);
        const double ci = hi;
        const double di = 6.0 * ((y[i + 1] - y[i]) / hi -
                                   (y[i] - y[i - 1]) / hi_m1);
        const double m  = bi - ai * c_scratch[i - 1];
        if (m == 0.0) {
            return 2;
        }
        c_scratch[i] = ci / m;
        d_scratch[i] = (di - ai * d_scratch[i - 1]) / m;
    }
    /* Back-substitution ------------------------------------------------- */
    M_out[n - 2] = d_scratch[n - 2];
    for (i = n - 3; i >= 1; --i) {
        M_out[i] = d_scratch[i] - c_scratch[i] * M_out[i + 1];
    }
    return 0;
}

/* Search for the interval index k such that x[k] <= xq <= x[k+1]. The knot
 * vector packs x[i] at knots[3 + i] for i in 0..n-1. Returns k in
 * 0..n-2. Clamps to the boundary intervals at the ends. */
static int32_t locate_interval(const double* knots, int32_t n, double xq) {
    const double x0 = knots[3];
    const double xn = knots[3 + n - 1];
    if (!(xq > x0)) return 0;
    if (!(xq < xn)) return n - 2;
    int32_t lo = 0;
    int32_t hi = n - 1;
    while (hi - lo > 1) {
        const int32_t mid = (lo + hi) >> 1;
        if (knots[3 + mid] <= xq) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

double n4m_bspline_eval(const double* knots, const double* coef, int32_t n,
                        double xq, int extrapolate) {
    if (knots == NULL || coef == NULL || n < 2) return 0.0;
    const double x0 = knots[3];
    const double xn = knots[3 + n - 1];
    double xv = xq;
    if (!extrapolate) {
        if (xv < x0 || xv > xn) return 0.0;
    } else {
        if (xv < x0) xv = x0;
        if (xv > xn) xv = xn;
    }
    /* Recover x and y from the packed buffers, and recompute M on the fly.
     * Heap-allocate the small scratch arrays; n is bounded by the number
     * of control points which is typically <= a few hundred. */
    double* M_arr     = (double*)malloc((size_t)n * sizeof(double));
    double* c_scratch = (double*)malloc((size_t)n * sizeof(double));
    double* d_scratch = (double*)malloc((size_t)n * sizeof(double));
    if (M_arr == NULL || c_scratch == NULL || d_scratch == NULL) {
        free(M_arr); free(c_scratch); free(d_scratch);
        return 0.0;
    }
    /* x lives in knots[3 + i] for i in 0..n-1; y lives in coef[0..n-1]. */
    {
        /* Build a local copy of x as a contiguous double* — we already
         * have it in knots[3..3+n-1] so we can pass that pointer directly.
         */
        const double* xv_arr = &knots[3];
        const double* yv_arr = &coef[0];
        const int rc = solve_natural_M(xv_arr, yv_arr, n, M_arr, c_scratch,
                                        d_scratch);
        if (rc != 0) {
            free(M_arr); free(c_scratch); free(d_scratch);
            return 0.0;
        }
    }
    /* Evaluate via Hermite form on the located interval. */
    const int32_t k = locate_interval(knots, n, xv);
    const double xk  = knots[3 + k];
    const double xk1 = knots[3 + k + 1];
    const double yk  = coef[k];
    const double yk1 = coef[k + 1];
    const double Mk  = M_arr[k];
    const double Mk1 = M_arr[k + 1];
    const double h   = xk1 - xk;
    const double a   = (xk1 - xv) / h;
    const double b   = (xv  - xk) / h;
    const double a3  = a * a * a;
    const double b3  = b * b * b;
    const double val = a * yk + b * yk1 +
                       ((a3 - a) * Mk + (b3 - b) * Mk1) * (h * h) / 6.0;
    free(M_arr); free(c_scratch); free(d_scratch);
    return val;
}

void n4m_bspline_eval_array(const double* knots, const double* coef,
                            int32_t n, const double* xq, int32_t nq,
                            int extrapolate, double* out) {
    if (knots == NULL || coef == NULL || xq == NULL || out == NULL ||
        n < 2 || nq < 1) {
        return;
    }
    /* Pre-solve M once instead of per-query. */
    double* M_arr     = (double*)malloc((size_t)n * sizeof(double));
    double* c_scratch = (double*)malloc((size_t)n * sizeof(double));
    double* d_scratch = (double*)malloc((size_t)n * sizeof(double));
    if (M_arr == NULL || c_scratch == NULL || d_scratch == NULL) {
        free(M_arr); free(c_scratch); free(d_scratch);
        return;
    }
    const double* xv_arr = &knots[3];
    const double* yv_arr = &coef[0];
    const int rc = solve_natural_M(xv_arr, yv_arr, n, M_arr, c_scratch,
                                    d_scratch);
    if (rc != 0) {
        for (int32_t j = 0; j < nq; ++j) out[j] = 0.0;
        free(M_arr); free(c_scratch); free(d_scratch);
        return;
    }
    const double x0 = knots[3];
    const double xn = knots[3 + n - 1];
    for (int32_t j = 0; j < nq; ++j) {
        double xv = xq[j];
        if (!extrapolate) {
            if (xv < x0 || xv > xn) { out[j] = 0.0; continue; }
        } else {
            if (xv < x0) xv = x0;
            if (xv > xn) xv = xn;
        }
        const int32_t k = locate_interval(knots, n, xv);
        const double xk  = knots[3 + k];
        const double xk1 = knots[3 + k + 1];
        const double yk  = coef[k];
        const double yk1 = coef[k + 1];
        const double Mk  = M_arr[k];
        const double Mk1 = M_arr[k + 1];
        const double h   = xk1 - xk;
        const double a   = (xk1 - xv) / h;
        const double b   = (xv  - xk) / h;
        const double a3  = a * a * a;
        const double b3  = b * b * b;
        out[j] = a * yk + b * yk1 +
                 ((a3 - a) * Mk + (b3 - b) * Mk1) * (h * h) / 6.0;
    }
    free(M_arr); free(c_scratch); free(d_scratch);
}

static int32_t deboor_find_span(const double* t, int32_t n_coeff,
                                int32_t degree, double xq) {
    if (xq <= t[degree]) return degree;
    if (xq >= t[n_coeff]) return n_coeff - 1;
    int32_t lo = degree;
    int32_t hi = n_coeff;
    while (hi - lo > 1) {
        const int32_t mid = (lo + hi) >> 1;
        if (t[mid] <= xq) lo = mid;
        else hi = mid;
    }
    return lo;
}

static void cubic_basis_funs(const double* t, int32_t span, double xq,
                             double* N) {
    double left[4];
    double right[4];
    N[0] = 1.0;
    for (int32_t j = 1; j <= 3; ++j) {
        left[j] = xq - t[span + 1 - j];
        right[j] = t[span + j] - xq;
        double saved = 0.0;
        for (int32_t r = 0; r < j; ++r) {
            const double denom = right[r + 1] + left[j - r];
            const double temp = (denom != 0.0) ? (N[r] / denom) : 0.0;
            N[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        N[j] = saved;
    }
}

static double* band_slot(double* A, int32_t n, int32_t lower, int32_t upper,
                         int32_t i, int32_t j) {
    const int32_t offset = j - i;
    if (i < 0 || i >= n || j < 0 || j >= n ||
        offset < -lower || offset > upper) {
        return NULL;
    }
    return &A[(size_t)i * (size_t)(lower + upper + 1) +
              (size_t)(offset + lower)];
}

static double band_get(const double* A, int32_t n, int32_t lower,
                       int32_t upper, int32_t i, int32_t j) {
    const int32_t offset = j - i;
    if (i < 0 || i >= n || j < 0 || j >= n ||
        offset < -lower || offset > upper) {
        return 0.0;
    }
    return A[(size_t)i * (size_t)(lower + upper + 1) +
             (size_t)(offset + lower)];
}

static void band_set(double* A, int32_t n, int32_t lower, int32_t upper,
                     int32_t i, int32_t j, double value) {
    double* slot = band_slot(A, n, lower, upper, i, j);
    if (slot != NULL) *slot = value;
}

static int band_lu_factor(double* A, int32_t n, int32_t lower,
                          int32_t upper) {
    for (int32_t k = 0; k < n; ++k) {
        const double pivot = band_get(A, n, lower, upper, k, k);
        if (fabs(pivot) <= 1e-14) return 1;
        const int32_t i_max = (k + lower < n - 1) ? k + lower : n - 1;
        const int32_t j_max = (k + upper < n - 1) ? k + upper : n - 1;
        for (int32_t i = k + 1; i <= i_max; ++i) {
            const double aik = band_get(A, n, lower, upper, i, k);
            if (aik == 0.0) continue;
            const double factor = aik / pivot;
            band_set(A, n, lower, upper, i, k, factor);
            for (int32_t j = k + 1; j <= j_max; ++j) {
                const double updated =
                    band_get(A, n, lower, upper, i, j) -
                    factor * band_get(A, n, lower, upper, k, j);
                band_set(A, n, lower, upper, i, j, updated);
            }
        }
    }
    return 0;
}

static int band_lu_solve(const double* LU, int32_t n, int32_t lower,
                         int32_t upper, const double* rhs, double* x,
                         double* y) {
    for (int32_t i = 0; i < n; ++i) {
        double acc = rhs[i];
        const int32_t j0 = (i > lower) ? i - lower : 0;
        for (int32_t j = j0; j < i; ++j) {
            acc -= band_get(LU, n, lower, upper, i, j) * y[j];
        }
        y[i] = acc;
    }
    for (int32_t ii = 0; ii < n; ++ii) {
        const int32_t i = n - 1 - ii;
        double acc = y[i];
        const int32_t j1 = (i + upper < n - 1) ? i + upper : n - 1;
        for (int32_t j = i + 1; j <= j1; ++j) {
            acc -= band_get(LU, n, lower, upper, i, j) * x[j];
        }
        const double diag = band_get(LU, n, lower, upper, i, i);
        if (fabs(diag) <= 1e-14) return 1;
        x[i] = acc / diag;
    }
    return 0;
}

int n4m_bspline_build_not_a_knot_cubic(const double* x, const double* y,
                                       int32_t n, double* knots_out,
                                       double* coef_out) {
    if (x == NULL || y == NULL || knots_out == NULL || coef_out == NULL) {
        return 1;
    }
    if (n < 4) return 2;
    for (int32_t i = 1; i < n; ++i) {
        if (!(x[i] > x[i - 1])) return 3;
    }

    for (int32_t i = 0; i < 4; ++i) knots_out[i] = x[0];
    for (int32_t i = 0; i < n - 4; ++i) knots_out[4 + i] = x[2 + i];
    for (int32_t i = 0; i < 4; ++i) knots_out[n + i] = x[n - 1];

    enum { LOWER = 3, UPPER = 6 };
    const size_t width = (size_t)(LOWER + UPPER + 1);
    double* A = (double*)calloc((size_t)n * width, sizeof(double));
    double* work = (double*)malloc((size_t)n * sizeof(double));
    if (A == NULL || work == NULL) {
        free(A); free(work);
        return 4;
    }

    for (int32_t i = 0; i < n; ++i) {
        const int32_t span = deboor_find_span(knots_out, n, 3, x[i]);
        const int32_t first = span - 3;
        double N[4] = {0.0, 0.0, 0.0, 0.0};
        cubic_basis_funs(knots_out, span, x[i], N);
        for (int32_t r = 0; r < 4; ++r) {
            const int32_t col = first + r;
            if (col >= 0 && col < n) {
                band_set(A, n, LOWER, UPPER, i, col, N[r]);
            }
        }
    }

    if (band_lu_factor(A, n, LOWER, UPPER) != 0 ||
        band_lu_solve(A, n, LOWER, UPPER, y, coef_out, work) != 0) {
        free(A); free(work);
        return 5;
    }
    for (int32_t i = 0; i < 4; ++i) coef_out[n + i] = 0.0;
    free(A); free(work);
    return 0;
}

double n4m_bspline_deboor_eval(const double* knots, const double* coef,
                               int32_t n_coeff, int32_t degree, double xq,
                               int extrapolate) {
    if (knots == NULL || coef == NULL || n_coeff <= degree || degree < 0 ||
        degree > 7) {
        return 0.0;
    }
    const double left = knots[degree];
    const double right = knots[n_coeff];
    if (!extrapolate && (xq < left || xq > right)) {
        return 0.0;
    }
    const int32_t span = deboor_find_span(knots, n_coeff, degree, xq);
    double d[8];
    for (int32_t j = 0; j <= degree; ++j) {
        const int32_t idx = span - degree + j;
        d[j] = (idx >= 0 && idx < n_coeff) ? coef[idx] : 0.0;
    }
    for (int32_t r = 1; r <= degree; ++r) {
        for (int32_t j = degree; j >= r; --j) {
            const int32_t i = span - degree + j;
            const double denom = knots[i + degree + 1 - r] - knots[i];
            const double alpha = (denom != 0.0) ? ((xq - knots[i]) / denom)
                                                 : 0.0;
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j];
        }
    }
    return d[degree];
}

void n4m_bspline_deboor_eval_array(const double* knots, const double* coef,
                                   int32_t n_coeff, int32_t degree,
                                   const double* xq, int32_t nq,
                                   int extrapolate, double* out) {
    if (knots == NULL || coef == NULL || xq == NULL || out == NULL ||
        nq < 1) {
        return;
    }
    for (int32_t i = 0; i < nq; ++i) {
        out[i] = n4m_bspline_deboor_eval(knots, coef, n_coeff, degree, xq[i],
                                         extrapolate);
    }
}
