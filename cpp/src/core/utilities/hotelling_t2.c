/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Hotelling T² reference implementation.
 *
 * The PCA step uses the shared Jacobi SVD (`core/common/svd.{c,h}`). The
 * UCL r (n-1)(n+1) / (n (n-r)) F_{1-alpha}(r, n-r) is solved directly: with
 * kappa = (n^2 - 1) / n, a UCL G corresponds to
 *
 *     Y = G / (kappa + G) ~ Beta(r / 2, (n - r) / 2),
 *
 * whose tails are regularised incomplete beta functions (continued fraction,
 * Numerical Recipes §6.4), evaluated in logarithms from G / kappa so that no
 * beta argument goes through a complement close to 1. The small tail is the
 * one inverted (upper for alpha <= 0.5, lower otherwise), by bisection on
 * the normal doubles.
 */

#include "hotelling_t2.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "column_centring.h"
#include "core/common/svd.h"

/* ------------------------------------------------------------------------- */
/* Lanczos approximation (g = 7, nine terms; relative error about 1e-15 on   */
/* Gamma for x >= 0.5): Gamma(x) = sqrt(2 pi) t^(x - 0.5) e^-t S(x), with    */
/* t = x + 6.5. Kept in this split form so that ln B(a, b) cancels its large */
/* terms analytically.                                                       */
/* ------------------------------------------------------------------------- */
static double lanczos_sum(double x) {
    static const double c[9] = {
        0.99999999999980993,  676.5203681218851,     -1259.1392167224028,
        771.32342877765313,   -176.61502916214059,   12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
    };
    double s = c[0];
    for (int i = 1; i < 9; ++i) s += c[i] / (x - 1.0 + (double)i);
    return s;
}

static double lngamma_ref(double x) {
    const double t = x + 6.5;
    return 0.91893853320467274178 + (x - 0.5) * log(t) - t + log(lanczos_sum(x));
}

/* ln B(a, b), a, b >= 0.5. With b >= a, ln Gamma(b) - ln Gamma(a + b) is
 * -(b - 0.5) log1p(a / (b + 6.5)) - a ln(a + b + 6.5) + a + ln(S(b) / S(a + b)):
 * no difference of two terms of order b ln b, so a large b (many
 * observations) keeps the relative accuracy of the tails. */
static double lbeta_ref(double a, double b) {
    if (a > b) {
        const double t = a;
        a = b;
        b = t;
    }
    const double tb = b + 6.5;
    return lngamma_ref(a) - (b - 0.5) * log1p(a / tb) - a * log(a + tb) + a +
           log(lanczos_sum(b) / lanczos_sum(a + b));
}

/* Continued fraction of I_x(a, b) (Numerical Recipes §6.4, modified Lentz).
 * It needs O(sqrt(max(a, b))) terms; NaN when it has not converged within
 * the cap, so a truncated value is never used. */
static double betacf_ref(double a, double b, double x) {
    const int    MAX_IT = 20000;
    const double FPMIN  = 1e-300;
    const double EPS    = 3.0e-16;
    const double qab = a + b;
    const double qap = a + 1.0;
    const double qam = a - 1.0;
    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    if (fabs(d) < FPMIN) d = FPMIN;
    d = 1.0 / d;
    double h = d;
    for (int m = 1; m <= MAX_IT; ++m) {
        const int m2 = 2 * m;
        double aa = (double)m * (b - (double)m) * x /
                    ((qam + (double)m2) * (a + (double)m2));
        d = 1.0 + aa * d;
        if (fabs(d) < FPMIN) d = FPMIN;
        c = 1.0 + aa / c;
        if (fabs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + (double)m) * (qab + (double)m) * x /
              ((a + (double)m2) * (qap + (double)m2));
        d = 1.0 + aa * d;
        if (fabs(d) < FPMIN) d = FPMIN;
        c = 1.0 + aa / c;
        if (fabs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (fabs(del - 1.0) < EPS) return h;
    }
    return NAN;
}

/* Logarithms of both tails of Y ~ Beta(a, b) at y = s / (1 + s), s > 0,
 * given s and ln s. ln y = ln s - log1p(s) and ln(1 - y) = -log1p(s) are
 * taken without forming 1 - y or y (which round to 1, or underflow, at the
 * extremes), and the tail on the side of y is computed directly, the other
 * one as log1p(-exp(.)) of it (that side is at least about 0.3). Returns
 * NAN in *log_lower when the continued fraction has not converged. */
static void beta_log_tails(double a, double b, double s, double log_s,
                           double* log_lower, double* log_upper) {
    const double log_x = -log1p(s);
    const double log_y = log_s + log_x;
    const double log_front = a * log_y + b * log_x - lbeta_ref(a, b);
    const double y = s / (1.0 + s);
    if (y < (a + 1.0) / (a + b + 2.0)) {
        *log_lower = log_front + log(betacf_ref(a, b, y) / a);
        *log_upper = log1p(-fmin(exp(*log_lower), 1.0));
    } else {
        *log_upper = log_front + log(betacf_ref(b, a, 1.0 / (1.0 + s)) / b);
        *log_lower = log1p(-fmin(exp(*log_upper), 1.0));
    }
    if (isnan(*log_upper)) *log_lower = NAN;
}

/* Position of G with respect to the UCL quantile: +1 below it, 0 at or above
 * it, -1 when the tail could not be evaluated. With kappa = (n^2 - 1) / n,
 * the Hotelling limit G = r (n-1)(n+1) / (n (n-r)) F, F ~ F(r, n-r), maps to
 * Y = G / (kappa + G) ~ Beta(r/2, (n-r)/2), so s = G / kappa. For
 * alpha <= 0.5 the upper tail is compared with ln alpha; above 0.5 the lower
 * tail is compared with ln(1 - alpha), exact there: the small tail is always
 * the one inverted (alpha = 1 - 2^-30 gave a UCL 128 times too large through
 * a beta argument that rounded to 1). */
typedef struct {
    double a, b, kappa, log_kappa, log_target;
    int upper;
} ucl_target;

static int below_quantile(const ucl_target* t, double G) {
    double log_lower, log_upper;
    beta_log_tails(t->a, t->b, G / t->kappa, log(G) - t->log_kappa, &log_lower, &log_upper);
    if (isnan(log_lower)) return -1;
    return t->upper ? (log_upper > t->log_target) : (log_lower < t->log_target);
}

/* The Hotelling UCL itself, root of the tail equation in G on the normal
 * doubles [DBL_MIN, DBL_MAX]: the bracket is grown or shrunk by factors of
 * two from 1, saturating at DBL_MAX and DBL_MIN, then bisected with
 * lo + (hi - lo) / 2. N4M_ERR_NUMERICAL_FAILURE only when the UCL is not a
 * finite normal double (alpha = 1e-308, n = 3, r = 1: 1.333e308 is
 * returned; the former doubling of F overflowed first) or a tail could not be
 * evaluated. */
static n4m_status_t hotelling_ucl_ref(double alpha, int64_t n, int32_t r, double* ucl) {
    ucl_target t;
    t.a = (double)r / 2.0;
    t.b = (double)(n - r) / 2.0;
    t.kappa = (double)(n - 1) * ((double)(n + 1) / (double)n);
    t.log_kappa = log(t.kappa);
    t.upper = alpha <= 0.5;
    t.log_target = t.upper ? log(alpha) : log(1.0 - alpha);
    double lo = 1.0, hi = 1.0;
    int side = below_quantile(&t, 1.0);
    if (side < 0) return N4M_ERR_NUMERICAL_FAILURE;
    if (side == 1) {
        while (side == 1) {
            if (hi == DBL_MAX) return N4M_ERR_NUMERICAL_FAILURE;
            lo = hi;
            hi = hi > DBL_MAX / 2.0 ? DBL_MAX : 2.0 * hi;
            side = below_quantile(&t, hi);
            if (side < 0) return N4M_ERR_NUMERICAL_FAILURE;
        }
    } else {
        while (side == 0) {
            if (lo == DBL_MIN) return N4M_ERR_NUMERICAL_FAILURE;
            hi = lo;
            lo = lo / 2.0 < DBL_MIN ? DBL_MIN : lo / 2.0;
            side = below_quantile(&t, lo);
            if (side < 0) return N4M_ERR_NUMERICAL_FAILURE;
        }
    }
    for (int it = 0; it < 200 && hi - lo > 2.0 * DBL_EPSILON * hi; ++it) {
        const double mid = lo + (hi - lo) / 2.0;
        side = below_quantile(&t, mid);
        if (side < 0) return N4M_ERR_NUMERICAL_FAILURE;
        if (side == 1) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    *ucl = lo + (hi - lo) / 2.0;
    return N4M_OK;
}

n4m_status_t n4m_util_hotelling_t2_impl(const double* X,
                                         int64_t rows, int64_t cols,
                                         int32_t n_components,
                                         double alpha,
                                         double* t2_per_sample,
                                         double* ucl_out,
                                         int32_t* rank_used) {
    if (X == NULL || t2_per_sample == NULL || ucl_out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 2 || cols < 1 || n_components < 1) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (n_components > cols || n_components >= rows) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (!(alpha > 0.0 && alpha < 1.0)) {
        return N4M_ERR_INVALID_ARGUMENT;
    }

    const int64_t n = rows;
    const int64_t p = cols;
    const int32_t k = n_components;

    /* The SVD below requires rows >= cols. */
    if (n < p) return N4M_ERR_INVALID_ARGUMENT;

    /* Corrected two-pass centring at one power-of-two scale
     * (column_centring.h): a constant column centres to exact zeros, and T²
     * and the rank are scale-invariant, so the scale is not needed here. */
    double* Xc = (double*)malloc((size_t)n * (size_t)p * sizeof(double));
    if (Xc == NULL) return N4M_ERR_OUT_OF_MEMORY;
    int scale_exp = 0;
    const n4m_status_t centred = n4m_util_centre_columns_pow2(X, n, p, Xc, &scale_exp);
    if (centred != N4M_OK) {
        free(Xc);
        return centred;
    }

    double* U  = (double*)malloc((size_t)n * (size_t)p * sizeof(double));
    double* S  = (double*)malloc((size_t)p * sizeof(double));
    /* Shared SVD (core/common/svd.h) requires a non-NULL Vt scratch even
     * when the caller does not need the right singular vectors. Allocate a
     * (p x p) buffer here purely as a sink. */
    double* Vt = (double*)malloc((size_t)p * (size_t)p * sizeof(double));
    if (U == NULL || S == NULL || Vt == NULL) {
        free(Xc); free(U); free(S); free(Vt);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    const n4m_status_t svd_st = n4m_svd_compact(Xc, n, p, U, S, Vt);
    if (svd_st != N4M_OK) {
        free(Xc); free(U); free(S); free(Vt);
        return svd_st;
    }

    /* Only resolved directions enter T²: singular values are accurate to
     * the absolute bound O(max(n,p) eps s_max) (svd.h), so a direction with
     * sigma_j <= 16 max(n,p) eps sigma_1 is rounding noise (e.g. an exactly
     * collinear column) and dividing by its variance would amplify it.
     * S is sorted, so the resolved directions are the leading r <= k. */
    const double tol = 16.0 * (double)(n > p ? n : p) * DBL_EPSILON * S[0];
    int32_t r = 0;
    while (r < k && S[r] > tol) ++r;

    /* T[i, j] = U[i, j] σ_j and λ_j = σ_j² / (n - 1), so T[i, j]² / λ_j =
     * (n - 1) U[i, j]²: computed without σ_j², which overflows or underflows
     * for extreme inputs (T² was NaN with status OK for |x| ~ 2^600). */
    const double nm1 = (double)(n - 1);
    for (int64_t i = 0; i < n; ++i) {
        double t2 = 0.0;
        for (int32_t j = 0; j < r; ++j) {
            const double u = U[(size_t)i * (size_t)p + (size_t)j];
            t2 += u * u;
        }
        t2_per_sample[i] = nm1 * t2;
    }

    /* UCL = r (n - 1)(n + 1) / (n (n - r)) * F^{-1}(1 - alpha; r, n - r),
     * with r the rank actually used (r = k for a full-rank request), solved
     * directly in the UCL (hotelling_ucl_ref). No resolved direction
     * (constant X): T² = 0 and UCL = 0. */
    double ucl = 0.0;
    if (r > 0) {
        const n4m_status_t ucl_st = hotelling_ucl_ref(alpha, n, r, &ucl);
        if (ucl_st != N4M_OK) {
            free(Xc); free(U); free(S); free(Vt);
            return ucl_st;
        }
    }
    *ucl_out = ucl;
    if (rank_used != NULL) *rank_used = r;

    free(Xc);
    free(U);
    free(S);
    free(Vt);
    return N4M_OK;
}
