/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Hotelling T² reference implementation.
 *
 * The PCA step uses the shared Jacobi SVD (`core/common/svd.{c,h}`). The
 * F-distribution upper-tail inversion uses the standard "lower regularised
 * incomplete beta" relation:
 *
 *     P(F_{d1, d2} <= F0) = I_{d1 F0 / (d1 F0 + d2)}(d1 / 2, d2 / 2).
 *
 * We compute the regularised incomplete beta via the continued-fraction
 * expansion (Numerical Recipes §6.4) and invert it by bisection. The
 * F-quantile is then the F0 that maps to the target CDF.
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
/* Log-gamma — Lanczos approximation (single-coefficient set).               */
/* ------------------------------------------------------------------------- */
static double lngamma_ref(double x) {
    static const double cof[6] = {
        76.18009172947146,    -86.50532032941677,
        24.01409824083091,    -1.231739572450155,
        0.1208650973866179e-2,-0.5395239384953e-5
    };
    double y = x;
    double tmp = x + 5.5;
    tmp -= (x + 0.5) * log(tmp);
    double ser = 1.000000000190015;
    for (int j = 0; j < 6; ++j) {
        y += 1.0;
        ser += cof[j] / y;
    }
    return -tmp + log(2.5066282746310005 * ser / x);
}

/* ------------------------------------------------------------------------- */
/* Regularised incomplete beta function I_x(a, b).                            */
/* ------------------------------------------------------------------------- */
static double betacf_ref(double a, double b, double x) {
    const int    MAX_IT = 200;
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
        if (fabs(del - 1.0) < EPS) break;
    }
    return h;
}

static double betai_ref(double a, double b, double x) {
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;
    const double bt = exp(lngamma_ref(a + b) - lngamma_ref(a) -
                           lngamma_ref(b) + a * log(x) + b * log(1.0 - x));
    if (x < (a + 1.0) / (a + b + 2.0)) {
        return bt * betacf_ref(a, b, x) / a;
    } else {
        return 1.0 - bt * betacf_ref(b, a, 1.0 - x) / b;
    }
}

/* Upper tail P(F_{d1,d2} > F0) = I_{d2 / (d2 + d1 F0)}(d2 / 2, d1 / 2). */
static double f_upper_tail_ref(double F0, double d1, double d2) {
    return betai_ref(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * F0));
}

/* F0 with P(F_{d1,d2} > F0) = alpha, 0 < alpha < 1, by bisection on a bracket
 * grown or shrunk by factors of two (no fixed cap: the former [1e-10, 1e12]
 * bracket silently returned about 1.1e12 for larger quantiles, e.g. d1 = 1,
 * d2 = 2, alpha = 1e-10, whose quantile is about 1e20). Working in the upper
 * tail keeps a small alpha exact, where 1 - alpha rounds to 1 below 2^-53.
 * Returns 0 when the quantile is not a finite normal double. */
static double f_quantile_upper_ref(double alpha, double d1, double d2) {
    double lo = 1.0, hi = 1.0;
    if (f_upper_tail_ref(1.0, d1, d2) > alpha) {
        while (f_upper_tail_ref(hi, d1, d2) > alpha) {
            lo = hi;
            hi *= 2.0;
            if (!isfinite(hi)) return 0.0;
        }
    } else {
        while (f_upper_tail_ref(lo, d1, d2) <= alpha) {
            hi = lo;
            lo *= 0.5;
            if (lo < DBL_MIN) return 0.0;
        }
    }
    for (int it = 0; it < 200 && hi - lo > 4.0 * DBL_EPSILON * hi; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (f_upper_tail_ref(mid, d1, d2) > alpha) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return 0.5 * (lo + hi);
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
     * with r the rank actually used (r = k for a full-rank request). No
     * resolved direction (constant X): T² = 0 and UCL = 0. A quantile or UCL
     * that is not a finite normal double (only for extreme alpha) is an
     * explicit N4M_ERR_NUMERICAL_FAILURE, never Inf or a capped value. */
    double ucl = 0.0;
    if (r > 0) {
        const double d1 = (double)r;
        const double d2 = (double)(n - r);
        const double Fq = f_quantile_upper_ref(alpha, d1, d2);
        ucl = (double)r * (double)(n - 1) * (double)(n + 1) /
              ((double)n * (double)(n - r)) * Fq;
        if (!(isfinite(ucl) && ucl >= DBL_MIN)) {
            free(Xc); free(U); free(S); free(Vt);
            return N4M_ERR_NUMERICAL_FAILURE;
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
