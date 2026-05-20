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

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

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

/* F-cdf at F0 with d1, d2 dof. */
static double f_cdf_ref(double F0, double d1, double d2) {
    if (F0 <= 0.0) return 0.0;
    const double x = (d1 * F0) / (d1 * F0 + d2);
    return betai_ref(d1 / 2.0, d2 / 2.0, x);
}

/* Invert F-cdf at probability p (0 < p < 1) via bisection. */
static double f_quantile_ref(double p, double d1, double d2) {
    if (p <= 0.0) return 0.0;
    if (p >= 1.0) return HUGE_VAL;
    /* Bracket: F_quantile grows with p; pick large upper bound. */
    double lo = 1e-10;
    double hi = 1.0;
    while (f_cdf_ref(hi, d1, d2) < p) {
        hi *= 2.0;
        if (hi > 1e12) break;
    }
    /* Bisection. */
    for (int it = 0; it < 200; ++it) {
        const double mid = 0.5 * (lo + hi);
        const double cdf = f_cdf_ref(mid, d1, d2);
        if (cdf < p) {
            lo = mid;
        } else {
            hi = mid;
        }
        if ((hi - lo) <= 1e-12 * (1.0 + 0.5 * (hi + lo))) break;
    }
    return 0.5 * (lo + hi);
}

c4a_status_t c4a_util_hotelling_t2_impl(const double* X,
                                         int64_t rows, int64_t cols,
                                         int32_t n_components,
                                         double alpha,
                                         double* t2_per_sample,
                                         double* ucl_out) {
    if (X == NULL || t2_per_sample == NULL || ucl_out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 2 || cols < 1 || n_components < 1) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (n_components > cols || n_components >= rows) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (!(alpha > 0.0 && alpha < 1.0)) {
        return C4A_ERR_INVALID_ARGUMENT;
    }

    const int64_t n = rows;
    const int64_t p = cols;
    const int32_t k = n_components;

    /* Centre by column means. */
    double* Xc = (double*)malloc((size_t)n * (size_t)p * sizeof(double));
    if (Xc == NULL) return C4A_ERR_OUT_OF_MEMORY;
    memcpy(Xc, X, (size_t)n * (size_t)p * sizeof(double));
    for (int64_t j = 0; j < p; ++j) {
        double s = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            s += Xc[(size_t)i * (size_t)p + (size_t)j];
        }
        const double m = s / (double)n;
        for (int64_t i = 0; i < n; ++i) {
            Xc[(size_t)i * (size_t)p + (size_t)j] -= m;
        }
    }

    /* SVD: rows >= cols required. */
    if (n < p) {
        /* The Jacobi SVD requires rows >= cols. Pad with zero rows
         * conceptually: trim to cols = min(p, n) and discard the trailing
         * zero singular values. */
        free(Xc);
        return C4A_ERR_INVALID_ARGUMENT;
    }
    double* U  = (double*)malloc((size_t)n * (size_t)p * sizeof(double));
    double* S  = (double*)malloc((size_t)p * sizeof(double));
    /* Shared SVD (core/common/svd.h) requires a non-NULL Vt scratch even
     * when the caller does not need the right singular vectors. Allocate a
     * (p x p) buffer here purely as a sink. */
    double* Vt = (double*)malloc((size_t)p * (size_t)p * sizeof(double));
    if (U == NULL || S == NULL || Vt == NULL) {
        free(Xc); free(U); free(S); free(Vt);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    const c4a_status_t svd_st = c4a_svd_compact(Xc, n, p, U, S, Vt);
    if (svd_st != C4A_OK) {
        free(Xc); free(U); free(S); free(Vt);
        return svd_st;
    }

    /* T = U * Σ; eigenvalues λ_j = σ_j² / (n - 1). */
    const double inv_nm1 = 1.0 / (double)(n - 1);
    for (int64_t i = 0; i < n; ++i) {
        double t2 = 0.0;
        for (int32_t j = 0; j < k; ++j) {
            const double sigma_j = S[j];
            const double lambda  = sigma_j * sigma_j * inv_nm1;
            if (lambda > 0.0) {
                const double t = U[(size_t)i * (size_t)p + (size_t)j] * sigma_j;
                t2 += (t * t) / lambda;
            }
        }
        t2_per_sample[i] = t2;
    }

    /* UCL = k (n - 1)(n + 1) / (n (n - k)) * F^{-1}(1 - alpha; k, n - k). */
    const double d1 = (double)k;
    const double d2 = (double)(n - k);
    const double Fq = f_quantile_ref(1.0 - alpha, d1, d2);
    *ucl_out = (double)k * (double)(n - 1) * (double)(n + 1) /
               ((double)n * (double)(n - k)) * Fq;

    free(Xc);
    free(U);
    free(S);
    free(Vt);
    return C4A_OK;
}
