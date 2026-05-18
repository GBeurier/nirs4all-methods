/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Q-residuals (squared prediction error) on PCA-reduced X.
 *
 * The Jackson-Mudholkar UCL needs the inverse standard normal at (1 - alpha).
 * We use the Beasley-Springer-Moro rational approximation (Wichura 1988
 * AS241 style coefficients) which is accurate to about 7 significant digits
 * over the central region — more than sufficient for the residual-statistic
 * limit which is itself an approximation.
 */

#include "q_residuals.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "core/common/svd.h"

/* ------------------------------------------------------------------------- */
/* Inverse standard normal — Wichura AS241 (compact form).                    */
/* ------------------------------------------------------------------------- */
static double inv_norm_ref(double p) {
    static const double a[6] = {
        -3.969683028665376e+01,  2.209460984245205e+02,
        -2.759285104469687e+02,  1.383577518672690e+02,
        -3.066479806614716e+01,  2.506628277459239e+00
    };
    static const double b[5] = {
        -5.447609879822406e+01,  1.615858368580409e+02,
        -1.556989798598866e+02,  6.680131188771972e+01,
        -1.328068155288572e+01
    };
    static const double c[6] = {
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00,  2.938163982698783e+00
    };
    static const double d[4] = {
         7.784695709041462e-03,  3.224671290700398e-01,
         2.445134137142996e+00,  3.754408661907416e+00
    };
    const double p_low  = 0.02425;
    const double p_high = 1.0 - p_low;
    double q, r, x;
    if (p < p_low) {
        q = sqrt(-2.0 * log(p));
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    } else if (p <= p_high) {
        q = p - 0.5;
        r = q * q;
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    } else {
        q = sqrt(-2.0 * log(1.0 - p));
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
             ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    return x;
}

c4a_status_t c4a_util_q_residuals_impl(const double* X,
                                        int64_t rows, int64_t cols,
                                        int32_t n_components,
                                        double alpha,
                                        double* q_per_sample,
                                        double* ucl_out) {
    if (X == NULL || q_per_sample == NULL || ucl_out == NULL) {
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

    if (n < p) {
        free(Xc);
        return C4A_ERR_INVALID_ARGUMENT;
    }
    double* U  = (double*)malloc((size_t)n * (size_t)p * sizeof(double));
    double* S  = (double*)malloc((size_t)p * sizeof(double));
    /* The shared SVD returns Vt (k x n) row-major where Vt[i, j] is the
     * j-th component of the i-th right singular vector. */
    double* Vt = (double*)malloc((size_t)p * (size_t)p * sizeof(double));
    /* c4a_svd_compact consumes (overwrites) its input matrix. Pass a copy
     * so the original centred Xc survives for the residual computation. */
    double* Xc_svd = (double*)malloc((size_t)n * (size_t)p * sizeof(double));
    if (U == NULL || S == NULL || Vt == NULL || Xc_svd == NULL) {
        free(Xc); free(U); free(S); free(Vt); free(Xc_svd);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    memcpy(Xc_svd, Xc, (size_t)n * (size_t)p * sizeof(double));
    const c4a_status_t svd_st = c4a_svd_compact(Xc_svd, n, p, U, S, Vt);
    free(Xc_svd);
    if (svd_st != C4A_OK) {
        free(Xc); free(U); free(S); free(Vt);
        return svd_st;
    }

    /* Reconstruct Xc_hat = sum_{j<k} U[:, j] * Σ[j] * Vt[j, :]. */
    double* Xc_hat = (double*)calloc((size_t)n * (size_t)p, sizeof(double));
    if (Xc_hat == NULL) {
        free(Xc); free(U); free(S); free(Vt);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    for (int32_t j = 0; j < k; ++j) {
        const double sigma_j = S[j];
        if (sigma_j == 0.0) continue;
        for (int64_t i = 0; i < n; ++i) {
            const double uij = U[(size_t)i * (size_t)p + (size_t)j];
            const double sij = uij * sigma_j;
            for (int64_t d = 0; d < p; ++d) {
                /* Vt is (p x p) row-major: Vt[j, d] is the d-th coord of
                 * the j-th right singular vector. */
                Xc_hat[(size_t)i * (size_t)p + (size_t)d] +=
                    sij * Vt[(size_t)j * (size_t)p + (size_t)d];
            }
        }
    }

    for (int64_t i = 0; i < n; ++i) {
        double q = 0.0;
        for (int64_t d = 0; d < p; ++d) {
            const double r = Xc[(size_t)i * (size_t)p + (size_t)d] -
                             Xc_hat[(size_t)i * (size_t)p + (size_t)d];
            q += r * r;
        }
        q_per_sample[i] = q;
    }

    /* Jackson-Mudholkar UCL using residual eigenvalues. */
    const double inv_nm1 = 1.0 / (double)(n - 1);
    const int64_t rank_cap = (n < p) ? n : p;
    double theta1 = 0.0, theta2 = 0.0, theta3 = 0.0;
    for (int64_t j = (int64_t)k; j < rank_cap; ++j) {
        const double lam = S[j] * S[j] * inv_nm1;
        theta1 += lam;
        theta2 += lam * lam;
        theta3 += lam * lam * lam;
    }
    double ucl = 0.0;
    if (theta1 > 0.0 && theta2 > 0.0) {
        const double h0 = 1.0 - (2.0 * theta1 * theta3) / (3.0 * theta2 * theta2);
        const double ca = inv_norm_ref(1.0 - alpha);
        const double term1 = ca * sqrt(2.0 * theta2 * h0 * h0) / theta1;
        const double term2 = (theta2 * h0 * (h0 - 1.0)) / (theta1 * theta1);
        const double base  = 1.0 + term1 + term2;
        if (base > 0.0 && h0 != 0.0) {
            ucl = theta1 * pow(base, 1.0 / h0);
        }
    }
    *ucl_out = ucl;

    free(Xc);
    free(U);
    free(S);
    free(Vt);
    free(Xc_hat);
    return C4A_OK;
}
