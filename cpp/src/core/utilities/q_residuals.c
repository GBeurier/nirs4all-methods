/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Q-residuals (squared prediction error) on PCA-reduced X.
 *
 * The Jackson-Mudholkar UCL needs the upper standard-normal quantile at
 * alpha. We use Acklam's rational approximation (relative error about
 * 1.15e-9), more than sufficient for a limit that is itself an
 * approximation.
 */

#include "q_residuals.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "column_centring.h"
#include "core/common/svd.h"

/* ------------------------------------------------------------------------- */
/* Upper standard-normal quantile z with P(Z > z) = alpha — Acklam's rational */
/* approximation (relative error about 1.15e-9). The tails take alpha and    */
/* 1 - alpha directly, so a small alpha is not rounded through 1 - alpha.    */
/* ------------------------------------------------------------------------- */
static double inv_norm_upper_ref(double alpha) {
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
    const double p_low = 0.02425;
    double q, r;
    if (alpha < p_low) {
        q = sqrt(-2.0 * log(alpha));
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    if (alpha <= 1.0 - p_low) {
        q = 0.5 - alpha;
        r = q * q;
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    }
    q = sqrt(-2.0 * log(1.0 - alpha));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
}

/* 2^e * v for a value of the scaled space: N4M_ERR_NUMERICAL_FAILURE when a
 * significant value (v > noise) does not map to a finite normal double. */
static n4m_status_t unscale(double v, int e, double noise, double* out) {
    *out = ldexp(v, e);
    if (v > noise && !(isfinite(*out) && *out >= DBL_MIN)) return N4M_ERR_NUMERICAL_FAILURE;
    return N4M_OK;
}

n4m_status_t n4m_util_q_residuals_impl(const double* X,
                                        int64_t rows, int64_t cols,
                                        int32_t n_components,
                                        double alpha,
                                        double* q_per_sample,
                                        double* ucl_out) {
    if (X == NULL || q_per_sample == NULL || ucl_out == NULL) {
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
    const int64_t compact = n < p ? n : p;

    /* Everything below runs on Xc = (X - mean) 2^-e, max|Xc| in [0.5, 1)
     * (corrected two-pass centring shared with Hotelling T²: a constant
     * column centres to exact zeros). Q and the UCL are homogeneous of
     * degree 2 in X, so they are published as 2^(2e) times their scaled
     * values: the eigenvalue moments theta_a no longer overflow or underflow
     * (X = 2^200 [[2,1],[2,-1],[-2,1],[-2,-1]], k = 1 used to give UCL 0
     * instead of 1.29e121 with status OK), and the results are exactly
     * equivariant under power-of-two scaling. */
    double* Xc = (double*)malloc((size_t)n * (size_t)p * sizeof(double));
    if (Xc == NULL) return N4M_ERR_OUT_OF_MEMORY;
    int e = 0;
    const n4m_status_t centred = n4m_util_centre_columns_pow2(X, n, p, Xc, &e);
    if (centred != N4M_OK) {
        free(Xc);
        return centred;
    }

    double* U  = (double*)malloc((size_t)n * (size_t)compact * sizeof(double));
    double* S  = (double*)malloc((size_t)compact * sizeof(double));
    /* The shared SVD returns Vt (min(n,p) x p) row-major. */
    double* Vt = (double*)malloc((size_t)compact * (size_t)p * sizeof(double));
    /* n4m_svd_compact consumes (overwrites) its input matrix. Pass a copy
     * so the original centred Xc survives for the residual computation. */
    double* Xc_svd = (double*)malloc((size_t)n * (size_t)p * sizeof(double));
    if (U == NULL || S == NULL || Vt == NULL || Xc_svd == NULL) {
        free(Xc); free(U); free(S); free(Vt); free(Xc_svd);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    memcpy(Xc_svd, Xc, (size_t)n * (size_t)p * sizeof(double));
    const n4m_status_t svd_st = n4m_svd_compact(Xc_svd, n, p, U, S, Vt);
    free(Xc_svd);
    if (svd_st != N4M_OK) {
        free(Xc); free(U); free(S); free(Vt);
        return svd_st;
    }

    /* Singular values are accurate to the absolute bound O(max(n,p) eps
     * sigma_1) (svd.h); tol is the operational threshold also used by
     * Hotelling T². When no residual direction is above it (k = p, a
     * constant X, or an X of numerical rank <= k), the reconstructed residual
     * is rounding noise: Q = 0 and UCL = 0 exactly, the documented
     * "residual subspace unresolved" state (e.g. ten rows of the constant
     * 0.1 used to give Q = 1.9e-65 everywhere next to UCL 0). Otherwise the
     * residual is resolved and the UCL is a positive normal double. */
    const double tol = 16.0 * (double)(n > p ? n : p) * DBL_EPSILON * S[0];
    if (!(k < compact && S[k] > tol)) {
        for (int64_t i = 0; i < n; ++i) q_per_sample[i] = 0.0;
        *ucl_out = 0.0;
        free(Xc); free(U); free(S); free(Vt);
        return N4M_OK;
    }

    /* Reconstruct Xc_hat = sum_{j<k} U[:, j] * Σ[j] * Vt[j, :]. */
    double* Xc_hat = (double*)calloc((size_t)n * (size_t)p, sizeof(double));
    if (Xc_hat == NULL) {
        free(Xc); free(U); free(S); free(Vt);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    for (int32_t j = 0; j < k; ++j) {
        const double sigma_j = S[j];
        if (sigma_j == 0.0) continue;
        for (int64_t i = 0; i < n; ++i) {
            const double uij = U[(size_t)i * (size_t)compact + (size_t)j];
            const double sij = uij * sigma_j;
            for (int64_t d = 0; d < p; ++d) {
                /* Vt is (compact x p) row-major: Vt[j, d] is the d-th coord of
                 * the j-th right singular vector. */
                Xc_hat[(size_t)i * (size_t)p + (size_t)d] +=
                    sij * Vt[(size_t)j * (size_t)p + (size_t)d];
            }
        }
    }

    /* Q_i at or below tol^2 (scaled space) is rounding noise: it may round
     * towards 0 when unscaled; a larger one that is not a normal double is
     * refused. */
    n4m_status_t st = N4M_OK;
    for (int64_t i = 0; i < n && st == N4M_OK; ++i) {
        double q = 0.0;
        for (int64_t d = 0; d < p; ++d) {
            const double r = Xc[(size_t)i * (size_t)p + (size_t)d] -
                             Xc_hat[(size_t)i * (size_t)p + (size_t)d];
            q += r * r;
        }
        st = unscale(q, 2 * e, tol * tol, &q_per_sample[i]);
    }

    /* Jackson-Mudholkar UCL from the residual eigenvalues, in the scaled
     * space: sigma_1 <= sqrt(n p), and sigma_k > tol >= 16 eps because
     * sigma_1 >= max|Xc| >= 0.5, so theta_1..3 and theta_2^2 are normal
     * doubles. Y = (Q / theta_1)^h0 is approximately normal with mean
     * 1 + h0 (h0 - 1) B and standard deviation |h0| sqrt(2 B), B = theta_2 /
     * theta_1^2. Y increases with Q when h0 > 0 and decreases when h0 < 0, so
     * the upper alpha-quantile of Q maps to the upper quantile of Y for h0 > 0
     * and to the lower one for h0 < 0; both are 1 + g with
     *     g = h0 C,  C = A + (h0 - 1) B,  A = ca sqrt(2 theta_2) / theta_1
     * (Jackson & Mudholkar 1979; Nomikos & MacGregor 1995, appendix B). The
     * former |h0| A kept the upper quantile of Y for h0 < 0 (a 14-column
     * Hadamard example with h0 = -0.0676 gave 0.365 instead of 5.229).
     * base^(1/h0) = exp(log1p(h0 C) / h0) tends to exp(C) = exp(A - B) as h0
     * tends to 0 from either side (h0 = 0 takes that limit).
     * The base 1 + g is positive whenever h0 ca >= 0 (for h0 > 0, h0 <= 1/3
     * and B <= 1 give g >= h0 (h0 - 1) B >= -2/9; for h0 < 0 both terms are
     * >= 0) and whenever h0 < 0 with ca <= sqrt(2), i.e. alpha >=
     * 1 - Phi(sqrt(2)) ~ 0.0786 (1 + g >= (1 - |h0| sqrt(B))^2 + |h0| B).
     * It can be non-positive only for h0 > 0 with alpha > 0.5, or h0 < 0
     * with alpha < 0.0786 (e.g. residual eigenvalues 3.25^2 and 22 times 1,
     * h0 = -0.46, at alpha = 1e-6): the approximating normal then puts more
     * than alpha below Y = 0 (Q beyond every finite value), the approximation
     * has no alpha-quantile, and the result is an
     * explicit N4M_ERR_NUMERICAL_FAILURE. */
    if (st == N4M_OK) {
        const double inv_nm1 = 1.0 / (double)(n - 1);
        double theta1 = 0.0, theta2 = 0.0, theta3 = 0.0;
        for (int64_t j = (int64_t)k; j < compact; ++j) {
            const double lam = S[j] * S[j] * inv_nm1;
            theta1 += lam;
            theta2 += lam * lam;
            theta3 += lam * lam * lam;
        }
        const double h0 = 1.0 - (2.0 * theta1 * theta3) / (3.0 * theta2 * theta2);
        const double A = inv_norm_upper_ref(alpha) * sqrt(2.0 * theta2) / theta1;
        const double B = theta2 / (theta1 * theta1);
        const double C = A + (h0 - 1.0) * B;
        double ucl_scaled = 0.0;
        if (h0 == 0.0) {
            ucl_scaled = theta1 * exp(C);
        } else if (h0 * C > -1.0) {
            ucl_scaled = theta1 * exp(log1p(h0 * C) / h0);
        }
        st = (ucl_scaled > 0.0) ? unscale(ucl_scaled, 2 * e, 0.0, ucl_out)
                                : N4M_ERR_NUMERICAL_FAILURE;
    }

    free(Xc);
    free(U);
    free(S);
    free(Vt);
    free(Xc_hat);
    return st;
}
