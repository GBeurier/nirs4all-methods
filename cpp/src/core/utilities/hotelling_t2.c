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
 * whose small tail is inverted in logarithms from G / kappa by the shared
 * Beta root (core/common/beta_quantile.h, also used by the T2-PLS variable
 * selection limit).
 */

#include "hotelling_t2.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "column_centring.h"
#include "core/common/beta_quantile.h"
#include "core/common/svd.h"

/* The Hotelling UCL G: with kappa = (n^2 - 1) / n, G = r (n-1)(n+1) /
 * (n (n-r)) F, F ~ F(r, n-r), maps to Y = G / (kappa + G) ~
 * Beta(r/2, (n-r)/2), so G is the small-tail Beta root at scale kappa
 * (core/common/beta_quantile.h). For alpha <= 0.5 the upper tail is compared
 * with ln alpha; above 0.5 the lower tail with ln(1 - alpha), exact there
 * (alpha = 1 - 2^-30 gave a UCL 128 times too large through a beta argument
 * that rounded to 1). N4M_ERR_NUMERICAL_FAILURE only when the UCL is not a
 * finite normal double (alpha = 1e-308, n = 3, r = 1: 1.333e308 is
 * returned; the former doubling of F overflowed first) or a tail could not
 * be evaluated. */
static n4m_status_t hotelling_ucl_ref(double alpha, int64_t n, int32_t r, double* ucl) {
    const double kappa = (double)(n - 1) * ((double)(n + 1) / (double)n);
    const n4m_beta_root_status_t st = n4m_util_beta_small_tail_root(
        alpha, (double)r / 2.0, (double)(n - r) / 2.0, kappa, ucl);
    return st == N4M_BETA_ROOT_OK ? N4M_OK : N4M_ERR_NUMERICAL_FAILURE;
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

    /* Compact SVD supports both orientations; its inner dimension is min(n,p). */
    const int64_t compact = n < p ? n : p;

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

    double* U  = (double*)malloc((size_t)n * (size_t)compact * sizeof(double));
    double* S  = (double*)malloc((size_t)compact * sizeof(double));
    /* Shared SVD requires the compact right-singular-vector buffer even
     * when T² only needs the left singular vectors. */
    double* Vt = (double*)malloc((size_t)compact * (size_t)p * sizeof(double));
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
            const double u = U[(size_t)i * (size_t)compact + (size_t)j];
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
