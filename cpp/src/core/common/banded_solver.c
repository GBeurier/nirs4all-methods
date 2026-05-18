/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Pentadiagonal symmetric LDLT solver — see banded_solver.h for the
 * storage convention and contract.
 *
 * LDLT factorisation of a symmetric pentadiagonal A:
 *
 *   D[k] = A[k, k]
 *        - L[k, k-1]^2  * D[k-1]      if k >= 1
 *        - L[k, k-2]^2  * D[k-2]      if k >= 2
 *
 *   L[k+1, k] = (A[k+1, k] - L[k+1, k-1] * D[k-1] * L[k, k-1]) / D[k]
 *               where the inner term is present only for k >= 1.
 *
 *   L[k+2, k] = A[k+2, k] / D[k]
 *               (no inner sum: |k+2 - j| > 2 for j < k.)
 *
 * Storage:
 *   L[k * 2 + 0] = L[k+1, k]   (i.e. the *first* sub-diagonal element
 *                                created by step k of the factorisation)
 *   L[k * 2 + 1] = L[k+2, k]   (the *second* sub-diagonal element)
 *
 *   L[(n-2) * 2 + 1] and L[(n-1) * 2 + {0,1}] are unused (no k+1 or k+2
 *   to write to). We leave them as zeros.
 *
 * Solve L D L^T x = b in three passes:
 *   1. Forward L y = b:
 *        y[i] = b[i] - L[i, i-1] * y[i-1] - L[i, i-2] * y[i-2]
 *             = b[i] - L[(i-1) * 2 + 0] * y[i-1]
 *                    - L[(i-2) * 2 + 1] * y[i-2]
 *   2. Diagonal z = D^{-1} y.
 *   3. Backward L^T x = z:
 *        x[i] = z[i] - L[i+1, i] * x[i+1] - L[i+2, i] * x[i+2]
 *             = z[i] - L[i * 2 + 0] * x[i+1]
 *                    - L[i * 2 + 1] * x[i+2]
 *
 * Zero-pivot semantics: a zero D[k] returns C4A_ERR_NUMERICAL_FAILURE.
 * Near-zero pivots (D[k] of order DBL_EPSILON * |A[k,k]|) silently
 * propagate as fp arithmetic — the AsLS / AirPLS / ArPLS weights stay
 * strictly positive in practice, so this case does not arise.
 */

#include "banded_solver.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

c4a_status_t c4a_banded5_factor(const double* main_diag,
                                const double* super1, const double* super2,
                                int64_t n, c4a_banded5_t* out) {
    if (main_diag == NULL || super1 == NULL || super2 == NULL || out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (n < 1) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    out->n = 0;
    out->L = NULL;
    out->D = NULL;

    double* D = (double*)malloc((size_t)n * sizeof(double));
    double* L = (double*)calloc((size_t)n * (size_t)2, sizeof(double));
    if (D == NULL || L == NULL) {
        free(D);
        free(L);
        return C4A_ERR_OUT_OF_MEMORY;
    }

    /* k = 0: no prior pivots. */
    D[0] = main_diag[0];
    if (D[0] == 0.0) {
        free(D);
        free(L);
        return C4A_ERR_NUMERICAL_FAILURE;
    }
    if (n >= 2) {
        L[0 * 2 + 0] = super1[0] / D[0];                 /* L[1, 0] */
    }
    if (n >= 3) {
        L[0 * 2 + 1] = super2[0] / D[0];                 /* L[2, 0] */
    }

    /* k = 1: contribution from D[0] only. */
    if (n >= 2) {
        const double L10 = L[0 * 2 + 0];
        D[1] = main_diag[1] - L10 * L10 * D[0];
        if (D[1] == 0.0) {
            free(D);
            free(L);
            return C4A_ERR_NUMERICAL_FAILURE;
        }
        if (n >= 3) {
            /* L[2, 1] = (A[2, 1] - L[2, 0] * D[0] * L[1, 0]) / D[1] */
            const double L20 = L[0 * 2 + 1];
            L[1 * 2 + 0] = (super1[1] - L20 * D[0] * L10) / D[1];
        }
        if (n >= 4) {
            /* L[3, 1] = A[3, 1] / D[1] = super2[1] / D[1] */
            L[1 * 2 + 1] = super2[1] / D[1];
        }
    }

    /* k >= 2: both D[k-1] and D[k-2] contribute. */
    for (int64_t k = 2; k < n; ++k) {
        const double Lk_k1 = L[(k - 1) * 2 + 0];  /* L[k, k-1] */
        const double Lk_k2 = L[(k - 2) * 2 + 1];  /* L[k, k-2] */
        D[k] = main_diag[k]
             - Lk_k1 * Lk_k1 * D[k - 1]
             - Lk_k2 * Lk_k2 * D[k - 2];
        if (D[k] == 0.0) {
            free(D);
            free(L);
            return C4A_ERR_NUMERICAL_FAILURE;
        }
        if (k + 1 < n) {
            /* L[k+1, k] = (A[k+1, k] - L[k+1, k-1] * D[k-1] * L[k, k-1]) / D[k]
             * L[k+1, k-1] = L[(k-1) * 2 + 1]; L[k, k-1] = Lk_k1 (= L[(k-1)*2+0]) */
            const double Lkp1_km1 = L[(k - 1) * 2 + 1];
            L[k * 2 + 0] = (super1[k] - Lkp1_km1 * D[k - 1] * Lk_k1) / D[k];
        }
        if (k + 2 < n) {
            /* L[k+2, k] = A[k+2, k] / D[k] = super2[k] / D[k] */
            L[k * 2 + 1] = super2[k] / D[k];
        }
    }

    out->n = n;
    out->L = L;
    out->D = D;
    return C4A_OK;
}

c4a_status_t c4a_banded5_solve(const c4a_banded5_t* fact,
                                const double* b, double* x) {
    if (fact == NULL || b == NULL || x == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (fact->n < 1 || fact->L == NULL || fact->D == NULL) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    const int64_t n = fact->n;
    const double* L = fact->L;
    const double* D = fact->D;

    /* 1. Forward solve L y = b. We reuse x as the y-buffer (the back-solve
     *    overwrites it anyway).
     *    y[0] = b[0]
     *    y[1] = b[1] - L[1, 0] * y[0]
     *    y[i] = b[i] - L[i, i-1] * y[i-1] - L[i, i-2] * y[i-2]   (i >= 2) */
    x[0] = b[0];
    if (n >= 2) {
        x[1] = b[1] - L[0 * 2 + 0] * x[0];
    }
    for (int64_t i = 2; i < n; ++i) {
        x[i] = b[i]
             - L[(i - 1) * 2 + 0] * x[i - 1]
             - L[(i - 2) * 2 + 1] * x[i - 2];
    }

    /* 2. Diagonal solve: x[i] /= D[i]. */
    for (int64_t i = 0; i < n; ++i) {
        x[i] = x[i] / D[i];
    }

    /* 3. Back-solve L^T x = z.
     *    x[n-1] = z[n-1]
     *    x[n-2] = z[n-2] - L[n-1, n-2] * x[n-1]
     *    x[i]   = z[i]   - L[i+1, i] * x[i+1] - L[i+2, i] * x[i+2]   (i <= n-3) */
    if (n >= 2) {
        x[n - 2] = x[n - 2] - L[(n - 2) * 2 + 0] * x[n - 1];
    }
    for (int64_t i = n - 3; i >= 0; --i) {
        x[i] = x[i]
             - L[i * 2 + 0] * x[i + 1]
             - L[i * 2 + 1] * x[i + 2];
    }
    return C4A_OK;
}

void c4a_banded5_free(c4a_banded5_t* fact) {
    if (fact == NULL) return;
    free(fact->L);
    free(fact->D);
    fact->L = NULL;
    fact->D = NULL;
    fact->n = 0;
}
