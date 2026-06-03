/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Shared dense linear algebra helpers (Householder QR + companions).
 *
 * Bit-exact replacement for the in-line QR previously duplicated in
 * emsc.c, savitzky_golay.c (main coef build) and savitzky_golay.c
 * (sg_fit_edge_left / sg_fit_edge_right). Same algorithm, same
 * arithmetic order — kept identical so the 3 QR copies elimination
 * doesn't shift any fixture byte.
 */

#include "linalg.h"

#include <math.h>
#include <stddef.h>

n4m_status_t n4m_householder_qr(double* A, int64_t rows, int64_t cols,
                                double* tau) {
    if (A == NULL || tau == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 1 || cols < 1 || rows < cols) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    for (int64_t k = 0; k < cols; ++k) {
        /* Compute ||A[k:rows, k]||. */
        double sigma = 0.0;
        for (int64_t i = k; i < rows; ++i) {
            const double v = A[(size_t)i * (size_t)cols + (size_t)k];
            sigma += v * v;
        }
        sigma = sqrt(sigma);
        const double Akk = A[(size_t)k * (size_t)cols + (size_t)k];
        const double alpha = (Akk >= 0.0) ? -sigma : sigma;
        if (sigma == 0.0) {
            tau[k] = 0.0;
            continue;
        }
        const double vk0 = Akk - alpha;
        /* ||v||^2 = vk0^2 + (sigma^2 - Akk^2). The parenthesised term loses
         * precision via catastrophic cancellation when the column is near
         * axis-aligned (Akk ~= +/-sigma). Use the algebraically-equal,
         * cancellation-free form:
         *   ||v||^2 = (Akk - alpha)^2 + sigma^2 - Akk^2
         *           = alpha^2 + sigma^2 - 2*Akk*alpha   (alpha^2 = sigma^2)
         *           = 2*sigma^2 - 2*Akk*alpha
         *           = -2*alpha*(Akk - alpha) = -2*alpha*vk0.
         * Because alpha = -sign(Akk)*sigma, vk0 = Akk - alpha is a sum of
         * same-sign terms (no cancellation), and -2*alpha*vk0 stays exact. */
        const double v_norm_sq = -2.0 * alpha * vk0;
        const double t = 2.0 / v_norm_sq * vk0 * vk0;
        const double inv_vk0 = 1.0 / vk0;
        A[(size_t)k * (size_t)cols + (size_t)k] = alpha;
        for (int64_t i = k + 1; i < rows; ++i) {
            A[(size_t)i * (size_t)cols + (size_t)k] *= inv_vk0;
        }
        tau[k] = t;
        for (int64_t j = k + 1; j < cols; ++j) {
            double dot = A[(size_t)k * (size_t)cols + (size_t)j];
            for (int64_t i = k + 1; i < rows; ++i) {
                dot += A[(size_t)i * (size_t)cols + (size_t)k]
                     * A[(size_t)i * (size_t)cols + (size_t)j];
            }
            const double coef = t * dot;
            A[(size_t)k * (size_t)cols + (size_t)j] -= coef;
            for (int64_t i = k + 1; i < rows; ++i) {
                A[(size_t)i * (size_t)cols + (size_t)j] -=
                    coef * A[(size_t)i * (size_t)cols + (size_t)k];
            }
        }
    }
    return N4M_OK;
}

n4m_status_t n4m_apply_qt(const double* A, int64_t rows, int64_t cols,
                          const double* tau, double* b) {
    if (A == NULL || tau == NULL || b == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 1 || cols < 1 || rows < cols) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    for (int64_t k = 0; k < cols; ++k) {
        if (tau[k] == 0.0) continue;
        double dot = b[k];
        for (int64_t i = k + 1; i < rows; ++i) {
            dot += A[(size_t)i * (size_t)cols + (size_t)k] * b[i];
        }
        const double coef = tau[k] * dot;
        b[k] -= coef;
        for (int64_t i = k + 1; i < rows; ++i) {
            b[i] -= coef * A[(size_t)i * (size_t)cols + (size_t)k];
        }
    }
    return N4M_OK;
}

n4m_status_t n4m_back_solve_R(const double* A, int64_t rows, int64_t cols,
                              const double* y, double* x) {
    (void)rows;  /* only the storage stride matters; we read the upper block. */
    if (A == NULL || y == NULL || x == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 1 || cols < 1 || rows < cols) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    /* Reject a rank-deficient R relative to its largest diagonal: a diagonal
     * with |R_ii| <= rtol * max_j|R_jj| means R is (numerically) singular, so
     * the back-substitution would amplify it into a garbage huge solution.
     * Flag it as a numerical failure instead. rtol is a few ulps scaled by the
     * problem dimension — tight enough that well-conditioned solves are
     * unaffected, loose enough to catch exact / near-exact rank deficiency
     * (e.g. two identical design columns -> a zero pivot up to rounding). */
    double max_abs_diag = 0.0;
    for (int64_t i = 0; i < cols; ++i) {
        const double d = A[(size_t)i * (size_t)cols + (size_t)i];
        const double ad = (d < 0.0) ? -d : d;
        if (ad > max_abs_diag) {
            max_abs_diag = ad;
        }
    }
    const double rtol = 2.220446049250313e-16 * (double)cols * max_abs_diag;
    for (int64_t i = cols - 1; i >= 0; --i) {
        double s = y[i];
        for (int64_t j = i + 1; j < cols; ++j) {
            s -= A[(size_t)i * (size_t)cols + (size_t)j] * x[j];
        }
        const double diag = A[(size_t)i * (size_t)cols + (size_t)i];
        const double abs_diag = (diag < 0.0) ? -diag : diag;
        if (abs_diag <= rtol) {
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        x[i] = s / diag;
    }
    return N4M_OK;
}
