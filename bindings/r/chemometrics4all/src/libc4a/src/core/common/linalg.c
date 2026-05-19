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

c4a_status_t c4a_householder_qr(double* A, int64_t rows, int64_t cols,
                                double* tau) {
    if (A == NULL || tau == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 1 || cols < 1 || rows < cols) {
        return C4A_ERR_INVALID_ARGUMENT;
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
        const double v_norm_sq = vk0 * vk0 + (sigma * sigma - Akk * Akk);
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
    return C4A_OK;
}

c4a_status_t c4a_apply_qt(const double* A, int64_t rows, int64_t cols,
                          const double* tau, double* b) {
    if (A == NULL || tau == NULL || b == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 1 || cols < 1 || rows < cols) {
        return C4A_ERR_INVALID_ARGUMENT;
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
    return C4A_OK;
}

c4a_status_t c4a_back_solve_R(const double* A, int64_t rows, int64_t cols,
                              const double* y, double* x) {
    (void)rows;  /* only the storage stride matters; we read the upper block. */
    if (A == NULL || y == NULL || x == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 1 || cols < 1 || rows < cols) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    for (int64_t i = cols - 1; i >= 0; --i) {
        double s = y[i];
        for (int64_t j = i + 1; j < cols; ++j) {
            s -= A[(size_t)i * (size_t)cols + (size_t)j] * x[j];
        }
        const double diag = A[(size_t)i * (size_t)cols + (size_t)i];
        if (diag == 0.0) {
            return C4A_ERR_NUMERICAL_FAILURE;
        }
        x[i] = s / diag;
    }
    return C4A_OK;
}
