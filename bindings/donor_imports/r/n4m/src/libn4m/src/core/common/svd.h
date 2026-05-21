/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Shared dense SVD helpers used by Phase 9 feature-selection operators
 * (FlexiblePCA, FlexibleSVD) and potentially by Phase 8 / Phase 19 agents
 * (T^2 / Q residuals, OSC) that need a portable in-process SVD.
 *
 * Algorithm: one-sided Jacobi rotation SVD on a row-major (m x n) matrix.
 * The decomposition produced is the compact form:
 *
 *     A = U @ diag(S) @ Vt
 *
 * with U of shape (m x k), S of length k, Vt of shape (k x n), where
 * k = min(m, n). Singular values are sorted in descending order; sign is
 * canonicalised so that the largest-absolute-value element of each U
 * column is positive (`max_abs_element_positive` per component).
 *
 * The Jacobi algorithm rotates pairs of columns of A in place until the
 * off-diagonal entries of A^T A drop below an O(eps * ||A||) threshold.
 * It is numerically robust on small to medium dense matrices (parity
 * matrices in Phase 9 are at most 200 x 200) and has no BLAS / LAPACK
 * dependency, fitting the nirs4all-methods "pure C, no deps" constraint.
 *
 * Pure C, no dependencies. INTERNAL: never exposed by the public ABI.
 */
#ifndef N4M_CORE_COMMON_SVD_H
#define N4M_CORE_COMMON_SVD_H

#include <stdint.h>

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Compute the compact SVD of A (m x n, row-major). On success:
 *   - U  is filled with shape (m x k)  row-major, k = min(m, n).
 *   - S  is filled with length k singular values in descending order.
 *   - Vt is filled with shape (k x n) row-major (Vt[i, :] is the i-th
 *     right singular vector V[:, i] transposed).
 *
 * Singular vectors are sign-canonicalised: for each i in [0, k), the
 * element of U[:, i] with the largest absolute value is positive (ties
 * broken by smallest index). Vt[i, :] is flipped along with U[:, i] to
 * preserve A = U @ diag(S) @ Vt.
 *
 * The input A is consumed (modified in place) — pass a copy if the
 * caller needs to keep the original.
 *
 * Returns:
 *   - N4M_OK on success.
 *   - N4M_ERR_NULL_POINTER if any of A, U, S, Vt is NULL with m > 0 and
 *     n > 0.
 *   - N4M_ERR_INVALID_ARGUMENT for m < 1 or n < 1.
 *   - N4M_ERR_OUT_OF_MEMORY when an internal scratch allocation fails.
 *   - N4M_ERR_CONVERGENCE_FAILED if the Jacobi sweep limit is reached
 *     without the off-diagonal mass dropping below tolerance.
 */
n4m_status_t n4m_svd_compact(double* A, int64_t m, int64_t n,
                              double* U, double* S, double* Vt);

/* Exact leading-k SVD for wide matrices (m <= n) through the sample Gram
 * matrix A @ A.T. This avoids the expensive column-pair Jacobi sweep on the
 * original feature width while preserving full-SVD parity for small m.
 */
n4m_status_t n4m_svd_truncated_dual_wide(const double* A, int64_t m, int64_t n,
                                         int64_t k, double* U, double* S,
                                         double* Vt);

/* Exact leading-k SVD for tall matrices (m >= n) through the feature Gram
 * matrix A.T @ A. This avoids the full one-sided Jacobi sweep when callers
 * only need a small leading component count.
 */
n4m_status_t n4m_svd_truncated_tall_gram(const double* A, int64_t m,
                                         int64_t n, int64_t k, double* U,
                                         double* S, double* Vt);

/* Deterministic randomized leading-k SVD for large matrices where an exact
 * full decomposition is too expensive. The input is read-only.
 */
n4m_status_t n4m_svd_truncated_randomized(const double* A, int64_t m,
                                          int64_t n, int64_t k,
                                          double* U, double* S, double* Vt);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_COMMON_SVD_H */
