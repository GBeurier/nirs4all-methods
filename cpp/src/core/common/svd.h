/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Shared dense SVD helpers used by Phase 9 feature-selection operators
 * (FlexiblePCA, FlexibleSVD) and potentially by Phase 8 / Phase 19 agents
 * (T^2 / Q residuals, OSC) that need a portable in-process SVD.
 *
 * Algorithms: one-sided Jacobi rotation SVD on a row-major (m x n) matrix
 * (n4m_svd_compact), Gram + tridiagonal QL (truncated Gram paths,
 * n4m_symmetric_eigh) and a randomized sketch.
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
 * Accuracy: one-sided Jacobi keeps relative accuracy for graded or
 * block-decoupled inputs; small but genuine directions are never truncated.
 * Rank-deficient inputs: an exactly zero column gives an exactly zero
 * singular value with a zero U column (zero Vt row for m < n). A column that
 * two consecutive sweeps cancel to rounding level (rotations regenerating
 * noise parallel to another column, e.g. +-1 designs with opposite rows) is
 * frozen: its singular value is its residual norm (rounding level, not
 * forced to 0) and its U column an explicit orthonormal completion.
 *
 * Magnitude domain (all helpers in this header): entries must be finite
 * (otherwise N4M_ERR_NUMERICAL_FAILURE). Inputs whose largest magnitude lies
 * outside [2^-128, 2^128) are rescaled by an exact power of two, so results
 * are scale-equivariant over the whole double range; inside that band the
 * algorithms run unscaled. A published singular value / eigenvalue that is
 * significant (> max(m,n) * eps * the largest one) but overflows, or falls
 * below 2^-1042 (fewer than 32 significant bits), is an explicit
 * N4M_ERR_NUMERICAL_FAILURE; output buffers are then unspecified.
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
 *   - N4M_ERR_NUMERICAL_FAILURE for non-finite input or an output that is
 *     not representable (see the magnitude domain above).
 */
n4m_status_t n4m_svd_compact(double* A, int64_t m, int64_t n,
                              double* U, double* S, double* Vt);

/* Exact leading-k SVD for wide matrices (m <= n) through the sample Gram
 * matrix A @ A.T. This avoids the expensive column-pair Jacobi sweep on the
 * original feature width while preserving full-SVD parity for small m.
 * Squared singular values are accurate to O(eps * s_max^2) for coupled
 * inputs (relative accuracy kept on decoupled blocks). Same magnitude domain
 * and statuses as n4m_svd_compact.
 */
n4m_status_t n4m_svd_truncated_dual_wide(const double* A, int64_t m, int64_t n,
                                         int64_t k, double* U, double* S,
                                         double* Vt);

/* Exact leading-k SVD for tall matrices (m >= n) through the feature Gram
 * matrix A.T @ A. This avoids the full one-sided Jacobi sweep when callers
 * only need a small leading component count. Accuracy, domain and statuses
 * as n4m_svd_truncated_dual_wide.
 */
n4m_status_t n4m_svd_truncated_tall_gram(const double* A, int64_t m,
                                         int64_t n, int64_t k, double* U,
                                         double* S, double* Vt);

/* Deterministic randomized leading-k SVD for large matrices where an exact
 * full decomposition is too expensive. The input is read-only. Singular
 * values are resolved only to O(eps * s_max) absolutely: directions below
 * that level are not captured by the sketch and come back as 0 (sketch
 * columns that are linearly dependent at rounding level are dropped).
 * Same magnitude domain and statuses as n4m_svd_compact.
 */
n4m_status_t n4m_svd_truncated_randomized(const double* A, int64_t m,
                                          int64_t n, int64_t k,
                                          double* U, double* S, double* Vt);

/* Full symmetric eigendecomposition of a row-major n x n matrix.
 *
 * This is an internal helper, not part of the public C ABI. On success:
 *   - eigvals receives n eigenvalues.
 *   - V receives the row-major n x n eigenvector matrix, with eigenvectors
 *     stored as columns.
 *
 * The input A is read-only. Eigenpairs are not sorted; callers that use the
 * full basis, such as Ridge path scans, do not need a particular order.
 * Tridiagonal QL in the LAPACK dsteqr organisation: split into unreduced
 * blocks by the local test |e| <= eps (|d_m| + |d_m+1|), each block scaled
 * exactly when extreme (subnormal spectra are resolved, not flushed), 30n
 * total QL sweeps (else N4M_ERR_CONVERGENCE_FAILED). Same magnitude domain
 * and N4M_ERR_NUMERICAL_FAILURE rule as n4m_svd_compact (n * eps rank
 * threshold).
 */
n4m_status_t n4m_symmetric_eigh(const double* A, int64_t n, double* eigvals,
                                double* V);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_COMMON_SVD_H */
