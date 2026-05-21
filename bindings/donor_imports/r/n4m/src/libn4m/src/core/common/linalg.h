/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Shared dense linear algebra helpers (Householder QR factorisation +
 * companions). Extracted in Phase 5a so EMSC, SavitzkyGolay (main coef
 * build + sg_fit_edge_left/right) and the new Phase 5a baseline operators
 * (Detrend) all consume one implementation.
 *
 * Pure C, no dependencies. INTERNAL: never exposed by the public ABI.
 *
 * Storage convention: matrices are row-major doubles, shape (rows, cols)
 * with cols-element stride. `rows >= cols` is required for the QR.
 */
#ifndef N4M_CORE_COMMON_LINALG_H
#define N4M_CORE_COMMON_LINALG_H

#include <stdint.h>

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

/* In-place Householder QR factorisation of a (rows x cols) row-major matrix
 * A with `rows >= cols`. After the call:
 *   - The upper triangle of A[0:cols, 0:cols] holds R (cols x cols upper
 *     triangular).
 *   - The Householder vectors v_k for k = 0..cols-1 are stored in column k
 *     of A from row k+1 down; the implicit first component v_k[k] = 1.
 *     Concretely, v_k[i] (for i > k) is stored at A[i * cols + k].
 *   - tau[k] holds the Householder scalar 2 / ||v_k||^2 (scaled so a
 *     reflection on b is `b -= tau[k] * (v_k^T b) * v_k`).
 *
 * Returns:
 *   - N4M_OK on success.
 *   - N4M_ERR_NULL_POINTER if A or tau is NULL.
 *   - N4M_ERR_INVALID_ARGUMENT if rows < cols, rows < 1, cols < 1, or
 *     rows / cols overflow a row-major index.
 */
n4m_status_t n4m_householder_qr(double* A, int64_t rows, int64_t cols,
                                double* tau);

/* Apply Q^T (transpose of the orthogonal factor) to a length-rows vector b
 * in place. Uses the Householder vectors stored below the diagonal of A and
 * the scaling factors in tau. Both come from a successful n4m_householder_qr
 * call.
 *
 * Returns N4M_ERR_NULL_POINTER on null inputs, N4M_ERR_INVALID_ARGUMENT on
 * rows < cols or non-positive dimensions, N4M_OK otherwise.
 */
n4m_status_t n4m_apply_qt(const double* A, int64_t rows, int64_t cols,
                          const double* tau, double* b);

/* Back-substitute R x = y where R is the upper-triangular (cols x cols)
 * block at the top-left of A (row-major, stride cols). On success, x[0..cols)
 * holds the solution; on a zero diagonal entry (singular R) we return
 * N4M_ERR_NUMERICAL_FAILURE.
 *
 * `rows` is the storage stride of A (i.e. the original matrix passed to
 * n4m_householder_qr); only the [0:cols, 0:cols] upper triangle is read.
 *
 * Returns N4M_ERR_NULL_POINTER on null inputs, N4M_ERR_INVALID_ARGUMENT on
 * non-positive dimensions / rows < cols.
 */
n4m_status_t n4m_back_solve_R(const double* A, int64_t rows, int64_t cols,
                              const double* y, double* x);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_COMMON_LINALG_H */
