// SPDX-License-Identifier: CeCILL-2.1
//
// Internal linear-algebra dispatch layer (Phase 43).
//
// Three entry points cover every hot kernel in cpp/src/core/model.cpp:
//
//   matvec — out = A * x      (replaces matrix_vector_product)
//   gemm   — C = a op(A) op(B) + b C (cross-covariance + rotation GEMMs)
//   ger    — A += a * x * y^T (rank-1 deflations)
//
// When P4A_USE_BLAS is not defined the implementations are the same
// row-major double loops that lived in model.cpp pre-Phase 43, so the
// default-OFF build is bit-identical to the scalar engine.
//
// When P4A_USE_BLAS is defined we dispatch to the system CBLAS
// implementation (OpenBLAS, MKL, Accelerate, Netlib — whichever the
// CMake `find_package(BLAS)` resolved). All buffers stay row-major and
// contiguous; lda equals cols for every internal working buffer, so
// no transposition or copy is needed across the boundary.

#ifndef PLS4ALL_CORE_LINALG_HPP
#define PLS4ALL_CORE_LINALG_HPP

#include <cstddef>
#include <vector>

#if defined(P4A_USE_BLAS)
#  if defined(__APPLE__) && __has_include(<Accelerate/Accelerate.h>) && !__has_include(<cblas.h>)
#    include <Accelerate/Accelerate.h>
#  else
#    include <cblas.h>
#  endif
#endif

namespace pls4all {
namespace linalg {

// ---------------------------------------------------------------------------
// Transpose flag. Defined as a tiny enum on the reference path so the call
// sites can stay #ifdef-free.
// ---------------------------------------------------------------------------
#if defined(P4A_USE_BLAS)
using Trans_t = CBLAS_TRANSPOSE;
constexpr CBLAS_TRANSPOSE Trans_No  = CblasNoTrans;
constexpr CBLAS_TRANSPOSE Trans_Yes = CblasTrans;
#else
enum Trans_t { Trans_No = 0, Trans_Yes = 1 };
#endif

// ---------------------------------------------------------------------------
// gemv: y = alpha * op(A) * x + beta * y
//   A is (rows x cols) row-major contiguous, lda = cols.
//   op(A) = A    if trans == Trans_No  -> output length is rows
//   op(A) = A^T  if trans == Trans_Yes -> output length is cols
// Caller owns sizing y to match op(A).
// ---------------------------------------------------------------------------
inline void gemv(Trans_t trans,
                 std::size_t rows, std::size_t cols,
                 double alpha,
                 const double* A,
                 const double* x,
                 double beta,
                 double* y) {
    if (rows == 0 || cols == 0) {
        return;
    }
#if defined(P4A_USE_BLAS)
    cblas_dgemv(CblasRowMajor, trans,
                static_cast<int>(rows), static_cast<int>(cols),
                alpha,
                A, static_cast<int>(cols),
                x, 1,
                beta,
                y, 1);
#else
    const std::size_t out_len = (trans == Trans_No) ? rows : cols;
    const std::size_t in_len  = (trans == Trans_No) ? cols : rows;
    for (std::size_t i = 0; i < out_len; ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j < in_len; ++j) {
            const double a_ij = (trans == Trans_No)
                ? A[i * cols + j]
                : A[j * cols + i];
            s += a_ij * x[j];
        }
        y[i] = beta * y[i] + alpha * s;
    }
#endif
}

// ---------------------------------------------------------------------------
// matvec: out = A * x  (thin wrapper around gemv with NoTrans, alpha=1, beta=0).
// `out` is resized to `rows` and overwritten.
// ---------------------------------------------------------------------------
inline void matvec(const std::vector<double>& A,
                   std::size_t rows,
                   std::size_t cols,
                   const std::vector<double>& x,
                   std::vector<double>& out) {
    out.assign(rows, 0.0);
    gemv(Trans_No, rows, cols, 1.0, A.data(), x.data(), 0.0, out.data());
}

// ---------------------------------------------------------------------------
// gemm: C = alpha * op(A) * op(B) + beta * C
//
//   op(A) is (M x K) row-major, op(B) is (K x N) row-major,
//   C is (M x N) row-major.
//   lda, ldb, ldc are the row strides in elements of A, B, C as STORED
//   (i.e. cols of the unmodified buffer, not of op(A)).
//
// Reference path supports any combination of Trans_No / Trans_Yes on
// either operand; this is required because some call sites compute
// X^T Y (op_a = Yes), others compute W^T P (op_a = Yes), and the
// rotations step computes W * ptw_inv (op_a = No, op_b = No).
// ---------------------------------------------------------------------------
inline void gemm(Trans_t trans_a,
                 Trans_t trans_b,
                 std::size_t M, std::size_t N, std::size_t K,
                 double alpha,
                 const double* A, std::size_t lda,
                 const double* B, std::size_t ldb,
                 double beta,
                 double* C, std::size_t ldc) {
    if (M == 0 || N == 0) {
        return;
    }
#if defined(P4A_USE_BLAS)
    cblas_dgemm(CblasRowMajor, trans_a, trans_b,
                static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                alpha,
                A, static_cast<int>(lda),
                B, static_cast<int>(ldb),
                beta,
                C, static_cast<int>(ldc));
#else
    // Reference: scalar triple-loop, all four (op_a, op_b) combinations.
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            double& cij = C[i * ldc + j];
            cij = beta * cij;
            if (K == 0 || alpha == 0.0) {
                continue;
            }
            double s = 0.0;
            for (std::size_t k = 0; k < K; ++k) {
                // op(A)[i,k] for row-major source A with lda
                const double a_ik = (trans_a == Trans_No)
                    ? A[i * lda + k]
                    : A[k * lda + i];
                const double b_kj = (trans_b == Trans_No)
                    ? B[k * ldb + j]
                    : B[j * ldb + k];
                s += a_ik * b_kj;
            }
            cij += alpha * s;
        }
    }
#endif
}

// ---------------------------------------------------------------------------
// ger: A += alpha * x * y^T  with A (M x N) row-major contiguous.
// Used for rank-1 deflations (alpha is typically -1.0).
// ---------------------------------------------------------------------------
inline void ger(std::size_t M, std::size_t N,
                double alpha,
                const double* x,   // length M
                const double* y,   // length N
                double* A, std::size_t lda) {
    if (M == 0 || N == 0 || alpha == 0.0) {
        return;
    }
#if defined(P4A_USE_BLAS)
    cblas_dger(CblasRowMajor,
               static_cast<int>(M), static_cast<int>(N),
               alpha,
               x, 1,
               y, 1,
               A, static_cast<int>(lda));
#else
    for (std::size_t i = 0; i < M; ++i) {
        const double ax = alpha * x[i];
        for (std::size_t j = 0; j < N; ++j) {
            A[i * lda + j] += ax * y[j];
        }
    }
#endif
}

}  // namespace linalg
}  // namespace pls4all

#endif  // PLS4ALL_CORE_LINALG_HPP
