// SPDX-License-Identifier: CeCILL-2.1
//
// cuBLAS dispatch implementation (Phase 45).
//
// libp4a's matrices are row-major contiguous doubles. cuBLAS is
// column-major. The standard trick to bridge them without copying:
// a row-major (rows x cols) buffer is bit-equivalent to a
// column-major (cols x rows) buffer. So we tell cuBLAS to operate on
// the transposed view, swap M and N where needed, and flip operand
// order in GEMM.
//
// For gemv:  row-major  y = op(A) * x
//            cuBLAS sees A as column-major (cols x rows).
//            For row-major NoTrans -> cuBLAS Trans on (cols x rows)
//            For row-major Trans   -> cuBLAS NoTrans on (cols x rows)
//            cuBLAS gemv signature: y = alpha * op(A) * x + beta * y
//            where m=cols (rows of stored A), n=rows (cols of stored A).
//
// For gemm:  row-major  C = op(A) * op(B)
//            Use identity: (op(A) * op(B))^T = op(B)^T * op(A)^T
//            cuBLAS sees the row-major C as a column-major C^T.
//            Compute C^T_cm = op(B)^T_cm * op(A)^T_cm with B as first
//            operand and A as second, swapping trans flags accordingly.

#include "cuda_dispatch.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <new>
#include <stdexcept>
#include <string>

namespace pls4all {
namespace cuda_dispatch {

namespace {

struct CublasState {
    cublasHandle_t handle{};
    bool available{false};

    CublasState() noexcept {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
            return;
        }
        if (cudaSetDevice(0) != cudaSuccess) {
            return;
        }
        if (cublasCreate_v2(&handle) != CUBLAS_STATUS_SUCCESS) {
            return;
        }
        available = true;
    }

    ~CublasState() {
        if (available) {
            cublasDestroy_v2(handle);
        }
    }

    CublasState(const CublasState&)            = delete;
    CublasState& operator=(const CublasState&) = delete;
};

CublasState& state() {
    static CublasState s;
    return s;
}

// RAII device buffer. cudaFree on a null pointer is documented as a
// no-op so the destructor is safe.
template <typename T>
class DevicePtr {
public:
    explicit DevicePtr(std::size_t n) : ptr_(nullptr) {
        if (n == 0) {
            return;
        }
        const cudaError_t st = cudaMalloc(reinterpret_cast<void**>(&ptr_),
                                           n * sizeof(T));
        if (st != cudaSuccess) {
            ptr_ = nullptr;
            throw std::bad_alloc();
        }
    }
    ~DevicePtr() noexcept { if (ptr_) cudaFree(ptr_); }
    DevicePtr(const DevicePtr&)            = delete;
    DevicePtr& operator=(const DevicePtr&) = delete;

    T* get() const noexcept { return ptr_; }

private:
    T* ptr_;
};

void copy_h2d(void* dst, const void* src, std::size_t bytes) {
    if (bytes == 0) {
        return;
    }
    if (cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy H2D failed");
    }
}

void copy_d2h(void* dst, const void* src, std::size_t bytes) {
    if (bytes == 0) {
        return;
    }
    if (cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy D2H failed");
    }
}

void check_cublas(cublasStatus_t st, const char* what) {
    if (st != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLAS error in ") + what);
    }
}

}  // namespace

bool cuda_runtime_available() noexcept {
    return state().available;
}

// ---------------------------------------------------------------------------
// gemv: row-major  y = alpha * op(A) * x + beta * y
//   A is (rows x cols) row-major; bit-equivalent to (cols x rows) column-major
//   For row-major NoTrans -> cuBLAS sees A^cm and we ask for op = Trans
//   For row-major Trans   -> we ask for op = NoTrans
//   cuBLAS signature: cublasDgemv_v2(h, op, m, n, alpha, A_cm, lda_cm, x, incx,
//                                     beta, y, incy)
//     where m, n are dimensions of the *stored* matrix A_cm = (cols x rows).
//     -> m = cols, n = rows, lda_cm = cols (column-major leading dim).
// ---------------------------------------------------------------------------
void gemv(int trans,
          std::size_t rows, std::size_t cols,
          double alpha,
          const double* A,
          const double* x,
          double beta,
          double* y) {
    if (!cuda_runtime_available()) {
        throw std::runtime_error("CUDA backend requested but no GPU available");
    }
    if (rows == 0 || cols == 0) {
        return;
    }
    const std::size_t out_len = (trans == 0) ? rows : cols;
    const std::size_t in_len  = (trans == 0) ? cols : rows;
    const std::size_t A_elems = rows * cols;

    DevicePtr<double> dA(A_elems);
    DevicePtr<double> dx(in_len);
    DevicePtr<double> dy(out_len);

    copy_h2d(dA.get(), A, A_elems * sizeof(double));
    copy_h2d(dx.get(), x, in_len  * sizeof(double));
    if (beta != 0.0) {
        copy_h2d(dy.get(), y, out_len * sizeof(double));
    }

    const cublasOperation_t op = (trans == 0) ? CUBLAS_OP_T : CUBLAS_OP_N;
    check_cublas(
        cublasDgemv_v2(state().handle,
                       op,
                       static_cast<int>(cols),   // m of stored (cols x rows)
                       static_cast<int>(rows),   // n of stored (cols x rows)
                       &alpha,
                       dA.get(),
                       static_cast<int>(cols),   // lda_cm
                       dx.get(), 1,
                       &beta,
                       dy.get(), 1),
        "cublasDgemv_v2");

    copy_d2h(y, dy.get(), out_len * sizeof(double));
}

// ---------------------------------------------------------------------------
// gemm: row-major  C = alpha * op(A) * op(B) + beta * C
//   With A (Ar x Ac) and B (Br x Bc) row-major. Their column-major view is
//   (Ac x Ar) and (Bc x Br). Identity:
//       C_row = op(A) * op(B)    =>    C_col = op(B)^T_cm * op(A)^T_cm
//   So we call cuBLAS with B first, A second, and flip each operand's trans:
//       op_cm(B) = flip(trans_b); op_cm(A) = flip(trans_a)
//   Where flip(NoTrans) = Trans and flip(Trans) = NoTrans.
//   The dimensions:
//       m_cm = N  (col count of stored C_row = row count of C_col)
//       n_cm = M
//       k_cm = K
//       lda_cm = leading dim of stored B = Bc = ldb_row
//       ldb_cm = leading dim of stored A = Ac = lda_row
//       ldc_cm = leading dim of stored C = Cc = ldc_row
// ---------------------------------------------------------------------------
void gemm(int trans_a, int trans_b,
          std::size_t M, std::size_t N, std::size_t K,
          double alpha,
          const double* A, std::size_t lda,
          const double* B, std::size_t ldb,
          double beta,
          double* C, std::size_t ldc) {
    if (!cuda_runtime_available()) {
        throw std::runtime_error("CUDA backend requested but no GPU available");
    }
    if (M == 0 || N == 0) {
        return;
    }
    // op(A) is M x K, stored A is (Ar x Ac) where:
    //   trans_a == 0: Ar = M, Ac = K  -> A_elems = M*K
    //   trans_a == 1: Ar = K, Ac = M  -> A_elems = M*K  (same product)
    const std::size_t A_rows = (trans_a == 0) ? M : K;
    const std::size_t A_cols = (trans_a == 0) ? K : M;
    const std::size_t B_rows = (trans_b == 0) ? K : N;
    const std::size_t B_cols = (trans_b == 0) ? N : K;
    const std::size_t A_elems = A_rows * A_cols;
    const std::size_t B_elems = B_rows * B_cols;
    const std::size_t C_elems = M * N;

    DevicePtr<double> dA(A_elems);
    DevicePtr<double> dB(B_elems);
    DevicePtr<double> dC(C_elems);

    copy_h2d(dA.get(), A, A_elems * sizeof(double));
    copy_h2d(dB.get(), B, B_elems * sizeof(double));
    if (beta != 0.0) {
        copy_h2d(dC.get(), C, C_elems * sizeof(double));
    }

    // op_cm of each operand equals the row-major op directly (no flip).
    // Derivation: (op_row(A) * op_row(B))^T = (op_row(B))^T * (op_row(A))^T
    // The transpose of a row-major matrix M_row stored at addr p is the
    // column-major matrix M^cm sharing the same memory; therefore
    // (op_row(M))^T = op_row(M^cm). So we pass op_row directly.
    const cublasOperation_t op_a_cm =
        (trans_a == 0) ? CUBLAS_OP_N : CUBLAS_OP_T;
    const cublasOperation_t op_b_cm =
        (trans_b == 0) ? CUBLAS_OP_N : CUBLAS_OP_T;

    check_cublas(
        cublasDgemm_v2(state().handle,
                       op_b_cm, op_a_cm,
                       static_cast<int>(N),       // m_cm = N_row
                       static_cast<int>(M),       // n_cm = M_row
                       static_cast<int>(K),       // k_cm = K_row
                       &alpha,
                       dB.get(), static_cast<int>(ldb),  // ldb_row
                       dA.get(), static_cast<int>(lda),  // lda_row
                       &beta,
                       dC.get(), static_cast<int>(ldc)),
        "cublasDgemm_v2");

    copy_d2h(C, dC.get(), C_elems * sizeof(double));
}

// ---------------------------------------------------------------------------
// ger: row-major  A += alpha * x * y^T
//   A is (M x N) row-major, bit-equivalent to (N x M) column-major.
//   The row-major outer product `alpha * x * y^T` produces an (M x N) update.
//   In column-major view that is `alpha * y * x^T` (because (x y^T)^T = y x^T)
//   producing an (N x M) update. cuBLAS dger:
//       A = A + alpha * x * y^T  (column-major)
//   So we swap x and y and pass m=N, n=M, lda=N.
// ---------------------------------------------------------------------------
void ger(std::size_t M, std::size_t N,
         double alpha,
         const double* x,   // length M, the row-major outer-product left vec
         const double* y,   // length N, the row-major outer-product right vec
         double* A, std::size_t lda) {
    if (!cuda_runtime_available()) {
        throw std::runtime_error("CUDA backend requested but no GPU available");
    }
    if (M == 0 || N == 0 || alpha == 0.0) {
        return;
    }
    const std::size_t A_elems = M * lda;  // upper bound; M*N for contiguous

    DevicePtr<double> dA(A_elems);
    DevicePtr<double> dx(M);
    DevicePtr<double> dy(N);

    copy_h2d(dA.get(), A, A_elems * sizeof(double));
    copy_h2d(dx.get(), x, M * sizeof(double));
    copy_h2d(dy.get(), y, N * sizeof(double));

    // Column-major view: A_cm = (N x M), update is alpha * y_cm * x_cm^T
    check_cublas(
        cublasDger_v2(state().handle,
                      static_cast<int>(N),   // m_cm
                      static_cast<int>(M),   // n_cm
                      &alpha,
                      dy.get(), 1,           // x_cm (length N)
                      dx.get(), 1,           // y_cm (length M)
                      dA.get(), static_cast<int>(lda)),
        "cublasDger_v2");

    copy_d2h(A, dA.get(), A_elems * sizeof(double));
}

}  // namespace cuda_dispatch
}  // namespace pls4all
