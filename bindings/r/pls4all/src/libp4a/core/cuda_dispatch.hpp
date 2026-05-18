// SPDX-License-Identifier: CECILL-2.1
//
// Internal cuBLAS dispatch surface (Phase 45). Included only when
// P4A_USE_CUDA is defined — the header itself is safe to include
// from any translation unit, but the implementation in
// cuda_dispatch.cpp requires the CUDA toolkit headers and links
// against cuBLAS + cudart.
//
// The runtime ownership model is a single process-wide cuBLAS
// handle, lazily initialised on first dispatch call. Phase 45b
// will add a per-stream pool for concurrent fits.

#ifndef PLS4ALL_CORE_CUDA_DISPATCH_HPP
#define PLS4ALL_CORE_CUDA_DISPATCH_HPP

#include <cstddef>

namespace pls4all {
namespace cuda_dispatch {

// Returns true if a CUDA-capable GPU was detected at runtime and the
// process-wide cuBLAS handle was created successfully. Memoised after
// first call; safe to invoke from p4a_backend_is_available without
// fearing repeated device init.
bool cuda_runtime_available() noexcept;

// linalg-compatible entry points. The `trans` arguments use the same
// 0 = no-transpose, 1 = transpose convention as the Trans_t enum in
// linalg.hpp. The functions throw std::runtime_error on cudaMemcpy /
// cuBLAS errors. Callers in libp4a's hot path do not propagate
// exceptions across the C ABI; the linalg.hpp dispatch is wrapped in
// the same try/catch the existing C ABI surface uses.
//
// **Contiguous-buffer contract (Phase 45)**: all matrix arguments are
// assumed to be row-major *contiguous* with `lda == cols` (and same
// for ldb / ldc). The host→device copy uses `rows * cols` bytes, not
// `rows * lda` bytes — a caller passing a strided buffer would silently
// truncate. Every internal libp4a call site honours this; Phase 45b
// will extend the dispatch surface to accept strided inputs.
void gemv(int trans,
          std::size_t rows, std::size_t cols,
          double alpha,
          const double* A,
          const double* x,
          double beta,
          double* y);

void gemm(int trans_a, int trans_b,
          std::size_t M, std::size_t N, std::size_t K,
          double alpha,
          const double* A, std::size_t lda,
          const double* B, std::size_t ldb,
          double beta,
          double* C, std::size_t ldc);

void ger(std::size_t M, std::size_t N,
         double alpha,
         const double* x,   // length M
         const double* y,   // length N
         double* A, std::size_t lda);

}  // namespace cuda_dispatch
}  // namespace pls4all

#endif  // PLS4ALL_CORE_CUDA_DISPATCH_HPP
