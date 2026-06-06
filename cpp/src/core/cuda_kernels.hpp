// SPDX-License-Identifier: CECILL-2.1

#ifndef PLS4ALL_CORE_CUDA_KERNELS_HPP
#define PLS4ALL_CORE_CUDA_KERNELS_HPP

#include <cuda_runtime.h>

#include <cstddef>

namespace n4m {
namespace cuda_dispatch {

cudaError_t pls1_moment_normalize_signs_many(const double* weights,
                                             std::size_t p,
                                             std::size_t batch,
                                             double* signed_weights,
                                             cudaStream_t stream) noexcept;

}  // namespace cuda_dispatch
}  // namespace n4m

#endif  // PLS4ALL_CORE_CUDA_KERNELS_HPP
