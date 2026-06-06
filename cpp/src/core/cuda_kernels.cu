// SPDX-License-Identifier: CECILL-2.1

#include "cuda_kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <limits>

namespace n4m {
namespace cuda_dispatch {
namespace {

constexpr int kSignNormalizeBlockSize = 256;

__global__ void pls1_moment_normalize_signs_many_kernel(
    const double* weights,
    int p,
    int batch,
    double* signed_weights) {
    const int job = static_cast<int>(blockIdx.x);
    if (job >= batch) {
        return;
    }

    const double* const src =
        weights + static_cast<std::size_t>(job) * static_cast<std::size_t>(p);
    double* const dst =
        signed_weights +
        static_cast<std::size_t>(job) * static_cast<std::size_t>(p);

    double best_abs = -1.0;
    int best_idx = 0;
    for (int feature = static_cast<int>(threadIdx.x);
         feature < p;
         feature += static_cast<int>(blockDim.x)) {
        const double abs_value = fabs(src[feature]);
        if (abs_value > best_abs ||
            (abs_value == best_abs && feature < best_idx)) {
            best_abs = abs_value;
            best_idx = feature;
        }
    }

    __shared__ double shared_abs[kSignNormalizeBlockSize];
    __shared__ int shared_idx[kSignNormalizeBlockSize];
    shared_abs[threadIdx.x] = best_abs;
    shared_idx[threadIdx.x] = best_idx;
    __syncthreads();

    for (int stride = static_cast<int>(blockDim.x) / 2;
         stride > 0;
         stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            const double candidate_abs =
                shared_abs[threadIdx.x + static_cast<unsigned int>(stride)];
            const int candidate_idx =
                shared_idx[threadIdx.x + static_cast<unsigned int>(stride)];
            const bool candidate_wins =
                candidate_abs > shared_abs[threadIdx.x] ||
                (candidate_abs == shared_abs[threadIdx.x] &&
                 candidate_idx < shared_idx[threadIdx.x]);
            if (candidate_wins) {
                shared_abs[threadIdx.x] = candidate_abs;
                shared_idx[threadIdx.x] = candidate_idx;
            }
        }
        __syncthreads();
    }

    const double sign =
        (src[shared_idx[0]] < 0.0) ? -1.0 : 1.0;
    for (int feature = static_cast<int>(threadIdx.x);
         feature < p;
         feature += static_cast<int>(blockDim.x)) {
        dst[feature] = sign * src[feature];
    }
}

}  // namespace

cudaError_t pls1_moment_normalize_signs_many(const double* weights,
                                             std::size_t p,
                                             std::size_t batch,
                                             double* signed_weights,
                                             cudaStream_t stream) noexcept {
    if (p == 0 || batch == 0) {
        return cudaSuccess;
    }
    if (weights == nullptr || signed_weights == nullptr ||
        p > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        batch > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return cudaErrorInvalidValue;
    }

    const dim3 grid(static_cast<unsigned int>(batch));
    const dim3 block(kSignNormalizeBlockSize);
    pls1_moment_normalize_signs_many_kernel<<<grid, block, 0, stream>>>(
        weights, static_cast<int>(p), static_cast<int>(batch),
        signed_weights);
    return cudaGetLastError();
}

}  // namespace cuda_dispatch
}  // namespace n4m
