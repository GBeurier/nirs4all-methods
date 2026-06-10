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

constexpr int kScalarPrepBlockSize = 256;

// Records the first failure across the tile. Only the thread that wins the
// atomicCAS on the code slot writes the job/component, so the captured reason is
// the earliest component (kernels run in component order on one stream) and a
// failing job within it. Mirrors the legacy host loops' early-return reasons.
__device__ inline void record_batch_fail(int* fail, int code, int local,
                                         int comp) {
    if (atomicCAS(&fail[0], kPls1MomentBatchOk, code) == kPls1MomentBatchOk) {
        fail[1] = local;
        fail[2] = comp;
    }
}

__global__ void pls1_moment_prepare_scale_kernel(const double* norm_sq,
                                                 const double* cur_yy,
                                                 double eps,
                                                 int comp,
                                                 int batch,
                                                 double* scale_out,
                                                 int* fail) {
    const int local =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
        static_cast<int>(threadIdx.x);
    if (local >= batch) {
        return;
    }
    const double yy = cur_yy[local];
    if (yy <= eps || !isfinite(yy)) {
        record_batch_fail(fail, kPls1MomentBatchYResidualVanished, local, comp);
        scale_out[local] = 0.0;
        return;
    }
    double ns = norm_sq[local];
    if (ns < 0.0 && ns > -eps) {
        ns = 0.0;
    }
    const double s_norm = sqrt(ns);
    if (s_norm <= eps || !isfinite(s_norm)) {
        record_batch_fail(fail, kPls1MomentBatchXWeightsVanished, local, comp);
        scale_out[local] = 0.0;
        return;
    }
    scale_out[local] = 1.0 / (s_norm + eps * yy);
}

__global__ void pls1_moment_prepare_loadings_kernel(const double* tt,
                                                    const double* qdot,
                                                    double eps,
                                                    int comp,
                                                    int batch,
                                                    double* q_out,
                                                    double* inv_tt_out,
                                                    double* sqrt_tt_out,
                                                    double* defl_out,
                                                    int* fail) {
    const int local =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
        static_cast<int>(threadIdx.x);
    if (local >= batch) {
        return;
    }
    const std::size_t q_idx =
        static_cast<std::size_t>(comp) * static_cast<std::size_t>(batch) +
        static_cast<std::size_t>(local);
    const double t = tt[local];
    if (t <= eps || !isfinite(t)) {
        record_batch_fail(fail, kPls1MomentBatchXScoreVanished, local, comp);
        q_out[q_idx] = 0.0;
        inv_tt_out[local] = 0.0;
        sqrt_tt_out[local] = 0.0;
        defl_out[local] = 0.0;
        return;
    }
    const double q_load = qdot[local] / t;
    if (!isfinite(q_load)) {
        record_batch_fail(fail, kPls1MomentBatchYLoadingNotFinite, local, comp);
        q_out[q_idx] = 0.0;
        inv_tt_out[local] = 0.0;
        sqrt_tt_out[local] = 0.0;
        defl_out[local] = 0.0;
        return;
    }
    q_out[q_idx] = q_load;
    inv_tt_out[local] = 1.0 / t;
    sqrt_tt_out[local] = sqrt(t);
    defl_out[local] = -t * q_load;
}

__global__ void pls1_moment_update_yy_kernel(const double* tt,
                                             const double* q_load,
                                             int comp,
                                             int batch,
                                             double* cur_yy) {
    const int local =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
        static_cast<int>(threadIdx.x);
    if (local >= batch) {
        return;
    }
    const std::size_t q_idx =
        static_cast<std::size_t>(comp) * static_cast<std::size_t>(batch) +
        static_cast<std::size_t>(local);
    const double t = tt[local];
    const double q = q_load[q_idx];
    double yy = cur_yy[local] - t * q * q;
    if (yy < 0.0 && yy > -1e-9) {
        yy = 0.0;
    }
    cur_yy[local] = yy;
}

__global__ void add_scaled_identity_kernel(double* A, int n, double lambda) {
    const int i =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
        static_cast<int>(threadIdx.x);
    if (i >= n) {
        return;
    }
    A[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
      static_cast<std::size_t>(i)] += lambda;
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

namespace {

dim3 scalar_prep_grid(std::size_t batch) {
    return dim3(static_cast<unsigned int>(
        (batch + static_cast<std::size_t>(kScalarPrepBlockSize) - 1) /
        static_cast<std::size_t>(kScalarPrepBlockSize)));
}

bool scalar_prep_batch_invalid(std::size_t batch) {
    return batch > static_cast<std::size_t>(std::numeric_limits<int>::max());
}

}  // namespace

cudaError_t pls1_moment_prepare_scale_many(const double* norm_sq,
                                           const double* cur_yy,
                                           double eps,
                                           int comp,
                                           std::size_t batch,
                                           double* scale_out,
                                           int* fail,
                                           cudaStream_t stream) noexcept {
    if (batch == 0) {
        return cudaSuccess;
    }
    if (norm_sq == nullptr || cur_yy == nullptr || scale_out == nullptr ||
        fail == nullptr || scalar_prep_batch_invalid(batch)) {
        return cudaErrorInvalidValue;
    }
    const dim3 block(kScalarPrepBlockSize);
    pls1_moment_prepare_scale_kernel<<<scalar_prep_grid(batch), block, 0,
                                       stream>>>(
        norm_sq, cur_yy, eps, comp, static_cast<int>(batch), scale_out, fail);
    return cudaGetLastError();
}

cudaError_t pls1_moment_prepare_loadings_many(const double* tt,
                                              const double* qdot,
                                              double eps,
                                              int comp,
                                              std::size_t batch,
                                              double* q_out,
                                              double* inv_tt_out,
                                              double* sqrt_tt_out,
                                              double* defl_out,
                                              int* fail,
                                              cudaStream_t stream) noexcept {
    if (batch == 0) {
        return cudaSuccess;
    }
    if (tt == nullptr || qdot == nullptr || q_out == nullptr ||
        inv_tt_out == nullptr || sqrt_tt_out == nullptr ||
        defl_out == nullptr || fail == nullptr ||
        scalar_prep_batch_invalid(batch)) {
        return cudaErrorInvalidValue;
    }
    const dim3 block(kScalarPrepBlockSize);
    pls1_moment_prepare_loadings_kernel<<<scalar_prep_grid(batch), block, 0,
                                          stream>>>(
        tt, qdot, eps, comp, static_cast<int>(batch), q_out, inv_tt_out,
        sqrt_tt_out, defl_out, fail);
    return cudaGetLastError();
}

cudaError_t pls1_moment_update_yy_many(const double* tt,
                                       const double* q_load,
                                       int comp,
                                       std::size_t batch,
                                       double* cur_yy,
                                       cudaStream_t stream) noexcept {
    if (batch == 0) {
        return cudaSuccess;
    }
    if (tt == nullptr || q_load == nullptr || cur_yy == nullptr ||
        scalar_prep_batch_invalid(batch)) {
        return cudaErrorInvalidValue;
    }
    const dim3 block(kScalarPrepBlockSize);
    pls1_moment_update_yy_kernel<<<scalar_prep_grid(batch), block, 0, stream>>>(
        tt, q_load, comp, static_cast<int>(batch), cur_yy);
    return cudaGetLastError();
}

cudaError_t add_scaled_identity(double* A,
                                std::size_t n,
                                double lambda,
                                cudaStream_t stream) noexcept {
    if (n == 0) {
        return cudaSuccess;
    }
    if (A == nullptr ||
        n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return cudaErrorInvalidValue;
    }
    constexpr int kBlock = 256;
    const dim3 grid(static_cast<unsigned int>(
        (n + static_cast<std::size_t>(kBlock) - 1) /
        static_cast<std::size_t>(kBlock)));
    add_scaled_identity_kernel<<<grid, dim3(kBlock), 0, stream>>>(
        A, static_cast<int>(n), lambda);
    return cudaGetLastError();
}

}  // namespace cuda_dispatch
}  // namespace n4m
