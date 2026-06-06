// SPDX-License-Identifier: CECILL-2.1

#ifndef PLS4ALL_CORE_CUDA_KERNELS_HPP
#define PLS4ALL_CORE_CUDA_KERNELS_HPP

#include <cuda_runtime.h>

#include <cstddef>

namespace n4m {
namespace cuda_dispatch {

// Failure codes recorded by the many-batched scalar-preparation kernels in the
// shared 3-int fail buffer {code, job-local index, component}. A code of 0 means
// no job in the tile has failed; any non-zero code aborts the tile with status 1
// and mirrors one of the host-side guards in the legacy scalar-glue loop.
enum Pls1MomentBatchFailCode : int {
    kPls1MomentBatchOk = 0,
    kPls1MomentBatchYResidualVanished = 1,
    kPls1MomentBatchXWeightsVanished = 2,
    kPls1MomentBatchXScoreVanished = 3,
    kPls1MomentBatchYLoadingNotFinite = 4,
};

cudaError_t pls1_moment_normalize_signs_many(const double* weights,
                                             std::size_t p,
                                             std::size_t batch,
                                             double* signed_weights,
                                             cudaStream_t stream) noexcept;

// Stage 1 (after the s-norm gemm): checks the running Y residual and the X-weight
// norm per job, then writes the normalization scale 1/(||s|| + eps*yy) used to
// turn s into the unit weight vector. Records the first Y-residual / X-weight
// failure into `fail`.
cudaError_t pls1_moment_prepare_scale_many(const double* norm_sq,
                                           const double* cur_yy,
                                           double eps,
                                           int comp,
                                           std::size_t batch,
                                           double* scale_out,
                                           int* fail,
                                           cudaStream_t stream) noexcept;

// Stage 2 (after the tt=w'Cw and qdot=w's gemms): checks the X score and Y
// loading per job, stores the Y loading q into the device Q accumulator
// (component-major: q_out[comp*batch + job]) and prepares the three deflation
// scales 1/tt, sqrt(tt) and -tt*q. Records the first X-score / Y-loading failure.
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
                                              cudaStream_t stream) noexcept;

// Stage 3 (after the score deflation): updates the device Y residual
// yy -= tt*q*q with the same tiny-negative clamp as the legacy host loop.
cudaError_t pls1_moment_update_yy_many(const double* tt,
                                       const double* q_load,
                                       int comp,
                                       std::size_t batch,
                                       double* cur_yy,
                                       cudaStream_t stream) noexcept;

}  // namespace cuda_dispatch
}  // namespace n4m

#endif  // PLS4ALL_CORE_CUDA_KERNELS_HPP
