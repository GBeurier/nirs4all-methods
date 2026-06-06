// SPDX-License-Identifier: CECILL-2.1
//
// cuBLAS dispatch implementation (Phase 45).
//
// libn4m's matrices are row-major contiguous doubles. cuBLAS is
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

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace n4m {
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

struct Pls1MomentWorkspaceView {
    double* dC{nullptr};
    double* ds{nullptr};
    double* dw{nullptr};
    double* dcw{nullptr};
    double* dp_load{nullptr};
    double* dW{nullptr};
    double* dP{nullptr};
};

struct Pls1MomentBatchWorkspaceView {
    double* dC{nullptr};
    double* ds{nullptr};
    double* dw{nullptr};
    double* dcw{nullptr};
    double* dp_load{nullptr};
    double* douter{nullptr};
    double* dW{nullptr};
    double* dP{nullptr};
    double* dscale{nullptr};
    double* dnorm_sq{nullptr};
    double* dtt{nullptr};
    double* dqdot{nullptr};
    double* dsign{nullptr};
};

constexpr std::size_t kMaxParallelFoldStreams = 4;
constexpr std::size_t kDefaultPlsBatchTileBytes =
    static_cast<std::size_t>(512) * 1024 * 1024;

template <typename T>
class ReusableDeviceBuffer {
public:
    ReusableDeviceBuffer() = default;
    ~ReusableDeviceBuffer() noexcept {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }

    ReusableDeviceBuffer(const ReusableDeviceBuffer&) = delete;
    ReusableDeviceBuffer& operator=(const ReusableDeviceBuffer&) = delete;

    void ensure(std::size_t n) {
        if (n <= capacity_) {
            return;
        }
        T* next = nullptr;
        if (n != 0) {
            const cudaError_t st = cudaMalloc(
                reinterpret_cast<void**>(&next), n * sizeof(T));
            if (st != cudaSuccess) {
                throw std::bad_alloc();
            }
        }
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
        ptr_ = next;
        capacity_ = n;
    }

    T* get() const noexcept { return ptr_; }

private:
    T* ptr_{nullptr};
    std::size_t capacity_{0};
};

class ReusablePls1MomentWorkspace {
public:
    Pls1MomentWorkspaceView ensure(std::size_t p,
                                   std::size_t max_components) {
        dC_.ensure(p * p);
        ds_.ensure(p);
        dw_.ensure(p);
        dcw_.ensure(p);
        dp_load_.ensure(p);
        dW_.ensure(p * max_components);
        dP_.ensure(p * max_components);
        return {
            dC_.get(),
            ds_.get(),
            dw_.get(),
            dcw_.get(),
            dp_load_.get(),
            dW_.get(),
            dP_.get(),
        };
    }

private:
    ReusableDeviceBuffer<double> dC_;
    ReusableDeviceBuffer<double> ds_;
    ReusableDeviceBuffer<double> dw_;
    ReusableDeviceBuffer<double> dcw_;
    ReusableDeviceBuffer<double> dp_load_;
    ReusableDeviceBuffer<double> dW_;
    ReusableDeviceBuffer<double> dP_;
};

class ReusablePls1MomentBatchWorkspace {
public:
    Pls1MomentBatchWorkspaceView ensure(std::size_t n_jobs,
                                        std::size_t p,
                                        std::size_t max_components) {
        dC_.ensure(n_jobs * p * p);
        ds_.ensure(n_jobs * p);
        dw_.ensure(n_jobs * p);
        dcw_.ensure(n_jobs * p);
        dp_load_.ensure(n_jobs * p);
        douter_.ensure(n_jobs * p);
        dW_.ensure(n_jobs * p * max_components);
        dP_.ensure(n_jobs * p * max_components);
        dscale_.ensure(n_jobs);
        dnorm_sq_.ensure(n_jobs);
        dtt_.ensure(n_jobs);
        dqdot_.ensure(n_jobs);
        dsign_.ensure(n_jobs);
        return {
            dC_.get(),
            ds_.get(),
            dw_.get(),
            dcw_.get(),
            dp_load_.get(),
            douter_.get(),
            dW_.get(),
            dP_.get(),
            dscale_.get(),
            dnorm_sq_.get(),
            dtt_.get(),
            dqdot_.get(),
            dsign_.get(),
        };
    }

private:
    ReusableDeviceBuffer<double> dC_;
    ReusableDeviceBuffer<double> ds_;
    ReusableDeviceBuffer<double> dw_;
    ReusableDeviceBuffer<double> dcw_;
    ReusableDeviceBuffer<double> dp_load_;
    ReusableDeviceBuffer<double> douter_;
    ReusableDeviceBuffer<double> dW_;
    ReusableDeviceBuffer<double> dP_;
    ReusableDeviceBuffer<double> dscale_;
    ReusableDeviceBuffer<double> dnorm_sq_;
    ReusableDeviceBuffer<double> dtt_;
    ReusableDeviceBuffer<double> dqdot_;
    ReusableDeviceBuffer<double> dsign_;
};

Pls1MomentWorkspaceView workspace_view(DevicePtr<double>& dC,
                                       DevicePtr<double>& ds,
                                       DevicePtr<double>& dw,
                                       DevicePtr<double>& dcw,
                                       DevicePtr<double>& dp_load,
                                       DevicePtr<double>& dW,
                                       DevicePtr<double>& dP) noexcept {
    return {
        dC.get(),
        ds.get(),
        dw.get(),
        dcw.get(),
        dp_load.get(),
        dW.get(),
        dP.get(),
    };
}

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

void copy_h2d_stream(void* dst, const void* src, std::size_t bytes,
                     cudaStream_t stream) {
    if (bytes == 0) {
        return;
    }
    if (cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream) !=
        cudaSuccess) {
        throw std::runtime_error("cudaMemcpyAsync H2D failed");
    }
}

void copy_d2h_stream_sync(void* dst, const void* src, std::size_t bytes,
                          cudaStream_t stream) {
    if (bytes == 0) {
        return;
    }
    if (cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream) !=
        cudaSuccess) {
        throw std::runtime_error("cudaMemcpyAsync D2H failed");
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        throw std::runtime_error("cudaStreamSynchronize failed after D2H");
    }
}

void check_cublas(cublasStatus_t st, const char* what) {
    if (st != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLAS error in ") + what);
    }
}

void set_error(std::string* out, const char* message) {
    if (out != nullptr) {
        *out = message;
    }
}

bool cuda_pls_parallel_folds_requested(std::size_t n_jobs) {
    if (n_jobs <= 1) {
        return false;
    }
    const char* env = std::getenv("N4M_CUDA_PLS_PARALLEL_FOLDS");
    if (env == nullptr) {
        return false;
    }
    const std::string value(env);
    return value == "1" || value == "true" || value == "TRUE" ||
           value == "on" || value == "ON" || value == "yes" ||
           value == "YES";
}

bool truthy_env(const char* name) {
    const char* env = std::getenv(name);
    if (env == nullptr) {
        return false;
    }
    const std::string value(env);
    return value == "1" || value == "true" || value == "TRUE" ||
           value == "on" || value == "ON" || value == "yes" ||
           value == "YES";
}

bool cuda_pls_many_legacy_requested() {
    return truthy_env("N4M_CUDA_PLS_MANY_LEGACY");
}

bool cuda_pls_many_batched_requested() {
    return truthy_env("N4M_CUDA_PLS_MANY_BATCHED") &&
           !cuda_pls_many_legacy_requested();
}

std::size_t cuda_pls_batch_tile_budget_bytes() {
    const char* env = std::getenv("N4M_CUDA_PLS_BATCH_MAX_BYTES");
    if (env == nullptr || env[0] == '\0') {
        return kDefaultPlsBatchTileBytes;
    }
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(env, &end, 10);
    if (end == env || parsed == 0ULL) {
        return kDefaultPlsBatchTileBytes;
    }
    const auto max_size =
        static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max());
    if (parsed > max_size) {
        return kDefaultPlsBatchTileBytes;
    }
    return static_cast<std::size_t>(parsed);
}

bool mul_overflows(std::size_t a, std::size_t b) {
    return b != 0 &&
           a > (std::numeric_limits<std::size_t>::max() / b);
}

bool add_overflows(std::size_t a, std::size_t b) {
    return a > (std::numeric_limits<std::size_t>::max() - b);
}

bool checked_add_mul(std::size_t& acc,
                     std::size_t a,
                     std::size_t b) {
    if (mul_overflows(a, b)) {
        return false;
    }
    const std::size_t term = a * b;
    if (add_overflows(acc, term)) {
        return false;
    }
    acc += term;
    return true;
}

bool pls_batch_elems_per_job(std::size_t p,
                             std::size_t max_components,
                             std::size_t* out) {
    std::size_t elems = 0;
    if (!checked_add_mul(elems, p, p)) return false;                  // C
    if (!checked_add_mul(elems, 5, p)) return false;                  // s/w/Cw/p/outer
    if (!checked_add_mul(elems, 2 * max_components, p)) return false; // W/P
    if (!checked_add_mul(elems, 5, 1)) return false;                  // scalar batch buffers
    *out = elems;
    return true;
}

std::size_t choose_pls_batch_tile_jobs(std::size_t n_jobs,
                                       std::size_t p,
                                       std::size_t max_components) {
    std::size_t elems_per_job = 0;
    if (!pls_batch_elems_per_job(p, max_components, &elems_per_job) ||
        mul_overflows(elems_per_job, sizeof(double))) {
        return 1;
    }
    const std::size_t bytes_per_job = elems_per_job * sizeof(double);
    if (bytes_per_job == 0) {
        return 1;
    }

    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    std::size_t budget = cuda_pls_batch_tile_budget_bytes();
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess &&
        free_bytes > 0) {
        budget = std::min(budget, free_bytes / 3);
    }
    const std::size_t by_budget = std::max<std::size_t>(1, budget / bytes_per_job);
    const std::size_t by_cublas =
        static_cast<std::size_t>(std::numeric_limits<int>::max());
    return std::max<std::size_t>(
        1, std::min({n_jobs, by_budget, by_cublas}));
}

long long as_cublas_stride(std::size_t value) {
    const auto max_ll =
        static_cast<unsigned long long>(std::numeric_limits<long long>::max());
    if (static_cast<unsigned long long>(value) > max_ll) {
        throw std::runtime_error("cuBLAS strided-batched stride exceeds range");
    }
    return static_cast<long long>(value);
}

void cublas_copy_contiguous(const double* src,
                            double* dst,
                            std::size_t n,
                            const char* what) {
    std::size_t offset = 0;
    std::size_t remaining = n;
    while (remaining > 0) {
        const std::size_t chunk =
            std::min<std::size_t>(
                remaining,
                static_cast<std::size_t>(std::numeric_limits<int>::max()));
        const int chunk_i = static_cast<int>(chunk);
        check_cublas(
            cublasDcopy_v2(
                state().handle, chunk_i, src + offset, 1, dst + offset, 1),
            what);
        offset += chunk;
        remaining -= chunk;
    }
}

class CudaStream {
public:
    CudaStream() {
        if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) !=
            cudaSuccess) {
            stream_ = nullptr;
            throw std::runtime_error("cudaStreamCreateWithFlags failed");
        }
    }

    ~CudaStream() noexcept {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }

    CudaStream(const CudaStream&)            = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    cudaStream_t get() const noexcept { return stream_; }

private:
    cudaStream_t stream_{nullptr};
};

class LocalCublasHandle {
public:
    explicit LocalCublasHandle(cudaStream_t stream) {
        if (cudaSetDevice(0) != cudaSuccess) {
            throw std::runtime_error("cudaSetDevice failed");
        }
        if (cublasCreate_v2(&handle_) != CUBLAS_STATUS_SUCCESS) {
            handle_ = nullptr;
            throw std::runtime_error("cublasCreate_v2 failed");
        }
        check_cublas(cublasSetStream_v2(handle_, stream),
                     "cublasSetStream_v2");
    }

    ~LocalCublasHandle() noexcept {
        if (handle_ != nullptr) {
            cublasDestroy_v2(handle_);
        }
    }

    LocalCublasHandle(const LocalCublasHandle&)            = delete;
    LocalCublasHandle& operator=(const LocalCublasHandle&) = delete;

    cublasHandle_t get() const noexcept { return handle_; }

private:
    cublasHandle_t handle_{nullptr};
};

int pls1_moment_components_with_workspace(std::size_t p,
                                          std::size_t max_components,
                                          const double* C,
                                          const double* s,
                                          double yy,
                                          double eps,
                                          double* W,
                                          double* P,
                                          double* Q,
                                          double* yy_out,
                                          cublasHandle_t handle,
                                          cudaStream_t stream,
                                          Pls1MomentWorkspaceView workspace,
                                          std::string* error) {
    const int pi = static_cast<int>(p);
    const int ai = static_cast<int>(max_components);
    double* const dC = workspace.dC;
    double* const ds = workspace.ds;
    double* const dw = workspace.dw;
    double* const dcw = workspace.dcw;
    double* const dp_load = workspace.dp_load;
    double* const dW = workspace.dW;
    double* const dP = workspace.dP;
    copy_h2d_stream(dC, C, p * p * sizeof(double), stream);
    copy_h2d_stream(ds, s, p * sizeof(double), stream);

    double current_yy = yy;
    const double one = 1.0;
    const double zero = 0.0;

    for (std::size_t comp = 0; comp < max_components; ++comp) {
        if (current_yy <= eps || !std::isfinite(current_yy)) {
            set_error(error, "CUDA PLS1 moment Y residual vanished");
            return 1;
        }

        check_cublas(
            cublasDcopy_v2(handle, pi, ds, 1, dw, 1),
            "cublasDcopy_v2");
        double inv_yy = 1.0 / current_yy;
        check_cublas(
            cublasDscal_v2(handle, pi, &inv_yy, dw, 1),
            "cublasDscal_v2");

        double w_norm = 0.0;
        check_cublas(
            cublasDnrm2_v2(handle, pi, dw, 1, &w_norm),
            "cublasDnrm2_v2");
        if (w_norm <= eps || !std::isfinite(w_norm)) {
            set_error(error, "CUDA PLS1 moment X weights vanished");
            return 1;
        }
        double inv_w_norm = 1.0 / (w_norm + eps);
        check_cublas(
            cublasDscal_v2(handle, pi, &inv_w_norm, dw, 1),
            "cublasDscal_v2");

        int sign_idx = 0;
        check_cublas(
            cublasIdamax_v2(handle, pi, dw, 1, &sign_idx),
            "cublasIdamax_v2");
        if (sign_idx <= 0) {
            set_error(error, "CUDA PLS1 moment sign index failed");
            return 1;
        }
        double sign_value = 0.0;
        copy_d2h_stream_sync(
            &sign_value,
            dw + static_cast<std::size_t>(sign_idx - 1),
            sizeof(double), stream);
        if (sign_value < 0.0) {
            double minus_one = -1.0;
            check_cublas(
                cublasDscal_v2(handle, pi, &minus_one, dw, 1),
                "cublasDscal_v2");
        }

        check_cublas(
            cublasDgemv_v2(handle,
                           CUBLAS_OP_T,
                           pi,
                           pi,
                           &one,
                           dC,
                           pi,
                           dw,
                           1,
                           &zero,
                           dcw,
                           1),
            "cublasDgemv_v2");

        double tt = 0.0;
        check_cublas(
            cublasDdot_v2(handle, pi, dw, 1,
                          dcw, 1, &tt),
            "cublasDdot_v2");
        if (tt <= eps || !std::isfinite(tt)) {
            set_error(error, "CUDA PLS1 moment X score vanished");
            return 1;
        }

        double q_dot = 0.0;
        check_cublas(
            cublasDdot_v2(handle, pi, dw, 1,
                          ds, 1, &q_dot),
            "cublasDdot_v2");
        const double q_load = q_dot / tt;
        if (!std::isfinite(q_load)) {
            set_error(error, "CUDA PLS1 moment Y loading is not finite");
            return 1;
        }
        Q[comp] = q_load;

        check_cublas(
            cublasDcopy_v2(handle, pi, dcw, 1,
                           dp_load, 1),
            "cublasDcopy_v2");
        double inv_tt = 1.0 / tt;
        check_cublas(
            cublasDscal_v2(handle, pi, &inv_tt, dp_load, 1),
            "cublasDscal_v2");

        check_cublas(
            cublasDcopy_v2(handle, pi, dw, 1,
                           dW + comp, ai),
            "cublasDcopy_v2");
        check_cublas(
            cublasDcopy_v2(handle, pi, dp_load, 1,
                           dP + comp, ai),
            "cublasDcopy_v2");

        double neg_tt = -tt;
        check_cublas(
            cublasDger_v2(handle,
                          pi,
                          pi,
                          &neg_tt,
                          dp_load,
                          1,
                          dp_load,
                          1,
                          dC,
                          pi),
            "cublasDger_v2");

        double score_alpha = -tt * q_load;
        check_cublas(
            cublasDaxpy_v2(handle, pi, &score_alpha,
                           dp_load, 1, ds, 1),
            "cublasDaxpy_v2");

        current_yy -= tt * q_load * q_load;
        if (current_yy < 0.0 && current_yy > -1e-9) {
            current_yy = 0.0;
        }
    }
    if (yy_out != nullptr) {
        *yy_out = current_yy;
    }
    copy_d2h_stream_sync(W, dW, p * max_components * sizeof(double),
                         stream);
    copy_d2h_stream_sync(P, dP, p * max_components * sizeof(double),
                         stream);
    return 0;
}

int pls1_moment_components_with_local_stream(std::size_t p,
                                             std::size_t max_components,
                                             const double* C,
                                             const double* s,
                                             double yy,
                                             double eps,
                                             double* W,
                                             double* P,
                                             double* Q,
                                             double* yy_out,
                                             std::string* error) {
    CudaStream stream;
    LocalCublasHandle handle(stream.get());
    DevicePtr<double> dC(p * p);
    DevicePtr<double> ds(p);
    DevicePtr<double> dw(p);
    DevicePtr<double> dcw(p);
    DevicePtr<double> dp_load(p);
    DevicePtr<double> dW(p * max_components);
    DevicePtr<double> dP(p * max_components);

    const int status = pls1_moment_components_with_workspace(
        p, max_components, C, s, yy, eps, W, P, Q, yy_out,
        handle.get(), stream.get(),
        workspace_view(dC, ds, dw, dcw, dp_load, dW, dP), error);
    if (cudaStreamSynchronize(stream.get()) != cudaSuccess) {
        set_error(error, "CUDA PLS1 moment fold stream synchronization failed");
        return 2;
    }
    return status;
}

int pls1_moment_components_many_sequential(std::size_t n_jobs,
                                           std::size_t p,
                                           std::size_t max_components,
                                           const double* const* C,
                                           const double* const* s,
                                           const double* yy,
                                           double eps,
                                           double* const* W,
                                           double* const* P,
                                           double* const* Q,
                                           double* yy_out,
                                           std::string* error) {
    thread_local ReusablePls1MomentWorkspace reusable_workspace;
    const Pls1MomentWorkspaceView workspace =
        reusable_workspace.ensure(p, max_components);

    for (std::size_t job = 0; job < n_jobs; ++job) {
        double local_yy_out = 0.0;
        const int status = pls1_moment_components_with_workspace(
            p, max_components, C[job], s[job], yy[job], eps,
            W[job], P[job], Q[job],
            yy_out == nullptr ? &local_yy_out : &yy_out[job],
            state().handle, nullptr, workspace, error);
        if (status != 0) {
            if (error != nullptr && !error->empty()) {
                *error += " in fold workspace job " + std::to_string(job);
            }
            return status;
        }
    }
    return 0;
}

int pls1_moment_components_many_batched_tiled(std::size_t n_jobs,
                                              std::size_t p,
                                              std::size_t max_components,
                                              const double* const* C,
                                              const double* const* s,
                                              const double* yy,
                                              double eps,
                                              double* const* W,
                                              double* const* P,
                                              double* const* Q,
                                              double* yy_out,
                                              std::string* error) {
    const int pi = static_cast<int>(p);
    const std::size_t pp = p * p;
    const std::size_t pa = p * max_components;
    const long long stride_C = as_cublas_stride(pp);
    const long long stride_vec = as_cublas_stride(p);

    thread_local ReusablePls1MomentBatchWorkspace reusable_workspace;
    const std::size_t max_tile_jobs =
        choose_pls_batch_tile_jobs(n_jobs, p, max_components);

    std::vector<double> hC;
    std::vector<double> hs;
    std::vector<double> hW;
    std::vector<double> hP;
    std::vector<double> current_yy;
    std::vector<double> h_norm_sq;
    std::vector<double> h_scale;
    std::vector<double> h_tt;
    std::vector<double> h_qdot;
    std::vector<double> h_q_load;
    std::vector<double> h_inv_tt;
    std::vector<double> h_sqrt_tt;
    std::vector<double> h_sign;

    const double one = 1.0;
    const double zero = 0.0;
    const double minus_one = -1.0;

    for (std::size_t begin = 0; begin < n_jobs; begin += max_tile_jobs) {
        const std::size_t batch =
            std::min(max_tile_jobs, n_jobs - begin);
        if (batch > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            set_error(error, "CUDA PLS1 moment batch exceeds cuBLAS int range");
            return 2;
        }
        const int batch_i = static_cast<int>(batch);
        const Pls1MomentBatchWorkspaceView workspace =
            reusable_workspace.ensure(batch, p, max_components);
        const std::size_t batch_vec_elems = batch * p;

        hC.assign(batch * pp, 0.0);
        hs.assign(batch * p, 0.0);
        current_yy.assign(batch, 0.0);
        h_norm_sq.assign(batch, 0.0);
        h_scale.assign(batch, 0.0);
        h_tt.assign(batch, 0.0);
        h_qdot.assign(batch, 0.0);
        h_q_load.assign(batch, 0.0);
        h_inv_tt.assign(batch, 0.0);
        h_sqrt_tt.assign(batch, 0.0);
        h_sign.assign(batch, 0.0);
        for (std::size_t local = 0; local < batch; ++local) {
            const std::size_t job = begin + local;
            std::copy(C[job], C[job] + pp, hC.data() + local * pp);
            std::copy(s[job], s[job] + p, hs.data() + local * p);
            current_yy[local] = yy[job];
        }
        copy_h2d(workspace.dC, hC.data(), hC.size() * sizeof(double));
        copy_h2d(workspace.ds, hs.data(), hs.size() * sizeof(double));

        for (std::size_t comp = 0; comp < max_components; ++comp) {
            for (std::size_t local = 0; local < batch; ++local) {
                const std::size_t job = begin + local;
                if (current_yy[local] <= eps ||
                    !std::isfinite(current_yy[local])) {
                    if (error != nullptr) {
                        *error = "CUDA PLS1 moment Y residual vanished in "
                                 "batched job " + std::to_string(job);
                    }
                    return 1;
                }
            }

            check_cublas(
                cublasDgemmStridedBatched(
                    state().handle,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    1,
                    1,
                    pi,
                    &one,
                    workspace.ds,
                    pi,
                    stride_vec,
                    workspace.ds,
                    pi,
                    stride_vec,
                    &zero,
                    workspace.dnorm_sq,
                    1,
                    1,
                    batch_i),
                "cublasDgemmStridedBatched(s norm)");

            copy_d2h(h_norm_sq.data(), workspace.dnorm_sq,
                     h_norm_sq.size() * sizeof(double));
            for (std::size_t local = 0; local < batch; ++local) {
                const std::size_t job = begin + local;
                double norm_sq = h_norm_sq[local];
                if (norm_sq < 0.0 && norm_sq > -eps) {
                    norm_sq = 0.0;
                }
                const double s_norm = std::sqrt(norm_sq);
                if (s_norm <= eps || !std::isfinite(s_norm)) {
                    if (error != nullptr) {
                        *error = "CUDA PLS1 moment X weights vanished in "
                                 "batched job " + std::to_string(job);
                    }
                    return 1;
                }
                h_scale[local] = 1.0 / (s_norm + eps * current_yy[local]);
            }
            copy_h2d(workspace.dscale, h_scale.data(),
                     h_scale.size() * sizeof(double));
            check_cublas(
                cublasDdgmm(
                    state().handle,
                    CUBLAS_SIDE_RIGHT,
                    pi,
                    batch_i,
                    workspace.ds,
                    pi,
                    workspace.dscale,
                    1,
                    workspace.dw,
                    pi),
                "cublasDdgmm(s normalized weights)");

            for (std::size_t local = 0; local < batch; ++local) {
                const std::size_t job = begin + local;
                double* const dw_j = workspace.dw + local * p;
                int sign_idx = 0;
                check_cublas(
                    cublasIdamax_v2(state().handle, pi, dw_j, 1, &sign_idx),
                    "cublasIdamax_v2");
                if (sign_idx <= 0) {
                    if (error != nullptr) {
                        *error = "CUDA PLS1 moment sign index failed in "
                                 "batched job " + std::to_string(job);
                    }
                    return 1;
                }
                check_cublas(
                    cublasDcopy_v2(
                        state().handle, 1,
                        dw_j + static_cast<std::size_t>(sign_idx - 1), 1,
                        workspace.dsign + local, 1),
                    "cublasDcopy_v2(sign gather)");
            }
            copy_d2h(h_sign.data(), workspace.dsign,
                     h_sign.size() * sizeof(double));
            for (std::size_t local = 0; local < batch; ++local) {
                double* const dw_j = workspace.dw + local * p;
                if (h_sign[local] < 0.0) {
                    double neg_one = -1.0;
                    check_cublas(
                        cublasDscal_v2(state().handle, pi, &neg_one, dw_j, 1),
                        "cublasDscal_v2");
                }
            }

            check_cublas(
                cublasDgemmStridedBatched(
                    state().handle,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    pi,
                    1,
                    pi,
                    &one,
                    workspace.dC,
                    pi,
                    stride_C,
                    workspace.dw,
                    pi,
                    stride_vec,
                    &zero,
                    workspace.dcw,
                    pi,
                    stride_vec,
                    batch_i),
                "cublasDgemmStridedBatched(C*w)");

            check_cublas(
                cublasDgemmStridedBatched(
                    state().handle,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    1,
                    1,
                    pi,
                    &one,
                    workspace.dw,
                    pi,
                    stride_vec,
                    workspace.dcw,
                    pi,
                    stride_vec,
                    &zero,
                    workspace.dtt,
                    1,
                    1,
                    batch_i),
                "cublasDgemmStridedBatched(w*Cw)");
            check_cublas(
                cublasDgemmStridedBatched(
                    state().handle,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    1,
                    1,
                    pi,
                    &one,
                    workspace.dw,
                    pi,
                    stride_vec,
                    workspace.ds,
                    pi,
                    stride_vec,
                    &zero,
                    workspace.dqdot,
                    1,
                    1,
                    batch_i),
                "cublasDgemmStridedBatched(w*s)");
            copy_d2h(h_tt.data(), workspace.dtt,
                     h_tt.size() * sizeof(double));
            copy_d2h(h_qdot.data(), workspace.dqdot,
                     h_qdot.size() * sizeof(double));

            for (std::size_t local = 0; local < batch; ++local) {
                const std::size_t job = begin + local;
                const double tt = h_tt[local];
                if (tt <= eps || !std::isfinite(tt)) {
                    if (error != nullptr) {
                        *error = "CUDA PLS1 moment X score vanished in "
                                 "batched job " + std::to_string(job);
                    }
                    return 1;
                }
                const double q_load = h_qdot[local] / tt;
                if (!std::isfinite(q_load)) {
                    if (error != nullptr) {
                        *error = "CUDA PLS1 moment Y loading is not finite in "
                                 "batched job " + std::to_string(job);
                    }
                    return 1;
                }
                Q[job][comp] = q_load;
                h_q_load[local] = q_load;
                h_inv_tt[local] = 1.0 / tt;
                h_sqrt_tt[local] = std::sqrt(tt);
            }

            copy_h2d(workspace.dscale, h_inv_tt.data(),
                     h_inv_tt.size() * sizeof(double));
            check_cublas(
                cublasDdgmm(
                    state().handle,
                    CUBLAS_SIDE_RIGHT,
                    pi,
                    batch_i,
                    workspace.dcw,
                    pi,
                    workspace.dscale,
                    1,
                    workspace.dp_load,
                    pi),
                "cublasDdgmm(Cw loadings)");

            copy_h2d(workspace.dscale, h_sqrt_tt.data(),
                     h_sqrt_tt.size() * sizeof(double));
            check_cublas(
                cublasDdgmm(
                    state().handle,
                    CUBLAS_SIDE_RIGHT,
                    pi,
                    batch_i,
                    workspace.dp_load,
                    pi,
                    workspace.dscale,
                    1,
                    workspace.douter,
                    pi),
                "cublasDdgmm(C deflation vector)");

            for (std::size_t local = 0; local < batch; ++local) {
                h_scale[local] = -h_tt[local] * h_q_load[local];
            }
            copy_h2d(workspace.dscale, h_scale.data(),
                     h_scale.size() * sizeof(double));
            check_cublas(
                cublasDdgmm(
                    state().handle,
                    CUBLAS_SIDE_RIGHT,
                    pi,
                    batch_i,
                    workspace.dp_load,
                    pi,
                    workspace.dscale,
                    1,
                    workspace.dcw,
                    pi),
                "cublasDdgmm(score deflation vector)");
            std::size_t score_update_offset = 0;
            std::size_t score_update_remaining = batch * p;
            while (score_update_remaining > 0) {
                const std::size_t chunk =
                    std::min<std::size_t>(
                        score_update_remaining,
                        static_cast<std::size_t>(
                            std::numeric_limits<int>::max()));
                const int chunk_i = static_cast<int>(chunk);
                check_cublas(
                    cublasDaxpy_v2(
                        state().handle, chunk_i, &one,
                        workspace.dcw + score_update_offset, 1,
                        workspace.ds + score_update_offset, 1),
                    "cublasDaxpy_v2(score deflation)");
                score_update_offset += chunk;
                score_update_remaining -= chunk;
            }

            const std::size_t component_tile_offset = comp * batch_vec_elems;
            cublas_copy_contiguous(
                workspace.dw, workspace.dW + component_tile_offset,
                batch_vec_elems, "cublasDcopy_v2(W tile)");
            cublas_copy_contiguous(
                workspace.dp_load, workspace.dP + component_tile_offset,
                batch_vec_elems, "cublasDcopy_v2(P tile)");

            for (std::size_t local = 0; local < batch; ++local) {
                const double tt = h_tt[local];
                const double q_load = h_q_load[local];

                current_yy[local] -= tt * q_load * q_load;
                if (current_yy[local] < 0.0 &&
                    current_yy[local] > -1e-9) {
                    current_yy[local] = 0.0;
                }
            }

            check_cublas(
                cublasDgemmStridedBatched(
                    state().handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    pi,
                    pi,
                    1,
                    &minus_one,
                    workspace.douter,
                    pi,
                    stride_vec,
                    workspace.douter,
                    pi,
                    stride_vec,
                    &one,
                    workspace.dC,
                    pi,
                    stride_C,
                    batch_i),
                "cublasDgemmStridedBatched(C deflation)");
        }

        hW.assign(batch * pa, 0.0);
        hP.assign(batch * pa, 0.0);
        copy_d2h(hW.data(), workspace.dW, hW.size() * sizeof(double));
        copy_d2h(hP.data(), workspace.dP, hP.size() * sizeof(double));
        for (std::size_t local = 0; local < batch; ++local) {
            const std::size_t job = begin + local;
            for (std::size_t comp = 0; comp < max_components; ++comp) {
                const double* const w_comp =
                    hW.data() + comp * batch_vec_elems + local * p;
                const double* const p_comp =
                    hP.data() + comp * batch_vec_elems + local * p;
                for (std::size_t feature = 0; feature < p; ++feature) {
                    W[job][feature * max_components + comp] = w_comp[feature];
                    P[job][feature * max_components + comp] = p_comp[feature];
                }
            }
            if (yy_out != nullptr) {
                yy_out[job] = current_yy[local];
            }
        }
    }
    return 0;
}

}  // namespace

bool cuda_runtime_available() noexcept {
    return state().available;
}

int pls1_moment_components(std::size_t p,
                           std::size_t max_components,
                           const double* C,
                           const double* s,
                           double yy,
                           double eps,
                           double* W,
                           double* P,
                           double* Q,
                           double* yy_out,
                           std::string* error) {
    try {
        if (error != nullptr) {
            error->clear();
        }
        if (!cuda_runtime_available()) {
            set_error(error, "CUDA PLS1 moment component loop: no GPU available");
            return 2;
        }
        if (p == 0 || max_components == 0 || C == nullptr || s == nullptr ||
            W == nullptr || P == nullptr || Q == nullptr) {
            set_error(error, "CUDA PLS1 moment component loop received invalid buffers");
            return 2;
        }
        if (p > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            set_error(error, "CUDA PLS1 moment component loop feature count exceeds cuBLAS int range");
            return 2;
        }
        thread_local ReusablePls1MomentWorkspace reusable_workspace;
        const Pls1MomentWorkspaceView workspace =
            reusable_workspace.ensure(p, max_components);
        return pls1_moment_components_with_workspace(
            p, max_components, C, s, yy, eps, W, P, Q, yy_out,
            state().handle, nullptr, workspace, error);
    } catch (const std::bad_alloc&) {
        set_error(error, "CUDA PLS1 moment component loop ran out of memory");
        return 2;
    } catch (const std::exception& ex) {
        if (error != nullptr) {
            *error = std::string("CUDA PLS1 moment component loop failed: ") +
                     ex.what();
        }
        return 2;
    } catch (...) {
        set_error(error, "CUDA PLS1 moment component loop failed");
        return 2;
    }
}

int pls1_moment_components_many(std::size_t n_jobs,
                                std::size_t p,
                                std::size_t max_components,
                                const double* const* C,
                                const double* const* s,
                                const double* yy,
                                double eps,
                                double* const* W,
                                double* const* P,
                                double* const* Q,
                                double* yy_out,
                                bool parallel_folds,
                                bool many_batched,
                                bool* used_parallel_folds,
                                bool* used_many_batched,
                                std::string* error) {
    try {
        if (error != nullptr) {
            error->clear();
        }
        if (used_parallel_folds != nullptr) {
            *used_parallel_folds = false;
        }
        if (used_many_batched != nullptr) {
            *used_many_batched = false;
        }
        if (n_jobs == 0) {
            return 0;
        }
        if (!cuda_runtime_available()) {
            set_error(error, "CUDA PLS1 moment fold workspace: no GPU available");
            return 2;
        }
        if (p == 0 || max_components == 0 || C == nullptr || s == nullptr ||
            yy == nullptr || W == nullptr || P == nullptr || Q == nullptr) {
            set_error(error, "CUDA PLS1 moment fold workspace received invalid buffers");
            return 2;
        }
        if (p > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            set_error(error, "CUDA PLS1 moment fold workspace feature count exceeds cuBLAS int range");
            return 2;
        }
        for (std::size_t job = 0; job < n_jobs; ++job) {
            if (C[job] == nullptr || s[job] == nullptr ||
                W[job] == nullptr || P[job] == nullptr || Q[job] == nullptr) {
                set_error(error, "CUDA PLS1 moment fold workspace received a null job buffer");
                return 2;
            }
        }

        const bool use_many_batched =
            !cuda_pls_many_legacy_requested() &&
            (many_batched || cuda_pls_many_batched_requested()) &&
            n_jobs > 1;
        if (use_many_batched) {
            std::string batched_error;
            try {
                const int batched_status =
                    pls1_moment_components_many_batched_tiled(
                        n_jobs, p, max_components, C, s, yy, eps,
                        W, P, Q, yy_out, &batched_error);
                if (batched_status == 0 || batched_status == 1) {
                    if (batched_status == 0 && used_many_batched != nullptr) {
                        *used_many_batched = true;
                    }
                    if (batched_status != 0 && error != nullptr) {
                        *error = batched_error;
                    }
                    return batched_status;
                }
            } catch (const std::bad_alloc&) {
                // Fall back to the requested parallel or sequential path below.
            } catch (const std::runtime_error&) {
                // Fall back when the runtime lacks a strided-batched primitive
                // or a transient cuBLAS/device error affects only this path.
            }
        }

        const bool use_parallel_folds =
            parallel_folds || cuda_pls_parallel_folds_requested(n_jobs);
        if (use_parallel_folds) {
            if (used_parallel_folds != nullptr) {
                *used_parallel_folds = true;
            }
            const std::size_t max_in_flight =
                std::min(n_jobs, kMaxParallelFoldStreams);
            for (std::size_t begin = 0; begin < n_jobs; begin += max_in_flight) {
                const std::size_t batch =
                    std::min(max_in_flight, n_jobs - begin);
                std::vector<int> statuses(batch, 2);
                std::vector<std::string> errors(batch);
                std::vector<std::thread> workers;
                workers.reserve(batch);
                try {
                    for (std::size_t local = 0; local < batch; ++local) {
                        const std::size_t job = begin + local;
                        workers.emplace_back([&, local, job]() {
                            double local_yy_out = 0.0;
                            statuses[local] =
                                pls1_moment_components_with_local_stream(
                                    p, max_components, C[job], s[job],
                                    yy[job], eps, W[job], P[job], Q[job],
                                    yy_out == nullptr ? &local_yy_out
                                                      : &yy_out[job],
                                    &errors[local]);
                        });
                    }
                } catch (...) {
                    for (auto& worker : workers) {
                        if (worker.joinable()) worker.join();
                    }
                    throw;
                }
                for (auto& worker : workers) {
                    worker.join();
                }
                for (std::size_t local = 0; local < batch; ++local) {
                    const std::size_t job = begin + local;
                    if (statuses[local] != 0) {
                        if (error != nullptr) {
                            if (!errors[local].empty()) {
                                *error = errors[local] +
                                         " in parallel fold job " +
                                         std::to_string(job);
                            } else {
                                *error =
                                    "CUDA PLS1 moment parallel fold job failed";
                            }
                        }
                        return statuses[local];
                    }
                }
            }
            return 0;
        }

        return pls1_moment_components_many_sequential(
            n_jobs, p, max_components, C, s, yy, eps,
            W, P, Q, yy_out, error);
    } catch (const std::bad_alloc&) {
        set_error(error, "CUDA PLS1 moment fold workspace ran out of memory");
        return 2;
    } catch (const std::exception& ex) {
        if (error != nullptr) {
            *error = std::string("CUDA PLS1 moment fold workspace failed: ") +
                     ex.what();
        }
        return 2;
    } catch (...) {
        set_error(error, "CUDA PLS1 moment fold workspace failed");
        return 2;
    }
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
}  // namespace n4m
