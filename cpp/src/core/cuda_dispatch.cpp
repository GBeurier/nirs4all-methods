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
#include "cuda_kernels.hpp"

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
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
    cusolverDnHandle_t solver{};
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
        if (cusolverDnCreate(&solver) != CUSOLVER_STATUS_SUCCESS) {
            cublasDestroy_v2(handle);
            handle = nullptr;
            return;
        }
        available = true;
    }

    ~CublasState() {
        if (available) {
            cusolverDnDestroy(solver);
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
    double* dinv_tt{nullptr};
    double* dsqrt_tt{nullptr};
    double* ddefl{nullptr};
    double* dcur_yy{nullptr};
    double* dQ{nullptr};
    int* dfail{nullptr};
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

class ReusableDualWorkspace {
public:
    void ensure(std::size_t n, std::size_t q, int lwork) {
        dA_.ensure(n * n);
        dB_.ensure(n * q);
        dWork_.ensure(static_cast<std::size_t>(lwork));
        dInfo_.ensure(1);
    }

    double* dA() const noexcept { return dA_.get(); }
    double* dB() const noexcept { return dB_.get(); }
    double* dWork() const noexcept { return dWork_.get(); }
    int* dInfo() const noexcept { return dInfo_.get(); }

private:
    ReusableDeviceBuffer<double> dA_;
    ReusableDeviceBuffer<double> dB_;
    ReusableDeviceBuffer<double> dWork_;
    ReusableDeviceBuffer<int> dInfo_;
};

class ReusableMomentBuildWorkspace {
public:
    void ensure(std::size_t x_elems,
                std::size_t y_elems,
                std::size_t xtx_elems,
                std::size_t xty_elems,
                std::size_t yty_elems) {
        dX_.ensure(x_elems);
        dY_.ensure(y_elems);
        dXtX_.ensure(xtx_elems);
        dXtY_.ensure(xty_elems);
        dYtY_.ensure(yty_elems);
    }

    double* dX() const noexcept { return dX_.get(); }
    double* dY() const noexcept { return dY_.get(); }
    double* dXtX() const noexcept { return dXtX_.get(); }
    double* dXtY() const noexcept { return dXtY_.get(); }
    double* dYtY() const noexcept { return dYtY_.get(); }

private:
    ReusableDeviceBuffer<double> dX_;
    ReusableDeviceBuffer<double> dY_;
    ReusableDeviceBuffer<double> dXtX_;
    ReusableDeviceBuffer<double> dXtY_;
    ReusableDeviceBuffer<double> dYtY_;
};

class ReusableGramBuildWorkspace {
public:
    void ensure(std::size_t x_elems, std::size_t k_elems) {
        dX_.ensure(x_elems);
        dK_.ensure(k_elems);
    }

    double* dX() const noexcept { return dX_.get(); }
    double* dK() const noexcept { return dK_.get(); }

private:
    ReusableDeviceBuffer<double> dX_;
    ReusableDeviceBuffer<double> dK_;
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
        dinv_tt_.ensure(n_jobs);
        dsqrt_tt_.ensure(n_jobs);
        ddefl_.ensure(n_jobs);
        dcur_yy_.ensure(n_jobs);
        dQ_.ensure(n_jobs * max_components);
        dfail_.ensure(3);
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
            dinv_tt_.get(),
            dsqrt_tt_.get(),
            ddefl_.get(),
            dcur_yy_.get(),
            dQ_.get(),
            dfail_.get(),
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
    ReusableDeviceBuffer<double> dinv_tt_;
    ReusableDeviceBuffer<double> dsqrt_tt_;
    ReusableDeviceBuffer<double> ddefl_;
    ReusableDeviceBuffer<double> dcur_yy_;
    ReusableDeviceBuffer<double> dQ_;
    ReusableDeviceBuffer<int> dfail_;
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

void check_cusolver(cusolverStatus_t st, const char* what) {
    if (st != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuSOLVER error in ") + what);
    }
}

void check_cuda(cudaError_t st, const char* what) {
    if (st != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error in ") + what + ": " +
            cudaGetErrorString(st));
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
    if (!checked_add_mul(elems, 4, 1)) return false;                  // scalar batch buffers
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

void copy_d2d_contiguous(const double* src,
                         double* dst,
                         std::size_t n,
                         const char* what) {
    if (n == 0) {
        return;
    }
    if (mul_overflows(n, sizeof(double))) {
        throw std::runtime_error(std::string("CUDA copy size overflow in ") +
                                 what);
    }
    check_cuda(
        cudaMemcpyAsync(dst, src, n * sizeof(double),
                        cudaMemcpyDeviceToDevice, nullptr),
        what);
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

    // Per-component scalar work (norm/scale/tt/qdot/q-load/inv-tt/sqrt-tt and the
    // running Y residual) now stays resident on the device: it is prepared by the
    // three scalar kernels below instead of being copied host<->device every
    // component. Only the final W/P/Q tiles, the residual and a 3-int failure
    // flag are read back, once per tile.
    std::vector<double> hC;
    std::vector<double> hs;
    std::vector<double> hW;
    std::vector<double> hP;
    std::vector<double> hQ;
    std::vector<double> h_cur_yy;

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
        for (std::size_t local = 0; local < batch; ++local) {
            const std::size_t job = begin + local;
            std::copy(C[job], C[job] + pp, hC.data() + local * pp);
            std::copy(s[job], s[job] + p, hs.data() + local * p);
        }
        copy_h2d(workspace.dC, hC.data(), hC.size() * sizeof(double));
        copy_h2d(workspace.ds, hs.data(), hs.size() * sizeof(double));
        copy_h2d(workspace.dcur_yy, yy + begin, batch * sizeof(double));
        check_cuda(cudaMemset(workspace.dfail, 0, 3 * sizeof(int)),
                   "reset PLS1 moment batch fail flag");

        for (std::size_t comp = 0; comp < max_components; ++comp) {
            const int comp_i = static_cast<int>(comp);

            // The s-norm gemm fills dnorm_sq with ||s||^2 per job; the Y-residual
            // and X-weight guards plus the normalization scale are applied on the
            // device in pls1_moment_prepare_scale_many, so the running residual
            // never round-trips to the host.
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

            check_cuda(
                pls1_moment_prepare_scale_many(
                    workspace.dnorm_sq, workspace.dcur_yy, eps, comp_i, batch,
                    workspace.dscale, workspace.dfail, nullptr),
                "PLS1 many-batched scale preparation");
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

            check_cuda(
                pls1_moment_normalize_signs_many(
                    workspace.dw, p, batch, workspace.douter, nullptr),
                "PLS1 many-batched sign normalization");
            double* signed_w = workspace.douter;

            const std::size_t component_tile_offset = comp * batch_vec_elems;
            copy_d2d_contiguous(
                signed_w, workspace.dW + component_tile_offset,
                batch_vec_elems, "cudaMemcpyAsync(W tile)");

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
                    signed_w,
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
                    signed_w,
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
                    signed_w,
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
            // X-score / Y-loading guards, the Y loading stored into the device Q
            // accumulator, and the three deflation scales 1/tt, sqrt(tt) and
            // -tt*q are all prepared on the device; tt/qdot never read back.
            check_cuda(
                pls1_moment_prepare_loadings_many(
                    workspace.dtt, workspace.dqdot, eps, comp_i, batch,
                    workspace.dQ, workspace.dinv_tt, workspace.dsqrt_tt,
                    workspace.ddefl, workspace.dfail, nullptr),
                "PLS1 many-batched loading preparation");

            check_cublas(
                cublasDdgmm(
                    state().handle,
                    CUBLAS_SIDE_RIGHT,
                    pi,
                    batch_i,
                    workspace.dcw,
                    pi,
                    workspace.dinv_tt,
                    1,
                    workspace.dp_load,
                    pi),
                "cublasDdgmm(Cw loadings)");

            check_cublas(
                cublasDdgmm(
                    state().handle,
                    CUBLAS_SIDE_RIGHT,
                    pi,
                    batch_i,
                    workspace.dp_load,
                    pi,
                    workspace.dsqrt_tt,
                    1,
                    workspace.douter,
                    pi),
                "cublasDdgmm(C deflation vector)");

            check_cublas(
                cublasDdgmm(
                    state().handle,
                    CUBLAS_SIDE_RIGHT,
                    pi,
                    batch_i,
                    workspace.dp_load,
                    pi,
                    workspace.ddefl,
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

            copy_d2d_contiguous(
                workspace.dp_load, workspace.dP + component_tile_offset,
                batch_vec_elems, "cudaMemcpyAsync(P tile)");

            // Update the device-resident Y residual yy -= tt*q*q (with the same
            // tiny-negative clamp) before the next component's residual guard.
            check_cuda(
                pls1_moment_update_yy_many(
                    workspace.dtt, workspace.dQ, comp_i, batch,
                    workspace.dcur_yy, nullptr),
                "PLS1 many-batched Y residual update");

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

        int h_fail[3] = {kPls1MomentBatchOk, 0, 0};
        copy_d2h(h_fail, workspace.dfail, sizeof(h_fail));
        if (h_fail[0] != kPls1MomentBatchOk) {
            // The scalar kernels recorded the earliest component's failure; later
            // components of this tile produced don't-care outputs that the caller
            // discards on status 1. Rebuild the legacy host error message.
            if (error != nullptr) {
                const char* reason = "Y loading is not finite";
                switch (h_fail[0]) {
                    case kPls1MomentBatchYResidualVanished:
                        reason = "Y residual vanished";
                        break;
                    case kPls1MomentBatchXWeightsVanished:
                        reason = "X weights vanished";
                        break;
                    case kPls1MomentBatchXScoreVanished:
                        reason = "X score vanished";
                        break;
                    default:
                        break;
                }
                const std::size_t job =
                    begin + static_cast<std::size_t>(h_fail[1]);
                *error = std::string("CUDA PLS1 moment ") + reason +
                         " in batched job " + std::to_string(job);
            }
            return 1;
        }

        hW.assign(batch * pa, 0.0);
        hP.assign(batch * pa, 0.0);
        hQ.assign(batch * max_components, 0.0);
        copy_d2h(hW.data(), workspace.dW, hW.size() * sizeof(double));
        copy_d2h(hP.data(), workspace.dP, hP.size() * sizeof(double));
        copy_d2h(hQ.data(), workspace.dQ, hQ.size() * sizeof(double));
        if (yy_out != nullptr) {
            h_cur_yy.assign(batch, 0.0);
            copy_d2h(h_cur_yy.data(), workspace.dcur_yy,
                     h_cur_yy.size() * sizeof(double));
        }
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
                Q[job][comp] = hQ[comp * batch + local];
            }
            if (yy_out != nullptr) {
                yy_out[job] = h_cur_yy[local];
            }
        }
    }
    return 0;
}

}  // namespace

bool cuda_runtime_available() noexcept {
    return state().available;
}

int build_moments_device(std::size_t n,
                         std::size_t p,
                         std::size_t q,
                         const double* X,
                         const double* Y,
                         double* XtX,
                         double* XtY,
                         double* YtY,
                         std::string* error) {
    try {
        if (error != nullptr) {
            error->clear();
        }
        if (!cuda_runtime_available()) {
            set_error(error, "moment build: no GPU available");
            return 2;
        }
        if (n == 0 || p == 0 || q == 0 || X == nullptr || Y == nullptr ||
            XtX == nullptr || XtY == nullptr) {
            set_error(error, "moment build received invalid buffers");
            return 2;
        }
        if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            p > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            q > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            set_error(error, "moment build dimensions exceed cuBLAS int range");
            return 2;
        }
        if (mul_overflows(n, p) || mul_overflows(n, q) ||
            mul_overflows(p, p) || mul_overflows(p, q) ||
            (YtY != nullptr && mul_overflows(q, q))) {
            set_error(error, "moment build dimensions are too large");
            return 2;
        }

        const std::size_t x_elems = n * p;
        const std::size_t y_elems = n * q;
        const std::size_t xtx_elems = p * p;
        const std::size_t xty_elems = p * q;
        const std::size_t yty_elems = (YtY != nullptr) ? (q * q) : 0U;
        if (mul_overflows(x_elems, sizeof(double)) ||
            mul_overflows(y_elems, sizeof(double)) ||
            mul_overflows(xtx_elems, sizeof(double)) ||
            mul_overflows(xty_elems, sizeof(double)) ||
            (YtY != nullptr && mul_overflows(yty_elems, sizeof(double)))) {
            set_error(error, "moment build buffers are too large");
            return 2;
        }

        thread_local ReusableMomentBuildWorkspace ws;
        ws.ensure(x_elems, y_elems, xtx_elems, xty_elems, yty_elems);

        copy_h2d(ws.dX(), X, x_elems * sizeof(double));
        copy_h2d(ws.dY(), Y, y_elems * sizeof(double));

        const double alpha = 1.0;
        const double beta = 0.0;
        const int ni = static_cast<int>(n);
        const int pi = static_cast<int>(p);
        const int qi = static_cast<int>(q);

        check_cublas(
            cublasDgemm_v2(state().handle,
                           CUBLAS_OP_N, CUBLAS_OP_T,
                           pi, pi, ni,
                           &alpha,
                           ws.dX(), pi,
                           ws.dX(), pi,
                           &beta,
                           ws.dXtX(), pi),
            "moment build XtX");
        check_cublas(
            cublasDgemm_v2(state().handle,
                           CUBLAS_OP_N, CUBLAS_OP_T,
                           qi, pi, ni,
                           &alpha,
                           ws.dY(), qi,
                           ws.dX(), pi,
                           &beta,
                           ws.dXtY(), qi),
            "moment build XtY");
        if (YtY != nullptr) {
            check_cublas(
                cublasDgemm_v2(state().handle,
                               CUBLAS_OP_N, CUBLAS_OP_T,
                               qi, qi, ni,
                               &alpha,
                               ws.dY(), qi,
                               ws.dY(), qi,
                               &beta,
                               ws.dYtY(), qi),
                "moment build YtY");
        }

        copy_d2h(XtX, ws.dXtX(), xtx_elems * sizeof(double));
        copy_d2h(XtY, ws.dXtY(), xty_elems * sizeof(double));
        if (YtY != nullptr) {
            copy_d2h(YtY, ws.dYtY(), yty_elems * sizeof(double));
        }
        return 0;
    } catch (const std::bad_alloc&) {
        set_error(error, "moment build ran out of memory");
        return 2;
    } catch (const std::exception& ex) {
        if (error != nullptr) {
            *error = std::string("moment build failed: ") + ex.what();
        }
        return 2;
    } catch (...) {
        set_error(error, "moment build failed");
        return 2;
    }
}

int build_gram_device(std::size_t n,
                      std::size_t p,
                      const double* Xw,
                      double* K,
                      std::string* error) {
    try {
        if (error != nullptr) {
            error->clear();
        }
        if (!cuda_runtime_available()) {
            set_error(error, "gram build: no GPU available");
            return 2;
        }
        if (n == 0 || p == 0 || Xw == nullptr || K == nullptr) {
            set_error(error, "gram build received invalid buffers");
            return 2;
        }
        if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            p > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            set_error(error, "gram build dimensions exceed cuBLAS int range");
            return 2;
        }
        if (mul_overflows(n, p) || mul_overflows(n, n)) {
            set_error(error, "gram build dimensions are too large");
            return 2;
        }

        const std::size_t x_elems = n * p;
        const std::size_t k_elems = n * n;
        if (mul_overflows(x_elems, sizeof(double)) ||
            mul_overflows(k_elems, sizeof(double))) {
            set_error(error, "gram build buffers are too large");
            return 2;
        }

        thread_local ReusableGramBuildWorkspace ws;
        ws.ensure(x_elems, k_elems);

        copy_h2d(ws.dX(), Xw, x_elems * sizeof(double));

        const double alpha = 1.0;
        const double beta = 0.0;
        const int ni = static_cast<int>(n);
        const int pi = static_cast<int>(p);

        check_cublas(
            cublasDgemm_v2(state().handle,
                           CUBLAS_OP_T, CUBLAS_OP_N,
                           ni, ni, pi,
                           &alpha,
                           ws.dX(), pi,
                           ws.dX(), pi,
                           &beta,
                           ws.dK(), ni),
            "gram build K");

        copy_d2h(K, ws.dK(), k_elems * sizeof(double));
        return 0;
    } catch (const std::bad_alloc&) {
        set_error(error, "gram build ran out of memory");
        return 2;
    } catch (const std::exception& ex) {
        if (error != nullptr) {
            *error = std::string("gram build failed: ") + ex.what();
        }
        return 2;
    } catch (...) {
        set_error(error, "gram build failed");
        return 2;
    }
}

int spd_solve(std::size_t n,
              std::size_t q,
              const double* A,
              const double* B,
              double* X,
              std::string* error) {
    try {
        if (error != nullptr) {
            error->clear();
        }
        if (!cuda_runtime_available()) {
            set_error(error, "ridge SPD solve: no GPU available");
            return 2;
        }
        if (n == 0 || q == 0 || A == nullptr || B == nullptr || X == nullptr) {
            set_error(error, "ridge SPD solve received invalid buffers");
            return 2;
        }
        if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            q > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            set_error(error, "ridge SPD solve dimensions exceed cuSOLVER int range");
            return 2;
        }
        if (mul_overflows(n, n) || mul_overflows(n, q)) {
            set_error(error, "ridge SPD solve dimensions are too large");
            return 2;
        }
        const std::size_t a_elems = n * n;
        const std::size_t bx_elems = n * q;
        if (mul_overflows(a_elems, sizeof(double)) ||
            mul_overflows(bx_elems, sizeof(double))) {
            set_error(error, "ridge SPD solve buffers are too large");
            return 2;
        }

        const int ni = static_cast<int>(n);
        const int qi = static_cast<int>(q);
        DevicePtr<double> dA(a_elems);
        DevicePtr<double> dB(bx_elems);
        DevicePtr<int> dInfo(1);

        copy_h2d(dA.get(), A, a_elems * sizeof(double));

        std::vector<double> Bcm(bx_elems, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t target = 0; target < q; ++target) {
                Bcm[target * n + i] = B[i * q + target];
            }
        }
        copy_h2d(dB.get(), Bcm.data(), bx_elems * sizeof(double));

        // A is fully materialized and symmetric, so either triangle is valid.
        const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
        int lwork = 0;
        check_cusolver(
            cusolverDnDpotrf_bufferSize(state().solver, uplo, ni, dA.get(), ni,
                                        &lwork),
            "potrf_bufferSize");
        DevicePtr<double> dWork(static_cast<std::size_t>(lwork));
        check_cusolver(
            cusolverDnDpotrf(state().solver, uplo, ni, dA.get(), ni,
                             dWork.get(), lwork, dInfo.get()),
            "potrf");
        int info = 0;
        copy_d2h(&info, dInfo.get(), sizeof(int));
        if (info > 0) {
            set_error(error, "ridge SPD solve: matrix not positive definite");
            return 1;
        }
        if (info < 0) {
            set_error(error, "ridge SPD solve: potrf invalid argument");
            return 2;
        }

        check_cusolver(
            cusolverDnDpotrs(state().solver, uplo, ni, qi, dA.get(), ni,
                             dB.get(), ni, dInfo.get()),
            "potrs");
        copy_d2h(&info, dInfo.get(), sizeof(int));
        if (info != 0) {
            set_error(error, "ridge SPD solve: potrs failed");
            return 2;
        }

        std::vector<double> Xcm(bx_elems, 0.0);
        copy_d2h(Xcm.data(), dB.get(), bx_elems * sizeof(double));
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t target = 0; target < q; ++target) {
                X[i * q + target] = Xcm[target * n + i];
            }
        }
        return 0;
    } catch (const std::bad_alloc&) {
        set_error(error, "ridge SPD solve ran out of memory");
        return 2;
    } catch (const std::exception& ex) {
        if (error != nullptr) {
            *error = std::string("ridge SPD solve failed: ") + ex.what();
        }
        return 2;
    } catch (...) {
        set_error(error, "ridge SPD solve failed");
        return 2;
    }
}

struct PreparedDualRidge {
    std::size_t n{0};
    std::size_t q{0};
    int lwork{0};
    ReusableDeviceBuffer<double> dK;
    ReusableDeviceBuffer<double> dB0;
};

PreparedDualRidge* prepare_dual_ridge(std::size_t n,
                                      std::size_t q,
                                      const double* K,
                                      const double* B,
                                      std::string* error) {
    try {
        if (error != nullptr) {
            error->clear();
        }
        if (!cuda_runtime_available()) {
            set_error(error, "prepared dual Ridge: no GPU available");
            return nullptr;
        }
        if (n == 0 || q == 0 || K == nullptr || B == nullptr) {
            set_error(error, "prepared dual Ridge received invalid buffers");
            return nullptr;
        }
        if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            q > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            set_error(error,
                      "prepared dual Ridge dimensions exceed cuSOLVER int range");
            return nullptr;
        }
        if (mul_overflows(n, n) || mul_overflows(n, q)) {
            set_error(error, "prepared dual Ridge dimensions are too large");
            return nullptr;
        }
        const std::size_t k_elems = n * n;
        const std::size_t b_elems = n * q;
        if (mul_overflows(k_elems, sizeof(double)) ||
            mul_overflows(b_elems, sizeof(double))) {
            set_error(error, "prepared dual Ridge buffers are too large");
            return nullptr;
        }

        auto h = std::make_unique<PreparedDualRidge>();
        h->n = n;
        h->q = q;
        h->dK.ensure(k_elems);
        copy_h2d(h->dK.get(), K, k_elems * sizeof(double));

        std::vector<double> Bcm(b_elems, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t target = 0; target < q; ++target) {
                Bcm[target * n + i] = B[i * q + target];
            }
        }
        h->dB0.ensure(b_elems);
        copy_h2d(h->dB0.get(), Bcm.data(), b_elems * sizeof(double));

        const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
        const int ni = static_cast<int>(n);
        check_cusolver(
            cusolverDnDpotrf_bufferSize(state().solver, uplo, ni,
                                        h->dK.get(), ni, &h->lwork),
            "prepared dual Ridge potrf_bufferSize");
        return h.release();
    } catch (const std::bad_alloc&) {
        set_error(error, "prepared dual Ridge ran out of memory");
        return nullptr;
    } catch (const std::exception& ex) {
        if (error != nullptr) {
            *error = std::string("prepared dual Ridge failed: ") + ex.what();
        }
        return nullptr;
    } catch (...) {
        set_error(error, "prepared dual Ridge failed");
        return nullptr;
    }
}

int dual_ridge_solve(PreparedDualRidge* handle,
                     double lambda,
                     double* X,
                     std::string* error) {
    try {
        if (error != nullptr) {
            error->clear();
        }
        if (!cuda_runtime_available()) {
            set_error(error, "prepared dual Ridge solve: no GPU available");
            return 2;
        }
        if (handle == nullptr || X == nullptr || handle->n == 0 ||
            handle->q == 0 || handle->lwork < 0) {
            set_error(error, "prepared dual Ridge solve received invalid handle");
            return 2;
        }
        const std::size_t n = handle->n;
        const std::size_t q = handle->q;
        const std::size_t k_elems = n * n;
        const std::size_t b_elems = n * q;
        const int ni = static_cast<int>(n);
        const int qi = static_cast<int>(q);
        const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

        thread_local ReusableDualWorkspace ws;
        // A and B are fully overwritten before every solve. If a previous fold
        // made the thread-local buffers larger, cuSOLVER only sees the leading
        // ni-by-ni / ni-by-qi regions with ld=ni.
        ws.ensure(n, q, handle->lwork);

        copy_d2d_contiguous(handle->dK.get(), ws.dA(), k_elems,
                            "dual_ridge K->A");
        check_cuda(add_scaled_identity(ws.dA(), n, lambda, nullptr),
                   "add_scaled_identity");
        copy_d2d_contiguous(handle->dB0.get(), ws.dB(), b_elems,
                            "dual_ridge B0->B");

        check_cusolver(
            cusolverDnDpotrf(state().solver, uplo, ni, ws.dA(), ni,
                             ws.dWork(), handle->lwork, ws.dInfo()),
            "prepared dual Ridge potrf");
        int info = 0;
        copy_d2h(&info, ws.dInfo(), sizeof(int));
        if (info > 0) {
            set_error(error,
                      "prepared dual Ridge solve: matrix not positive definite");
            return 1;
        }
        if (info < 0) {
            set_error(error,
                      "prepared dual Ridge solve: potrf invalid argument");
            return 2;
        }

        check_cusolver(
            cusolverDnDpotrs(state().solver, uplo, ni, qi, ws.dA(), ni,
                             ws.dB(), ni, ws.dInfo()),
            "prepared dual Ridge potrs");
        copy_d2h(&info, ws.dInfo(), sizeof(int));
        if (info != 0) {
            set_error(error, "prepared dual Ridge solve: potrs failed");
            return 2;
        }

        std::vector<double> Xcm(b_elems, 0.0);
        copy_d2h(Xcm.data(), ws.dB(), b_elems * sizeof(double));
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t target = 0; target < q; ++target) {
                X[i * q + target] = Xcm[target * n + i];
            }
        }
        return 0;
    } catch (const std::bad_alloc&) {
        set_error(error, "prepared dual Ridge solve ran out of memory");
        return 2;
    } catch (const std::exception& ex) {
        if (error != nullptr) {
            *error = std::string("prepared dual Ridge solve failed: ") +
                     ex.what();
        }
        return 2;
    } catch (...) {
        set_error(error, "prepared dual Ridge solve failed");
        return 2;
    }
}

void destroy_dual_ridge(PreparedDualRidge* handle) noexcept {
    delete handle;
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
