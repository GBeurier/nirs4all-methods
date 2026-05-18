// SPDX-License-Identifier: CECILL-2.1

#include "core/context.hpp"

#include <cstdarg>
#include <cstdio>

#if defined(C4A_USE_CUDA)
#  include "core/cuda_dispatch.hpp"
#endif

namespace chemometrics4all::core {

Context::Context() noexcept
    : seed_(0),
      backend_(C4A_BACKEND_AUTO),
      num_threads_(0),
      user_data_(nullptr),
      error_buf_{} {
    error_buf_[0] = '\0';
}

c4a_status_t Context::set_backend(c4a_backend_t backend) noexcept {
    switch (backend) {
        case C4A_BACKEND_AUTO:
        case C4A_BACKEND_REFERENCE_CPU:
            backend_ = backend;
            return C4A_OK;
        case C4A_BACKEND_CUDA:
#if defined(C4A_USE_CUDA)
            if (!::chemometrics4all::cuda_dispatch::cuda_runtime_available()) {
                set_errorf("C4A_BACKEND_CUDA: no CUDA-capable GPU available at runtime");
                return C4A_ERR_BACKEND_UNAVAILABLE;
            }
            backend_ = backend;
            return C4A_OK;
#else
            set_errorf("backend %d is not compiled into this build of libc4a", static_cast<int>(backend));
            return C4A_ERR_BACKEND_UNAVAILABLE;
#endif
        case C4A_BACKEND_NATIVE_CPU:
        case C4A_BACKEND_BLAS:
        case C4A_BACKEND_OPENMP:
        case C4A_BACKEND_OPENCL:
        case C4A_BACKEND_METAL:
            // Phase 0 ships REFERENCE_CPU only. Future backends fail
            // softly with a status code; the caller can re-try with
            // C4A_BACKEND_AUTO or REFERENCE_CPU.
            set_errorf("backend %d is not compiled into this build of libc4a", static_cast<int>(backend));
            return C4A_ERR_BACKEND_UNAVAILABLE;
        default:
            set_errorf("invalid backend value %d", static_cast<int>(backend));
            return C4A_ERR_INVALID_ARGUMENT;
    }
}

void Context::set_error(const char* msg) noexcept {
    if (msg == nullptr) {
        error_buf_[0] = '\0';
        return;
    }
    std::size_t i = 0;
    while (i + 1 < sizeof(error_buf_) && msg[i] != '\0') {
        error_buf_[i] = msg[i];
        ++i;
    }
    error_buf_[i] = '\0';
}

void Context::set_errorf(const char* fmt, ...) noexcept {
    if (fmt == nullptr) {
        error_buf_[0] = '\0';
        return;
    }
    std::va_list ap;
    va_start(ap, fmt);
    // vsnprintf always NUL-terminates if the buffer size > 0.
    const int written = std::vsnprintf(error_buf_, sizeof(error_buf_), fmt, ap);
    va_end(ap);
    if (written < 0) {
        // Format error — fall back to a safe sentinel string.
        std::snprintf(error_buf_, sizeof(error_buf_), "internal: failed to format error message");
    }
}

}  // namespace chemometrics4all::core
