// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for the version / status / dtype / backend / ABI
// queries. None of these functions allocate or call into user code, so they
// cannot in practice throw — but we add a defensive try/catch (...) on every
// path so a future implementation change cannot accidentally cross the
// C boundary.

#include <stddef.h>
#include <stdint.h>

#include "pls4all/p4a.h"

#include "core/status.hpp"
#include "core/version.hpp"

#if defined(N4M_USE_CUDA)
#  include "core/cuda_dispatch.hpp"
#endif

#if defined(__GNUC__) || defined(__clang__)
#  define N4M_NO_UBSAN __attribute__((no_sanitize("undefined")))
#else
#  define N4M_NO_UBSAN
#endif

extern "C" {

N4M_API N4M_NO_UBSAN const char* n4m_status_to_string(n4m_status_t s) {
    try {
        return ::n4m::core::status_to_string(s);
    } catch (...) {
        return "internal error";
    }
}

N4M_API int n4m_backend_is_available(n4m_backend_t backend) {
    try {
        switch (backend) {
            case N4M_BACKEND_AUTO:
            case N4M_BACKEND_REFERENCE_CPU:
                return 1;
            case N4M_BACKEND_BLAS:
#if defined(N4M_USE_BLAS)
                return 1;
#else
                return 0;
#endif
            case N4M_BACKEND_OPENMP:
#if defined(N4M_USE_OPENMP)
                return 1;
#else
                return 0;
#endif
            case N4M_BACKEND_CUDA:
#if defined(N4M_USE_CUDA)
                return ::n4m::cuda_dispatch::cuda_runtime_available() ? 1 : 0;
#else
                return 0;
#endif
            case N4M_BACKEND_NATIVE_CPU:
            case N4M_BACKEND_OPENCL:
            case N4M_BACKEND_METAL:
            default:
                return 0;
        }
    } catch (...) {
        return 0;
    }
}

N4M_API const char* n4m_backend_to_string(n4m_backend_t backend) {
    try {
        switch (backend) {
            case N4M_BACKEND_AUTO:          return "auto";
            case N4M_BACKEND_REFERENCE_CPU: return "reference_cpu";
            case N4M_BACKEND_NATIVE_CPU:    return "native_cpu";
            case N4M_BACKEND_BLAS:          return "blas";
            case N4M_BACKEND_OPENMP:        return "openmp";
            case N4M_BACKEND_CUDA:          return "cuda";
            case N4M_BACKEND_OPENCL:        return "opencl";
            case N4M_BACKEND_METAL:         return "metal";
        }
        return "unknown_backend";
    } catch (...) {
        return "unknown_backend";
    }
}

N4M_API size_t n4m_dtype_size(n4m_dtype_t dtype) {
    try {
        switch (dtype) {
            case N4M_DTYPE_F64: return 8;
            case N4M_DTYPE_F32: return 4;
            case N4M_DTYPE_I32: return 4;
            case N4M_DTYPE_I64: return 8;
            case N4M_DTYPE_UNKNOWN: return 0;
        }
        return 0;
    } catch (...) {
        return 0;
    }
}

N4M_API const char* n4m_dtype_to_string(n4m_dtype_t dtype) {
    try {
        switch (dtype) {
            case N4M_DTYPE_F64:     return "f64";
            case N4M_DTYPE_F32:     return "f32";
            case N4M_DTYPE_I32:     return "i32";
            case N4M_DTYPE_I64:     return "i64";
            case N4M_DTYPE_UNKNOWN: return "unknown";
        }
        return "unknown_dtype";
    } catch (...) {
        return "unknown_dtype";
    }
}

N4M_API uint32_t n4m_get_abi_version_major(void) {
    try { return ::n4m::core::abi_major(); } catch (...) { return 0; }
}
N4M_API uint32_t n4m_get_abi_version_minor(void) {
    try { return ::n4m::core::abi_minor(); } catch (...) { return 0; }
}
N4M_API uint32_t n4m_get_abi_version_patch(void) {
    try { return ::n4m::core::abi_patch(); } catch (...) { return 0; }
}
N4M_API uint32_t n4m_get_abi_version_int(void) {
    try { return ::n4m::core::abi_int(); } catch (...) { return 0; }
}

N4M_API const char* n4m_get_version_string(void) {
    try { return ::n4m::core::version_string(); } catch (...) { return ""; }
}
N4M_API const char* n4m_get_build_info(void) {
    try { return ::n4m::core::build_info(); } catch (...) { return ""; }
}
N4M_API const char* n4m_get_git_revision(void) {
    try { return ::n4m::core::git_revision(); } catch (...) { return ""; }
}

N4M_API n4m_status_t n4m_check_abi_compatibility(
    uint32_t header_abi_major,
    uint32_t header_abi_minor) {
    try {
        if (header_abi_major != ::n4m::core::abi_major()) {
            return N4M_ERR_ABI_MISMATCH;
        }
        if (header_abi_minor > ::n4m::core::abi_minor()) {
            return N4M_ERR_VERSION_INCOMPATIBLE;
        }
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

}  // extern "C"

#undef N4M_NO_UBSAN
