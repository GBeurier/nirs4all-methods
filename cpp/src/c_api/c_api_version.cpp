// SPDX-License-Identifier: CeCILL-2.1
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

#if defined(P4A_USE_CUDA)
#  include "core/cuda_dispatch.hpp"
#endif

#if defined(__GNUC__) || defined(__clang__)
#  define P4A_NO_UBSAN __attribute__((no_sanitize("undefined")))
#else
#  define P4A_NO_UBSAN
#endif

extern "C" {

P4A_API P4A_NO_UBSAN const char* p4a_status_to_string(p4a_status_t s) {
    try {
        return ::pls4all::core::status_to_string(s);
    } catch (...) {
        return "internal error";
    }
}

P4A_API int p4a_backend_is_available(p4a_backend_t backend) {
    try {
        switch (backend) {
            case P4A_BACKEND_AUTO:
            case P4A_BACKEND_REFERENCE_CPU:
                return 1;
            case P4A_BACKEND_BLAS:
#if defined(P4A_USE_BLAS)
                return 1;
#else
                return 0;
#endif
            case P4A_BACKEND_OPENMP:
#if defined(P4A_USE_OPENMP)
                return 1;
#else
                return 0;
#endif
            case P4A_BACKEND_CUDA:
#if defined(P4A_USE_CUDA)
                return ::pls4all::cuda_dispatch::cuda_runtime_available() ? 1 : 0;
#else
                return 0;
#endif
            case P4A_BACKEND_NATIVE_CPU:
            case P4A_BACKEND_OPENCL:
            case P4A_BACKEND_METAL:
            default:
                return 0;
        }
    } catch (...) {
        return 0;
    }
}

P4A_API const char* p4a_backend_to_string(p4a_backend_t backend) {
    try {
        switch (backend) {
            case P4A_BACKEND_AUTO:          return "auto";
            case P4A_BACKEND_REFERENCE_CPU: return "reference_cpu";
            case P4A_BACKEND_NATIVE_CPU:    return "native_cpu";
            case P4A_BACKEND_BLAS:          return "blas";
            case P4A_BACKEND_OPENMP:        return "openmp";
            case P4A_BACKEND_CUDA:          return "cuda";
            case P4A_BACKEND_OPENCL:        return "opencl";
            case P4A_BACKEND_METAL:         return "metal";
        }
        return "unknown_backend";
    } catch (...) {
        return "unknown_backend";
    }
}

P4A_API size_t p4a_dtype_size(p4a_dtype_t dtype) {
    try {
        switch (dtype) {
            case P4A_DTYPE_F64: return 8;
            case P4A_DTYPE_F32: return 4;
            case P4A_DTYPE_I32: return 4;
            case P4A_DTYPE_I64: return 8;
            case P4A_DTYPE_UNKNOWN: return 0;
        }
        return 0;
    } catch (...) {
        return 0;
    }
}

P4A_API const char* p4a_dtype_to_string(p4a_dtype_t dtype) {
    try {
        switch (dtype) {
            case P4A_DTYPE_F64:     return "f64";
            case P4A_DTYPE_F32:     return "f32";
            case P4A_DTYPE_I32:     return "i32";
            case P4A_DTYPE_I64:     return "i64";
            case P4A_DTYPE_UNKNOWN: return "unknown";
        }
        return "unknown_dtype";
    } catch (...) {
        return "unknown_dtype";
    }
}

P4A_API uint32_t p4a_get_abi_version_major(void) {
    try { return ::pls4all::core::abi_major(); } catch (...) { return 0; }
}
P4A_API uint32_t p4a_get_abi_version_minor(void) {
    try { return ::pls4all::core::abi_minor(); } catch (...) { return 0; }
}
P4A_API uint32_t p4a_get_abi_version_patch(void) {
    try { return ::pls4all::core::abi_patch(); } catch (...) { return 0; }
}
P4A_API uint32_t p4a_get_abi_version_int(void) {
    try { return ::pls4all::core::abi_int(); } catch (...) { return 0; }
}

P4A_API const char* p4a_get_version_string(void) {
    try { return ::pls4all::core::version_string(); } catch (...) { return ""; }
}
P4A_API const char* p4a_get_build_info(void) {
    try { return ::pls4all::core::build_info(); } catch (...) { return ""; }
}
P4A_API const char* p4a_get_git_revision(void) {
    try { return ::pls4all::core::git_revision(); } catch (...) { return ""; }
}

P4A_API p4a_status_t p4a_check_abi_compatibility(
    uint32_t header_abi_major,
    uint32_t header_abi_minor) {
    try {
        if (header_abi_major != ::pls4all::core::abi_major()) {
            return P4A_ERR_ABI_MISMATCH;
        }
        if (header_abi_minor > ::pls4all::core::abi_minor()) {
            return P4A_ERR_VERSION_INCOMPATIBLE;
        }
        return P4A_OK;
    } catch (...) {
        return P4A_ERR_INTERNAL;
    }
}

}  // extern "C"

#undef P4A_NO_UBSAN
