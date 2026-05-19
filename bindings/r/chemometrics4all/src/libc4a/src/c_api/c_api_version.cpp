// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for the version / status / dtype / backend / ABI
// queries. None of these functions allocate or call into user code, so they
// cannot in practice throw — but we add a defensive try/catch (...) on every
// path so a future implementation change cannot accidentally cross the
// C boundary.

#include <stddef.h>
#include <stdint.h>

#include "chemometrics4all/c4a.h"

#include "core/status.hpp"
#include "core/version.hpp"

#if defined(C4A_USE_CUDA)
#  include "core/cuda_dispatch.hpp"
#endif

#if defined(__GNUC__) || defined(__clang__)
#  define C4A_NO_UBSAN __attribute__((no_sanitize("undefined")))
#else
#  define C4A_NO_UBSAN
#endif

extern "C" {

C4A_API C4A_NO_UBSAN const char* c4a_status_to_string(c4a_status_t s) {
    try {
        return ::chemometrics4all::core::status_to_string(s);
    } catch (...) {
        return "internal error";
    }
}

C4A_API int c4a_backend_is_available(c4a_backend_t backend) {
    try {
        switch (backend) {
            case C4A_BACKEND_AUTO:
            case C4A_BACKEND_REFERENCE_CPU:
                return 1;
            case C4A_BACKEND_BLAS:
#if defined(C4A_USE_BLAS)
                return 1;
#else
                return 0;
#endif
            case C4A_BACKEND_OPENMP:
#if defined(C4A_USE_OPENMP)
                return 1;
#else
                return 0;
#endif
            case C4A_BACKEND_CUDA:
#if defined(C4A_USE_CUDA)
                return ::chemometrics4all::cuda_dispatch::cuda_runtime_available() ? 1 : 0;
#else
                return 0;
#endif
            case C4A_BACKEND_NATIVE_CPU:
            case C4A_BACKEND_OPENCL:
            case C4A_BACKEND_METAL:
            default:
                return 0;
        }
    } catch (...) {
        return 0;
    }
}

C4A_API const char* c4a_backend_to_string(c4a_backend_t backend) {
    try {
        switch (backend) {
            case C4A_BACKEND_AUTO:          return "auto";
            case C4A_BACKEND_REFERENCE_CPU: return "reference_cpu";
            case C4A_BACKEND_NATIVE_CPU:    return "native_cpu";
            case C4A_BACKEND_BLAS:          return "blas";
            case C4A_BACKEND_OPENMP:        return "openmp";
            case C4A_BACKEND_CUDA:          return "cuda";
            case C4A_BACKEND_OPENCL:        return "opencl";
            case C4A_BACKEND_METAL:         return "metal";
        }
        return "unknown_backend";
    } catch (...) {
        return "unknown_backend";
    }
}

C4A_API size_t c4a_dtype_size(c4a_dtype_t dtype) {
    try {
        switch (dtype) {
            case C4A_DTYPE_F64: return 8;
            case C4A_DTYPE_F32: return 4;
            case C4A_DTYPE_I32: return 4;
            case C4A_DTYPE_I64: return 8;
            case C4A_DTYPE_UNKNOWN: return 0;
        }
        return 0;
    } catch (...) {
        return 0;
    }
}

C4A_API const char* c4a_dtype_to_string(c4a_dtype_t dtype) {
    try {
        switch (dtype) {
            case C4A_DTYPE_F64:     return "f64";
            case C4A_DTYPE_F32:     return "f32";
            case C4A_DTYPE_I32:     return "i32";
            case C4A_DTYPE_I64:     return "i64";
            case C4A_DTYPE_UNKNOWN: return "unknown";
        }
        return "unknown_dtype";
    } catch (...) {
        return "unknown_dtype";
    }
}

C4A_API uint32_t c4a_get_abi_version_major(void) {
    try { return ::chemometrics4all::core::abi_major(); } catch (...) { return 0; }
}
C4A_API uint32_t c4a_get_abi_version_minor(void) {
    try { return ::chemometrics4all::core::abi_minor(); } catch (...) { return 0; }
}
C4A_API uint32_t c4a_get_abi_version_patch(void) {
    try { return ::chemometrics4all::core::abi_patch(); } catch (...) { return 0; }
}
C4A_API uint32_t c4a_get_abi_version_int(void) {
    try { return ::chemometrics4all::core::abi_int(); } catch (...) { return 0; }
}

C4A_API const char* c4a_get_version_string(void) {
    try { return ::chemometrics4all::core::version_string(); } catch (...) { return ""; }
}
C4A_API const char* c4a_get_build_info(void) {
    try { return ::chemometrics4all::core::build_info(); } catch (...) { return ""; }
}
C4A_API const char* c4a_get_git_revision(void) {
    try { return ::chemometrics4all::core::git_revision(); } catch (...) { return ""; }
}

C4A_API c4a_status_t c4a_check_abi_compatibility(
    uint32_t header_abi_major,
    uint32_t header_abi_minor) {
    try {
        if (header_abi_major != ::chemometrics4all::core::abi_major()) {
            return C4A_ERR_ABI_MISMATCH;
        }
        if (header_abi_minor > ::chemometrics4all::core::abi_minor()) {
            return C4A_ERR_VERSION_INCOMPATIBLE;
        }
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

}  // extern "C"

#undef C4A_NO_UBSAN
