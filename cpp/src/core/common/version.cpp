// SPDX-License-Identifier: CECILL-2.1
//
// Version / build-info strings. Both the project version and the ABI version
// are pulled from n4m_version.h via the macros below, so a single
// scripts/bump_version.sh edit propagates everywhere.

#include "core/common/version.hpp"

#include "n4m/n4m_version.h"

#ifndef N4M_BUILD_GIT_REVISION
#  define N4M_BUILD_GIT_REVISION ""
#endif
#ifndef N4M_BUILD_INFO
#  define N4M_BUILD_INFO ""
#endif

// Local stringification helpers (kept TU-local — not exposed in any header).
#define N4M_STR_INNER_(x) #x
#define N4M_STR_(x)       N4M_STR_INNER_(x)
#define N4M_ABI_VERSION_STRING_                       \
    N4M_STR_(N4M_ABI_VERSION_MAJOR) "."               \
    N4M_STR_(N4M_ABI_VERSION_MINOR) "."               \
    N4M_STR_(N4M_ABI_VERSION_PATCH)

// Report compilation, not runtime availability (CUDA may have no visible device).
// This closed, versioned suffix is consumed by thin language bindings. Retain
// N4M_BUILD_INFO verbatim as the descriptive prefix. Detect compiler-enabled
// OpenMP as well: external CXXFLAGS may enable it with the configured backend
// disabled. BLAS/CUDA flags describe the core's direct numerical dispatch.
#if defined(N4M_USE_BLAS)
#  define N4M_COMPILED_BLAS_ "1"
#else
#  define N4M_COMPILED_BLAS_ "0"
#endif
#if defined(N4M_USE_OPENMP) || defined(_OPENMP)
#  define N4M_COMPILED_OPENMP_ "1"
#else
#  define N4M_COMPILED_OPENMP_ "0"
#endif
#if defined(N4M_USE_CUDA)
#  define N4M_COMPILED_CUDA_ "1"
#else
#  define N4M_COMPILED_CUDA_ "0"
#endif

namespace n4m::core {

const char* version_string() noexcept {
    return N4M_PROJECT_VERSION_STRING "+abi." N4M_ABI_VERSION_STRING_;
}

const char* build_info() noexcept {
    return N4M_BUILD_INFO "\nn4m-build-capabilities-v1;blas=" N4M_COMPILED_BLAS_
                          ";openmp=" N4M_COMPILED_OPENMP_ ";cuda=" N4M_COMPILED_CUDA_;
}

const char* git_revision() noexcept {
    return N4M_BUILD_GIT_REVISION;
}

}  // namespace n4m::core
