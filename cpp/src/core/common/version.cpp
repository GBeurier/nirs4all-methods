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

namespace n4m::core {

const char* version_string() noexcept {
    return N4M_PROJECT_VERSION_STRING "+abi." N4M_ABI_VERSION_STRING_;
}

const char* build_info() noexcept {
    // Reports optional-feature flags so a caller can detect, at runtime,
    // capabilities that are compiled out. FITPACK is the load-bearing one:
    // when it is absent (no Fortran compiler at build time) the spline-smooth
    // augmenter is an intentional no-op (see cpp/src/core/augmentation/
    // splines/spline_smoothing.c + PRODUCTION_AUDIT.md §10.3), so a downstream
    // pipeline can check for "fitpack=1" before relying on it. N4M_HAVE_FITPACK
    // is a PRIVATE n4m_core define (n4m_targets.cmake); version.cpp compiles
    // into n4m_core, so it is visible here.
    return N4M_BUILD_INFO
#if defined(N4M_HAVE_FITPACK) && N4M_HAVE_FITPACK
        "fitpack=1";
#else
        "fitpack=0";
#endif
}

const char* git_revision() noexcept {
    return N4M_BUILD_GIT_REVISION;
}

}  // namespace n4m::core
