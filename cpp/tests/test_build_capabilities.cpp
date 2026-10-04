// SPDX-License-Identifier: CECILL-2.1
// Compile-time profile witnesses, including compiled CUDA without a device.
#include "n4m/n4m.h"
#include "core/common/version.hpp"

#include <iostream>
#include <string>

#ifndef N4M_BUILD_INFO
#  define N4M_BUILD_INFO ""
#endif

#if defined(N4M_TEST_REQUIRE_EXTERNAL_OPENMP) && !defined(_OPENMP)
#  error "External OpenMP compiler flag did not define _OPENMP"
#endif

int main() {
    std::string expected = N4M_BUILD_INFO;
    expected += "\nn4m-build-capabilities-v1;blas=";
#if defined(N4M_USE_BLAS)
    expected += "1";
#else
    expected += "0";
#endif
    expected += ";openmp=";
#if defined(N4M_USE_OPENMP) || defined(_OPENMP)
    expected += "1";
#else
    expected += "0";
#endif
    expected += ";cuda=";
#if defined(N4M_USE_CUDA)
    expected += "1";
#else
    expected += "0";
#endif
#if defined(N4M_TEST_BUILD_INFO_CORE)
    const char* actual = n4m::core::build_info();
#else
    const char* actual = n4m_get_build_info();
#endif
    if (actual == nullptr || expected != actual) {
        std::cerr << "compiled build profile mismatch\n";
        return 1;
    }
    // The return is a static native string, with stable lifetime across calls.
#if defined(N4M_TEST_BUILD_INFO_CORE)
    if (actual != n4m::core::build_info()) return 2;
#else
    if (actual != n4m_get_build_info()) return 2;
#endif
    return 0;
}
