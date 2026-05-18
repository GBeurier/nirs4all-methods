// SPDX-License-Identifier: CECILL-2.1
//
// Version / build-info strings. Both the project version and the ABI version
// are pulled from p4a_version.h via the macros below, so a single
// scripts/bump_version.sh edit propagates everywhere.

#include "core/version.hpp"

#include "pls4all/p4a_version.h"

#ifndef P4A_BUILD_GIT_REVISION
#  define P4A_BUILD_GIT_REVISION ""
#endif
#ifndef P4A_BUILD_INFO
#  define P4A_BUILD_INFO ""
#endif

// Local stringification helpers (kept TU-local — not exposed in any header).
#define P4A_STR_INNER_(x) #x
#define P4A_STR_(x)       P4A_STR_INNER_(x)
#define P4A_ABI_VERSION_STRING_                       \
    P4A_STR_(P4A_ABI_VERSION_MAJOR) "."               \
    P4A_STR_(P4A_ABI_VERSION_MINOR) "."               \
    P4A_STR_(P4A_ABI_VERSION_PATCH)

namespace pls4all::core {

const char* version_string() noexcept {
    return P4A_PROJECT_VERSION_STRING "+abi." P4A_ABI_VERSION_STRING_;
}

const char* build_info() noexcept {
    return P4A_BUILD_INFO;
}

const char* git_revision() noexcept {
    return P4A_BUILD_GIT_REVISION;
}

}  // namespace pls4all::core
