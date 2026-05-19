// SPDX-License-Identifier: CECILL-2.1
//
// Version / build-info strings. Both the project version and the ABI version
// are pulled from c4a_version.h via the macros below, so a single
// scripts/bump_version.sh edit propagates everywhere.

#include "core/version.hpp"

#include "chemometrics4all/c4a_version.h"

#ifndef C4A_BUILD_GIT_REVISION
#  define C4A_BUILD_GIT_REVISION ""
#endif
#ifndef C4A_BUILD_INFO
#  define C4A_BUILD_INFO ""
#endif

// Local stringification helpers (kept TU-local — not exposed in any header).
#define C4A_STR_INNER_(x) #x
#define C4A_STR_(x)       C4A_STR_INNER_(x)
#define C4A_ABI_VERSION_STRING_                       \
    C4A_STR_(C4A_ABI_VERSION_MAJOR) "."               \
    C4A_STR_(C4A_ABI_VERSION_MINOR) "."               \
    C4A_STR_(C4A_ABI_VERSION_PATCH)

namespace chemometrics4all::core {

const char* version_string() noexcept {
    return C4A_PROJECT_VERSION_STRING "+abi." C4A_ABI_VERSION_STRING_;
}

const char* build_info() noexcept {
    return C4A_BUILD_INFO;
}

const char* git_revision() noexcept {
    return C4A_BUILD_GIT_REVISION;
}

}  // namespace chemometrics4all::core
