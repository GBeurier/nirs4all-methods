// SPDX-License-Identifier: CeCILL-2.1
//
// Version / build-info strings. Configure-time substitutions land in PR 7
// (abi-stubs / CLI). For now everything is hard-coded against
// p4a_version.h so PRs 4-6 do not need to wire CMake's `configure_file`.

#include "core/version.hpp"

#include "pls4all/p4a_version.h"

#ifndef P4A_BUILD_GIT_REVISION
#  define P4A_BUILD_GIT_REVISION ""
#endif
#ifndef P4A_BUILD_INFO
#  define P4A_BUILD_INFO ""
#endif

namespace pls4all::core {

const char* version_string() noexcept {
    return P4A_PROJECT_VERSION_STRING "+abi.1.11.0";
}

const char* build_info() noexcept {
    return P4A_BUILD_INFO;
}

const char* git_revision() noexcept {
    return P4A_BUILD_GIT_REVISION;
}

}  // namespace pls4all::core
