// SPDX-License-Identifier: CeCILL-2.1

#include <cstring>

#include "pls4all/p4a.h"
#include "harness.hpp"

TEST(version, runtime_matches_compile_time) {
    CHECK_EQ(p4a_get_abi_version_major(), static_cast<uint32_t>(P4A_ABI_VERSION_MAJOR));
    CHECK_EQ(p4a_get_abi_version_minor(), static_cast<uint32_t>(P4A_ABI_VERSION_MINOR));
    CHECK_EQ(p4a_get_abi_version_patch(), static_cast<uint32_t>(P4A_ABI_VERSION_PATCH));
    CHECK_EQ(p4a_get_abi_version_int(),   static_cast<uint32_t>(P4A_ABI_VERSION_INT));
}

TEST(version, version_string_contains_project_and_abi) {
    const char* v = p4a_get_version_string();
    CHECK_NE(v, nullptr);
    CHECK_STR_CONTAINS(v, "0.40.0");
    CHECK_STR_CONTAINS(v, "abi.1.0.0");
}

TEST(version, abi_compat_same_major_minor_returns_ok) {
    CHECK_EQ(p4a_check_abi_compatibility(P4A_ABI_VERSION_MAJOR,
                                          P4A_ABI_VERSION_MINOR), P4A_OK);
}

TEST(version, abi_compat_major_mismatch_returns_abi_mismatch) {
    CHECK_EQ(p4a_check_abi_compatibility(P4A_ABI_VERSION_MAJOR + 1, 0),
              P4A_ERR_ABI_MISMATCH);
    CHECK_EQ(p4a_check_abi_compatibility(P4A_ABI_VERSION_MAJOR - 0u, 999),
              P4A_ERR_VERSION_INCOMPATIBLE);
}
