// SPDX-License-Identifier: CECILL-2.1

#include <cstdio>
#include <cstring>

#include "pls4all/p4a.h"
#include "harness.hpp"

TEST(version, runtime_matches_compile_time) {
    // The runtime accessors must agree with the compile-time macros in
    // p4a_version.h. We deliberately do NOT hard-code the expected values:
    // p4a_version.h is the canonical source of truth, and the only thing
    // worth checking here is the binding between compile-time macros and the
    // runtime ABI surface.
    CHECK_EQ(p4a_get_abi_version_major(), static_cast<uint32_t>(P4A_ABI_VERSION_MAJOR));
    CHECK_EQ(p4a_get_abi_version_minor(), static_cast<uint32_t>(P4A_ABI_VERSION_MINOR));
    CHECK_EQ(p4a_get_abi_version_patch(), static_cast<uint32_t>(P4A_ABI_VERSION_PATCH));
    CHECK_EQ(p4a_get_abi_version_int(),   static_cast<uint32_t>(P4A_ABI_VERSION_INT));
}

TEST(version, version_string_contains_project_and_abi) {
    const char* v = p4a_get_version_string();
    CHECK_NE(v, nullptr);
    // The version string must embed the canonical project semver string and
    // the runtime ABI version. Both are derived from p4a_version.h, so a
    // version bump via scripts/bump_version.sh propagates here automatically.
    CHECK_STR_CONTAINS(v, P4A_PROJECT_VERSION_STRING);
    char expected_abi[64];
    std::snprintf(expected_abi, sizeof(expected_abi), "abi.%u.%u.%u",
                  p4a_get_abi_version_major(),
                  p4a_get_abi_version_minor(),
                  p4a_get_abi_version_patch());
    CHECK_STR_CONTAINS(v, expected_abi);
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
