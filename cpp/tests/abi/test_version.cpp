// SPDX-License-Identifier: CECILL-2.1

#include <cstdio>
#include <cstring>

#include "pls4all/p4a.h"
#include "harness.hpp"

TEST(version, runtime_matches_compile_time) {
    // The runtime accessors must agree with the compile-time macros in
    // n4m_version.h. We deliberately do NOT hard-code the expected values:
    // n4m_version.h is the canonical source of truth, and the only thing
    // worth checking here is the binding between compile-time macros and the
    // runtime ABI surface.
    CHECK_EQ(n4m_get_abi_version_major(), static_cast<uint32_t>(N4M_ABI_VERSION_MAJOR));
    CHECK_EQ(n4m_get_abi_version_minor(), static_cast<uint32_t>(N4M_ABI_VERSION_MINOR));
    CHECK_EQ(n4m_get_abi_version_patch(), static_cast<uint32_t>(N4M_ABI_VERSION_PATCH));
    CHECK_EQ(n4m_get_abi_version_int(),   static_cast<uint32_t>(N4M_ABI_VERSION_INT));
}

TEST(version, version_string_contains_project_and_abi) {
    const char* v = n4m_get_version_string();
    CHECK_NE(v, nullptr);
    // The version string must embed the canonical project semver string and
    // the runtime ABI version. Both are derived from n4m_version.h, so a
    // version bump via scripts/bump_version.sh propagates here automatically.
    CHECK_STR_CONTAINS(v, N4M_PROJECT_VERSION_STRING);
    char expected_abi[64];
    std::snprintf(expected_abi, sizeof(expected_abi), "abi.%u.%u.%u",
                  n4m_get_abi_version_major(),
                  n4m_get_abi_version_minor(),
                  n4m_get_abi_version_patch());
    CHECK_STR_CONTAINS(v, expected_abi);
}

TEST(version, abi_compat_same_major_minor_returns_ok) {
    CHECK_EQ(n4m_check_abi_compatibility(N4M_ABI_VERSION_MAJOR,
                                          N4M_ABI_VERSION_MINOR), N4M_OK);
}

TEST(version, abi_compat_major_mismatch_returns_abi_mismatch) {
    CHECK_EQ(n4m_check_abi_compatibility(N4M_ABI_VERSION_MAJOR + 1, 0),
              N4M_ERR_ABI_MISMATCH);
    CHECK_EQ(n4m_check_abi_compatibility(N4M_ABI_VERSION_MAJOR - 0u, 999),
              N4M_ERR_VERSION_INCOMPATIBLE);
}
