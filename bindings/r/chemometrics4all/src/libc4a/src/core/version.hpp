// SPDX-License-Identifier: CECILL-2.1
//
// Compile-time constants surfaced through the c4a_get_*_version_* family.
// All values are derived from the canonical chemometrics4all/c4a_version.h so that the
// runtime library can never drift from the header consumers compile against.

#pragma once

#include <cstdint>

#include "chemometrics4all/c4a_version.h"

namespace chemometrics4all::core {

[[nodiscard]] const char* version_string() noexcept;
[[nodiscard]] const char* build_info() noexcept;
[[nodiscard]] const char* git_revision() noexcept;

constexpr std::uint32_t abi_major() noexcept {
    return static_cast<std::uint32_t>(C4A_ABI_VERSION_MAJOR);
}
constexpr std::uint32_t abi_minor() noexcept {
    return static_cast<std::uint32_t>(C4A_ABI_VERSION_MINOR);
}
constexpr std::uint32_t abi_patch() noexcept {
    return static_cast<std::uint32_t>(C4A_ABI_VERSION_PATCH);
}
constexpr std::uint32_t abi_int() noexcept {
    return static_cast<std::uint32_t>(C4A_ABI_VERSION_INT);
}

}  // namespace chemometrics4all::core
