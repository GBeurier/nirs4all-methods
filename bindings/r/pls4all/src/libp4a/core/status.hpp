// SPDX-License-Identifier: CECILL-2.1
//
// Internal helpers around the p4a_status_t enum.

#pragma once

#include "pls4all/p4a.h"

namespace pls4all::core {

// Returns the static NUL-terminated description for the given status code.
// Returns "unknown status" for any value outside the declared enum range.
// Never returns nullptr.
[[nodiscard]] const char* status_to_string(p4a_status_t s) noexcept;

}  // namespace pls4all::core
