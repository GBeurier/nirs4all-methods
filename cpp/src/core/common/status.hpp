// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
//
// Internal helpers around the n4m_status_t enum.

#pragma once

#include "n4m/n4m.h"

namespace n4m::core {

// Returns the static NUL-terminated description for the given status code.
// Returns "unknown status" for any value outside the declared enum range.
// Never returns nullptr.
[[nodiscard]] const char* status_to_string(n4m_status_t s) noexcept;

}  // namespace n4m::core
