// SPDX-License-Identifier: CECILL-2.1
//
// Internal helpers around the c4a_status_t enum.

#pragma once

#include "chemometrics4all/c4a.h"

namespace chemometrics4all::core {

// Returns the static NUL-terminated description for the given status code.
// Returns "unknown status" for any value outside the declared enum range.
// Never returns nullptr.
[[nodiscard]] const char* status_to_string(c4a_status_t s) noexcept;

}  // namespace chemometrics4all::core
