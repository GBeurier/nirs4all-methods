// SPDX-License-Identifier: CECILL-2.1
//
// Internal helpers for p4a_matrix_view_t validation. The struct itself is
// declared in pls4all/p4a.h and is part of the public C ABI; these helpers
// live behind the extern "C" wrappers in cpp/src/c_api/c_api_matrix.cpp.

#pragma once

#include "pls4all/p4a.h"

namespace pls4all::core {

// Returns the byte-size of a single element of the given dtype, or 0 for
// UNKNOWN / out-of-range values. (Pure helper shared with the version
// translation unit.)
[[nodiscard]] size_t dtype_size(p4a_dtype_t dtype) noexcept;

// Returns true if the dtype is one of the four valid public dtypes.
[[nodiscard]] bool dtype_is_valid(p4a_dtype_t dtype) noexcept;

// Validate a non-NULL view against the rules documented in p4a.h §4.
// The caller is responsible for handling the v == NULL case.
[[nodiscard]] p4a_status_t validate_nonnull_view(const p4a_matrix_view_t& v) noexcept;

}  // namespace pls4all::core
