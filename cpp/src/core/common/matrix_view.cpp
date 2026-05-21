// SPDX-License-Identifier: CECILL-2.1

#include "core/common/matrix_view.hpp"

namespace n4m::core {

size_t dtype_size(n4m_dtype_t dtype) noexcept {
    switch (dtype) {
        case N4M_DTYPE_F64: return 8;
        case N4M_DTYPE_F32: return 4;
        case N4M_DTYPE_I32: return 4;
        case N4M_DTYPE_I64: return 8;
        case N4M_DTYPE_UNKNOWN: return 0;
    }
    return 0;
}

bool dtype_is_valid(n4m_dtype_t dtype) noexcept {
    return dtype == N4M_DTYPE_F64 || dtype == N4M_DTYPE_F32
        || dtype == N4M_DTYPE_I32 || dtype == N4M_DTYPE_I64;
}

n4m_status_t validate_nonnull_view(const n4m_matrix_view_t& v) noexcept {
    if (!dtype_is_valid(v.dtype)) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (v.rows < 0 || v.cols < 0) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    // Negative strides are not supported (Phase 0 design choice). Zero
    // strides are only allowed when the corresponding dimension is <= 1
    // (a degenerate dimension that has no neighbour).
    if (v.row_stride < 0 || v.col_stride < 0) {
        return N4M_ERR_STRIDE_INVALID;
    }
    if (v.rows >= 2 && v.row_stride < 1) {
        return N4M_ERR_STRIDE_INVALID;
    }
    if (v.cols >= 2 && v.col_stride < 1) {
        return N4M_ERR_STRIDE_INVALID;
    }
    // For non-empty views, the data pointer must be non-NULL.
    if (v.rows > 0 && v.cols > 0 && v.data == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    return N4M_OK;
}

}  // namespace n4m::core
