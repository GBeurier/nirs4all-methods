// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for n4m_matrix_view_t.

#include <stddef.h>
#include <stdint.h>

#include "pls4all/p4a.h"

#include "core/common/matrix_view.hpp"

extern "C" {

N4M_API n4m_status_t n4m_matrix_view_init_rowmajor(
    n4m_matrix_view_t* out, void* data,
    int64_t rows, int64_t cols, n4m_dtype_t dtype) {
    if (out == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        if (!::n4m::core::dtype_is_valid(dtype)) return N4M_ERR_INVALID_ARGUMENT;
        if (rows < 0 || cols < 0)                    return N4M_ERR_INVALID_ARGUMENT;
        if (rows > 0 && cols > 0 && data == nullptr) return N4M_ERR_NULL_POINTER;

        out->data        = data;
        out->rows        = rows;
        out->cols        = cols;
        // Row-major: walking down a column = +cols elements; walking across
        // a row = +1 element. Use 1 (instead of 0) for degenerate dims so
        // _validate accepts the view unconditionally.
        out->row_stride  = cols > 0 ? cols : 1;
        out->col_stride  = 1;
        out->dtype       = dtype;
        out->reserved0   = 0;
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_matrix_view_init_colmajor(
    n4m_matrix_view_t* out, void* data,
    int64_t rows, int64_t cols, n4m_dtype_t dtype) {
    if (out == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        if (!::n4m::core::dtype_is_valid(dtype)) return N4M_ERR_INVALID_ARGUMENT;
        if (rows < 0 || cols < 0)                    return N4M_ERR_INVALID_ARGUMENT;
        if (rows > 0 && cols > 0 && data == nullptr) return N4M_ERR_NULL_POINTER;

        out->data        = data;
        out->rows        = rows;
        out->cols        = cols;
        out->row_stride  = 1;
        out->col_stride  = rows > 0 ? rows : 1;
        out->dtype       = dtype;
        out->reserved0   = 0;
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_matrix_view_init_strided(
    n4m_matrix_view_t* out, void* data,
    int64_t rows, int64_t cols,
    int64_t row_stride, int64_t col_stride,
    n4m_dtype_t dtype) {
    if (out == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        if (!::n4m::core::dtype_is_valid(dtype)) return N4M_ERR_INVALID_ARGUMENT;
        if (rows < 0 || cols < 0)                    return N4M_ERR_INVALID_ARGUMENT;
        if (row_stride < 0 || col_stride < 0)        return N4M_ERR_STRIDE_INVALID;
        if (rows > 0 && cols > 0 && data == nullptr) return N4M_ERR_NULL_POINTER;

        out->data        = data;
        out->rows        = rows;
        out->cols        = cols;
        out->row_stride  = row_stride;
        out->col_stride  = col_stride;
        out->dtype       = dtype;
        out->reserved0   = 0;
        // Run the standard validator on what we just built so the
        // caller cannot smuggle in an invalid (rows>=2, row_stride==0)
        // configuration. Init helpers and _validate share the same rules.
        return ::n4m::core::validate_nonnull_view(*out);
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_matrix_view_validate(const n4m_matrix_view_t* v) {
    if (v == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        return ::n4m::core::validate_nonnull_view(*v);
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

}  // extern "C"
