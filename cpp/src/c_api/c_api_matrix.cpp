// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for c4a_matrix_view_t.

#include <stddef.h>
#include <stdint.h>

#include "chemometrics4all/c4a.h"

#include "core/matrix_view.hpp"

extern "C" {

C4A_API c4a_status_t c4a_matrix_view_init_rowmajor(
    c4a_matrix_view_t* out, void* data,
    int64_t rows, int64_t cols, c4a_dtype_t dtype) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        if (!::chemometrics4all::core::dtype_is_valid(dtype)) return C4A_ERR_INVALID_ARGUMENT;
        if (rows < 0 || cols < 0)                    return C4A_ERR_INVALID_ARGUMENT;
        if (rows > 0 && cols > 0 && data == nullptr) return C4A_ERR_NULL_POINTER;

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
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_matrix_view_init_colmajor(
    c4a_matrix_view_t* out, void* data,
    int64_t rows, int64_t cols, c4a_dtype_t dtype) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        if (!::chemometrics4all::core::dtype_is_valid(dtype)) return C4A_ERR_INVALID_ARGUMENT;
        if (rows < 0 || cols < 0)                    return C4A_ERR_INVALID_ARGUMENT;
        if (rows > 0 && cols > 0 && data == nullptr) return C4A_ERR_NULL_POINTER;

        out->data        = data;
        out->rows        = rows;
        out->cols        = cols;
        out->row_stride  = 1;
        out->col_stride  = rows > 0 ? rows : 1;
        out->dtype       = dtype;
        out->reserved0   = 0;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_matrix_view_init_strided(
    c4a_matrix_view_t* out, void* data,
    int64_t rows, int64_t cols,
    int64_t row_stride, int64_t col_stride,
    c4a_dtype_t dtype) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        if (!::chemometrics4all::core::dtype_is_valid(dtype)) return C4A_ERR_INVALID_ARGUMENT;
        if (rows < 0 || cols < 0)                    return C4A_ERR_INVALID_ARGUMENT;
        if (row_stride < 0 || col_stride < 0)        return C4A_ERR_STRIDE_INVALID;
        if (rows > 0 && cols > 0 && data == nullptr) return C4A_ERR_NULL_POINTER;

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
        return ::chemometrics4all::core::validate_nonnull_view(*out);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_matrix_view_validate(const c4a_matrix_view_t* v) {
    if (v == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        return ::chemometrics4all::core::validate_nonnull_view(*v);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

}  // extern "C"
