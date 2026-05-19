// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for the Phase 21 FCKStaticTransformer operator.
//
// The transformer is stateless on its input — its kernel bank is constructed
// at `_create` time from `(filter_orders × filter_scales × kernel_size)` and
// is constant for the handle's lifetime. The exposed surface is therefore
// the canonical stateless triplet plus a shape-helper:
//
//   c4a_pp_fck_static_create / _transform / _destroy
//   c4a_pp_fck_static_output_cols
//
// The internal C engine lives in core/preprocessing/specialized/fck_static.c;
// the engine guarantees no exception ever escapes (pure C), but we still
// wrap each entry point in try/catch to match the contract of the other
// preprocessing wrappers (Phases 2-7, 10).

#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>

#include "chemometrics4all/c4a.h"

#include "core/matrix_view.hpp"
#include "core/preprocessing/specialized/fck_static.h"

// ---------------------------------------------------------------------------
// Opaque public handle.
// ---------------------------------------------------------------------------

struct c4a_pp_fck_static_handle_t {
    c4a_pp_fck_static_state_t* state;
};

namespace {

// Validate that a matrix view is non-NULL, F64, row-major contiguous.
// Returns the underlying double pointer + dimensions.
c4a_status_t require_rowmajor_f64(const c4a_matrix_view_t& v,
                                   const double*& out_ptr,
                                   std::int64_t& out_rows,
                                   std::int64_t& out_cols) noexcept {
    const c4a_status_t s = ::chemometrics4all::core::validate_nonnull_view(v);
    if (s != C4A_OK) {
        return s;
    }
    if (v.dtype != C4A_DTYPE_F64) {
        return C4A_ERR_DTYPE_MISMATCH;
    }
    if (v.col_stride != 1) {
        return C4A_ERR_STRIDE_INVALID;
    }
    if (v.rows > 0 && v.cols > 0 && v.row_stride != v.cols) {
        return C4A_ERR_STRIDE_INVALID;
    }
    out_ptr  = static_cast<const double*>(v.data);
    out_rows = v.rows;
    out_cols = v.cols;
    return C4A_OK;
}

c4a_status_t require_rowmajor_f64_mut(c4a_matrix_view_t& v,
                                       double*& out_ptr,
                                       std::int64_t& out_rows,
                                       std::int64_t& out_cols) noexcept {
    const c4a_status_t s = ::chemometrics4all::core::validate_nonnull_view(v);
    if (s != C4A_OK) {
        return s;
    }
    if (v.dtype != C4A_DTYPE_F64) {
        return C4A_ERR_DTYPE_MISMATCH;
    }
    if (v.col_stride != 1) {
        return C4A_ERR_STRIDE_INVALID;
    }
    if (v.rows > 0 && v.cols > 0 && v.row_stride != v.cols) {
        return C4A_ERR_STRIDE_INVALID;
    }
    out_ptr  = static_cast<double*>(v.data);
    out_rows = v.rows;
    out_cols = v.cols;
    return C4A_OK;
}

}  // namespace

extern "C" {

C4A_API c4a_status_t c4a_pp_fck_static_create(
    c4a_pp_fck_static_handle_t** out,
    int32_t kernel_size,
    const double* filter_orders,
    int32_t n_orders,
    const double* filter_scales,
    int32_t n_scales) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    if (filter_orders == nullptr || filter_scales == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    if (kernel_size < 3 || (kernel_size % 2) == 0 ||
        n_orders <= 0 || n_scales <= 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    // Scales must be positive, matching nirs4all.FCKStaticTransformer.
    for (int32_t i = 0; i < n_scales; ++i) {
        if (!(filter_scales[i] > 0.0)) {
            return C4A_ERR_INVALID_ARGUMENT;
        }
    }
    // L = n_orders * n_scales must fit in int32_t.
    const std::int64_t n_kernels =
        static_cast<std::int64_t>(n_orders) *
        static_cast<std::int64_t>(n_scales);
    if (n_kernels > static_cast<std::int64_t>(INT32_MAX)) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        c4a_pp_fck_static_state_t* s = c4a_pp_fck_static_state_new(
            kernel_size, filter_orders, n_orders,
            filter_scales, n_scales);
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_fck_static_handle_t* h =
            new (std::nothrow) c4a_pp_fck_static_handle_t{s};
        if (h == nullptr) {
            c4a_pp_fck_static_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_fck_static_destroy(c4a_pp_fck_static_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_fck_static_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_fck_static_transform(
    const c4a_pp_fck_static_handle_t* h,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out) {
    if (h == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        const double* xp = nullptr;
        double*       op = nullptr;
        std::int64_t  xr = 0, xc = 0, orr = 0, oc = 0;
        c4a_status_t  s  = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        s = require_rowmajor_f64_mut(out, op, orr, oc);
        if (s != C4A_OK) return s;
        const int32_t n_kernels =
            c4a_pp_fck_static_state_n_kernels(h->state);
        if (orr != xr ||
            oc  != static_cast<std::int64_t>(n_kernels) * xc) {
            return C4A_ERR_SHAPE_MISMATCH;
        }
        const int rc = c4a_pp_fck_static_state_apply(
            h->state, xp, xr, xc, op);
        if (rc == -1) return C4A_ERR_NULL_POINTER;
        if (rc == -2) return C4A_ERR_INVALID_ARGUMENT;
        if (rc != 0)  return C4A_ERR_INTERNAL;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_pp_fck_static_output_cols(int32_t n_kernels,
                                                    int32_t n_features,
                                                    int32_t* out) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = 0;
    if (n_kernels < 0 || n_features < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    const std::int64_t prod = static_cast<std::int64_t>(n_kernels) *
                               static_cast<std::int64_t>(n_features);
    if (prod > static_cast<std::int64_t>(INT32_MAX)) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    *out = static_cast<int32_t>(prod);
    return C4A_OK;
}

}  // extern "C"
