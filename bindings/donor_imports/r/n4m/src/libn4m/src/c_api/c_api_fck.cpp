// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for the Phase 21 FCKStaticTransformer operator.
//
// The transformer is stateless on its input — its kernel bank is constructed
// at `_create` time from `(filter_orders × filter_scales × kernel_size)` and
// is constant for the handle's lifetime. The exposed surface is therefore
// the canonical stateless triplet plus a shape-helper:
//
//   n4m_pp_fck_static_create / _transform / _destroy
//   n4m_pp_fck_static_output_cols
//
// The internal C engine lives in core/preprocessing/specialized/fck_static.c;
// the engine guarantees no exception ever escapes (pure C), but we still
// wrap each entry point in try/catch to match the contract of the other
// preprocessing wrappers (Phases 2-7, 10).

#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>

#include "n4m/n4m.h"

#include "core/matrix_view.hpp"
#include "core/preprocessing/specialized/fck_static.h"

// ---------------------------------------------------------------------------
// Opaque public handle.
// ---------------------------------------------------------------------------

struct n4m_pp_fck_static_handle_t {
    n4m_pp_fck_static_state_t* state;
};

namespace {

// Validate that a matrix view is non-NULL, F64, row-major contiguous.
// Returns the underlying double pointer + dimensions.
n4m_status_t require_rowmajor_f64(const n4m_matrix_view_t& v,
                                   const double*& out_ptr,
                                   std::int64_t& out_rows,
                                   std::int64_t& out_cols) noexcept {
    const n4m_status_t s = ::n4m::core::validate_nonnull_view(v);
    if (s != N4M_OK) {
        return s;
    }
    if (v.dtype != N4M_DTYPE_F64) {
        return N4M_ERR_DTYPE_MISMATCH;
    }
    if (v.col_stride != 1) {
        return N4M_ERR_STRIDE_INVALID;
    }
    if (v.rows > 0 && v.cols > 0 && v.row_stride != v.cols) {
        return N4M_ERR_STRIDE_INVALID;
    }
    out_ptr  = static_cast<const double*>(v.data);
    out_rows = v.rows;
    out_cols = v.cols;
    return N4M_OK;
}

n4m_status_t require_rowmajor_f64_mut(n4m_matrix_view_t& v,
                                       double*& out_ptr,
                                       std::int64_t& out_rows,
                                       std::int64_t& out_cols) noexcept {
    const n4m_status_t s = ::n4m::core::validate_nonnull_view(v);
    if (s != N4M_OK) {
        return s;
    }
    if (v.dtype != N4M_DTYPE_F64) {
        return N4M_ERR_DTYPE_MISMATCH;
    }
    if (v.col_stride != 1) {
        return N4M_ERR_STRIDE_INVALID;
    }
    if (v.rows > 0 && v.cols > 0 && v.row_stride != v.cols) {
        return N4M_ERR_STRIDE_INVALID;
    }
    out_ptr  = static_cast<double*>(v.data);
    out_rows = v.rows;
    out_cols = v.cols;
    return N4M_OK;
}

}  // namespace

extern "C" {

N4M_API n4m_status_t n4m_pp_fck_static_create(
    n4m_pp_fck_static_handle_t** out,
    int32_t kernel_size,
    const double* filter_orders,
    int32_t n_orders,
    const double* filter_scales,
    int32_t n_scales) {
    if (out == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    *out = nullptr;
    if (filter_orders == nullptr || filter_scales == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    if (kernel_size < 3 || (kernel_size % 2) == 0 ||
        n_orders <= 0 || n_scales <= 0) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    // Scales must be positive, matching nirs4all.FCKStaticTransformer.
    for (int32_t i = 0; i < n_scales; ++i) {
        if (!(filter_scales[i] > 0.0)) {
            return N4M_ERR_INVALID_ARGUMENT;
        }
    }
    // L = n_orders * n_scales must fit in int32_t.
    const std::int64_t n_kernels =
        static_cast<std::int64_t>(n_orders) *
        static_cast<std::int64_t>(n_scales);
    if (n_kernels > static_cast<std::int64_t>(INT32_MAX)) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    try {
        n4m_pp_fck_static_state_t* s = n4m_pp_fck_static_state_new(
            kernel_size, filter_orders, n_orders,
            filter_scales, n_scales);
        if (s == nullptr) {
            return N4M_ERR_OUT_OF_MEMORY;
        }
        n4m_pp_fck_static_handle_t* h =
            new (std::nothrow) n4m_pp_fck_static_handle_t{s};
        if (h == nullptr) {
            n4m_pp_fck_static_state_free(s);
            return N4M_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API void n4m_pp_fck_static_destroy(n4m_pp_fck_static_handle_t* h) {
    if (h == nullptr) return;
    try {
        n4m_pp_fck_static_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

N4M_API n4m_status_t n4m_pp_fck_static_transform(
    const n4m_pp_fck_static_handle_t* h,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out) {
    if (h == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    try {
        const double* xp = nullptr;
        double*       op = nullptr;
        std::int64_t  xr = 0, xc = 0, orr = 0, oc = 0;
        n4m_status_t  s  = require_rowmajor_f64(X, xp, xr, xc);
        if (s != N4M_OK) return s;
        s = require_rowmajor_f64_mut(out, op, orr, oc);
        if (s != N4M_OK) return s;
        const int32_t n_kernels =
            n4m_pp_fck_static_state_n_kernels(h->state);
        if (orr != xr ||
            oc  != static_cast<std::int64_t>(n_kernels) * xc) {
            return N4M_ERR_SHAPE_MISMATCH;
        }
        const int rc = n4m_pp_fck_static_state_apply(
            h->state, xp, xr, xc, op);
        if (rc == -1) return N4M_ERR_NULL_POINTER;
        if (rc == -2) return N4M_ERR_INVALID_ARGUMENT;
        if (rc != 0)  return N4M_ERR_INTERNAL;
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_pp_fck_static_output_cols(int32_t n_kernels,
                                                    int32_t n_features,
                                                    int32_t* out) {
    if (out == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    *out = 0;
    if (n_kernels < 0 || n_features < 0) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const std::int64_t prod = static_cast<std::int64_t>(n_kernels) *
                               static_cast<std::int64_t>(n_features);
    if (prod > static_cast<std::int64_t>(INT32_MAX)) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    *out = static_cast<int32_t>(prod);
    return N4M_OK;
}

}  // extern "C"
