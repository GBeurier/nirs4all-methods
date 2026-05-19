// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for the Phase 10 resampling, cropping, and target
// discretization operators:
//   - Resampler              (stateful)
//   - CropTransformer        (stateless)
//   - ResampleTransformer    (stateless)
//   - IntegerKBinsDiscretizer (stateful, int32 output)
//   - RangeDiscretizer       (stateless, int32 output)
//
// The bodies delegate to the internal C engines under
// cpp/src/core/preprocessing/resampling/. Universal wrapper rules mirror
// c_api_preprocessing.cpp:
//   - Every entry point is wrapped in try/catch so no C++ exception ever
//     crosses the boundary.
//   - The opaque public type is a thin struct {engine_state*}.
//   - `_create` takes a pointer-to-pointer out-arg.
//   - `_destroy` is NULL-safe and never throws.
//   - `_fit` and `_transform` validate matrix views as row-major contiguous
//     F64; row counts and column counts are checked against the engine's
//     fitted shape where applicable.

#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>

#include "chemometrics4all/c4a.h"

#include "core/matrix_view.hpp"
#include "core/preprocessing/resampling/crop.h"
#include "core/preprocessing/resampling/kbins_discretizer.h"
#include "core/preprocessing/resampling/range_discretizer.h"
#include "core/preprocessing/resampling/resample_transformer.h"
#include "core/preprocessing/resampling/resampler.h"

// ---------------------------------------------------------------------------
// Opaque public handles.
// ---------------------------------------------------------------------------

struct c4a_pp_resampler_handle_t {
    c4a_pp_resampler_state_t* state;
};
struct c4a_pp_crop_handle_t {
    c4a_pp_crop_state_t* state;
};
struct c4a_pp_resample_handle_t {
    c4a_pp_resample_state_t* state;
};
struct c4a_pp_kbins_disc_handle_t {
    c4a_pp_kbins_disc_state_t* state;
};
struct c4a_pp_range_disc_handle_t {
    c4a_pp_range_disc_state_t* state;
};

namespace {

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

c4a_status_t require_rowmajor_i32_mut(c4a_matrix_view_t& v,
                                       std::int32_t*& out_ptr,
                                       std::int64_t& out_rows,
                                       std::int64_t& out_cols) noexcept {
    const c4a_status_t s = ::chemometrics4all::core::validate_nonnull_view(v);
    if (s != C4A_OK) {
        return s;
    }
    if (v.dtype != C4A_DTYPE_I32) {
        return C4A_ERR_DTYPE_MISMATCH;
    }
    if (v.col_stride != 1) {
        return C4A_ERR_STRIDE_INVALID;
    }
    if (v.rows > 0 && v.cols > 0 && v.row_stride != v.cols) {
        return C4A_ERR_STRIDE_INVALID;
    }
    out_ptr  = static_cast<std::int32_t*>(v.data);
    out_rows = v.rows;
    out_cols = v.cols;
    return C4A_OK;
}

}  // namespace

extern "C" {

// ===========================================================================
// Resampler (stateful)
// ===========================================================================

C4A_API c4a_status_t c4a_pp_resampler_create(c4a_pp_resampler_handle_t** out,
                                              const double* target_wl,
                                              int64_t n_target,
                                              int32_t method,
                                              double crop_min, double crop_max,
                                              int use_crop,
                                              double fill_value,
                                              int bounds_error,
                                              int extrapolate) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    if (method < 0 || method > 2) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (n_target != -1 && n_target < 1) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        c4a_pp_resampler_state_t* s = c4a_pp_resampler_state_new(
            target_wl, n_target, method, crop_min, crop_max, use_crop,
            fill_value, bounds_error, extrapolate);
        if (s == nullptr) {
            return C4A_ERR_INVALID_ARGUMENT;
        }
        c4a_pp_resampler_handle_t* h =
            new (std::nothrow) c4a_pp_resampler_handle_t{s};
        if (h == nullptr) {
            c4a_pp_resampler_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_resampler_destroy(c4a_pp_resampler_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_resampler_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_resampler_fit(c4a_pp_resampler_handle_t* h,
                                           const double* source_wl,
                                           int64_t n_source) {
    if (h == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        return c4a_pp_resampler_state_fit(h->state, source_wl, n_source);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_pp_resampler_is_fitted(
    const c4a_pp_resampler_handle_t* h, int* out_fitted) {
    if (h == nullptr || out_fitted == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        *out_fitted = c4a_pp_resampler_state_is_fitted(h->state);
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API int64_t c4a_pp_resampler_output_cols(const c4a_pp_resampler_handle_t* h) {
    if (h == nullptr) return 0;
    try {
        return c4a_pp_resampler_state_output_cols(h->state);
    } catch (...) {
        return 0;
    }
}

C4A_API c4a_status_t c4a_pp_resampler_transform(
    const c4a_pp_resampler_handle_t* h,
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
        if (xr != orr) {
            return C4A_ERR_SHAPE_MISMATCH;
        }
        return c4a_pp_resampler_state_apply(h->state, xp, xr, xc, oc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ===========================================================================
// CropTransformer (stateless)
// ===========================================================================

C4A_API c4a_status_t c4a_pp_crop_create(c4a_pp_crop_handle_t** out,
                                         int64_t start, int64_t end) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    try {
        c4a_pp_crop_state_t* s = c4a_pp_crop_state_new(start, end);
        if (s == nullptr) {
            return C4A_ERR_INVALID_ARGUMENT;
        }
        c4a_pp_crop_handle_t* h =
            new (std::nothrow) c4a_pp_crop_handle_t{s};
        if (h == nullptr) {
            c4a_pp_crop_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_crop_destroy(c4a_pp_crop_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_crop_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API int64_t c4a_pp_crop_output_cols(const c4a_pp_crop_handle_t* h,
                                         int64_t input_cols) {
    if (h == nullptr) return 0;
    try {
        return c4a_pp_crop_output_cols_helper(h->state, input_cols);
    } catch (...) {
        return 0;
    }
}

C4A_API c4a_status_t c4a_pp_crop_transform(const c4a_pp_crop_handle_t* h,
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
        if (xr != orr) {
            return C4A_ERR_SHAPE_MISMATCH;
        }
        return c4a_pp_crop_apply(h->state, xp, xr, xc, oc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ===========================================================================
// ResampleTransformer (stateless)
// ===========================================================================

C4A_API c4a_status_t c4a_pp_resample_create(c4a_pp_resample_handle_t** out,
                                             int64_t num_samples) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    try {
        c4a_pp_resample_state_t* s = c4a_pp_resample_state_new(num_samples);
        if (s == nullptr) {
            return C4A_ERR_INVALID_ARGUMENT;
        }
        c4a_pp_resample_handle_t* h =
            new (std::nothrow) c4a_pp_resample_handle_t{s};
        if (h == nullptr) {
            c4a_pp_resample_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_resample_destroy(c4a_pp_resample_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_resample_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API int64_t c4a_pp_resample_output_cols(const c4a_pp_resample_handle_t* h,
                                             int64_t input_cols) {
    if (h == nullptr) return 0;
    try {
        return c4a_pp_resample_output_cols_helper(h->state, input_cols);
    } catch (...) {
        return 0;
    }
}

C4A_API c4a_status_t c4a_pp_resample_transform(
    const c4a_pp_resample_handle_t* h,
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
        if (xr != orr) {
            return C4A_ERR_SHAPE_MISMATCH;
        }
        return c4a_pp_resample_apply(h->state, xp, xr, xc, oc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ===========================================================================
// IntegerKBinsDiscretizer (stateful, int32 output)
// ===========================================================================

C4A_API c4a_status_t c4a_pp_kbins_disc_create(c4a_pp_kbins_disc_handle_t** out,
                                               int32_t n_bins,
                                               int32_t strategy) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    if (n_bins < 2) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (strategy != 0 && strategy != 1) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        c4a_pp_kbins_disc_state_t* s =
            c4a_pp_kbins_disc_state_new(n_bins, strategy);
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_kbins_disc_handle_t* h =
            new (std::nothrow) c4a_pp_kbins_disc_handle_t{s};
        if (h == nullptr) {
            c4a_pp_kbins_disc_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_kbins_disc_destroy(c4a_pp_kbins_disc_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_kbins_disc_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_kbins_disc_fit(c4a_pp_kbins_disc_handle_t* h,
                                            c4a_matrix_view_t X) {
    if (h == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        const double* xp = nullptr;
        std::int64_t  xr = 0, xc = 0;
        const c4a_status_t s = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        return c4a_pp_kbins_disc_state_fit(h->state, xp, xr, xc);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_pp_kbins_disc_is_fitted(
    const c4a_pp_kbins_disc_handle_t* h, int* out_fitted) {
    if (h == nullptr || out_fitted == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        *out_fitted = c4a_pp_kbins_disc_state_is_fitted(h->state);
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_pp_kbins_disc_transform(
    const c4a_pp_kbins_disc_handle_t* h,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out) {
    if (h == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        const double*  xp = nullptr;
        std::int32_t*  op = nullptr;
        std::int64_t   xr = 0, xc = 0, orr = 0, oc = 0;
        c4a_status_t   s  = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        s = require_rowmajor_i32_mut(out, op, orr, oc);
        if (s != C4A_OK) return s;
        if (xr != orr || xc != oc) {
            return C4A_ERR_SHAPE_MISMATCH;
        }
        return c4a_pp_kbins_disc_state_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ===========================================================================
// RangeDiscretizer (stateless, int32 output)
// ===========================================================================

C4A_API c4a_status_t c4a_pp_range_disc_create(c4a_pp_range_disc_handle_t** out,
                                               const double* bins,
                                               int64_t n_edges) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    if (n_edges < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (n_edges > 0 && bins == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        c4a_pp_range_disc_state_t* s =
            c4a_pp_range_disc_state_new(bins, n_edges);
        if (s == nullptr) {
            return C4A_ERR_INVALID_ARGUMENT;
        }
        c4a_pp_range_disc_handle_t* h =
            new (std::nothrow) c4a_pp_range_disc_handle_t{s};
        if (h == nullptr) {
            c4a_pp_range_disc_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_range_disc_destroy(c4a_pp_range_disc_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_range_disc_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_range_disc_transform(
    const c4a_pp_range_disc_handle_t* h,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out) {
    if (h == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        const double*  xp = nullptr;
        std::int32_t*  op = nullptr;
        std::int64_t   xr = 0, xc = 0, orr = 0, oc = 0;
        c4a_status_t   s  = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        s = require_rowmajor_i32_mut(out, op, orr, oc);
        if (s != C4A_OK) return s;
        if (xr != orr || xc != oc) {
            return C4A_ERR_SHAPE_MISMATCH;
        }
        return c4a_pp_range_disc_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

}  // extern "C"
