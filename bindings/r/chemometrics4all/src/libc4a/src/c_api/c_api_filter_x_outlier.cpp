// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for the Phase 13 XOutlierFilter
// (c4a_filter_x_outlier_*). Six selectable methods over one operator.
//
// Per the Phase 13 brief the new c4a_filter_x_outlier_method_t enum
// belongs in c4a.h §18 and gets added at integration; this wrapper
// uses int32_t for the public method argument so the source compiles
// against an un-modified c4a.h.
//
// Universal rules (mirrored from c_api_filters_meta.cpp):
//   - Every entry point is wrapped in try/catch so no C++ exception ever
//     crosses the boundary.
//   - The opaque public type is a thin struct {engine_state*} that owns the
//     internal allocation; this decouples the public name from the internal
//     layout.
//   - `_create` takes a pointer-to-pointer out-arg; it returns
//     C4A_ERR_NULL_POINTER if `out` is NULL and C4A_ERR_INVALID_ARGUMENT for
//     invalid constructor parameters. It writes NULL to *out on every
//     failure.
//   - `_destroy` is NULL-safe and never throws.
//
// The shared filter stats struct uses byte-identical layout with the public
// c4a_filter_stats_t (declared in c4a.h §18). We rely on that and
// reinterpret_cast between them rather than re-copying.

#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>

#include "chemometrics4all/c4a.h"

#include "core/filters/x_outlier.h"
#include "core/matrix_view.hpp"

// Layout check: the internal struct mirrors the public one. Both have the
// same field order (n_samples, n_kept, n_excluded, exclusion_rate).
static_assert(sizeof(c4a_core_x_filter_stats_t) == sizeof(c4a_filter_stats_t),
              "c4a_core_x_filter_stats_t / c4a_filter_stats_t layout mismatch");
static_assert(offsetof(c4a_core_x_filter_stats_t, n_samples) ==
              offsetof(c4a_filter_stats_t, n_samples),
              "c4a_filter_stats_t::n_samples offset mismatch");
static_assert(offsetof(c4a_core_x_filter_stats_t, n_kept) ==
              offsetof(c4a_filter_stats_t, n_kept),
              "c4a_filter_stats_t::n_kept offset mismatch");
static_assert(offsetof(c4a_core_x_filter_stats_t, n_excluded) ==
              offsetof(c4a_filter_stats_t, n_excluded),
              "c4a_filter_stats_t::n_excluded offset mismatch");
static_assert(offsetof(c4a_core_x_filter_stats_t, exclusion_rate) ==
              offsetof(c4a_filter_stats_t, exclusion_rate),
              "c4a_filter_stats_t::exclusion_rate offset mismatch");

// ---------------------------------------------------------------------------
// Opaque public handle.
// ---------------------------------------------------------------------------

struct c4a_filter_x_outlier_handle_t {
    c4a_filter_x_outlier_state_t* state;
};

// ---------------------------------------------------------------------------
// Forward declarations for the Phase 13 ABI surface. Until integration adds
// these to c4a.h §18 they live here so the compiler sees a declaration
// before the definition (silences -Wmissing-declarations).
// ---------------------------------------------------------------------------
extern "C" {
C4A_API c4a_status_t c4a_filter_x_outlier_create(
    c4a_filter_x_outlier_handle_t** out,
    std::int32_t method,
    int          use_threshold,
    double       threshold,
    std::int32_t n_components,
    double       contamination,
    std::uint64_t seed,
    std::int32_t n_estimators,
    std::int64_t max_samples);
C4A_API void c4a_filter_x_outlier_destroy(c4a_filter_x_outlier_handle_t* h);
C4A_API c4a_status_t c4a_filter_x_outlier_fit(
    c4a_filter_x_outlier_handle_t* h, c4a_matrix_view_t X);
C4A_API c4a_status_t c4a_filter_x_outlier_is_fitted(
    const c4a_filter_x_outlier_handle_t* h, int* out_fitted);
C4A_API c4a_status_t c4a_filter_x_outlier_apply(
    const c4a_filter_x_outlier_handle_t* h,
    c4a_matrix_view_t X,
    std::uint8_t* mask_out,
    c4a_filter_stats_t* stats_out);
}  // extern "C"

namespace {

c4a_status_t require_rowmajor_f64(const c4a_matrix_view_t& v,
                                    const double*& out_ptr,
                                    std::int64_t& out_rows,
                                    std::int64_t& out_cols) noexcept {
    const c4a_status_t s = ::chemometrics4all::core::validate_nonnull_view(v);
    if (s != C4A_OK) return s;
    if (v.dtype != C4A_DTYPE_F64)         return C4A_ERR_DTYPE_MISMATCH;
    if (v.col_stride != 1)                return C4A_ERR_STRIDE_INVALID;
    if (v.rows > 0 && v.cols > 0 && v.row_stride != v.cols) {
        return C4A_ERR_STRIDE_INVALID;
    }
    out_ptr  = static_cast<const double*>(v.data);
    out_rows = v.rows;
    out_cols = v.cols;
    return C4A_OK;
}

}  // namespace

extern "C" {

// ---------------------------------------------------------------------------
// XOutlierFilter — 5 ABI entry points.
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_filter_x_outlier_create(
    c4a_filter_x_outlier_handle_t** out,
    std::int32_t method,
    int          use_threshold,
    double       threshold,
    std::int32_t n_components,
    double       contamination,
    std::uint64_t seed,
    std::int32_t n_estimators,
    std::int64_t max_samples) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    try {
        c4a_filter_x_outlier_state_t* s = c4a_filter_x_outlier_state_new(
            method, use_threshold, threshold, n_components,
            contamination, seed, n_estimators, max_samples);
        if (s == nullptr) {
            return C4A_ERR_INVALID_ARGUMENT;
        }
        c4a_filter_x_outlier_handle_t* h =
            new (std::nothrow) c4a_filter_x_outlier_handle_t{s};
        if (h == nullptr) {
            c4a_filter_x_outlier_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_filter_x_outlier_destroy(c4a_filter_x_outlier_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_filter_x_outlier_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_filter_x_outlier_fit(
    c4a_filter_x_outlier_handle_t* h,
    c4a_matrix_view_t X) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;
        std::int64_t  xr = 0, xc = 0;
        const c4a_status_t s = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        return c4a_filter_x_outlier_state_fit(h->state, xp, xr, xc);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_filter_x_outlier_is_fitted(
    const c4a_filter_x_outlier_handle_t* h, int* out_fitted) {
    if (h == nullptr || out_fitted == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        *out_fitted = c4a_filter_x_outlier_state_is_fitted(h->state);
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_filter_x_outlier_apply(
    const c4a_filter_x_outlier_handle_t* h,
    c4a_matrix_view_t X,
    std::uint8_t* mask_out,
    c4a_filter_stats_t* stats_out) {
    if (h == nullptr || mask_out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        if (stats_out != nullptr) {
            stats_out->n_samples = 0;
            stats_out->n_kept = 0;
            stats_out->n_excluded = 0;
            stats_out->exclusion_rate = 0.0;
        }
        const double* xp = nullptr;
        std::int64_t  xr = 0, xc = 0;
        const c4a_status_t s = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        return c4a_filter_x_outlier_state_apply(
            h->state, xp, xr, xc, mask_out,
            reinterpret_cast<c4a_core_x_filter_stats_t*>(stats_out));
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

}  // extern "C"
