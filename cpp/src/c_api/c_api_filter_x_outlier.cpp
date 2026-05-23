// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for the Phase 13 XOutlierFilter
// (n4m_filter_x_outlier_*). Six selectable methods over one operator.
//
// Per the Phase 13 brief the new n4m_filter_x_outlier_method_t enum
// belongs in n4m.h §18 and gets added at integration; this wrapper
// uses int32_t for the public method argument so the source compiles
// against an un-modified n4m.h.
//
// Universal rules (mirrored from c_api_filters_meta.cpp):
//   - Every entry point is wrapped in try/catch so no C++ exception ever
//     crosses the boundary.
//   - The opaque public type is a thin struct {engine_state*} that owns the
//     internal allocation; this decouples the public name from the internal
//     layout.
//   - `_create` takes a pointer-to-pointer out-arg; it returns
//     N4M_ERR_NULL_POINTER if `out` is NULL and N4M_ERR_INVALID_ARGUMENT for
//     invalid constructor parameters. It writes NULL to *out on every
//     failure.
//   - `_destroy` is NULL-safe and never throws.
//
// The shared filter stats struct uses byte-identical layout with the public
// n4m_filter_stats_t (declared in n4m.h §18). We rely on that and
// reinterpret_cast between them rather than re-copying.

#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>

#include "n4m/n4m.h"

#include "core/filters/x_outlier.h"
#include "core/common/matrix_view.hpp"

// Layout check: the internal struct mirrors the public one. Both have the
// same field order (n_samples, n_kept, n_excluded, exclusion_rate).
static_assert(sizeof(n4m_core_x_filter_stats_t) == sizeof(n4m_filter_stats_t),
              "n4m_core_x_filter_stats_t / n4m_filter_stats_t layout mismatch");
static_assert(offsetof(n4m_core_x_filter_stats_t, n_samples) ==
              offsetof(n4m_filter_stats_t, n_samples),
              "n4m_filter_stats_t::n_samples offset mismatch");
static_assert(offsetof(n4m_core_x_filter_stats_t, n_kept) ==
              offsetof(n4m_filter_stats_t, n_kept),
              "n4m_filter_stats_t::n_kept offset mismatch");
static_assert(offsetof(n4m_core_x_filter_stats_t, n_excluded) ==
              offsetof(n4m_filter_stats_t, n_excluded),
              "n4m_filter_stats_t::n_excluded offset mismatch");
static_assert(offsetof(n4m_core_x_filter_stats_t, exclusion_rate) ==
              offsetof(n4m_filter_stats_t, exclusion_rate),
              "n4m_filter_stats_t::exclusion_rate offset mismatch");

// ---------------------------------------------------------------------------
// Opaque public handle.
// ---------------------------------------------------------------------------

struct n4m_filter_x_outlier_handle_t {
    n4m_filter_x_outlier_state_t* state;
};

// ---------------------------------------------------------------------------
// Forward declarations for the Phase 13 ABI surface. Until integration adds
// these to n4m.h §18 they live here so the compiler sees a declaration
// before the definition (silences -Wmissing-declarations).
// ---------------------------------------------------------------------------
extern "C" {
N4M_API n4m_status_t n4m_filter_x_outlier_create(
    n4m_filter_x_outlier_handle_t** out,
    std::int32_t method,
    int          use_threshold,
    double       threshold,
    std::int32_t n_components,
    double       contamination,
    std::uint64_t seed,
    std::int32_t n_estimators,
    std::int64_t max_samples);
N4M_API void n4m_filter_x_outlier_destroy(n4m_filter_x_outlier_handle_t* h);
N4M_API n4m_status_t n4m_filter_x_outlier_fit(
    n4m_filter_x_outlier_handle_t* h, n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_filter_x_outlier_is_fitted(
    const n4m_filter_x_outlier_handle_t* h, int* out_fitted);
N4M_API n4m_status_t n4m_filter_x_outlier_apply(
    const n4m_filter_x_outlier_handle_t* h,
    n4m_matrix_view_t X,
    std::uint8_t* mask_out,
    n4m_filter_stats_t* stats_out);
}  // extern "C"

namespace {

n4m_status_t require_rowmajor_f64(const n4m_matrix_view_t& v,
                                    const double*& out_ptr,
                                    std::int64_t& out_rows,
                                    std::int64_t& out_cols) noexcept {
    const n4m_status_t s = ::n4m::core::validate_nonnull_view(v);
    if (s != N4M_OK) return s;
    if (v.dtype != N4M_DTYPE_F64)         return N4M_ERR_DTYPE_MISMATCH;
    if (v.col_stride != 1)                return N4M_ERR_STRIDE_INVALID;
    if (v.rows > 0 && v.cols > 0 && v.row_stride != v.cols) {
        return N4M_ERR_STRIDE_INVALID;
    }
    out_ptr  = static_cast<const double*>(v.data);
    out_rows = v.rows;
    out_cols = v.cols;
    return N4M_OK;
}

}  // namespace

extern "C" {

// ---------------------------------------------------------------------------
// XOutlierFilter — 5 ABI entry points.
// ---------------------------------------------------------------------------

N4M_API n4m_status_t n4m_filter_x_outlier_create(
    n4m_filter_x_outlier_handle_t** out,
    std::int32_t method,
    int          use_threshold,
    double       threshold,
    std::int32_t n_components,
    double       contamination,
    std::uint64_t seed,
    std::int32_t n_estimators,
    std::int64_t max_samples) {
    if (out == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    *out = nullptr;
    try {
        n4m_filter_x_outlier_state_t* s = n4m_filter_x_outlier_state_new(
            method, use_threshold, threshold, n_components,
            contamination, seed, n_estimators, max_samples);
        if (s == nullptr) {
            return N4M_ERR_INVALID_ARGUMENT;
        }
        n4m_filter_x_outlier_handle_t* h =
            new (std::nothrow) n4m_filter_x_outlier_handle_t{s};
        if (h == nullptr) {
            n4m_filter_x_outlier_state_free(s);
            return N4M_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API void n4m_filter_x_outlier_destroy(n4m_filter_x_outlier_handle_t* h) {
    if (h == nullptr) return;
    try {
        n4m_filter_x_outlier_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

N4M_API n4m_status_t n4m_filter_x_outlier_fit(
    n4m_filter_x_outlier_handle_t* h,
    n4m_matrix_view_t X) {
    if (h == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;
        std::int64_t  xr = 0, xc = 0;
        const n4m_status_t s = require_rowmajor_f64(X, xp, xr, xc);
        if (s != N4M_OK) return s;
        return n4m_filter_x_outlier_state_fit(h->state, xp, xr, xc);
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_filter_x_outlier_is_fitted(
    const n4m_filter_x_outlier_handle_t* h, int* out_fitted) {
    if (h == nullptr || out_fitted == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    try {
        *out_fitted = n4m_filter_x_outlier_state_is_fitted(h->state);
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_filter_x_outlier_apply(
    const n4m_filter_x_outlier_handle_t* h,
    n4m_matrix_view_t X,
    std::uint8_t* mask_out,
    n4m_filter_stats_t* stats_out) {
    if (h == nullptr || mask_out == nullptr) {
        return N4M_ERR_NULL_POINTER;
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
        const n4m_status_t s = require_rowmajor_f64(X, xp, xr, xc);
        if (s != N4M_OK) return s;
        return n4m_filter_x_outlier_state_apply(
            h->state, xp, xr, xc, mask_out,
            reinterpret_cast<n4m_core_x_filter_stats_t*>(stats_out));
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

}  // extern "C"
