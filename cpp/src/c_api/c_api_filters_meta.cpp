// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for the Phase 14 meta sample filters
// (HighLeverage, SpectralQuality, Composite).
//
// IMPORTANT — Phase 12 / Phase 14 coordination:
//
// Phase 12 introduces the `c4a_filter_*` ABI category, including the shared
// result type `c4a_filter_stats_t` and the prefix wiring in c4a.h §13+.
// Phase 14 (this file) implements three additional filters that plug into
// the same category. To allow the two phases to land in parallel without
// stepping on each other's c4a.h edits, this file declares the new public
// symbols inline with extern "C" linkage and reuses Phase 12's
// `c4a_filter_stats_t` definition guarded by a feature macro so the merge
// is a no-op once both phases reach main.
//
// Until Phase 12 lands in c4a.h, the symbols are still exported by the
// shared library (the Linux version script uses `c4a_*` as a wildcard),
// but they are not yet visible through `#include "chemometrics4all/c4a.h"`.
// The test suite includes `c_api_filters_meta.h` directly to access them.
//
// Universal rules (mirroring c_api_preprocessing.cpp):
//   - Every entry point is wrapped in try/catch so no C++ exception ever
//     crosses the boundary.
//   - The opaque public types are thin structs {engine_state*} that own the
//     internal allocation; this decouples the public name from the internal
//     layout.
//   - `_create` takes a pointer-to-pointer out-arg; it returns
//     C4A_ERR_NULL_POINTER if `out` is NULL and C4A_ERR_INVALID_ARGUMENT
//     for invalid constructor parameters. It writes NULL to *out on every
//     failure.
//   - `_destroy` is NULL-safe and never throws.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <new>
#include <vector>

#include "chemometrics4all/c4a.h"

#include "core/filters/composite.h"
#include "core/filters/high_leverage.h"
#include "core/filters/spectral_quality.h"
#include "core/matrix_view.hpp"

// ---------------------------------------------------------------------------
// Opaque public handles.
// ---------------------------------------------------------------------------

struct c4a_filter_leverage_handle_t {
    c4a_filter_leverage_state_t* state;
};
struct c4a_filter_quality_handle_t {
    c4a_filter_quality_state_t* state;
};
struct c4a_filter_composite_handle_t {
    int mode;                                                       // ANY=0 / ALL=1
    std::vector<c4a_filter_leverage_handle_t*>  leverage_subs;
    std::vector<c4a_filter_quality_handle_t*>   quality_subs;
    // Phase 12 introduces y_outlier (and friends); coordination hook below
    // expands the vector list at merge time. Each kind has its own apply()
    // entry point because the Phase 12 handles are opaque to us.
};

namespace {

// Validate that a matrix view is non-NULL, F64, row-major contiguous.
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

void zero_stats(c4a_filter_stats_t* stats) noexcept {
    if (stats == nullptr) return;
    stats->n_samples      = 0;
    stats->n_kept         = 0;
    stats->n_excluded     = 0;
    stats->exclusion_rate = 0.0;
}

void fill_stats(c4a_filter_stats_t* stats, int64_t ns, int64_t nk,
                int64_t nx) noexcept {
    if (stats == nullptr) return;
    stats->n_samples      = ns;
    stats->n_kept         = nk;
    stats->n_excluded     = nx;
    stats->exclusion_rate = (ns > 0) ? (static_cast<double>(nx) /
                                         static_cast<double>(ns)) : 0.0;
}

}  // namespace

extern "C" {

// ---------------------------------------------------------------------------
// HighLeverageFilter
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_filter_leverage_create(
    c4a_filter_leverage_handle_t** out,
    int32_t method,
    double  threshold_multiplier,
    int     use_absolute_threshold,
    double  absolute_threshold,
    int32_t n_components,
    int     center) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    try {
        c4a_filter_leverage_state_t* s = c4a_filter_leverage_state_new(
            method, threshold_multiplier,
            use_absolute_threshold, absolute_threshold,
            n_components, center);
        if (s == nullptr) {
            return C4A_ERR_INVALID_ARGUMENT;
        }
        c4a_filter_leverage_handle_t* h =
            new (std::nothrow) c4a_filter_leverage_handle_t{s};
        if (h == nullptr) {
            c4a_filter_leverage_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_filter_leverage_destroy(c4a_filter_leverage_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_filter_leverage_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_filter_leverage_fit(c4a_filter_leverage_handle_t* h,
                                              c4a_matrix_view_t X) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;
        std::int64_t  xr = 0, xc = 0;
        c4a_status_t s = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        return c4a_filter_leverage_state_fit(h->state, xp, xr, xc);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_filter_leverage_is_fitted(
    const c4a_filter_leverage_handle_t* h, int* out_fitted) {
    if (h == nullptr || out_fitted == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        *out_fitted = c4a_filter_leverage_state_is_fitted(h->state);
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_filter_leverage_apply(
    const c4a_filter_leverage_handle_t* h,
    c4a_matrix_view_t X,
    uint8_t* mask_out,
    c4a_filter_stats_t* stats_out) {
    if (h == nullptr || mask_out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        zero_stats(stats_out);
        const double* xp = nullptr;
        std::int64_t  xr = 0, xc = 0;
        c4a_status_t s = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        std::int64_t ns = 0, nk = 0, nx = 0;
        s = c4a_filter_leverage_state_apply(h->state, xp, xr, xc,
                                             mask_out, &ns, &nk, &nx);
        if (s != C4A_OK) return s;
        fill_stats(stats_out, ns, nk, nx);
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API double c4a_filter_leverage_threshold(
    const c4a_filter_leverage_handle_t* h) {
    if (h == nullptr) return 0.0;
    try {
        return c4a_filter_leverage_state_threshold(h->state);
    } catch (...) {
        return 0.0;
    }
}

// ---------------------------------------------------------------------------
// SpectralQualityFilter
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_filter_quality_create(
    c4a_filter_quality_handle_t** out,
    double max_nan_ratio,
    double max_zero_ratio,
    double min_variance,
    int    use_max,
    double max_value,
    int    use_min,
    double min_value,
    int    check_inf) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    try {
        c4a_filter_quality_state_t* s = c4a_filter_quality_state_new(
            max_nan_ratio, max_zero_ratio, min_variance,
            use_max, max_value, use_min, min_value, check_inf);
        if (s == nullptr) {
            return C4A_ERR_INVALID_ARGUMENT;
        }
        c4a_filter_quality_handle_t* h =
            new (std::nothrow) c4a_filter_quality_handle_t{s};
        if (h == nullptr) {
            c4a_filter_quality_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_filter_quality_destroy(c4a_filter_quality_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_filter_quality_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_filter_quality_apply(
    const c4a_filter_quality_handle_t* h,
    c4a_matrix_view_t X,
    uint8_t* mask_out,
    c4a_filter_stats_t* stats_out) {
    if (h == nullptr || mask_out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        zero_stats(stats_out);
        const double* xp = nullptr;
        std::int64_t  xr = 0, xc = 0;
        c4a_status_t s = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        std::int64_t ns = 0, nk = 0, nx = 0;
        s = c4a_filter_quality_state_apply(h->state, xp, xr, xc,
                                            mask_out, &ns, &nk, &nx);
        if (s != C4A_OK) return s;
        fill_stats(stats_out, ns, nk, nx);
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// CompositeFilter
// ---------------------------------------------------------------------------
//
// The composite holds references to its sub-filter handles. It does NOT own
// them — the caller must keep the sub-filter handles alive until the
// composite is destroyed and free them separately afterwards.

C4A_API c4a_status_t c4a_filter_composite_create(
    c4a_filter_composite_handle_t** out,
    c4a_composite_mode_t mode) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    if (mode != C4A_COMPOSITE_ANY && mode != C4A_COMPOSITE_ALL) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        auto* h = new (std::nothrow) c4a_filter_composite_handle_t{};
        if (h == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        h->mode = static_cast<int>(mode);
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_filter_composite_destroy(c4a_filter_composite_handle_t* h) {
    if (h == nullptr) return;
    try {
        delete h;  // does NOT free sub-filters
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_filter_composite_add_leverage(
    c4a_filter_composite_handle_t* h,
    c4a_filter_leverage_handle_t* sub) {
    if (h == nullptr || sub == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        h->leverage_subs.push_back(sub);
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_OUT_OF_MEMORY;
    }
}

C4A_API c4a_status_t c4a_filter_composite_add_quality(
    c4a_filter_composite_handle_t* h,
    c4a_filter_quality_handle_t* sub) {
    if (h == nullptr || sub == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        h->quality_subs.push_back(sub);
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_OUT_OF_MEMORY;
    }
}

C4A_API c4a_status_t c4a_filter_composite_apply(
    const c4a_filter_composite_handle_t* h,
    c4a_matrix_view_t X,
    uint8_t* mask_out,
    c4a_filter_stats_t* stats_out) {
    if (h == nullptr || mask_out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        zero_stats(stats_out);
        const double* xp = nullptr;
        std::int64_t  xr = 0, xc = 0;
        c4a_status_t s = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;

        const std::size_t n_lev = h->leverage_subs.size();
        const std::size_t n_q   = h->quality_subs.size();
        const std::size_t n_sub = n_lev + n_q;

        // Each sub-filter contributes a per-row keep mask of length xr.
        std::vector<std::vector<std::uint8_t>> submasks(n_sub);
        std::vector<const std::uint8_t*> submask_ptrs(n_sub, nullptr);

        std::size_t idx = 0;
        for (auto* sub : h->leverage_subs) {
            submasks[idx].assign(static_cast<std::size_t>(xr), 0);
            std::int64_t ns = 0, nk = 0, nx = 0;
            const c4a_status_t st = c4a_filter_leverage_state_apply(
                sub->state, xp, xr, xc,
                submasks[idx].data(), &ns, &nk, &nx);
            if (st != C4A_OK) return st;
            submask_ptrs[idx] = submasks[idx].data();
            ++idx;
        }
        for (auto* sub : h->quality_subs) {
            submasks[idx].assign(static_cast<std::size_t>(xr), 0);
            std::int64_t ns = 0, nk = 0, nx = 0;
            const c4a_status_t st = c4a_filter_quality_state_apply(
                sub->state, xp, xr, xc,
                submasks[idx].data(), &ns, &nk, &nx);
            if (st != C4A_OK) return st;
            submask_ptrs[idx] = submasks[idx].data();
            ++idx;
        }

        std::int64_t ns = 0, nk = 0, nx = 0;
        const c4a_status_t st = c4a_filter_composite_combine(
            h->mode,
            submask_ptrs.empty() ? nullptr : submask_ptrs.data(),
            static_cast<int>(n_sub),
            xr,
            mask_out,
            &ns, &nk, &nx);
        if (st != C4A_OK) return st;
        fill_stats(stats_out, ns, nk, nx);
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

}  // extern "C"
