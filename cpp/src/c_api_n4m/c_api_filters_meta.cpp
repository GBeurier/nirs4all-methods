// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for the Phase 14 meta sample filters
// (HighLeverage, SpectralQuality, Composite). The public symbols declared
// in n4m.h §19 are implemented here on top of the engines under
// core/filters/. Each entry point wraps the engine call in try/catch,
// validates the matrix view contract, and routes through `fill_stats` so
// the shared `n4m_filter_stats_t` (declared in n4m.h §18) is populated
// consistently — including `exclusion_rate` — across leverage, quality
// and composite.
//
// Universal rules (mirroring c_api_preprocessing.cpp):
//   - Every entry point is wrapped in try/catch so no C++ exception ever
//     crosses the boundary.
//   - The opaque public types are thin structs {engine_state*} that own the
//     internal allocation; this decouples the public name from the internal
//     layout.
//   - `_create` takes a pointer-to-pointer out-arg; it returns
//     N4M_ERR_NULL_POINTER if `out` is NULL and N4M_ERR_INVALID_ARGUMENT
//     for invalid constructor parameters. It writes NULL to *out on every
//     failure.
//   - `_destroy` is NULL-safe and never throws.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <new>
#include <vector>

#include "n4m/n4m.h"

#include "core/filters/composite.h"
#include "core/filters/high_leverage.h"
#include "core/filters/spectral_quality.h"
#include "core/common/matrix_view.hpp"

// ---------------------------------------------------------------------------
// Opaque public handles.
// ---------------------------------------------------------------------------

struct n4m_filter_leverage_handle_t {
    n4m_filter_leverage_state_t* state;
};
struct n4m_filter_quality_handle_t {
    n4m_filter_quality_state_t* state;
};
struct n4m_filter_composite_handle_t {
    int mode;                                                       // ANY=0 / ALL=1
    // One vector per supported sub-filter kind. Each kind has its own
    // apply() entry point because the underlying handles are opaque to
    // each other; the composite re-evaluates every sub-filter on apply.
    std::vector<n4m_filter_leverage_handle_t*>  leverage_subs;
    std::vector<n4m_filter_quality_handle_t*>   quality_subs;
};

namespace {

// Validate that a matrix view is non-NULL, F64, row-major contiguous.
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

void zero_stats(n4m_filter_stats_t* stats) noexcept {
    if (stats == nullptr) return;
    stats->n_samples      = 0;
    stats->n_kept         = 0;
    stats->n_excluded     = 0;
    stats->exclusion_rate = 0.0;
}

void fill_stats(n4m_filter_stats_t* stats, int64_t ns, int64_t nk,
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

N4M_API n4m_status_t n4m_filter_leverage_create(
    n4m_filter_leverage_handle_t** out,
    int32_t method,
    double  threshold_multiplier,
    int     use_absolute_threshold,
    double  absolute_threshold,
    int32_t n_components,
    int     center) {
    if (out == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    *out = nullptr;
    try {
        n4m_filter_leverage_state_t* s = n4m_filter_leverage_state_new(
            method, threshold_multiplier,
            use_absolute_threshold, absolute_threshold,
            n_components, center);
        if (s == nullptr) {
            return N4M_ERR_INVALID_ARGUMENT;
        }
        n4m_filter_leverage_handle_t* h =
            new (std::nothrow) n4m_filter_leverage_handle_t{s};
        if (h == nullptr) {
            n4m_filter_leverage_state_free(s);
            return N4M_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API void n4m_filter_leverage_destroy(n4m_filter_leverage_handle_t* h) {
    if (h == nullptr) return;
    try {
        n4m_filter_leverage_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

N4M_API n4m_status_t n4m_filter_leverage_fit(n4m_filter_leverage_handle_t* h,
                                              n4m_matrix_view_t X) {
    if (h == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;
        std::int64_t  xr = 0, xc = 0;
        n4m_status_t s = require_rowmajor_f64(X, xp, xr, xc);
        if (s != N4M_OK) return s;
        return n4m_filter_leverage_state_fit(h->state, xp, xr, xc);
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_filter_leverage_is_fitted(
    const n4m_filter_leverage_handle_t* h, int* out_fitted) {
    if (h == nullptr || out_fitted == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    try {
        *out_fitted = n4m_filter_leverage_state_is_fitted(h->state);
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_filter_leverage_apply(
    const n4m_filter_leverage_handle_t* h,
    n4m_matrix_view_t X,
    uint8_t* mask_out,
    n4m_filter_stats_t* stats_out) {
    if (h == nullptr || mask_out == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    try {
        zero_stats(stats_out);
        const double* xp = nullptr;
        std::int64_t  xr = 0, xc = 0;
        n4m_status_t s = require_rowmajor_f64(X, xp, xr, xc);
        if (s != N4M_OK) return s;
        std::int64_t ns = 0, nk = 0, nx = 0;
        s = n4m_filter_leverage_state_apply(h->state, xp, xr, xc,
                                             mask_out, &ns, &nk, &nx);
        if (s != N4M_OK) return s;
        fill_stats(stats_out, ns, nk, nx);
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API double n4m_filter_leverage_threshold(
    const n4m_filter_leverage_handle_t* h) {
    if (h == nullptr) return 0.0;
    try {
        return n4m_filter_leverage_state_threshold(h->state);
    } catch (...) {
        return 0.0;
    }
}

// ---------------------------------------------------------------------------
// SpectralQualityFilter
// ---------------------------------------------------------------------------

N4M_API n4m_status_t n4m_filter_quality_create(
    n4m_filter_quality_handle_t** out,
    double max_nan_ratio,
    double max_zero_ratio,
    double min_variance,
    int    use_max,
    double max_value,
    int    use_min,
    double min_value,
    int    check_inf) {
    if (out == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    *out = nullptr;
    try {
        n4m_filter_quality_state_t* s = n4m_filter_quality_state_new(
            max_nan_ratio, max_zero_ratio, min_variance,
            use_max, max_value, use_min, min_value, check_inf);
        if (s == nullptr) {
            return N4M_ERR_INVALID_ARGUMENT;
        }
        n4m_filter_quality_handle_t* h =
            new (std::nothrow) n4m_filter_quality_handle_t{s};
        if (h == nullptr) {
            n4m_filter_quality_state_free(s);
            return N4M_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API void n4m_filter_quality_destroy(n4m_filter_quality_handle_t* h) {
    if (h == nullptr) return;
    try {
        n4m_filter_quality_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

N4M_API n4m_status_t n4m_filter_quality_apply(
    const n4m_filter_quality_handle_t* h,
    n4m_matrix_view_t X,
    uint8_t* mask_out,
    n4m_filter_stats_t* stats_out) {
    if (h == nullptr || mask_out == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    try {
        zero_stats(stats_out);
        const double* xp = nullptr;
        std::int64_t  xr = 0, xc = 0;
        n4m_status_t s = require_rowmajor_f64(X, xp, xr, xc);
        if (s != N4M_OK) return s;
        std::int64_t ns = 0, nk = 0, nx = 0;
        s = n4m_filter_quality_state_apply(h->state, xp, xr, xc,
                                            mask_out, &ns, &nk, &nx);
        if (s != N4M_OK) return s;
        fill_stats(stats_out, ns, nk, nx);
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// CompositeFilter
// ---------------------------------------------------------------------------
//
// The composite holds references to its sub-filter handles. It does NOT own
// them — the caller must keep the sub-filter handles alive until the
// composite is destroyed and free them separately afterwards.

N4M_API n4m_status_t n4m_filter_composite_create(
    n4m_filter_composite_handle_t** out,
    int32_t mode) {
    if (out == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    *out = nullptr;
    if (mode != static_cast<int32_t>(N4M_COMPOSITE_ANY)
        && mode != static_cast<int32_t>(N4M_COMPOSITE_ALL)) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    try {
        auto* h = new (std::nothrow) n4m_filter_composite_handle_t{};
        if (h == nullptr) {
            return N4M_ERR_OUT_OF_MEMORY;
        }
        h->mode = static_cast<int>(mode);
        *out = h;
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API void n4m_filter_composite_destroy(n4m_filter_composite_handle_t* h) {
    if (h == nullptr) return;
    try {
        delete h;  // does NOT free sub-filters
    } catch (...) {
        // swallow
    }
}

N4M_API n4m_status_t n4m_filter_composite_add_leverage(
    n4m_filter_composite_handle_t* h,
    n4m_filter_leverage_handle_t* sub) {
    if (h == nullptr || sub == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    try {
        h->leverage_subs.push_back(sub);
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_OUT_OF_MEMORY;
    }
}

N4M_API n4m_status_t n4m_filter_composite_add_quality(
    n4m_filter_composite_handle_t* h,
    n4m_filter_quality_handle_t* sub) {
    if (h == nullptr || sub == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    try {
        h->quality_subs.push_back(sub);
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_OUT_OF_MEMORY;
    }
}

N4M_API n4m_status_t n4m_filter_composite_apply(
    const n4m_filter_composite_handle_t* h,
    n4m_matrix_view_t X,
    uint8_t* mask_out,
    n4m_filter_stats_t* stats_out) {
    if (h == nullptr || mask_out == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    try {
        zero_stats(stats_out);
        const double* xp = nullptr;
        std::int64_t  xr = 0, xc = 0;
        n4m_status_t s = require_rowmajor_f64(X, xp, xr, xc);
        if (s != N4M_OK) return s;

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
            const n4m_status_t st = n4m_filter_leverage_state_apply(
                sub->state, xp, xr, xc,
                submasks[idx].data(), &ns, &nk, &nx);
            if (st != N4M_OK) return st;
            submask_ptrs[idx] = submasks[idx].data();
            ++idx;
        }
        for (auto* sub : h->quality_subs) {
            submasks[idx].assign(static_cast<std::size_t>(xr), 0);
            std::int64_t ns = 0, nk = 0, nx = 0;
            const n4m_status_t st = n4m_filter_quality_state_apply(
                sub->state, xp, xr, xc,
                submasks[idx].data(), &ns, &nk, &nx);
            if (st != N4M_OK) return st;
            submask_ptrs[idx] = submasks[idx].data();
            ++idx;
        }

        std::int64_t ns = 0, nk = 0, nx = 0;
        const n4m_status_t st = n4m_filter_composite_combine(
            h->mode,
            submask_ptrs.empty() ? nullptr : submask_ptrs.data(),
            static_cast<int>(n_sub),
            xr,
            mask_out,
            &ns, &nk, &nx);
        if (st != N4M_OK) return st;
        fill_stats(stats_out, ns, nk, nx);
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

}  // extern "C"
