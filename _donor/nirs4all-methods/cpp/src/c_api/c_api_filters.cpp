// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for the Phase 12 filter operators (new ABI category
// `n4m_filter_*`). The single operator landed in Phase 12 is YOutlierFilter
// with four detection methods (IQR, Z-score, percentile, MAD).
//
// The public contract now mirrors the sklearn fit/transform split:
//   - `_create`     builds an unfitted handle.
//   - `_fit`        learns the bounds (idempotent — can be called more
//                   than once to refit).
//   - `_apply`      writes the keep-mask + stats using the fitted bounds.
//                   Returns N4M_ERR_NOT_FITTED when called before `_fit`.
//   - `_is_fitted`  reports the fitted state (0/1).
//   - `_destroy`    NULL-safe deleter.
//
// Universal rules of the wrappers (mirrored from c_api_preprocessing.cpp):
//   - Every entry point is wrapped in try/catch so no C++ exception ever
//     crosses the boundary.
//   - The opaque public type is a thin struct {engine_state*} that owns the
//     internal allocation; this decouples the public name from the internal
//     layout.
//   - `_create` takes a pointer-to-pointer out-arg; it returns
//     N4M_ERR_NULL_POINTER if `out` is NULL and N4M_ERR_INVALID_ARGUMENT for
//     invalid constructor parameters that the engine rejects. It writes
//     NULL to *out on every failure.
//   - `_destroy` is NULL-safe and never throws.
//   - `_apply` validates pointers, allocates no caller-visible memory (the
//     mask + stats buffers are caller-owned), and writes counts back through
//     the caller's stats pointer.
//
// The `n4m_filter_stats_t` public layout and the engine's
// `n4m_core_filter_stats_t` layout are byte-identical (same field order,
// same types). We rely on that and reinterpret_cast between them rather
// than re-copying.

#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>

#include "n4m/n4m.h"

#include "core/filters/y_outlier.h"

// Compile-time guard: confirm the engine struct matches the public layout.
static_assert(sizeof(n4m_core_filter_stats_t) == sizeof(n4m_filter_stats_t),
              "n4m_core_filter_stats_t / n4m_filter_stats_t layout mismatch");
static_assert(offsetof(n4m_core_filter_stats_t, n_samples) ==
              offsetof(n4m_filter_stats_t, n_samples),
              "n4m_filter_stats_t::n_samples offset mismatch");
static_assert(offsetof(n4m_core_filter_stats_t, n_kept) ==
              offsetof(n4m_filter_stats_t, n_kept),
              "n4m_filter_stats_t::n_kept offset mismatch");
static_assert(offsetof(n4m_core_filter_stats_t, n_excluded) ==
              offsetof(n4m_filter_stats_t, n_excluded),
              "n4m_filter_stats_t::n_excluded offset mismatch");
static_assert(offsetof(n4m_core_filter_stats_t, exclusion_rate) ==
              offsetof(n4m_filter_stats_t, exclusion_rate),
              "n4m_filter_stats_t::exclusion_rate offset mismatch");

// ---------------------------------------------------------------------------
// Opaque public handle for YOutlierFilter.
// ---------------------------------------------------------------------------

struct n4m_filter_y_outlier_handle_t {
    n4m_filter_y_outlier_state_t* state;
};

extern "C" {

// ---------------------------------------------------------------------------
// YOutlierFilter
// ---------------------------------------------------------------------------

N4M_API n4m_status_t n4m_filter_y_outlier_create(
    n4m_filter_y_outlier_handle_t** out,
    int32_t method,
    double threshold,
    double lower_pct, double upper_pct) {
    if (out == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    *out = nullptr;
    if (method != static_cast<int32_t>(N4M_Y_OUTLIER_IQR)    &&
        method != static_cast<int32_t>(N4M_Y_OUTLIER_ZSCORE) &&
        method != static_cast<int32_t>(N4M_Y_OUTLIER_PERCENTILE) &&
        method != static_cast<int32_t>(N4M_Y_OUTLIER_MAD)) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (threshold <= 0.0) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (!(lower_pct >= 0.0 && upper_pct <= 100.0 && lower_pct < upper_pct)) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    try {
        n4m_filter_y_outlier_state_t* s = n4m_filter_y_outlier_state_new(
            method, threshold, lower_pct, upper_pct);
        if (s == nullptr) {
            return N4M_ERR_OUT_OF_MEMORY;
        }
        n4m_filter_y_outlier_handle_t* h =
            new (std::nothrow) n4m_filter_y_outlier_handle_t{s};
        if (h == nullptr) {
            n4m_filter_y_outlier_state_free(s);
            return N4M_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API void n4m_filter_y_outlier_destroy(n4m_filter_y_outlier_handle_t* h) {
    if (h == nullptr) return;
    try {
        n4m_filter_y_outlier_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

N4M_API n4m_status_t n4m_filter_y_outlier_fit(
    n4m_filter_y_outlier_handle_t* h,
    const double* y, std::int64_t n) {
    if (h == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    if (n < 0) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (y == nullptr && n > 0) {
        return N4M_ERR_NULL_POINTER;
    }
    try {
        return n4m_filter_y_outlier_state_fit(h->state, y, n);
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_filter_y_outlier_apply(
    const n4m_filter_y_outlier_handle_t* h,
    const double* y, std::int64_t n,
    std::uint8_t* mask_out,
    n4m_filter_stats_t* stats_out) {
    if (h == nullptr || mask_out == nullptr || stats_out == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    if (n < 0) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (y == nullptr && n > 0) {
        return N4M_ERR_NULL_POINTER;
    }
    try {
        return n4m_filter_y_outlier_state_apply(
            h->state, y, n, mask_out,
            reinterpret_cast<n4m_core_filter_stats_t*>(stats_out));
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_filter_y_outlier_is_fitted(
    const n4m_filter_y_outlier_handle_t* h, int* out) {
    if (h == nullptr || out == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    try {
        return n4m_filter_y_outlier_state_is_fitted(h->state, out);
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

}  // extern "C"
