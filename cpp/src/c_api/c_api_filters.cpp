// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for the Phase 12 filter operators (new ABI category
// `c4a_filter_*`). The single operator landed in Phase 12 is YOutlierFilter
// with four detection methods (IQR, Z-score, percentile, MAD).
//
// The public contract now mirrors the sklearn fit/transform split:
//   - `_create`     builds an unfitted handle.
//   - `_fit`        learns the bounds (idempotent — can be called more
//                   than once to refit).
//   - `_apply`      writes the keep-mask + stats using the fitted bounds.
//                   Returns C4A_ERR_NOT_FITTED when called before `_fit`.
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
//     C4A_ERR_NULL_POINTER if `out` is NULL and C4A_ERR_INVALID_ARGUMENT for
//     invalid constructor parameters that the engine rejects. It writes
//     NULL to *out on every failure.
//   - `_destroy` is NULL-safe and never throws.
//   - `_apply` validates pointers, allocates no caller-visible memory (the
//     mask + stats buffers are caller-owned), and writes counts back through
//     the caller's stats pointer.
//
// The `c4a_filter_stats_t` public layout and the engine's
// `c4a_core_filter_stats_t` layout are byte-identical (same field order,
// same types). We rely on that and reinterpret_cast between them rather
// than re-copying.

#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>

#include "chemometrics4all/c4a.h"

#include "core/filters/y_outlier.h"

// Compile-time guard: confirm the engine struct matches the public layout.
static_assert(sizeof(c4a_core_filter_stats_t) == sizeof(c4a_filter_stats_t),
              "c4a_core_filter_stats_t / c4a_filter_stats_t layout mismatch");
static_assert(offsetof(c4a_core_filter_stats_t, n_samples) ==
              offsetof(c4a_filter_stats_t, n_samples),
              "c4a_filter_stats_t::n_samples offset mismatch");
static_assert(offsetof(c4a_core_filter_stats_t, n_kept) ==
              offsetof(c4a_filter_stats_t, n_kept),
              "c4a_filter_stats_t::n_kept offset mismatch");
static_assert(offsetof(c4a_core_filter_stats_t, n_excluded) ==
              offsetof(c4a_filter_stats_t, n_excluded),
              "c4a_filter_stats_t::n_excluded offset mismatch");
static_assert(offsetof(c4a_core_filter_stats_t, exclusion_rate) ==
              offsetof(c4a_filter_stats_t, exclusion_rate),
              "c4a_filter_stats_t::exclusion_rate offset mismatch");

// ---------------------------------------------------------------------------
// Opaque public handle for YOutlierFilter.
// ---------------------------------------------------------------------------

struct c4a_filter_y_outlier_handle_t {
    c4a_filter_y_outlier_state_t* state;
};

extern "C" {

// ---------------------------------------------------------------------------
// YOutlierFilter
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_filter_y_outlier_create(
    c4a_filter_y_outlier_handle_t** out,
    int32_t method,
    double threshold,
    double lower_pct, double upper_pct) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    if (method != static_cast<int32_t>(C4A_Y_OUTLIER_IQR)    &&
        method != static_cast<int32_t>(C4A_Y_OUTLIER_ZSCORE) &&
        method != static_cast<int32_t>(C4A_Y_OUTLIER_PERCENTILE) &&
        method != static_cast<int32_t>(C4A_Y_OUTLIER_MAD)) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (threshold <= 0.0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (!(lower_pct >= 0.0 && upper_pct <= 100.0 && lower_pct < upper_pct)) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        c4a_filter_y_outlier_state_t* s = c4a_filter_y_outlier_state_new(
            method, threshold, lower_pct, upper_pct);
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_filter_y_outlier_handle_t* h =
            new (std::nothrow) c4a_filter_y_outlier_handle_t{s};
        if (h == nullptr) {
            c4a_filter_y_outlier_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_filter_y_outlier_destroy(c4a_filter_y_outlier_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_filter_y_outlier_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_filter_y_outlier_fit(
    c4a_filter_y_outlier_handle_t* h,
    const double* y, std::int64_t n) {
    if (h == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    if (n < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (y == nullptr && n > 0) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        return c4a_filter_y_outlier_state_fit(h->state, y, n);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_filter_y_outlier_apply(
    const c4a_filter_y_outlier_handle_t* h,
    const double* y, std::int64_t n,
    std::uint8_t* mask_out,
    c4a_filter_stats_t* stats_out) {
    if (h == nullptr || mask_out == nullptr || stats_out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    if (n < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (y == nullptr && n > 0) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        return c4a_filter_y_outlier_state_apply(
            h->state, y, n, mask_out,
            reinterpret_cast<c4a_core_filter_stats_t*>(stats_out));
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_filter_y_outlier_is_fitted(
    const c4a_filter_y_outlier_handle_t* h, int* out) {
    if (h == nullptr || out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        return c4a_filter_y_outlier_state_is_fitted(h->state, out);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

}  // extern "C"
