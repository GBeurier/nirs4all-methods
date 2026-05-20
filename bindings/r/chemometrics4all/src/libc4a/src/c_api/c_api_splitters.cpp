// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for the Phase 11 sample-partitioning splitters
// (c4a_split_* category). Each operator delegates to its engine under
// cpp/src/core/splitters/. The boundary rules mirror the preprocessing
// wrappers in c_api_preprocessing.cpp:
//
//   - Every entry point is wrapped in try/catch so no C++ exception ever
//     crosses the boundary.
//   - The opaque public handle is a thin struct {engine_state*} that owns
//     the internal allocation.
//   - `_create` takes a pointer-to-pointer out-arg; returns
//     C4A_ERR_NULL_POINTER if `out` is NULL, C4A_ERR_INVALID_ARGUMENT for
//     invalid constructor parameters, C4A_ERR_OUT_OF_MEMORY on allocation
//     failure. Writes NULL to *out on every failure.
//   - `_destroy` is NULL-safe and never throws.
//   - `_split` writes train/test index arrays into a caller-provided
//     `c4a_split_result_t*`. The result must be destroyed via
//     `c4a_split_result_destroy` after consumption.
//
// The opaque public types `c4a_split_<op>_handle_t` and `c4a_split_result_t`
// are declared in c4a.h (§17). The definitions live in the engine headers
// and this TU.
//
// NOTE: c4a.h declarations for §17 are appended centrally by the
// integration step that ships this phase. This wrapper assumes the public
// surface is already in place at build time.

#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>

#include "chemometrics4all/c4a.h"

#include "core/matrix_view.hpp"
#include "core/splitters/binned_strat_group_kfold.h"
#include "core/splitters/kbins_stratified.h"
#include "core/splitters/kennard_stone.h"
#include "core/splitters/kmeans.h"
#include "core/splitters/split_common.h"
#include "core/splitters/split_splitter.h"
#include "core/splitters/spxy.h"
#include "core/splitters/spxy_fold.h"
#include "core/splitters/spxy_g_fold.h"
#include "core/splitters/systematic_circular.h"

// ---------------------------------------------------------------------------
// Opaque public handles.
// ---------------------------------------------------------------------------
struct c4a_split_kennard_stone_handle_t {
    c4a_split_ks_state_t* state;
};
struct c4a_split_spxy_handle_t {
    c4a_split_spxy_state_t* state;
};
struct c4a_split_spxy_fold_handle_t {
    c4a_split_spxy_fold_state_t* state;
};
struct c4a_split_spxy_g_fold_handle_t {
    c4a_split_spxy_g_fold_state_t* state;
};
struct c4a_split_kmeans_handle_t {
    c4a_split_kmeans_state_t* state;
};
struct c4a_split_kbins_stratified_handle_t {
    c4a_split_kbins_state_t* state;
};
struct c4a_split_binned_strat_group_kfold_handle_t {
    c4a_split_bsgk_state_t* state;
};
struct c4a_split_systematic_circular_handle_t {
    c4a_split_syscirc_state_t* state;
};
struct c4a_split_split_splitter_handle_t {
    c4a_split_splt_state_t* state;
};

namespace {

c4a_status_t require_rowmajor_f64(const c4a_matrix_view_t& v,
                                   const double*& out_ptr,
                                   std::int64_t& out_rows,
                                   std::int64_t& out_cols) noexcept {
    const c4a_status_t s = ::chemometrics4all::core::validate_nonnull_view(v);
    if (s != C4A_OK) return s;
    if (v.dtype != C4A_DTYPE_F64) return C4A_ERR_DTYPE_MISMATCH;
    if (v.col_stride != 1) return C4A_ERR_STRIDE_INVALID;
    if (v.rows > 0 && v.cols > 0 && v.row_stride != v.cols) {
        return C4A_ERR_STRIDE_INVALID;
    }
    out_ptr  = static_cast<const double*>(v.data);
    out_rows = v.rows;
    out_cols = v.cols;
    return C4A_OK;
}

// y views: accept (rows x cols) row-major F64; rows must match X.rows.
c4a_status_t require_y_view(const c4a_matrix_view_t& v, std::int64_t expected_rows,
                             const double*& out_ptr,
                             std::int64_t& out_cols) noexcept {
    const double* yp = nullptr;
    std::int64_t  yr = 0, yc = 0;
    c4a_status_t s = require_rowmajor_f64(v, yp, yr, yc);
    if (s != C4A_OK) return s;
    if (yr != expected_rows) return C4A_ERR_SHAPE_MISMATCH;
    out_ptr  = yp;
    out_cols = yc;
    return C4A_OK;
}

}  // namespace

extern "C" {

// ---------------------------------------------------------------------------
// Common: result destructor.
// ---------------------------------------------------------------------------

C4A_API void c4a_split_result_destroy(c4a_split_result_t* r) {
    try {
        c4a_split_result_free(r);
    } catch (...) {
        // best-effort cleanup
    }
}

// ---------------------------------------------------------------------------
// KennardStone
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_split_kennard_stone_create(
    c4a_split_kennard_stone_handle_t** out, double test_size) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (!(test_size > 0.0) || !(test_size < 1.0)) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        c4a_split_ks_state_t* s = c4a_split_ks_state_new(test_size);
        if (s == nullptr) return C4A_ERR_OUT_OF_MEMORY;
        c4a_split_kennard_stone_handle_t* h =
            new (std::nothrow) c4a_split_kennard_stone_handle_t{s};
        if (h == nullptr) {
            c4a_split_ks_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API void c4a_split_kennard_stone_destroy(c4a_split_kennard_stone_handle_t* h) {
    if (h == nullptr) return;
    try { c4a_split_ks_state_free(h->state); delete h; } catch (...) {}
}

C4A_API c4a_status_t c4a_split_kennard_stone_split(
    const c4a_split_kennard_stone_handle_t* h,
    c4a_matrix_view_t X,
    c4a_split_result_t* out) {
    if (h == nullptr || out == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;
        std::int64_t  xr = 0, xc = 0;
        c4a_status_t s = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        c4a_split_result_clear(out);
        return c4a_split_ks_apply(h->state, xp, xr, xc, out);
    } catch (...) { return C4A_ERR_INTERNAL; }
}

// ---------------------------------------------------------------------------
// SPXY
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_split_spxy_create(c4a_split_spxy_handle_t** out,
                                            double test_size) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (!(test_size > 0.0) || !(test_size < 1.0)) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        c4a_split_spxy_state_t* s = c4a_split_spxy_state_new(test_size);
        if (s == nullptr) return C4A_ERR_OUT_OF_MEMORY;
        c4a_split_spxy_handle_t* h = new (std::nothrow) c4a_split_spxy_handle_t{s};
        if (h == nullptr) { c4a_split_spxy_state_free(s); return C4A_ERR_OUT_OF_MEMORY; }
        *out = h;
        return C4A_OK;
    } catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API void c4a_split_spxy_destroy(c4a_split_spxy_handle_t* h) {
    if (h == nullptr) return;
    try { c4a_split_spxy_state_free(h->state); delete h; } catch (...) {}
}

C4A_API c4a_status_t c4a_split_spxy_split(const c4a_split_spxy_handle_t* h,
                                           c4a_matrix_view_t X,
                                           c4a_matrix_view_t Y,
                                           c4a_split_result_t* out) {
    if (h == nullptr || out == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;
        std::int64_t  xr = 0, xc = 0;
        c4a_status_t s = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        const double* yp = nullptr;
        std::int64_t  yc = 0;
        s = require_y_view(Y, xr, yp, yc);
        if (s != C4A_OK) return s;
        c4a_split_result_clear(out);
        return c4a_split_spxy_apply(h->state, xp, xr, xc, yp, yc, out);
    } catch (...) { return C4A_ERR_INTERNAL; }
}

// ---------------------------------------------------------------------------
// SPXYFold
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_split_spxy_fold_create(c4a_split_spxy_fold_handle_t** out,
                                                 int32_t n_splits,
                                                 int32_t y_metric) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (n_splits < 2) return C4A_ERR_INVALID_ARGUMENT;
    if (y_metric < 0 || y_metric > 2) return C4A_ERR_INVALID_ARGUMENT;
    try {
        c4a_split_spxy_fold_state_t* s = c4a_split_spxy_fold_state_new(n_splits, y_metric);
        if (s == nullptr) return C4A_ERR_OUT_OF_MEMORY;
        c4a_split_spxy_fold_handle_t* h = new (std::nothrow) c4a_split_spxy_fold_handle_t{s};
        if (h == nullptr) { c4a_split_spxy_fold_state_free(s); return C4A_ERR_OUT_OF_MEMORY; }
        *out = h;
        return C4A_OK;
    } catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API void c4a_split_spxy_fold_destroy(c4a_split_spxy_fold_handle_t* h) {
    if (h == nullptr) return;
    try { c4a_split_spxy_fold_state_free(h->state); delete h; } catch (...) {}
}

C4A_API c4a_status_t c4a_split_spxy_fold_n_splits(const c4a_split_spxy_fold_handle_t* h,
                                                   int32_t* out) {
    if (h == nullptr || out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = h->state->n_splits;
    return C4A_OK;
}

C4A_API c4a_status_t c4a_split_spxy_fold_split_fold(
    const c4a_split_spxy_fold_handle_t* h,
    c4a_matrix_view_t X,
    c4a_matrix_view_t Y,
    int32_t fold_idx,
    c4a_split_result_t* out) {
    if (h == nullptr || out == nullptr) return C4A_ERR_NULL_POINTER;
    if (fold_idx < 0 || fold_idx >= h->state->n_splits) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        const double* xp = nullptr;
        std::int64_t  xr = 0, xc = 0;
        c4a_status_t s = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        const double* yp = nullptr;
        std::int64_t  yc = 0;
        if (h->state->use_y != 0) {
            s = require_y_view(Y, xr, yp, yc);
            if (s != C4A_OK) return s;
        }
        std::int32_t* fa = new (std::nothrow) std::int32_t[static_cast<std::size_t>(xr)];
        if (fa == nullptr) return C4A_ERR_OUT_OF_MEMORY;
        s = c4a_split_spxy_fold_assign(h->state, xp, xr, xc, yp, yc, fa);
        if (s != C4A_OK) { delete[] fa; return s; }
        c4a_split_result_clear(out);
        s = c4a_split_spxy_fold_extract(fa, xr, fold_idx, out);
        delete[] fa;
        return s;
    } catch (...) { return C4A_ERR_INTERNAL; }
}

// ---------------------------------------------------------------------------
// SPXYGFold (group-aware k-fold)
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_split_spxy_g_fold_create(c4a_split_spxy_g_fold_handle_t** out,
                                                    int32_t n_splits,
                                                    int32_t y_metric,
                                                    int32_t aggregation) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (n_splits < 2) return C4A_ERR_INVALID_ARGUMENT;
    if (y_metric < 0 || y_metric > 2) return C4A_ERR_INVALID_ARGUMENT;
    if (aggregation != 0 && aggregation != 1) return C4A_ERR_INVALID_ARGUMENT;
    try {
        c4a_split_spxy_g_fold_state_t* s = c4a_split_spxy_g_fold_state_new(
            n_splits, y_metric, aggregation);
        if (s == nullptr) return C4A_ERR_OUT_OF_MEMORY;
        c4a_split_spxy_g_fold_handle_t* h = new (std::nothrow) c4a_split_spxy_g_fold_handle_t{s};
        if (h == nullptr) { c4a_split_spxy_g_fold_state_free(s); return C4A_ERR_OUT_OF_MEMORY; }
        *out = h;
        return C4A_OK;
    } catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API void c4a_split_spxy_g_fold_destroy(c4a_split_spxy_g_fold_handle_t* h) {
    if (h == nullptr) return;
    try { c4a_split_spxy_g_fold_state_free(h->state); delete h; } catch (...) {}
}

C4A_API c4a_status_t c4a_split_spxy_g_fold_n_splits(
    const c4a_split_spxy_g_fold_handle_t* h, int32_t* out) {
    if (h == nullptr || out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = h->state->n_splits;
    return C4A_OK;
}

C4A_API c4a_status_t c4a_split_spxy_g_fold_split_fold(
    const c4a_split_spxy_g_fold_handle_t* h,
    c4a_matrix_view_t X,
    c4a_matrix_view_t Y,
    const int64_t* groups, int64_t groups_len,
    int32_t fold_idx,
    c4a_split_result_t* out) {
    if (h == nullptr || groups == nullptr || out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    if (fold_idx < 0 || fold_idx >= h->state->n_splits) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        const double* xp = nullptr;
        std::int64_t  xr = 0, xc = 0;
        c4a_status_t s = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        if (groups_len != xr) return C4A_ERR_SHAPE_MISMATCH;
        const double* yp = nullptr;
        std::int64_t  yc = 0;
        if (h->state->use_y != 0) {
            s = require_y_view(Y, xr, yp, yc);
            if (s != C4A_OK) return s;
        }
        std::int32_t* fa = new (std::nothrow) std::int32_t[static_cast<std::size_t>(xr)];
        if (fa == nullptr) return C4A_ERR_OUT_OF_MEMORY;
        s = c4a_split_spxy_g_fold_assign(h->state, xp, xr, xc, yp, yc,
                                          groups, fa);
        if (s != C4A_OK) { delete[] fa; return s; }
        c4a_split_result_clear(out);
        s = c4a_split_spxy_g_fold_extract(fa, xr, fold_idx, out);
        delete[] fa;
        return s;
    } catch (...) { return C4A_ERR_INTERNAL; }
}

// ---------------------------------------------------------------------------
// KMeans
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_split_kmeans_create(c4a_split_kmeans_handle_t** out,
                                               double test_size, uint64_t seed,
                                               int32_t max_iter) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (!(test_size > 0.0) || !(test_size < 1.0)) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (max_iter < 0) return C4A_ERR_INVALID_ARGUMENT;
    try {
        c4a_split_kmeans_state_t* s =
            c4a_split_kmeans_state_new(test_size, seed, max_iter);
        if (s == nullptr) return C4A_ERR_OUT_OF_MEMORY;
        c4a_split_kmeans_handle_t* h = new (std::nothrow) c4a_split_kmeans_handle_t{s};
        if (h == nullptr) { c4a_split_kmeans_state_free(s); return C4A_ERR_OUT_OF_MEMORY; }
        *out = h;
        return C4A_OK;
    } catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API void c4a_split_kmeans_destroy(c4a_split_kmeans_handle_t* h) {
    if (h == nullptr) return;
    try { c4a_split_kmeans_state_free(h->state); delete h; } catch (...) {}
}

C4A_API c4a_status_t c4a_split_kmeans_split(const c4a_split_kmeans_handle_t* h,
                                              c4a_matrix_view_t X,
                                              c4a_split_result_t* out) {
    if (h == nullptr || out == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;
        std::int64_t  xr = 0, xc = 0;
        c4a_status_t s = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        c4a_split_result_clear(out);
        return c4a_split_kmeans_apply(h->state, xp, xr, xc, out);
    } catch (...) { return C4A_ERR_INTERNAL; }
}

// ---------------------------------------------------------------------------
// KBinsStratified
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_split_kbins_stratified_create(
    c4a_split_kbins_stratified_handle_t** out, double test_size, uint64_t seed,
    int32_t n_bins, int32_t strategy) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (!(test_size > 0.0) || !(test_size < 1.0)) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (n_bins < 2) return C4A_ERR_INVALID_ARGUMENT;
    if (strategy != C4A_SPLIT_KBINS_UNIFORM &&
        strategy != C4A_SPLIT_KBINS_QUANTILE) return C4A_ERR_INVALID_ARGUMENT;
    try {
        c4a_split_kbins_state_t* s =
            c4a_split_kbins_state_new(test_size, seed, n_bins, strategy);
        if (s == nullptr) return C4A_ERR_OUT_OF_MEMORY;
        c4a_split_kbins_stratified_handle_t* h =
            new (std::nothrow) c4a_split_kbins_stratified_handle_t{s};
        if (h == nullptr) { c4a_split_kbins_state_free(s); return C4A_ERR_OUT_OF_MEMORY; }
        *out = h;
        return C4A_OK;
    } catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API void c4a_split_kbins_stratified_destroy(c4a_split_kbins_stratified_handle_t* h) {
    if (h == nullptr) return;
    try { c4a_split_kbins_state_free(h->state); delete h; } catch (...) {}
}

C4A_API c4a_status_t c4a_split_kbins_stratified_split(
    const c4a_split_kbins_stratified_handle_t* h,
    c4a_matrix_view_t Y,
    c4a_split_result_t* out) {
    if (h == nullptr || out == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        const double* yp = nullptr;
        std::int64_t  yr = 0, yc = 0;
        c4a_status_t s = require_rowmajor_f64(Y, yp, yr, yc);
        if (s != C4A_OK) return s;
        if (yc != 1) return C4A_ERR_SHAPE_MISMATCH;
        c4a_split_result_clear(out);
        return c4a_split_kbins_apply(h->state, yp, yr, out);
    } catch (...) { return C4A_ERR_INTERNAL; }
}

// ---------------------------------------------------------------------------
// BinnedStratifiedGroupKFold
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_split_binned_strat_group_kfold_create(
    c4a_split_binned_strat_group_kfold_handle_t** out,
    int32_t n_splits, int32_t n_bins, int32_t strategy,
    int32_t shuffle, uint64_t seed) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (n_splits < 2 || n_bins < 2) return C4A_ERR_INVALID_ARGUMENT;
    if (strategy != C4A_SPLIT_KBINS_UNIFORM &&
        strategy != C4A_SPLIT_KBINS_QUANTILE) return C4A_ERR_INVALID_ARGUMENT;
    try {
        c4a_split_bsgk_state_t* s =
            c4a_split_bsgk_state_new(n_splits, n_bins, strategy, shuffle, seed);
        if (s == nullptr) return C4A_ERR_OUT_OF_MEMORY;
        c4a_split_binned_strat_group_kfold_handle_t* h =
            new (std::nothrow) c4a_split_binned_strat_group_kfold_handle_t{s};
        if (h == nullptr) { c4a_split_bsgk_state_free(s); return C4A_ERR_OUT_OF_MEMORY; }
        *out = h;
        return C4A_OK;
    } catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API void c4a_split_binned_strat_group_kfold_destroy(
    c4a_split_binned_strat_group_kfold_handle_t* h) {
    if (h == nullptr) return;
    try { c4a_split_bsgk_state_free(h->state); delete h; } catch (...) {}
}

C4A_API c4a_status_t c4a_split_binned_strat_group_kfold_n_splits(
    const c4a_split_binned_strat_group_kfold_handle_t* h, int32_t* out) {
    if (h == nullptr || out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = h->state->n_splits;
    return C4A_OK;
}

C4A_API c4a_status_t c4a_split_binned_strat_group_kfold_split_fold(
    const c4a_split_binned_strat_group_kfold_handle_t* h,
    c4a_matrix_view_t Y,
    const int64_t* groups, int64_t groups_len,
    int32_t fold_idx,
    c4a_split_result_t* out) {
    if (h == nullptr || groups == nullptr || out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    if (fold_idx < 0 || fold_idx >= h->state->n_splits) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        const double* yp = nullptr;
        std::int64_t  yr = 0, yc = 0;
        c4a_status_t s = require_rowmajor_f64(Y, yp, yr, yc);
        if (s != C4A_OK) return s;
        if (yc != 1) return C4A_ERR_SHAPE_MISMATCH;
        if (groups_len != yr) return C4A_ERR_SHAPE_MISMATCH;
        std::int32_t* fa = new (std::nothrow) std::int32_t[static_cast<std::size_t>(yr)];
        if (fa == nullptr) return C4A_ERR_OUT_OF_MEMORY;
        s = c4a_split_bsgk_assign(h->state, yp, yr, groups, fa);
        if (s != C4A_OK) { delete[] fa; return s; }
        c4a_split_result_clear(out);
        s = c4a_split_bsgk_extract(fa, yr, fold_idx, out);
        delete[] fa;
        return s;
    } catch (...) { return C4A_ERR_INTERNAL; }
}

// ---------------------------------------------------------------------------
// SystematicCircular
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_split_systematic_circular_create(
    c4a_split_systematic_circular_handle_t** out, double test_size, uint64_t seed) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (!(test_size > 0.0) || !(test_size < 1.0)) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        c4a_split_syscirc_state_t* s =
            c4a_split_syscirc_state_new(test_size, seed);
        if (s == nullptr) return C4A_ERR_OUT_OF_MEMORY;
        c4a_split_systematic_circular_handle_t* h =
            new (std::nothrow) c4a_split_systematic_circular_handle_t{s};
        if (h == nullptr) { c4a_split_syscirc_state_free(s); return C4A_ERR_OUT_OF_MEMORY; }
        *out = h;
        return C4A_OK;
    } catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API void c4a_split_systematic_circular_destroy(
    c4a_split_systematic_circular_handle_t* h) {
    if (h == nullptr) return;
    try { c4a_split_syscirc_state_free(h->state); delete h; } catch (...) {}
}

C4A_API c4a_status_t c4a_split_systematic_circular_split(
    const c4a_split_systematic_circular_handle_t* h,
    c4a_matrix_view_t Y,
    c4a_split_result_t* out) {
    if (h == nullptr || out == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        const double* yp = nullptr;
        std::int64_t  yr = 0, yc = 0;
        c4a_status_t s = require_rowmajor_f64(Y, yp, yr, yc);
        if (s != C4A_OK) return s;
        if (yc != 1) return C4A_ERR_SHAPE_MISMATCH;
        c4a_split_result_clear(out);
        return c4a_split_syscirc_apply(h->state, yp, yr, out);
    } catch (...) { return C4A_ERR_INTERNAL; }
}

// ---------------------------------------------------------------------------
// SPlit (data twinning)
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_split_split_splitter_create(
    c4a_split_split_splitter_handle_t** out, double test_size, uint64_t seed) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (!(test_size > 0.0) || !(test_size < 1.0)) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        c4a_split_splt_state_t* s =
            c4a_split_splt_state_new(test_size, seed);
        if (s == nullptr) return C4A_ERR_OUT_OF_MEMORY;
        c4a_split_split_splitter_handle_t* h =
            new (std::nothrow) c4a_split_split_splitter_handle_t{s};
        if (h == nullptr) { c4a_split_splt_state_free(s); return C4A_ERR_OUT_OF_MEMORY; }
        *out = h;
        return C4A_OK;
    } catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API void c4a_split_split_splitter_destroy(c4a_split_split_splitter_handle_t* h) {
    if (h == nullptr) return;
    try { c4a_split_splt_state_free(h->state); delete h; } catch (...) {}
}

C4A_API c4a_status_t c4a_split_split_splitter_split(
    const c4a_split_split_splitter_handle_t* h,
    c4a_matrix_view_t X,
    c4a_split_result_t* out) {
    if (h == nullptr || out == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;
        std::int64_t  xr = 0, xc = 0;
        c4a_status_t s = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        c4a_split_result_clear(out);
        return c4a_split_splt_apply(h->state, xp, xr, xc, out);
    } catch (...) { return C4A_ERR_INTERNAL; }
}

}  // extern "C"
