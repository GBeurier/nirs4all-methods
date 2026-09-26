// SPDX-License-Identifier: CECILL-2.1
// Closed sample-filter role over existing native outlier kernels.
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

#include "n4m/n4m.h"
#include "core/common/matrix_view.hpp"

struct n4m_sample_filter_t {
    int32_t kind = -1;
    bool fitted = false;
    n4m_filter_y_outlier_handle_t* y = nullptr;
    n4m_filter_x_outlier_handle_t* x = nullptr;
    n4m_filter_leverage_handle_t* leverage = nullptr;
    n4m_filter_quality_handle_t* quality = nullptr;
    n4m_filter_composite_handle_t* composite = nullptr;
    std::vector<n4m_sample_filter_t*> children;
};

namespace {

bool i32(std::int64_t v) noexcept {
    return v >= std::numeric_limits<std::int32_t>::min() &&
           v <= std::numeric_limits<std::int32_t>::max();
}

bool bit(std::int64_t v) noexcept { return v == 0 || v == 1; }

n4m_status_t view(const n4m_matrix_view_t* v) noexcept {
    if (v == nullptr) return N4M_ERR_NULL_POINTER;
    n4m_status_t st = n4m::core::validate_nonnull_view(*v);
    if (st != N4M_OK) return st;
    if (v->dtype != N4M_DTYPE_F64) return N4M_ERR_DTYPE_MISMATCH;
    if (v->col_stride != 1 || (v->rows > 0 && v->cols > 0 && v->row_stride != v->cols)) {
        return N4M_ERR_STRIDE_INVALID;
    }
    return N4M_OK;
}

n4m_status_t views(const n4m_matrix_view_t* x, const n4m_matrix_view_t* y,
                   bool y_required) noexcept {
    n4m_status_t st = view(x);
    if (st != N4M_OK) return st;
    if (y == nullptr) return y_required ? N4M_ERR_NULL_POINTER : N4M_OK;
    st = view(y);
    if (st != N4M_OK) return st;
    return (y->cols == 1 && y->rows == x->rows) ? N4M_OK : N4M_ERR_SHAPE_MISMATCH;
}

void destroy(n4m_sample_filter_t* h) noexcept {
    if (h == nullptr) return;
    // Composite holds borrowed direct child pointers: release it first.
    n4m_outlier_detection_composite_destroy(h->composite);
    for (auto* child : h->children) destroy(child);
    n4m_outlier_detection_y_outlier_destroy(h->y);
    n4m_outlier_detection_x_outlier_destroy(h->x);
    n4m_outlier_detection_high_leverage_destroy(h->leverage);
    n4m_outlier_detection_spectral_quality_destroy(h->quality);
    delete h;
}

n4m_status_t create(int32_t kind, const int64_t* i, int32_t ni,
                    const double* d, int32_t nd, uint64_t seed,
                    n4m_sample_filter_t** out) {
    if (out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = nullptr;
    if (ni < 0 || nd < 0 || (ni > 0 && i == nullptr) || (nd > 0 && d == nullptr)) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    for (int32_t j = 0; j < nd; ++j) {
        if (!std::isfinite(d[j])) return N4M_ERR_INVALID_ARGUMENT;
    }
    n4m_sample_filter_t* h = new (std::nothrow) n4m_sample_filter_t;
    if (h == nullptr) return N4M_ERR_OUT_OF_MEMORY;
    h->kind = kind;
    n4m_status_t st = N4M_ERR_INVALID_ARGUMENT;
    switch (kind) {
        case N4M_SAMPLE_FILTER_Y_OUTLIER:
            if (ni == 1 && nd == 3 && i32(i[0])) {
                st = n4m_outlier_detection_y_outlier_create(
                    &h->y, static_cast<int32_t>(i[0]), d[0], d[1], d[2]);
            }
            break;
        case N4M_SAMPLE_FILTER_X_OUTLIER:
            if (ni == 5 && nd == 2 && i32(i[0]) && bit(i[1]) &&
                i32(i[2]) && i32(i[3])) {
                st = n4m_outlier_detection_x_outlier_create(
                    &h->x, static_cast<int32_t>(i[0]), static_cast<int>(i[1]),
                    d[0], static_cast<int32_t>(i[2]), d[1], seed,
                    static_cast<int32_t>(i[3]), i[4]);
            }
            break;
        case N4M_SAMPLE_FILTER_HIGH_LEVERAGE:
            if (ni == 4 && nd == 2 && i32(i[0]) && bit(i[1]) &&
                i32(i[2]) && bit(i[3])) {
                st = n4m_outlier_detection_high_leverage_create(
                    &h->leverage, static_cast<int32_t>(i[0]), d[0],
                    static_cast<int>(i[1]), d[1], static_cast<int32_t>(i[2]),
                    static_cast<int>(i[3]));
            }
            break;
        case N4M_SAMPLE_FILTER_SPECTRAL_QUALITY:
            if (ni == 3 && nd == 5 && bit(i[0]) && bit(i[1]) && bit(i[2])) {
                st = n4m_outlier_detection_spectral_quality_create(
                    &h->quality, d[0], d[1], d[2], static_cast<int>(i[0]),
                    d[3], static_cast<int>(i[1]), d[4], static_cast<int>(i[2]));
            }
            break;
        case N4M_SAMPLE_FILTER_COMPOSITE:
            if (ni == 1 && nd == 0 && i32(i[0])) {
                st = n4m_outlier_detection_composite_create(
                    &h->composite, static_cast<int32_t>(i[0]));
            }
            break;
        default:
            break;
    }
    if (st != N4M_OK) {
        destroy(h);
        return st;
    }
    *out = h;
    return N4M_OK;
}

}  // namespace

extern "C" N4M_API n4m_status_t n4m_sample_filter_create(
    int32_t kind, const int64_t* i, int32_t ni, const double* d, int32_t nd,
    uint64_t seed, n4m_sample_filter_t** out) {
    try { return create(kind, i, ni, d, nd, seed, out); }
    catch (const std::bad_alloc&) { if (out) *out = nullptr; return N4M_ERR_OUT_OF_MEMORY; }
    catch (...) { if (out) *out = nullptr; return N4M_ERR_INTERNAL; }
}

extern "C" N4M_API n4m_status_t n4m_sample_filter_add_child(
    n4m_sample_filter_t* parent, int32_t kind, const int64_t* i, int32_t ni,
    const double* d, int32_t nd, uint64_t seed) {
    if (parent == nullptr) return N4M_ERR_NULL_POINTER;
    if (parent->kind != N4M_SAMPLE_FILTER_COMPOSITE ||
        (kind != N4M_SAMPLE_FILTER_HIGH_LEVERAGE &&
         kind != N4M_SAMPLE_FILTER_SPECTRAL_QUALITY)) return N4M_ERR_INVALID_ARGUMENT;
    try {
        n4m_sample_filter_t* child = nullptr;
        n4m_status_t st = create(kind, i, ni, d, nd, seed, &child);
        if (st != N4M_OK) return st;
        try { parent->children.push_back(child); }
        catch (...) { destroy(child); throw; }
        st = kind == N4M_SAMPLE_FILTER_HIGH_LEVERAGE
            ? n4m_outlier_detection_composite_add_leverage(parent->composite, child->leverage)
            : n4m_outlier_detection_composite_add_quality(parent->composite, child->quality);
        if (st != N4M_OK) {
            parent->children.pop_back();
            destroy(child);
            return st;
        }
        parent->fitted = false;
        return N4M_OK;
    } catch (const std::bad_alloc&) { return N4M_ERR_OUT_OF_MEMORY; }
      catch (...) { return N4M_ERR_INTERNAL; }
}

extern "C" N4M_API n4m_status_t n4m_sample_filter_fit(
    n4m_sample_filter_t* h, const n4m_matrix_view_t* x,
    const n4m_matrix_view_t* y) {
    if (h == nullptr) return N4M_ERR_NULL_POINTER;
    n4m_status_t st = views(x, y, h->kind == N4M_SAMPLE_FILTER_Y_OUTLIER);
    if (st != N4M_OK) return st;
    h->fitted = false;
    switch (h->kind) {
        case N4M_SAMPLE_FILTER_Y_OUTLIER:
            st = n4m_outlier_detection_y_outlier_fit(
                h->y, static_cast<const double*>(y->data), y->rows);
            break;
        case N4M_SAMPLE_FILTER_X_OUTLIER:
            st = n4m_outlier_detection_x_outlier_fit(h->x, *x);
            break;
        case N4M_SAMPLE_FILTER_HIGH_LEVERAGE:
            st = n4m_outlier_detection_high_leverage_fit(h->leverage, *x);
            break;
        case N4M_SAMPLE_FILTER_SPECTRAL_QUALITY:
            st = N4M_OK;
            break;
        case N4M_SAMPLE_FILTER_COMPOSITE:
            if (h->children.empty()) return N4M_ERR_INVALID_ARGUMENT;
            for (auto* child : h->children) {
                st = n4m_sample_filter_fit(child, x, y);
                if (st != N4M_OK) return st;
            }
            break;
        default: return N4M_ERR_INVALID_ARGUMENT;
    }
    h->fitted = st == N4M_OK;
    return st;
}

extern "C" N4M_API n4m_status_t n4m_sample_filter_apply(
    const n4m_sample_filter_t* h, const n4m_matrix_view_t* x,
    const n4m_matrix_view_t* y, uint8_t* mask, n4m_filter_stats_t* stats) {
    if (h == nullptr || mask == nullptr || stats == nullptr) return N4M_ERR_NULL_POINTER;
    if (!h->fitted) return N4M_ERR_NOT_FITTED;
    n4m_status_t st = views(x, y, h->kind == N4M_SAMPLE_FILTER_Y_OUTLIER);
    if (st != N4M_OK) return st;
    switch (h->kind) {
        case N4M_SAMPLE_FILTER_Y_OUTLIER:
            return n4m_outlier_detection_y_outlier_apply(
                h->y, static_cast<const double*>(y->data), y->rows, mask, stats);
        case N4M_SAMPLE_FILTER_X_OUTLIER:
            return n4m_outlier_detection_x_outlier_apply(h->x, *x, mask, stats);
        case N4M_SAMPLE_FILTER_HIGH_LEVERAGE:
            return n4m_outlier_detection_high_leverage_apply(h->leverage, *x, mask, stats);
        case N4M_SAMPLE_FILTER_SPECTRAL_QUALITY:
            return n4m_outlier_detection_spectral_quality_apply(h->quality, *x, mask, stats);
        case N4M_SAMPLE_FILTER_COMPOSITE:
            return n4m_outlier_detection_composite_apply(h->composite, *x, mask, stats);
        default: return N4M_ERR_INVALID_ARGUMENT;
    }
}

extern "C" N4M_API n4m_status_t n4m_sample_filter_is_fitted(
    const n4m_sample_filter_t* h, int* out) {
    if (h == nullptr || out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = h->fitted ? 1 : 0;
    return N4M_OK;
}

extern "C" N4M_API void n4m_sample_filter_destroy(n4m_sample_filter_t* h) {
    destroy(h);
}
