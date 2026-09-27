// SPDX-License-Identifier: CECILL-2.1
#include <limits>
#include <new>

#include "core/common/matrix_view.hpp"
#include "core/spectral_encoding.hpp"
#include "n4m/n4m.h"

struct n4m_spectral_encoder_t {
    n4m::core::SpectralEncoding value;
};
namespace {
n4m_status_t validate(n4m_matrix_view_t v) {
    auto status = n4m::core::validate_nonnull_view(v);
    if (status != N4M_OK)
        return status;
    if (v.dtype != N4M_DTYPE_F64)
        return N4M_ERR_DTYPE_MISMATCH;
    if (v.rows < 1 || v.cols < 1 || v.rows > std::numeric_limits<int64_t>::max() / v.cols / 8)
        return N4M_ERR_INVALID_ARGUMENT;
    return N4M_OK;
}
std::vector<double> copy(n4m_matrix_view_t v) {
    std::vector<double> x(static_cast<std::size_t>(v.rows * v.cols));
    auto* data = static_cast<const double*>(v.data);
    for (int64_t i = 0; i < v.rows; ++i)
        for (int64_t j = 0; j < v.cols; ++j)
            x[static_cast<std::size_t>(i * v.cols + j)] = data[i * v.row_stride + j * v.col_stride];
    return x;
}
void write(const std::vector<double>& x, n4m_matrix_view_t v) {
    auto* data = static_cast<double*>(v.data);
    for (int64_t i = 0; i < v.rows; ++i)
        for (int64_t j = 0; j < v.cols; ++j)
            data[i * v.row_stride + j * v.col_stride] = x[static_cast<std::size_t>(i * v.cols + j)];
}
}  // namespace
extern "C" {
n4m_status_t n4m_decomposition_spectral_create(int32_t kind,
                                               int32_t width,
                                               int32_t rank,
                                               double overlap,
                                               int32_t standardize,
                                               int32_t snv,
                                               int32_t iterations,
                                               double tolerance,
                                               n4m_spectral_encoder_t** out) {
    if (!out)
        return N4M_ERR_NULL_POINTER;
    *out = nullptr;
    if (kind < 0 || kind > 1 || width < 1 || rank < 1 || iterations < 1
        || !(overlap >= 0 && overlap < 1) || !(tolerance >= 0 && tolerance <= 1) || standardize < 0
        || standardize > 1 || snv < 0 || snv > 1 || (kind == 1 && snv))
        return N4M_ERR_INVALID_ARGUMENT;
    try {
        auto* h = new n4m_spectral_encoder_t;
        auto& v = h->value;
        v.kind = kind;
        v.width = width;
        v.rank = rank;
        v.overlap = overlap;
        v.standardize = standardize != 0;
        v.snv = snv != 0;
        v.iterations = iterations;
        v.tolerance = tolerance;
        *out = h;
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}
void n4m_decomposition_spectral_destroy(n4m_spectral_encoder_t* h) {
    delete h;
}
n4m_status_t n4m_decomposition_spectral_fit(n4m_spectral_encoder_t* h, n4m_matrix_view_t x) {
    if (!h)
        return N4M_ERR_NULL_POINTER;
    auto status = validate(x);
    if (status != N4M_OK)
        return status;
    try {
        auto next = h->value;
        status =
            next.fit(copy(x), static_cast<std::size_t>(x.rows), static_cast<std::size_t>(x.cols));
        if (status == N4M_OK)
            h->value = std::move(next);
        return status;
    } catch (const std::bad_alloc&) {
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}
n4m_status_t n4m_decomposition_spectral_output_cols(const n4m_spectral_encoder_t* h, int64_t* out) {
    if (!h || !out)
        return N4M_ERR_NULL_POINTER;
    if (!h->value.outputs)
        return N4M_ERR_INVALID_ARGUMENT;
    *out = static_cast<int64_t>(h->value.outputs);
    return N4M_OK;
}
n4m_status_t n4m_decomposition_spectral_transform(const n4m_spectral_encoder_t* h,
                                                  n4m_matrix_view_t x,
                                                  n4m_matrix_view_t out) {
    if (!h)
        return N4M_ERR_NULL_POINTER;
    auto status = validate(x);
    if (status != N4M_OK)
        return status;
    status = validate(out);
    if (status != N4M_OK)
        return status;
    if (x.rows != out.rows || x.cols != static_cast<int64_t>(h->value.features)
        || out.cols != static_cast<int64_t>(h->value.outputs))
        return N4M_ERR_SHAPE_MISMATCH;
    try {
        std::vector<double> result;
        status = h->value.transform(copy(x), static_cast<std::size_t>(x.rows), result);
        if (status == N4M_OK)
            write(result, out);
        return status;
    } catch (const std::bad_alloc&) {
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}
n4m_status_t n4m_decomposition_spectral_export_affine(const n4m_spectral_encoder_t* h,
                                                      n4m_matrix_view_t op,
                                                      n4m_matrix_view_t offset) {
    if (!h)
        return N4M_ERR_NULL_POINTER;
    const auto& v = h->value;
    if (v.kind != 0 || v.snv || !v.outputs)
        return N4M_ERR_INVALID_ARGUMENT;
    auto status = validate(op);
    if (status != N4M_OK)
        return status;
    status = validate(offset);
    if (status != N4M_OK)
        return status;
    if (op.rows != static_cast<int64_t>(v.outputs) || op.cols != static_cast<int64_t>(v.features)
        || offset.rows != 1 || offset.cols != op.rows)
        return N4M_ERR_SHAPE_MISMATCH;
    try {
        std::vector<double> b(v.outputs, 0);
        for (std::size_t a = 0; a < v.outputs; ++a)
            for (std::size_t j = 0; j < v.features; ++j)
                b[a] -= v.basis[a * v.features + j] * v.center[j];
        write(v.basis, op);
        write(b, offset);
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}
}  // extern C
