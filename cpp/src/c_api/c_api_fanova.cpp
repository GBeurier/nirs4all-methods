// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
#include "n4m/fanova.h"
#include "core/common/matrix_view.hpp"
#include "core/component_analysis/fanova.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>

#if N4M_WITH_RFR
namespace {
n4m_status_t validate_view(const n4m_matrix_view_t& view) {
    const auto status = n4m::core::validate_nonnull_view(view);
    if (status != N4M_OK)
        return status;
    if (view.dtype != N4M_DTYPE_F64)
        return N4M_ERR_DTYPE_MISMATCH;
    if (view.rows <= 0 || view.cols <= 0)
        return N4M_ERR_INVALID_ARGUMENT;
    constexpr int64_t max_offset = std::numeric_limits<std::ptrdiff_t>::max() / sizeof(double);
    if ((view.rows > 1 && view.row_stride > max_offset / (view.rows - 1))
        || (view.cols > 1 && view.col_stride > max_offset / (view.cols - 1)))
        return N4M_ERR_STRIDE_INVALID;
    if ((view.rows - 1) * view.row_stride
        > max_offset - (view.cols - 1) * view.col_stride)
        return N4M_ERR_STRIDE_INVALID;
    return N4M_OK;
}
double at(const n4m_matrix_view_t& v, int64_t i, int64_t j) {
    double value;
    const auto offset = static_cast<size_t>(i * v.row_stride + j * v.col_stride) * sizeof(double);
    std::memcpy(&value, static_cast<const unsigned char*>(v.data) + offset, sizeof(value));
    return value;
}
struct Range { uintptr_t begin, end; };
bool range(const void* ptr, uint64_t bytes, Range& result) {
    result.begin = reinterpret_cast<uintptr_t>(ptr);
    if (bytes > std::numeric_limits<uintptr_t>::max() - result.begin)
        return false;
    result.end = result.begin + static_cast<uintptr_t>(bytes);
    return true;
}
bool overlaps(const Range& a, const Range& b) {
    return a.begin < b.end && b.begin < a.end;
}
}  // namespace
#endif

extern "C" {
N4M_API n4m_status_t n4m_fanova_options_init_v1(n4m_fanova_options_v1_t* options) {
    if (!options)
        return N4M_ERR_NULL_POINTER;
    *options = {sizeof(*options), 32, 8, 2, 0, 0, 2, 0, 1000000};
    return N4M_OK;
}
N4M_API int n4m_fanova_backend_available_v1(void) {
#if N4M_WITH_RFR
    return 1;
#else
    return 0;
#endif
}

N4M_API n4m_status_t n4m_fanova_fit_v1(
    const n4m_matrix_view_t* X, const n4m_matrix_view_t* y,
    const n4m_matrix_view_t* bounds, const uint32_t* types,
    const n4m_matrix_view_t* X_validation, const n4m_matrix_view_t* y_validation,
    const n4m_fanova_options_v1_t* options,
    double* main_effects, int64_t main_effects_size,
    double* pair_effects, int64_t pair_effects_size,
    n4m_fanova_summary_v1_t* summary) {
    if (!X || !y || !bounds || !types || !options || !main_effects || !pair_effects || !summary)
        return N4M_ERR_NULL_POINTER;
    if (options->struct_size != sizeof(*options) || summary->struct_size != sizeof(*summary))
        return N4M_ERR_ABI_MISMATCH;
#if !N4M_WITH_RFR
    (void)X_validation; (void)y_validation;
    (void)main_effects_size; (void)pair_effects_size;
    return N4M_ERR_BACKEND_UNAVAILABLE;
#else
    if ((X_validation == nullptr) != (y_validation == nullptr))
        return N4M_ERR_INVALID_ARGUMENT;
    for (const auto* v : {X, y, bounds, X_validation, y_validation}) {
        if (v) {
            const auto status = validate_view(*v);
            if (status != N4M_OK)
                return status;
        }
    }
    if (X->rows < 2 || X->rows > 8192 || X->cols > 32 || X->rows > 262144 / X->cols
        || (X_validation && X_validation->rows > 8192))
        return N4M_ERR_UNSUPPORTED;
    const int64_t p = X->cols;
    if (y->rows != X->rows || y->cols != 1 || bounds->rows != p || bounds->cols != 2
        || (X_validation && (X_validation->cols != p || y_validation->cols != 1
                            || y_validation->rows != X_validation->rows))
        || main_effects_size != p * 2 || pair_effects_size != p * p * 2)
        return N4M_ERR_SHAPE_MISMATCH;
    if (options->num_trees < 1 || options->num_trees > 128
        || options->max_depth < 1 || options->max_depth > 12
        || options->min_samples_leaf < 1 || options->min_samples_leaf > 8192
        || options->max_features < 0 || options->max_features > p
        || (options->max_order != 1 && options->max_order != 2)
        || options->reserved != 0 || options->max_marginal_evaluations < 1
        || options->max_marginal_evaluations > 10000000)
        return N4M_ERR_INVALID_ARGUMENT;
    // Conservative byte extents also reject aliasing through strided input gaps.
    std::array<Range, 3> outputs;
    if (!range(main_effects, static_cast<uint64_t>(main_effects_size) * sizeof(double), outputs[0])
        || !range(pair_effects, static_cast<uint64_t>(pair_effects_size) * sizeof(double), outputs[1])
        || !range(summary, sizeof(*summary), outputs[2]))
        return N4M_ERR_INVALID_ARGUMENT;
    for (size_t i = 0; i < outputs.size(); ++i)
        for (size_t j = i + 1; j < outputs.size(); ++j)
            if (overlaps(outputs[i], outputs[j]))
                return N4M_ERR_INVALID_ARGUMENT;
    auto overlaps_output = [&](const void* data, uint64_t bytes) {
        Range input;
        if (!range(data, bytes, input))
            return true;
        for (const auto& output : outputs)
            if (overlaps(input, output))
                return true;
        return false;
    };
    if (overlaps_output(types, static_cast<uint64_t>(p) * sizeof(uint32_t)) || overlaps_output(options, sizeof(*options)))
        return N4M_ERR_INVALID_ARGUMENT;
    for (const auto* v : {X, y, bounds, X_validation, y_validation}) {
        if (v && overlaps_output(v->data, static_cast<uint64_t>(
                (v->rows - 1) * v->row_stride + (v->cols - 1) * v->col_stride + 1) * sizeof(double)))
            return N4M_ERR_INVALID_ARGUMENT;
        if (v && overlaps_output(v, sizeof(*v)))
            return N4M_ERR_INVALID_ARGUMENT;
    }
    for (int64_t j = 0; j < p; ++j) {
        const double lo = at(*bounds, j, 0), hi = at(*bounds, j, 1);
        if (!std::isfinite(lo) || !std::isfinite(hi) || types[j] > 128
            || (types[j] == 0 && !(lo < hi))
            || (types[j] != 0 && (lo != 0 || hi != types[j] - 1)))
            return N4M_ERR_INVALID_ARGUMENT;
    }
    for (const auto* v : {X, X_validation}) {
        if (!v)
            continue;
        for (int64_t i = 0; i < v->rows; ++i) {
            for (int64_t j = 0; j < p; ++j) {
                const double value = at(*v, i, j);
                if (!std::isfinite(value) || value < at(*bounds, j, 0)
                    || value > at(*bounds, j, 1)
                    || (types[j] && value != std::floor(value)))
                    return N4M_ERR_INVALID_ARGUMENT;
            }
        }
    }
    for (const auto* v : {y, y_validation})
        if (v)
            for (int64_t i = 0; i < v->rows; ++i)
                if (!std::isfinite(at(*v, i, 0)))
                    return N4M_ERR_INVALID_ARGUMENT;
    try {
        n4m::fanova::Result result;
        const auto status = n4m::fanova::fit(*X, *y, *bounds, types, X_validation, y_validation,
                                            *options, result);
        if (status != N4M_OK)
            return status;
        std::copy(result.main_effects.begin(), result.main_effects.end(), main_effects);
        std::copy(result.pair_effects.begin(), result.pair_effects.end(), pair_effects);
        *summary = result.summary;
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
#endif
}
}  // extern "C"
