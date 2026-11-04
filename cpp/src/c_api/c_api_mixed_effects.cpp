// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <utility>
#include <vector>

#include "core/common/matrix_view.hpp"
#include "n4m/mixed_effects.h"

#ifndef N4M_WITH_MIXEFF
#    define N4M_WITH_MIXEFF 0
#endif

#if N4M_WITH_MIXEFF
#    if defined(__GNUC__) && !defined(_WIN32)
#        define N4M_MIXEFF_LOCAL __attribute__((visibility("hidden")))
#    else
#        define N4M_MIXEFF_LOCAL
#    endif
extern "C" N4M_MIXEFF_LOCAL int32_t n4m_mixeff_fit_bridge(const double*,
                                                          const double*,
                                                          const int64_t*,
                                                          size_t,
                                                          size_t,
                                                          size_t,
                                                          const n4m_mixed_effects_options_v1_t*,
                                                          double*,
                                                          double*,
                                                          double*,
                                                          double*,
                                                          n4m_mixed_effects_summary_v1_t*);

namespace {
n4m_status_t validate(const n4m_matrix_view_t& v, n4m_dtype_t dtype) {
    const auto status = n4m::core::validate_nonnull_view(v);
    if (status != N4M_OK)
        return status;
    if (v.dtype != dtype)
        return N4M_ERR_DTYPE_MISMATCH;
    if (v.rows <= 0 || v.cols <= 0)
        return N4M_ERR_SHAPE_MISMATCH;
    constexpr int64_t limit = std::numeric_limits<std::ptrdiff_t>::max() / 8;
    if ((v.rows > 1 && v.row_stride > limit / (v.rows - 1))
        || (v.cols > 1 && v.col_stride > limit / (v.cols - 1)))
        return N4M_ERR_STRIDE_INVALID;
    const int64_t row = (v.rows - 1) * v.row_stride;
    const int64_t col = (v.cols - 1) * v.col_stride;
    return row > limit - col ? N4M_ERR_STRIDE_INVALID : N4M_OK;
}
template<typename T>
T at(const n4m_matrix_view_t& v, int64_t i, int64_t j) {
    T value;
    const auto offset = static_cast<size_t>(i * v.row_stride + j * v.col_stride) * sizeof(T);
    std::memcpy(&value, static_cast<const unsigned char*>(v.data) + offset, sizeof(T));
    return value;
}
struct Range {
    const void* data;
    size_t bytes;
};
bool overlapping(const std::array<Range, 5>& ranges) {
    for (size_t i = 0; i < ranges.size(); ++i) {
        const auto a = reinterpret_cast<uintptr_t>(ranges[i].data);
        if (a > std::numeric_limits<uintptr_t>::max() - ranges[i].bytes)
            return true;
        for (size_t j = 0; j < i; ++j) {
            const auto b = reinterpret_cast<uintptr_t>(ranges[j].data);
            if (b > std::numeric_limits<uintptr_t>::max() - ranges[j].bytes)
                return true;
            if (a < b + ranges[j].bytes && b < a + ranges[i].bytes)
                return true;
        }
    }
    return false;
}
}  // namespace
#endif

extern "C" {
N4M_API n4m_status_t n4m_mixed_effects_options_init_v1(n4m_mixed_effects_options_v1_t* options) {
    if (!options)
        return N4M_ERR_NULL_POINTER;
    *options = {sizeof(*options), 1, 0, 1000, 1e-8, 1e-10, 1e-8};
    return N4M_OK;
}
N4M_API int n4m_mixed_effects_backend_available_v1(void) {
    return N4M_WITH_MIXEFF ? 1 : 0;
}
N4M_API n4m_status_t n4m_mixed_effects_fit_v1(const n4m_matrix_view_t* X,
                                              const n4m_matrix_view_t* y,
                                              const n4m_matrix_view_t* groups,
                                              const n4m_mixed_effects_options_v1_t* options,
                                              double* beta,
                                              int64_t beta_size,
                                              double* covariance,
                                              int64_t covariance_size,
                                              double* standard_errors,
                                              int64_t standard_errors_size,
                                              double* variances,
                                              int64_t variances_size,
                                              n4m_mixed_effects_summary_v1_t* summary) {
    if (!X || !y || !groups || !options || !beta || !covariance || !standard_errors || !variances
        || !summary)
        return N4M_ERR_NULL_POINTER;
    if (options->struct_size != sizeof(*options) || summary->struct_size != sizeof(*summary))
        return N4M_ERR_ABI_MISMATCH;
#if !N4M_WITH_MIXEFF
    (void)beta_size;
    (void)covariance_size;
    (void)standard_errors_size;
    (void)variances_size;
    return N4M_ERR_BACKEND_UNAVAILABLE;
#else
    for (const auto& pair : {std::pair<const n4m_matrix_view_t*, n4m_dtype_t>{X, N4M_DTYPE_F64},
                             {y, N4M_DTYPE_F64},
                             {groups, N4M_DTYPE_I64}}) {
        const auto status = validate(*pair.first, pair.second);
        if (status != N4M_OK)
            return status;
    }
    if (y->rows != X->rows || y->cols != 1 || groups->rows != X->rows)
        return N4M_ERR_SHAPE_MISMATCH;
    const int64_t n = X->rows, p = X->cols, k = groups->cols;
    if (n > 8192 || p > 64 || k > 8 || n * p > 262144)
        return N4M_ERR_UNSUPPORTED;
    if ((options->reml != 0 && options->reml != 1) || options->reserved != 0
        || options->max_evaluations < 1 || options->max_evaluations > 100000
        || !std::isfinite(options->tolerance) || options->tolerance <= 0 || options->tolerance >= 1
        || !std::isfinite(options->rank_tolerance) || options->rank_tolerance <= 0
        || options->rank_tolerance >= 1 || !std::isfinite(options->boundary_tolerance)
        || options->boundary_tolerance <= 0 || options->boundary_tolerance >= 1)
        return N4M_ERR_INVALID_ARGUMENT;
    if (beta_size != p || covariance_size != p * p || standard_errors_size != p
        || variances_size != k + 1)
        return N4M_ERR_SHAPE_MISMATCH;
    if (overlapping({Range{beta, static_cast<size_t>(p) * 8},
                     Range{covariance, static_cast<size_t>(p * p) * 8},
                     Range{standard_errors, static_cast<size_t>(p) * 8},
                     Range{variances, static_cast<size_t>(k + 1) * 8},
                     Range{summary, sizeof(*summary)}}))
        return N4M_ERR_INVALID_ARGUMENT;
    try {
        std::vector<double> packed_x(static_cast<size_t>(n * p)), packed_y(static_cast<size_t>(n));
        std::vector<int64_t> packed_groups(static_cast<size_t>(n * k));
        for (int64_t i = 0; i < n; ++i) {
            packed_y[static_cast<size_t>(i)] = at<double>(*y, i, 0);
            if (!std::isfinite(packed_y[static_cast<size_t>(i)]))
                return N4M_ERR_INVALID_ARGUMENT;
            for (int64_t j = 0; j < p; ++j) {
                const auto value = at<double>(*X, i, j);
                if (!std::isfinite(value))
                    return N4M_ERR_INVALID_ARGUMENT;
                packed_x[static_cast<size_t>(i * p + j)] = value;
            }
            for (int64_t j = 0; j < k; ++j) {
                const auto value = at<int64_t>(*groups, i, j);
                if (value < 0)
                    return N4M_ERR_INVALID_ARGUMENT;
                packed_groups[static_cast<size_t>(i * k + j)] = value;
            }
        }
        std::vector<double> b(static_cast<size_t>(p)), cov(static_cast<size_t>(p * p)),
            se(static_cast<size_t>(p)), var(static_cast<size_t>(k + 1));
        n4m_mixed_effects_summary_v1_t result{};
        const int32_t code = n4m_mixeff_fit_bridge(packed_x.data(),
                                                   packed_y.data(),
                                                   packed_groups.data(),
                                                   static_cast<size_t>(n),
                                                   static_cast<size_t>(p),
                                                   static_cast<size_t>(k),
                                                   options,
                                                   b.data(),
                                                   cov.data(),
                                                   se.data(),
                                                   var.data(),
                                                   &result);
        if (code != N4M_OK)
            return static_cast<n4m_status_t>(code);
        std::copy(b.begin(), b.end(), beta);
        std::copy(cov.begin(), cov.end(), covariance);
        std::copy(se.begin(), se.end(), standard_errors);
        std::copy(var.begin(), var.end(), variances);
        *summary = result;
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
#endif
}
}
