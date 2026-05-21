// SPDX-License-Identifier: CECILL-2.1

#include "core/sipls_selection.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <numeric>
#include <vector>

#include "core/cross_validation.hpp"
#include "core/matrix_view.hpp"
#include "core/status.hpp"

namespace {

[[nodiscard]] std::size_t idx(std::size_t row, std::size_t cols, std::size_t col) noexcept {
    return row * cols + col;
}

[[nodiscard]] double read_value(const n4m_matrix_view_t& view,
                                std::size_t row,
                                std::size_t col) noexcept {
    const std::int64_t off =
        static_cast<std::int64_t>(row) * view.row_stride +
        static_cast<std::int64_t>(col) * view.col_stride;
    const auto uoff = static_cast<std::size_t>(off);
    if (view.dtype == N4M_DTYPE_F64) {
        const auto* ptr = static_cast<const double*>(view.data);
        return ptr[uoff];
    }
    const auto* ptr = static_cast<const float*>(view.data);
    return static_cast<double>(ptr[uoff]);
}

[[nodiscard]] n4m_matrix_view_t rowmajor_f64_view(std::vector<double>& values,
                                                  std::int64_t rows,
                                                  std::int64_t cols) noexcept {
    n4m_matrix_view_t view{};
    view.data = values.data();
    view.rows = rows;
    view.cols = cols;
    view.row_stride = cols > 0 ? cols : 1;
    view.col_stride = 1;
    view.dtype = N4M_DTYPE_F64;
    return view;
}

[[nodiscard]] n4m_status_t validate_float_view(::n4m::core::Context& ctx,
                                               const n4m_matrix_view_t& view,
                                               const char* name) noexcept {
    const n4m_status_t status = ::n4m::core::validate_nonnull_view(view);
    if (status != N4M_OK) {
        ctx.set_errorf("%s matrix view is invalid: %s",
                       name,
                       ::n4m::core::status_to_string(status));
        return status;
    }
    if (view.dtype != N4M_DTYPE_F64 && view.dtype != N4M_DTYPE_F32) {
        ctx.set_errorf("%s dtype must be f64 or f32", name);
        return N4M_ERR_DTYPE_MISMATCH;
    }
    return N4M_OK;
}

[[nodiscard]] std::vector<std::int64_t> make_intervals(std::int64_t n_features,
                                                       std::int32_t interval_width) {
    std::vector<std::int64_t> intervals;
    if (n_features <= 0 || interval_width <= 0) {
        return intervals;
    }
    for (std::int64_t start = 0; start < n_features; start += interval_width) {
        const std::int64_t remaining = n_features - start;
        const std::int64_t length =
            std::min<std::int64_t>(static_cast<std::int64_t>(interval_width), remaining);
        intervals.push_back(start);
        intervals.push_back(length);
    }
    return intervals;
}

[[nodiscard]] std::size_t interval_length(const std::vector<std::int64_t>& intervals,
                                          std::int64_t interval_index) noexcept {
    return static_cast<std::size_t>(intervals[static_cast<std::size_t>(2 * interval_index + 1)]);
}

[[nodiscard]] std::size_t active_column_count(const std::vector<std::int64_t>& intervals,
                                              const std::vector<std::int64_t>& active) noexcept {
    std::size_t out = 0;
    for (const std::int64_t interval_index : active) {
        out += interval_length(intervals, interval_index);
    }
    return out;
}

[[nodiscard]] std::size_t minimum_column_count_for_combination(
    const std::vector<std::int64_t>& intervals,
    std::int32_t combination_size) {
    std::vector<std::size_t> lengths;
    lengths.reserve(intervals.size() / 2U);
    for (std::size_t interval = 0; interval < intervals.size() / 2U; ++interval) {
        lengths.push_back(static_cast<std::size_t>(intervals[2U * interval + 1U]));
    }
    std::sort(lengths.begin(), lengths.end());
    std::size_t out = 0;
    for (std::int32_t i = 0; i < combination_size; ++i) {
        out += lengths[static_cast<std::size_t>(i)];
    }
    return out;
}

[[nodiscard]] n4m_status_t validate_request(::n4m::core::Context& ctx,
                                            const ::n4m::core::Config& cfg,
                                            const n4m_matrix_view_t& X,
                                            const n4m_matrix_view_t& Y,
                                            const ::n4m::core::ValidationPlan& plan,
                                            std::int32_t interval_width,
                                            std::int32_t combination_size,
                                            const std::vector<std::int64_t>& intervals) {
    n4m_status_t status = validate_float_view(ctx, X, "X");
    if (status != N4M_OK) {
        return status;
    }
    status = validate_float_view(ctx, Y, "Y");
    if (status != N4M_OK) {
        return status;
    }
    if (X.rows <= 0 || X.cols <= 0 || Y.cols <= 0) {
        ctx.set_error("siPLS matrices must be non-empty");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (X.rows != Y.rows) {
        ctx.set_errorf("X rows (%lld) must match Y rows (%lld)",
                       static_cast<long long>(X.rows),
                       static_cast<long long>(Y.rows));
        return N4M_ERR_SHAPE_MISMATCH;
    }
    if (plan.n_samples != X.rows) {
        ctx.set_errorf("validation plan n_samples (%lld) must match X rows (%lld)",
                       static_cast<long long>(plan.n_samples),
                       static_cast<long long>(X.rows));
        return N4M_ERR_SHAPE_MISMATCH;
    }
    if (plan.folds.empty()) {
        ctx.set_error("siPLS requires at least one validation fold");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (interval_width < 1 || interval_width > X.cols) {
        ctx.set_errorf("interval_width must be in [1, %lld]; got %d",
                       static_cast<long long>(X.cols),
                       static_cast<int>(interval_width));
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const auto n_intervals = static_cast<std::int32_t>(intervals.size() / 2U);
    if (combination_size < 1 || combination_size > n_intervals) {
        ctx.set_errorf("combination_size must be in [1, %d]; got %d",
                       static_cast<int>(n_intervals),
                       static_cast<int>(combination_size));
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (cfg.n_components < 1 || cfg.n_components > X.cols) {
        ctx.set_errorf("n_components must be in [1, %lld]; got %d",
                       static_cast<long long>(X.cols),
                       static_cast<int>(cfg.n_components));
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (minimum_column_count_for_combination(intervals, combination_size) <
        static_cast<std::size_t>(cfg.n_components)) {
        ctx.set_error("combination_size can leave fewer columns than n_components");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t copy_active_columns(::n4m::core::Context& ctx,
                                               const n4m_matrix_view_t& X,
                                               const std::vector<std::int64_t>& intervals,
                                               const std::vector<std::int64_t>& active,
                                               std::vector<double>& out,
                                               std::int64_t& out_cols) {
    const std::size_t rows = static_cast<std::size_t>(X.rows);
    const std::size_t cols = active_column_count(intervals, active);
    if (cols == 0U) {
        ctx.set_error("siPLS active interval combination is empty");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (cols > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
        rows > std::numeric_limits<std::size_t>::max() / cols) {
        ctx.set_error("siPLS active matrix shape is too large");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    out.assign(rows * cols, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        std::size_t dest_col = 0;
        for (const std::int64_t interval_index : active) {
            const std::size_t offset = static_cast<std::size_t>(2 * interval_index);
            const std::size_t start = static_cast<std::size_t>(intervals[offset]);
            const std::size_t length = static_cast<std::size_t>(intervals[offset + 1U]);
            for (std::size_t local_col = 0; local_col < length; ++local_col) {
                out[idx(row, cols, dest_col)] = read_value(X, row, start + local_col);
                ++dest_col;
            }
        }
    }
    out_cols = static_cast<std::int64_t>(cols);
    return N4M_OK;
}

[[nodiscard]] n4m_status_t cv_rmse(::n4m::core::Context& ctx,
                                   const ::n4m::core::Config& cfg,
                                   const n4m_matrix_view_t& X,
                                   const n4m_matrix_view_t& Y,
                                   const ::n4m::core::ValidationPlan& plan,
                                   const std::vector<std::int64_t>& intervals,
                                   const std::vector<std::int64_t>& active,
                                   double& rmse) {
    std::vector<double> active_x;
    std::int64_t active_cols = 0;
    n4m_status_t status = copy_active_columns(ctx, X, intervals, active, active_x, active_cols);
    if (status != N4M_OK) {
        return status;
    }
    n4m_matrix_view_t active_x_view = rowmajor_f64_view(active_x, X.rows, active_cols);
    ::n4m::core::CrossValidationResult cv{};
    status = ::n4m::core::cross_validate_regression(ctx, cfg, active_x_view, Y, plan, cv);
    if (status != N4M_OK) {
        return status;
    }
    rmse = cv.metrics.rmse;
    return N4M_OK;
}

[[nodiscard]] bool advance_combination(std::vector<std::int64_t>& combination,
                                       std::int64_t n_intervals) noexcept {
    const auto k = static_cast<std::int64_t>(combination.size());
    for (std::int64_t i = k - 1; i >= 0; --i) {
        if (combination[static_cast<std::size_t>(i)] < n_intervals - k + i) {
            ++combination[static_cast<std::size_t>(i)];
            for (std::int64_t j = i + 1; j < k; ++j) {
                combination[static_cast<std::size_t>(j)] =
                    combination[static_cast<std::size_t>(j - 1)] + 1;
            }
            return true;
        }
    }
    return false;
}

[[nodiscard]] std::vector<std::int64_t> selected_features(
    const std::vector<std::int64_t>& intervals,
    const std::vector<std::int64_t>& active) {
    std::vector<std::int64_t> features;
    for (const std::int64_t interval_index : active) {
        const std::size_t offset = static_cast<std::size_t>(2 * interval_index);
        const std::int64_t start = intervals[offset];
        const std::int64_t length = intervals[offset + 1U];
        for (std::int64_t feature = start; feature < start + length; ++feature) {
            features.push_back(feature);
        }
    }
    return features;
}

}  // namespace

namespace n4m::core {

n4m_status_t select_by_sipls(Context& ctx,
                             const Config& cfg,
                             const n4m_matrix_view_t& X,
                             const n4m_matrix_view_t& Y,
                             const ValidationPlan& plan,
                             std::int32_t interval_width,
                             std::int32_t combination_size,
                             SiplsSelectionResult& out) {
    try {
        out = SiplsSelectionResult{};
        std::vector<std::int64_t> intervals = make_intervals(X.cols, interval_width);
        n4m_status_t status = validate_request(ctx,
                                               cfg,
                                               X,
                                               Y,
                                               plan,
                                               interval_width,
                                               combination_size,
                                               intervals);
        if (status != N4M_OK) {
            return status;
        }

        const std::int64_t n_intervals = static_cast<std::int64_t>(intervals.size() / 2U);
        std::vector<std::int64_t> combination(static_cast<std::size_t>(combination_size), 0);
        std::iota(combination.begin(), combination.end(), static_cast<std::int64_t>(0));
        std::vector<std::int64_t> best_combination = combination;
        double best_rmse = std::numeric_limits<double>::infinity();
        std::int32_t combination_index = 0;

        for (;;) {
            double rmse = 0.0;
            status = cv_rmse(ctx, cfg, X, Y, plan, intervals, combination, rmse);
            if (status != N4M_OK) {
                out = SiplsSelectionResult{};
                return status;
            }

            out.candidate_interval_indices.insert(out.candidate_interval_indices.end(),
                                                  combination.begin(),
                                                  combination.end());
            out.rmse_by_combination.push_back(rmse);
            if (rmse < best_rmse) {
                best_rmse = rmse;
                best_combination = combination;
                out.best_combination_index = combination_index;
            }

            if (!advance_combination(combination, n_intervals)) {
                break;
            }
            if (combination_index == std::numeric_limits<std::int32_t>::max()) {
                ctx.set_error("siPLS combination count is too large");
                out = SiplsSelectionResult{};
                return N4M_ERR_INVALID_ARGUMENT;
            }
            ++combination_index;
        }

        out.intervals = std::move(intervals);
        out.selected_interval_indices = best_combination;
        out.selected_feature_indices = selected_features(out.intervals, best_combination);
        out.n_intervals = static_cast<std::int32_t>(out.intervals.size() / 2U);
        out.interval_width = interval_width;
        out.combination_size = combination_size;
        out.n_combinations = static_cast<std::int32_t>(out.rmse_by_combination.size());
        ctx.clear_error();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while running siPLS selection");
        out = SiplsSelectionResult{};
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while running siPLS selection");
        out = SiplsSelectionResult{};
        return N4M_ERR_INTERNAL;
    }
}

}  // namespace n4m::core
