// SPDX-License-Identifier: CECILL-2.1

#include "core/interval_selection.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <utility>
#include <vector>

#include "core/cross_validation.hpp"
#include "core/matrix_view.hpp"
#include "core/status.hpp"

namespace {

void append_metrics_row(const ::pls4all::core::RegressionMetrics& metrics,
                        std::vector<double>& out) {
    out.push_back(metrics.rmse);
    out.push_back(metrics.mae);
    out.push_back(metrics.bias);
    out.push_back(metrics.r2);
    out.push_back(metrics.q2);
    out.push_back(metrics.slope);
    out.push_back(metrics.intercept);
    out.push_back(metrics.rpd);
    out.push_back(metrics.rpiq);
}

[[nodiscard]] bool checked_matrix_size(std::int64_t rows,
                                       std::int64_t cols,
                                       std::size_t& out) noexcept {
    if (rows < 0 || cols < 0) {
        return false;
    }
    const auto urows = static_cast<std::size_t>(rows);
    const auto ucols = static_cast<std::size_t>(cols);
    if (ucols != 0U &&
        urows > std::numeric_limits<std::size_t>::max() / ucols) {
        return false;
    }
    out = urows * ucols;
    return true;
}

[[nodiscard]] double read_value(const p4a_matrix_view_t& view,
                                std::size_t row,
                                std::size_t col) noexcept {
    const std::int64_t off =
        static_cast<std::int64_t>(row) * view.row_stride +
        static_cast<std::int64_t>(col) * view.col_stride;
    const auto uoff = static_cast<std::size_t>(off);
    if (view.dtype == P4A_DTYPE_F64) {
        const auto* ptr = static_cast<const double*>(view.data);
        return ptr[uoff];
    }
    const auto* ptr = static_cast<const float*>(view.data);
    return static_cast<double>(ptr[uoff]);
}

[[nodiscard]] p4a_matrix_view_t rowmajor_f64_view(std::vector<double>& values,
                                                  std::int64_t rows,
                                                  std::int64_t cols) noexcept {
    p4a_matrix_view_t view{};
    view.data = values.data();
    view.rows = rows;
    view.cols = cols;
    view.row_stride = cols > 0 ? cols : 1;
    view.col_stride = 1;
    view.dtype = P4A_DTYPE_F64;
    return view;
}

[[nodiscard]] p4a_status_t validate_float_view(::pls4all::core::Context& ctx,
                                               const p4a_matrix_view_t& view,
                                               const char* name) noexcept {
    const p4a_status_t status = ::pls4all::core::validate_nonnull_view(view);
    if (status != P4A_OK) {
        ctx.set_errorf("%s matrix view is invalid: %s",
                       name,
                       ::pls4all::core::status_to_string(status));
        return status;
    }
    if (view.dtype != P4A_DTYPE_F64 && view.dtype != P4A_DTYPE_F32) {
        ctx.set_errorf("%s dtype must be f64 or f32", name);
        return P4A_ERR_DTYPE_MISMATCH;
    }
    return P4A_OK;
}

[[nodiscard]] p4a_status_t validate_interval_request(
    ::pls4all::core::Context& ctx,
    const ::pls4all::core::Config& cfg,
    const p4a_matrix_view_t& X,
    const p4a_matrix_view_t& Y,
    const ::pls4all::core::ValidationPlan& plan,
    std::int32_t interval_width,
    std::int32_t step) {
    p4a_status_t status = validate_float_view(ctx, X, "X");
    if (status != P4A_OK) {
        return status;
    }
    status = validate_float_view(ctx, Y, "Y");
    if (status != P4A_OK) {
        return status;
    }
    if (X.rows == 0 || X.cols == 0 || Y.cols == 0) {
        ctx.set_error("interval selection matrices must be non-empty");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (X.rows != Y.rows) {
        ctx.set_errorf("X rows (%lld) must match Y rows (%lld)",
                       static_cast<long long>(X.rows),
                       static_cast<long long>(Y.rows));
        return P4A_ERR_SHAPE_MISMATCH;
    }
    if (plan.n_samples != X.rows) {
        ctx.set_errorf("validation plan n_samples (%lld) must match X rows (%lld)",
                       static_cast<long long>(plan.n_samples),
                       static_cast<long long>(X.rows));
        return P4A_ERR_SHAPE_MISMATCH;
    }
    if (plan.folds.empty()) {
        ctx.set_error("interval selection requires at least one validation fold");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (interval_width < 1 || interval_width > X.cols) {
        ctx.set_errorf("interval_width must be in [1, %lld]; got %d",
                       static_cast<long long>(X.cols),
                       static_cast<int>(interval_width));
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (step < 1) {
        ctx.set_errorf("interval step must be >= 1; got %d", static_cast<int>(step));
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (cfg.n_components < 1 || cfg.n_components > interval_width) {
        ctx.set_errorf("n_components must be in [1, interval_width]; got %d for width %d",
                       static_cast<int>(cfg.n_components),
                       static_cast<int>(interval_width));
        return P4A_ERR_INVALID_ARGUMENT;
    }
    return P4A_OK;
}

[[nodiscard]] p4a_status_t copy_interval(::pls4all::core::Context& ctx,
                                         const p4a_matrix_view_t& X,
                                         std::int64_t start,
                                         std::int32_t width,
                                         std::vector<double>& out) {
    std::size_t n_values = 0;
    if (!checked_matrix_size(X.rows, width, n_values)) {
        ctx.set_error("interval selection matrix shape is too large");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    out.assign(n_values, 0.0);
    const auto rows = static_cast<std::size_t>(X.rows);
    const auto ustart = static_cast<std::size_t>(start);
    const auto uwidth = static_cast<std::size_t>(width);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t local_col = 0; local_col < uwidth; ++local_col) {
            out[row * uwidth + local_col] = read_value(X, row, ustart + local_col);
        }
    }
    return P4A_OK;
}

}  // namespace

namespace pls4all::core {

p4a_status_t cross_validate_intervals(Context& ctx,
                                      const Config& cfg,
                                      const p4a_matrix_view_t& X,
                                      const p4a_matrix_view_t& Y,
                                      const ValidationPlan& plan,
                                      std::int32_t interval_width,
                                      std::int32_t step,
                                      IntervalSelectionResult& out) {
    try {
        out = IntervalSelectionResult{};
        p4a_status_t status = validate_interval_request(ctx,
                                                        cfg,
                                                        X,
                                                        Y,
                                                        plan,
                                                        interval_width,
                                                        step);
        if (status != P4A_OK) {
            return status;
        }

        const std::int64_t p = X.cols;
        const std::int64_t width = interval_width;
        std::int32_t interval_index = 0;
        double best_rmse = std::numeric_limits<double>::infinity();

        for (std::int64_t start = 0; start + width <= p; start += step) {
            std::vector<double> interval_x;
            status = copy_interval(ctx, X, start, interval_width, interval_x);
            if (status != P4A_OK) {
                out = IntervalSelectionResult{};
                return status;
            }

            p4a_matrix_view_t interval_x_view = rowmajor_f64_view(interval_x,
                                                                  X.rows,
                                                                  interval_width);
            CrossValidationResult cv{};
            status = cross_validate_regression(ctx, cfg, interval_x_view, Y, plan, cv);
            if (status != P4A_OK) {
                out = IntervalSelectionResult{};
                return status;
            }

            out.intervals.push_back(start);
            out.intervals.push_back(width);
            out.rmse.push_back(cv.metrics.rmse);
            out.metrics_by_interval.push_back(cv.metrics);
            append_metrics_row(cv.metrics, out.metrics_matrix);
            if (cv.metrics.rmse < best_rmse) {
                best_rmse = cv.metrics.rmse;
                out.best_interval_index = interval_index;
                out.best_start = start;
                out.best_length = width;
            }
            ++interval_index;
        }

        if (out.intervals.empty()) {
            ctx.set_error("interval selection produced no candidate intervals");
            out = IntervalSelectionResult{};
            return P4A_ERR_INVALID_ARGUMENT;
        }

        out.interval_width = interval_width;
        out.step = step;
        ctx.clear_error();
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while running interval selection");
        out = IntervalSelectionResult{};
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while running interval selection");
        out = IntervalSelectionResult{};
        return P4A_ERR_INTERNAL;
    }
}

}  // namespace pls4all::core
