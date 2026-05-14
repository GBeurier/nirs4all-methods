// SPDX-License-Identifier: CeCILL-2.1

#include "core/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

#include "core/matrix_view.hpp"
#include "core/status.hpp"

namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();

[[nodiscard]] unsigned long long ull(std::size_t value) noexcept {
    return static_cast<unsigned long long>(value);
}

[[nodiscard]] double read_value(const p4a_matrix_view_t& view,
                                std::size_t row,
                                std::size_t col) noexcept {
    const std::int64_t off =
        static_cast<std::int64_t>(row) * view.row_stride +
        static_cast<std::int64_t>(col) * view.col_stride;
    const std::size_t uoff = static_cast<std::size_t>(off);
    if (view.dtype == P4A_DTYPE_F64) {
        const auto* ptr = static_cast<const double*>(view.data);
        return ptr[uoff];
    }
    const auto* ptr = static_cast<const float*>(view.data);
    return static_cast<double>(ptr[uoff]);
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
    if (view.rows == 0 || view.cols == 0) {
        ctx.set_errorf("%s must contain at least one value", name);
        return P4A_ERR_INVALID_ARGUMENT;
    }
    return P4A_OK;
}

[[nodiscard]] p4a_status_t copy_checked(::pls4all::core::Context& ctx,
                                        const p4a_matrix_view_t& view,
                                        const char* name,
                                        std::vector<double>& out) {
    const std::size_t rows = static_cast<std::size_t>(view.rows);
    const std::size_t cols = static_cast<std::size_t>(view.cols);
    if (cols != 0U &&
        rows > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) / cols) {
        ctx.set_errorf("%s shape is too large", name);
        return P4A_ERR_INVALID_ARGUMENT;
    }
    out.clear();
    out.resize(rows * cols);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            const double value = read_value(view, row, col);
            if (!std::isfinite(value)) {
                ctx.set_errorf("%s contains NaN or Inf at row %llu col %llu",
                               name,
                               ull(row),
                               ull(col));
                return P4A_ERR_INVALID_ARGUMENT;
            }
            out[row * cols + col] = value;
        }
    }
    return P4A_OK;
}

[[nodiscard]] double percentile_linear(std::vector<double> sorted,
                                       double probability) {
    std::sort(sorted.begin(), sorted.end());
    if (sorted.empty()) {
        return 0.0;
    }
    if (sorted.size() == 1U) {
        return sorted[0];
    }
    const double position = probability * static_cast<double>(sorted.size() - 1U);
    const std::size_t lo = static_cast<std::size_t>(std::floor(position));
    const std::size_t hi = std::min(lo + 1U, sorted.size() - 1U);
    const double frac = position - static_cast<double>(lo);
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

}  // namespace

namespace pls4all::core {

p4a_status_t compute_regression_metrics(Context& ctx,
                                        const p4a_matrix_view_t& observed,
                                        const p4a_matrix_view_t& predicted,
                                        RegressionMetrics& out) {
    try {
        out = RegressionMetrics{};
        p4a_status_t status = validate_float_view(ctx, observed, "observed");
        if (status != P4A_OK) {
            return status;
        }
        status = validate_float_view(ctx, predicted, "predicted");
        if (status != P4A_OK) {
            return status;
        }
        if (observed.rows != predicted.rows || observed.cols != predicted.cols) {
            ctx.set_errorf("observed shape (%lld, %lld) must match predicted shape (%lld, %lld)",
                           static_cast<long long>(observed.rows),
                           static_cast<long long>(observed.cols),
                           static_cast<long long>(predicted.rows),
                           static_cast<long long>(predicted.cols));
            return P4A_ERR_SHAPE_MISMATCH;
        }

        std::vector<double> y_true;
        std::vector<double> y_pred;
        status = copy_checked(ctx, observed, "observed", y_true);
        if (status != P4A_OK) {
            return status;
        }
        status = copy_checked(ctx, predicted, "predicted", y_pred);
        if (status != P4A_OK) {
            return status;
        }

        const std::size_t n = y_true.size();
        const double inv_n = 1.0 / static_cast<double>(n);
        double sum_true = 0.0;
        double sum_pred = 0.0;
        double sum_error = 0.0;
        double sum_abs_error = 0.0;
        double sum_squared_error = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double error = y_pred[i] - y_true[i];
            sum_true += y_true[i];
            sum_pred += y_pred[i];
            sum_error += error;
            sum_abs_error += std::fabs(error);
            sum_squared_error += error * error;
        }
        const double mean_true = sum_true * inv_n;
        const double mean_pred = sum_pred * inv_n;

        double tss = 0.0;
        double pred_ss = 0.0;
        double pred_true_cov = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double centered_true = y_true[i] - mean_true;
            const double centered_pred = y_pred[i] - mean_pred;
            tss += centered_true * centered_true;
            pred_ss += centered_pred * centered_pred;
            pred_true_cov += centered_pred * centered_true;
        }

        const double rmse = std::sqrt(sum_squared_error * inv_n);
        const double mae = sum_abs_error * inv_n;
        const double bias = sum_error * inv_n;
        const double r2 = (tss <= kEps)
            ? ((sum_squared_error <= kEps) ? 1.0 : 0.0)
            : (1.0 - sum_squared_error / tss);
        const double slope = (pred_ss <= kEps) ? 0.0 : (pred_true_cov / pred_ss);
        const double intercept = mean_true - slope * mean_pred;
        const double stddev = (n > 1U) ? std::sqrt(tss / static_cast<double>(n - 1U)) : 0.0;
        const double q1 = percentile_linear(y_true, 0.25);
        const double q3 = percentile_linear(y_true, 0.75);
        const double ratio_denominator = std::max(rmse, kEps);

        out.rows = observed.rows;
        out.cols = observed.cols;
        out.count = static_cast<std::int64_t>(n);
        out.rmse = rmse;
        out.mae = mae;
        out.bias = bias;
        out.r2 = r2;
        out.q2 = r2;
        out.slope = slope;
        out.intercept = intercept;
        out.rpd = stddev / ratio_denominator;
        out.rpiq = (q3 - q1) / ratio_denominator;
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while computing regression metrics");
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while computing regression metrics");
        return P4A_ERR_INTERNAL;
    }
}

}  // namespace pls4all::core
