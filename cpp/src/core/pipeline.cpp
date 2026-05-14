// SPDX-License-Identifier: CeCILL-2.1

#include "core/pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "core/matrix_view.hpp"
#include "core/status.hpp"

namespace {

[[nodiscard]] std::size_t idx(std::size_t row, std::size_t cols, std::size_t col) noexcept {
    return row * cols + col;
}

[[nodiscard]] unsigned long long ull(std::size_t value) noexcept {
    return static_cast<unsigned long long>(value);
}

[[nodiscard]] double read_value(const p4a_matrix_view_t& v,
                                std::size_t row,
                                std::size_t col) noexcept {
    const std::int64_t off =
        static_cast<std::int64_t>(row) * v.row_stride +
        static_cast<std::int64_t>(col) * v.col_stride;
    const std::size_t uoff = static_cast<std::size_t>(off);
    if (v.dtype == P4A_DTYPE_F64) {
        const auto* ptr = static_cast<const double*>(v.data);
        return ptr[uoff];
    }
    const auto* ptr = static_cast<const float*>(v.data);
    return static_cast<double>(ptr[uoff]);
}

void write_value(p4a_matrix_view_t& v,
                 std::size_t row,
                 std::size_t col,
                 double value) noexcept {
    const std::int64_t off =
        static_cast<std::int64_t>(row) * v.row_stride +
        static_cast<std::int64_t>(col) * v.col_stride;
    const std::size_t uoff = static_cast<std::size_t>(off);
    if (v.dtype == P4A_DTYPE_F64) {
        auto* ptr = static_cast<double*>(v.data);
        ptr[uoff] = value;
        return;
    }
    auto* ptr = static_cast<float*>(v.data);
    ptr[uoff] = static_cast<float>(value);
}

[[nodiscard]] p4a_status_t validate_float_view(::pls4all::core::Context& ctx,
                                               const p4a_matrix_view_t& v,
                                               const char* name) noexcept {
    const p4a_status_t status = ::pls4all::core::validate_nonnull_view(v);
    if (status != P4A_OK) {
        ctx.set_errorf("%s matrix view is invalid: %s",
                       name,
                       ::pls4all::core::status_to_string(status));
        return status;
    }
    if (v.dtype != P4A_DTYPE_F64 && v.dtype != P4A_DTYPE_F32) {
        ctx.set_errorf("%s dtype must be f64 or f32", name);
        return P4A_ERR_DTYPE_MISMATCH;
    }
    return P4A_OK;
}

[[nodiscard]] bool element_count(std::int64_t rows,
                                 std::int64_t cols,
                                 std::size_t& out) noexcept {
    if (rows < 0 || cols < 0) {
        return false;
    }
    const std::uint64_t urows = static_cast<std::uint64_t>(rows);
    const std::uint64_t ucols = static_cast<std::uint64_t>(cols);
    if (ucols != 0U &&
        urows > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) / ucols) {
        return false;
    }
    out = static_cast<std::size_t>(urows * ucols);
    return true;
}

[[nodiscard]] p4a_status_t copy_matrix_checked(::pls4all::core::Context& ctx,
                                               const p4a_matrix_view_t& src,
                                               const char* name,
                                               std::vector<double>& out) {
    std::size_t n_values = 0;
    if (!element_count(src.rows, src.cols, n_values)) {
        ctx.set_errorf("%s shape is invalid or too large", name);
        return P4A_ERR_INVALID_ARGUMENT;
    }
    out.assign(n_values, 0.0);
    const std::size_t rows = static_cast<std::size_t>(src.rows);
    const std::size_t cols = static_cast<std::size_t>(src.cols);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            const double value = read_value(src, row, col);
            if (!std::isfinite(value)) {
                ctx.set_errorf("%s contains NaN or Inf at row %llu col %llu",
                               name,
                               ull(row),
                               ull(col));
                return P4A_ERR_INVALID_ARGUMENT;
            }
            out[idx(row, cols, col)] = value;
        }
    }
    return P4A_OK;
}

void copy_to_output(const std::vector<double>& values,
                    std::size_t rows,
                    std::size_t cols,
                    p4a_matrix_view_t& out) noexcept {
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            write_value(out, row, col, values[idx(row, cols, col)]);
        }
    }
}

void fit_column_center(const std::vector<double>& values,
                       std::size_t rows,
                       std::size_t cols,
                       std::vector<double>& mean) {
    mean.assign(cols, 0.0);
    for (std::size_t col = 0; col < cols; ++col) {
        double sum = 0.0;
        for (std::size_t row = 0; row < rows; ++row) {
            sum += values[idx(row, cols, col)];
        }
        mean[col] = sum / static_cast<double>(rows);
    }
}

void fit_column_scale(const std::vector<double>& values,
                      const std::vector<double>& mean,
                      std::size_t rows,
                      std::size_t cols,
                      bool pareto,
                      std::vector<double>& scale) {
    scale.assign(cols, 1.0);
    const double denom = rows > 1U ? static_cast<double>(rows - 1U) : 1.0;
    for (std::size_t col = 0; col < cols; ++col) {
        double sumsq = 0.0;
        for (std::size_t row = 0; row < rows; ++row) {
            const double centered = values[idx(row, cols, col)] - mean[col];
            sumsq += centered * centered;
        }
        const double stddev = std::sqrt(sumsq / denom);
        const double raw_scale = pareto ? std::sqrt(stddev) : stddev;
        scale[col] = (raw_scale == 0.0 || !std::isfinite(raw_scale)) ? 1.0 : raw_scale;
    }
}

void apply_column_transform(const std::vector<double>& values,
                            const std::vector<double>& mean,
                            const std::vector<double>& scale,
                            std::size_t rows,
                            std::size_t cols,
                            std::vector<double>& out) {
    out.assign(rows * cols, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            out[idx(row, cols, col)] =
                (values[idx(row, cols, col)] - mean[col]) / scale[col];
        }
    }
}

void apply_snv(const std::vector<double>& values,
               std::size_t rows,
               std::size_t cols,
               std::vector<double>& out) {
    out.assign(rows * cols, 0.0);
    const double denom = cols > 1U ? static_cast<double>(cols - 1U) : 1.0;
    for (std::size_t row = 0; row < rows; ++row) {
        double mean = 0.0;
        for (std::size_t col = 0; col < cols; ++col) {
            mean += values[idx(row, cols, col)];
        }
        mean /= static_cast<double>(cols);
        double sumsq = 0.0;
        for (std::size_t col = 0; col < cols; ++col) {
            const double centered = values[idx(row, cols, col)] - mean;
            sumsq += centered * centered;
        }
        const double stddev = std::sqrt(sumsq / denom);
        const double scale = (stddev == 0.0 || !std::isfinite(stddev)) ? 1.0 : stddev;
        for (std::size_t col = 0; col < cols; ++col) {
            out[idx(row, cols, col)] = (values[idx(row, cols, col)] - mean) / scale;
        }
    }
}

}  // namespace

namespace pls4all::core {

p4a_status_t Pipeline::fit(Context& ctx, const p4a_matrix_view_t& X) {
    fitted_ = false;
    states_.clear();

    p4a_status_t status = validate_float_view(ctx, X, "X");
    if (status != P4A_OK) {
        return status;
    }
    if (X.rows < 1 || X.cols < 1) {
        ctx.set_error("pipeline fit requires at least 1 row and 1 column");
        return P4A_ERR_INVALID_ARGUMENT;
    }

    std::vector<double> current;
    status = copy_matrix_checked(ctx, X, "X", current);
    if (status != P4A_OK) {
        return status;
    }

    const std::size_t rows = static_cast<std::size_t>(X.rows);
    const std::size_t cols = static_cast<std::size_t>(X.cols);
    states_.reserve(entries_.size());

    for (const OperatorEntry& entry : entries_) {
        if (!entry.params.empty()) {
            ctx.set_error("phase-3a pipeline operators do not accept parameters");
            states_.clear();
            return P4A_ERR_INVALID_ARGUMENT;
        }

        OperatorState state;
        state.kind = entry.kind;
        state.n_features = X.cols;
        std::vector<double> next;

        switch (entry.kind) {
            case P4A_OP_IDENTITY:
                next = current;
                break;
            case P4A_OP_CENTER:
                fit_column_center(current, rows, cols, state.location);
                state.scale.assign(cols, 1.0);
                apply_column_transform(current, state.location, state.scale, rows, cols, next);
                break;
            case P4A_OP_AUTOSCALE:
                fit_column_center(current, rows, cols, state.location);
                fit_column_scale(current, state.location, rows, cols, false, state.scale);
                apply_column_transform(current, state.location, state.scale, rows, cols, next);
                break;
            case P4A_OP_PARETO_SCALE:
                fit_column_center(current, rows, cols, state.location);
                fit_column_scale(current, state.location, rows, cols, true, state.scale);
                apply_column_transform(current, state.location, state.scale, rows, cols, next);
                break;
            case P4A_OP_SNV:
                apply_snv(current, rows, cols, next);
                break;
            default:
                ctx.set_errorf("pipeline operator %d is not implemented in phase 3a",
                               static_cast<int>(entry.kind));
                states_.clear();
                return P4A_ERR_NOT_IMPLEMENTED;
        }

        states_.push_back(std::move(state));
        current.swap(next);
    }

    n_features_ = X.cols;
    fitted_ = true;
    ctx.clear_error();
    return P4A_OK;
}

p4a_status_t Pipeline::transform(Context& ctx,
                                 const p4a_matrix_view_t& X,
                                 p4a_matrix_view_t& out) const {
    if (!fitted_) {
        ctx.set_error("pipeline is not fitted");
        return P4A_ERR_NOT_FITTED;
    }

    p4a_status_t status = validate_float_view(ctx, X, "X");
    if (status != P4A_OK) {
        return status;
    }
    status = validate_float_view(ctx, out, "output");
    if (status != P4A_OK) {
        return status;
    }
    if (X.cols != n_features_) {
        ctx.set_errorf("X cols (%lld) must match fitted pipeline cols (%lld)",
                       static_cast<long long>(X.cols),
                       static_cast<long long>(n_features_));
        return P4A_ERR_SHAPE_MISMATCH;
    }
    if (out.rows != X.rows || out.cols != X.cols) {
        ctx.set_error("pipeline output shape must match X shape");
        return P4A_ERR_SHAPE_MISMATCH;
    }

    std::vector<double> current;
    status = copy_matrix_checked(ctx, X, "X", current);
    if (status != P4A_OK) {
        return status;
    }

    const std::size_t rows = static_cast<std::size_t>(X.rows);
    const std::size_t cols = static_cast<std::size_t>(X.cols);
    for (const OperatorState& state : states_) {
        std::vector<double> next;
        switch (state.kind) {
            case P4A_OP_IDENTITY:
                next = current;
                break;
            case P4A_OP_CENTER:
            case P4A_OP_AUTOSCALE:
            case P4A_OP_PARETO_SCALE:
                apply_column_transform(current, state.location, state.scale, rows, cols, next);
                break;
            case P4A_OP_SNV:
                apply_snv(current, rows, cols, next);
                break;
            default:
                ctx.set_error("fitted pipeline contains an unsupported operator");
                return P4A_ERR_INTERNAL;
        }
        current.swap(next);
    }

    copy_to_output(current, rows, cols, out);
    ctx.clear_error();
    return P4A_OK;
}

}  // namespace pls4all::core
