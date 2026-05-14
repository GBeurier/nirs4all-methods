// SPDX-License-Identifier: CeCILL-2.1

#include "core/lw_pls.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <utility>
#include <vector>

#include "core/matrix_view.hpp"
#include "core/model.hpp"
#include "core/status.hpp"

namespace {

[[nodiscard]] std::size_t idx(std::size_t row, std::size_t cols, std::size_t col) noexcept {
    return row * cols + col;
}

[[nodiscard]] bool checked_mul_size(std::size_t a,
                                    std::size_t b,
                                    std::size_t& out) noexcept {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        return false;
    }
    out = a * b;
    return true;
}

[[nodiscard]] double read_float(const p4a_matrix_view_t& view,
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

void resize_fill(std::vector<double>& values, std::size_t n, double fill) {
    values.clear();
    values.resize(n);
    std::fill(values.begin(), values.end(), fill);
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

[[nodiscard]] p4a_status_t copy_float_matrix(::pls4all::core::Context& ctx,
                                             const p4a_matrix_view_t& view,
                                             const char* name,
                                             std::vector<double>& out) {
    const auto rows = static_cast<std::size_t>(view.rows);
    const auto cols = static_cast<std::size_t>(view.cols);
    std::size_t total = 0;
    if (!checked_mul_size(rows, cols, total)) {
        ctx.set_errorf("%s matrix size overflows size_t", name);
        return P4A_ERR_INVALID_ARGUMENT;
    }
    resize_fill(out, total, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            const double value = read_float(view, row, col);
            if (!std::isfinite(value)) {
                ctx.set_errorf("%s contains NaN or Inf", name);
                return P4A_ERR_INVALID_ARGUMENT;
            }
            out[idx(row, cols, col)] = value;
        }
    }
    return P4A_OK;
}

[[nodiscard]] p4a_status_t standardize_for_distance(::pls4all::core::Context& ctx,
                                                    const std::vector<double>& X,
                                                    std::size_t rows,
                                                    std::size_t cols,
                                                    std::vector<double>& Xs) {
    std::size_t total = 0;
    if (!checked_mul_size(rows, cols, total)) {
        ctx.set_error("LW-PLS distance matrix size overflows size_t");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    resize_fill(Xs, total, 0.0);
    std::vector<double> mean(cols, 0.0);
    std::vector<double> scale(cols, 1.0);
    for (std::size_t col = 0; col < cols; ++col) {
        double sum = 0.0;
        for (std::size_t row = 0; row < rows; ++row) {
            sum += X[idx(row, cols, col)];
        }
        mean[col] = sum / static_cast<double>(rows);
        double sumsq = 0.0;
        for (std::size_t row = 0; row < rows; ++row) {
            const double delta = X[idx(row, cols, col)] - mean[col];
            sumsq += delta * delta;
        }
        double stddev = std::sqrt(sumsq / static_cast<double>(rows - 1U));
        if (stddev == 0.0 || !std::isfinite(stddev)) {
            stddev = 1.0;
        }
        scale[col] = stddev;
    }
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            Xs[idx(row, cols, col)] = (X[idx(row, cols, col)] - mean[col]) / scale[col];
        }
    }
    return P4A_OK;
}

}  // namespace

namespace pls4all::core {

p4a_status_t fit_predict_lw_pls(Context& ctx,
                               const Config& cfg,
                               const p4a_matrix_view_t& X,
                               const p4a_matrix_view_t& Y,
                               std::int32_t n_neighbors,
                               LwPlsResult& out) {
    try {
        out = LwPlsResult{};
        p4a_status_t status = validate_float_view(ctx, X, "X");
        if (status != P4A_OK) {
            return status;
        }
        status = validate_float_view(ctx, Y, "Y");
        if (status != P4A_OK) {
            return status;
        }
        if (X.rows != Y.rows) {
            ctx.set_error("X and Y row counts must match");
            return P4A_ERR_SHAPE_MISMATCH;
        }
        if (X.rows < 2 || X.cols < 1 || Y.cols < 1) {
            ctx.set_error("LW-PLS requires at least 2 rows, 1 X column and 1 Y column");
            return P4A_ERR_INVALID_ARGUMENT;
        }
        if (cfg.n_components < 1) {
            ctx.set_error("LW-PLS requires at least one PLS component");
            return P4A_ERR_INVALID_ARGUMENT;
        }
        if (n_neighbors < cfg.n_components + 1 || n_neighbors > X.rows) {
            ctx.set_error("LW-PLS n_neighbors must be in [n_components + 1, n_samples]");
            return P4A_ERR_INVALID_ARGUMENT;
        }
        if (X.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) ||
            Y.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
            ctx.set_error("LW-PLS dimensions exceed ABI limits");
            return P4A_ERR_INVALID_ARGUMENT;
        }

        const auto rows = static_cast<std::size_t>(X.rows);
        const auto cols = static_cast<std::size_t>(X.cols);
        const auto targets = static_cast<std::size_t>(Y.cols);
        const auto neighbors = static_cast<std::size_t>(n_neighbors);
        std::size_t pred_size = 0;
        std::size_t neighbor_size = 0;
        if (!checked_mul_size(rows, targets, pred_size) ||
            !checked_mul_size(rows, neighbors, neighbor_size)) {
            ctx.set_error("LW-PLS output size overflows size_t");
            return P4A_ERR_INVALID_ARGUMENT;
        }

        std::vector<double> x_values;
        std::vector<double> y_values;
        status = copy_float_matrix(ctx, X, "X", x_values);
        if (status != P4A_OK) {
            return status;
        }
        status = copy_float_matrix(ctx, Y, "Y", y_values);
        if (status != P4A_OK) {
            return status;
        }
        std::vector<double> standardized;
        status = standardize_for_distance(ctx, x_values, rows, cols, standardized);
        if (status != P4A_OK) {
            return status;
        }

        out.predictions.assign(pred_size, 0.0);
        out.neighbor_indices.assign(neighbor_size, 0);
        std::vector<std::pair<double, std::int64_t>> order(rows);
        std::vector<double> local_x(neighbors * cols, 0.0);
        std::vector<double> local_y(neighbors * targets, 0.0);
        std::vector<double> query(cols, 0.0);
        std::vector<double> query_pred(targets, 0.0);

        Config local_cfg = cfg;
        local_cfg.algorithm = P4A_ALGO_PLS_REGRESSION;
        local_cfg.deflation = P4A_DEFLATION_REGRESSION;
        local_cfg.center_x = 1;
        local_cfg.scale_x = 1;
        local_cfg.center_y = 1;
        local_cfg.scale_y = 1;

        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t other = 0; other < rows; ++other) {
                double distance = 0.0;
                for (std::size_t col = 0; col < cols; ++col) {
                    const double delta =
                        standardized[idx(other, cols, col)] -
                        standardized[idx(row, cols, col)];
                    distance += delta * delta;
                }
                order[other] = {distance, static_cast<std::int64_t>(other)};
            }
            std::sort(order.begin(), order.end());
            for (std::size_t nidx = 0; nidx < neighbors; ++nidx) {
                const auto selected = static_cast<std::size_t>(order[nidx].second);
                out.neighbor_indices[idx(row, neighbors, nidx)] = order[nidx].second;
                for (std::size_t col = 0; col < cols; ++col) {
                    local_x[idx(nidx, cols, col)] = x_values[idx(selected, cols, col)];
                }
                for (std::size_t target = 0; target < targets; ++target) {
                    local_y[idx(nidx, targets, target)] =
                        y_values[idx(selected, targets, target)];
                }
            }
            for (std::size_t col = 0; col < cols; ++col) {
                query[col] = x_values[idx(row, cols, col)];
            }

            p4a_matrix_view_t local_x_view =
                rowmajor_f64_view(local_x, static_cast<std::int64_t>(neighbors), X.cols);
            p4a_matrix_view_t local_y_view =
                rowmajor_f64_view(local_y, static_cast<std::int64_t>(neighbors), Y.cols);
            std::unique_ptr<Model> model;
            status = fit_model(ctx, local_cfg, local_x_view, local_y_view, model);
            if (status != P4A_OK) {
                out = LwPlsResult{};
                return status;
            }
            p4a_matrix_view_t query_view = rowmajor_f64_view(query, 1, X.cols);
            p4a_matrix_view_t pred_view = rowmajor_f64_view(query_pred, 1, Y.cols);
            status = predict_into(ctx, *model, query_view, pred_view);
            if (status != P4A_OK) {
                out = LwPlsResult{};
                return status;
            }
            for (std::size_t target = 0; target < targets; ++target) {
                out.predictions[idx(row, targets, target)] = query_pred[target];
            }
        }

        out.n_samples = X.rows;
        out.n_features = static_cast<std::int32_t>(X.cols);
        out.n_targets = static_cast<std::int32_t>(Y.cols);
        out.n_components = cfg.n_components;
        out.n_neighbors = n_neighbors;
        ctx.clear_error();
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while fitting LW-PLS");
        out = LwPlsResult{};
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while fitting LW-PLS");
        out = LwPlsResult{};
        return P4A_ERR_INTERNAL;
    }
}

}  // namespace pls4all::core
