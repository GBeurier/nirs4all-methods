// SPDX-License-Identifier: CECILL-2.1

#include "core/mb_pls.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <vector>

#include "core/common/matrix_view.hpp"
#include "core/model.hpp"
#include "core/common/status.hpp"

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

[[nodiscard]] double read_float(const n4m_matrix_view_t& view,
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

void resize_fill(std::vector<double>& values, std::size_t n, double fill) {
    values.clear();
    values.resize(n);
    std::fill(values.begin(), values.end(), fill);
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

[[nodiscard]] n4m_status_t copy_float_matrix(::n4m::core::Context& ctx,
                                             const n4m_matrix_view_t& view,
                                             const char* name,
                                             std::vector<double>& out) {
    const auto rows = static_cast<std::size_t>(view.rows);
    const auto cols = static_cast<std::size_t>(view.cols);
    std::size_t total = 0;
    if (!checked_mul_size(rows, cols, total)) {
        ctx.set_errorf("%s matrix size overflows size_t", name);
        return N4M_ERR_INVALID_ARGUMENT;
    }
    resize_fill(out, total, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            const double value = read_float(view, row, col);
            if (!std::isfinite(value)) {
                ctx.set_errorf("%s contains NaN or Inf", name);
                return N4M_ERR_INVALID_ARGUMENT;
            }
            out[idx(row, cols, col)] = value;
        }
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t transform_blocks(::n4m::core::Context& ctx,
                                            const n4m_matrix_view_t& X,
                                            const std::int64_t* block_sizes,
                                            std::size_t n_blocks,
                                            std::vector<double>& transformed,
                                            std::vector<double>& x_mean,
                                            std::vector<double>& x_scale,
                                            std::vector<double>& feature_weights,
                                            std::vector<double>& block_weights) {
    if (block_sizes == nullptr || n_blocks == 0) {
        ctx.set_error("MB-PLS requires at least one block size");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (X.rows < 2 || X.cols < 1) {
        ctx.set_error("MB-PLS requires at least 2 rows and 1 X column");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    std::int64_t total_cols = 0;
    for (std::size_t block = 0; block < n_blocks; ++block) {
        if (block_sizes[block] <= 0) {
            ctx.set_error("MB-PLS block sizes must be positive");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        if (total_cols > std::numeric_limits<std::int64_t>::max() - block_sizes[block]) {
            ctx.set_error("MB-PLS block sizes overflow int64");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        total_cols += block_sizes[block];
    }
    if (total_cols != X.cols) {
        ctx.set_error("MB-PLS block sizes must sum to X columns");
        return N4M_ERR_SHAPE_MISMATCH;
    }

    const auto rows = static_cast<std::size_t>(X.rows);
    const auto cols = static_cast<std::size_t>(X.cols);
    std::size_t total = 0;
    if (!checked_mul_size(rows, cols, total)) {
        ctx.set_error("MB-PLS transformed matrix size overflows size_t");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    resize_fill(transformed, total, 0.0);
    resize_fill(x_mean, cols, 0.0);
    resize_fill(x_scale, cols, 1.0);
    resize_fill(feature_weights, cols, 1.0);
    resize_fill(block_weights, n_blocks, 1.0);

    std::size_t start = 0;
    for (std::size_t block = 0; block < n_blocks; ++block) {
        const auto width = static_cast<std::size_t>(block_sizes[block]);
        const std::size_t stop = start + width;
        const double block_weight = 1.0 / std::sqrt(static_cast<double>(width));
        block_weights[block] = block_weight;
        for (std::size_t col = start; col < stop; ++col) {
            double sum = 0.0;
            for (std::size_t row = 0; row < rows; ++row) {
                const double value = read_float(X, row, col);
                if (!std::isfinite(value)) {
                    ctx.set_error("X contains NaN or Inf");
                    return N4M_ERR_INVALID_ARGUMENT;
                }
                sum += value;
            }
            const double mean = sum / static_cast<double>(rows);
            x_mean[col] = mean;
            double sumsq = 0.0;
            for (std::size_t row = 0; row < rows; ++row) {
                const double delta = read_float(X, row, col) - mean;
                sumsq += delta * delta;
            }
            double scale = std::sqrt(sumsq / static_cast<double>(rows - 1U));
            if (scale == 0.0 || !std::isfinite(scale)) {
                scale = 1.0;
            }
            x_scale[col] = scale;
            feature_weights[col] = block_weight;
            for (std::size_t row = 0; row < rows; ++row) {
                transformed[idx(row, cols, col)] =
                    (read_float(X, row, col) - mean) / scale * block_weight;
            }
        }
        start = stop;
    }
    return N4M_OK;
}

}  // namespace

namespace n4m::core {

n4m_status_t fit_predict_mb_pls(Context& ctx,
                               const Config& cfg,
                               const n4m_matrix_view_t& X,
                               const n4m_matrix_view_t& Y,
                               const std::int64_t* block_sizes,
                               std::size_t n_blocks,
                               MbPlsResult& out) {
    try {
        out = MbPlsResult{};
        n4m_status_t status = validate_float_view(ctx, X, "X");
        if (status != N4M_OK) {
            return status;
        }
        status = validate_float_view(ctx, Y, "Y");
        if (status != N4M_OK) {
            return status;
        }
        if (X.rows != Y.rows) {
            ctx.set_error("X and Y row counts must match");
            return N4M_ERR_SHAPE_MISMATCH;
        }
        if (Y.cols < 1) {
            ctx.set_error("Y must have at least one target column");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        if (X.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) ||
            Y.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) ||
            n_blocks > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
            ctx.set_error("MB-PLS dimensions exceed ABI limits");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        if (cfg.n_components < 1) {
            ctx.set_error("MB-PLS requires at least one PLS component");
            return N4M_ERR_INVALID_ARGUMENT;
        }

        std::vector<double> transformed;
        std::vector<double> feature_weights;
        status = transform_blocks(ctx,
                                  X,
                                  block_sizes,
                                  n_blocks,
                                  transformed,
                                  out.x_mean,
                                  out.x_scale,
                                  feature_weights,
                                  out.block_weights);
        if (status != N4M_OK) {
            out = MbPlsResult{};
            return status;
        }

        std::vector<double> y_values;
        status = copy_float_matrix(ctx, Y, "Y", y_values);
        if (status != N4M_OK) {
            out = MbPlsResult{};
            return status;
        }

        Config pls_cfg = cfg;
        pls_cfg.algorithm = N4M_ALGO_PLS_REGRESSION;
        pls_cfg.deflation = N4M_DEFLATION_REGRESSION;
        pls_cfg.center_x = 1;
        pls_cfg.scale_x = 0;
        pls_cfg.center_y = 1;
        pls_cfg.scale_y = 0;
        n4m_matrix_view_t Xt = rowmajor_f64_view(transformed, X.rows, X.cols);
        n4m_matrix_view_t Yt = rowmajor_f64_view(y_values, Y.rows, Y.cols);
        std::unique_ptr<Model> model;
        status = fit_model(ctx, pls_cfg, Xt, Yt, model);
        if (status != N4M_OK) {
            out = MbPlsResult{};
            return status;
        }

        const auto rows = static_cast<std::size_t>(X.rows);
        const auto cols = static_cast<std::size_t>(X.cols);
        const auto targets = static_cast<std::size_t>(Y.cols);
        std::size_t pred_size = 0;
        std::size_t coef_size = 0;
        if (!checked_mul_size(rows, targets, pred_size) ||
            !checked_mul_size(cols, targets, coef_size)) {
            ctx.set_error("MB-PLS output size overflows size_t");
            out = MbPlsResult{};
            return N4M_ERR_INVALID_ARGUMENT;
        }

        out.predictions.assign(pred_size, 0.0);
        n4m_matrix_view_t pred_view = rowmajor_f64_view(out.predictions, X.rows, Y.cols);
        status = predict_into(ctx, *model, Xt, pred_view);
        if (status != N4M_OK) {
            out = MbPlsResult{};
            return status;
        }

        out.coefficients.assign(coef_size, 0.0);
        out.intercept.assign(targets, 0.0);
        for (std::size_t feature = 0; feature < cols; ++feature) {
            const double factor = feature_weights[feature] / out.x_scale[feature];
            for (std::size_t target = 0; target < targets; ++target) {
                out.coefficients[idx(feature, targets, target)] =
                    model->coefficients[idx(feature, targets, target)] * factor;
            }
        }
        for (std::size_t target = 0; target < targets; ++target) {
            double value = model->y_mean[target];
            for (std::size_t feature = 0; feature < cols; ++feature) {
                value -= model->x_mean[feature] *
                         model->coefficients[idx(feature, targets, target)];
                value -= out.x_mean[feature] *
                         out.coefficients[idx(feature, targets, target)];
            }
            out.intercept[target] = value;
        }

        out.n_samples = X.rows;
        out.n_features = static_cast<std::int32_t>(X.cols);
        out.n_targets = static_cast<std::int32_t>(Y.cols);
        out.n_components = cfg.n_components;
        out.n_blocks = static_cast<std::int32_t>(n_blocks);
        ctx.clear_error();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while fitting MB-PLS");
        out = MbPlsResult{};
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while fitting MB-PLS");
        out = MbPlsResult{};
        return N4M_ERR_INTERNAL;
    }
}

}  // namespace n4m::core
