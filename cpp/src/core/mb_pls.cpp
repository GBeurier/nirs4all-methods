// SPDX-License-Identifier: CECILL-2.1

#include "core/mb_pls.hpp"

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

[[nodiscard]] p4a_status_t validate_block_sizes(::pls4all::core::Context& ctx,
                                                const p4a_matrix_view_t& X,
                                                const std::int64_t* block_sizes,
                                                std::size_t n_blocks) {
    if (block_sizes == nullptr || n_blocks == 0) {
        ctx.set_error("MB-PLS requires at least one block size");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (X.rows < 2 || X.cols < 1) {
        ctx.set_error("MB-PLS requires at least 2 rows and 1 X column");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    std::int64_t total_cols = 0;
    for (std::size_t block = 0; block < n_blocks; ++block) {
        if (block_sizes[block] <= 0) {
            ctx.set_error("MB-PLS block sizes must be positive");
            return P4A_ERR_INVALID_ARGUMENT;
        }
        if (total_cols > std::numeric_limits<std::int64_t>::max() - block_sizes[block]) {
            ctx.set_error("MB-PLS block sizes overflow int64");
            return P4A_ERR_INVALID_ARGUMENT;
        }
        total_cols += block_sizes[block];
    }
    if (total_cols != X.cols) {
        ctx.set_error("MB-PLS block sizes must sum to X columns");
        return P4A_ERR_SHAPE_MISMATCH;
    }

    return P4A_OK;
}

void column_means(const std::vector<double>& values,
                  std::size_t rows,
                  std::size_t cols,
                  std::vector<double>& means) {
    means.assign(cols, 0.0);
    if (rows == 0U) {
        return;
    }
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            means[col] += values[idx(row, cols, col)];
        }
    }
    const double denom = static_cast<double>(rows);
    for (double& mean : means) {
        mean /= denom;
    }
}

[[nodiscard]] bool invert_square_matrix(std::vector<double> a,
                                        std::size_t n,
                                        std::vector<double>& inverse) {
    inverse.assign(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        inverse[idx(i, n, i)] = 1.0;
    }
    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot = col;
        double pivot_abs = std::fabs(a[idx(col, n, col)]);
        for (std::size_t row = col + 1U; row < n; ++row) {
            const double candidate = std::fabs(a[idx(row, n, col)]);
            if (candidate > pivot_abs) {
                pivot = row;
                pivot_abs = candidate;
            }
        }
        if (pivot_abs <= std::numeric_limits<double>::epsilon()) {
            return false;
        }
        if (pivot != col) {
            for (std::size_t j = 0; j < n; ++j) {
                std::swap(a[idx(col, n, j)], a[idx(pivot, n, j)]);
                std::swap(inverse[idx(col, n, j)], inverse[idx(pivot, n, j)]);
            }
        }
        const double diag = a[idx(col, n, col)];
        for (std::size_t j = 0; j < n; ++j) {
            a[idx(col, n, j)] /= diag;
            inverse[idx(col, n, j)] /= diag;
        }
        for (std::size_t row = 0; row < n; ++row) {
            if (row == col) {
                continue;
            }
            const double factor = a[idx(row, n, col)];
            if (factor == 0.0) {
                continue;
            }
            for (std::size_t j = 0; j < n; ++j) {
                a[idx(row, n, j)] -= factor * a[idx(col, n, j)];
                inverse[idx(row, n, j)] -= factor * inverse[idx(col, n, j)];
            }
        }
    }
    return true;
}

}  // namespace

namespace pls4all::core {

p4a_status_t fit_predict_mb_pls(Context& ctx,
                               const Config& cfg,
                               const p4a_matrix_view_t& X,
                               const p4a_matrix_view_t& Y,
                               const std::int64_t* block_sizes,
                               std::size_t n_blocks,
                               MbPlsResult& out) {
    try {
        out = MbPlsResult{};
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
        if (Y.cols < 1) {
            ctx.set_error("Y must have at least one target column");
            return P4A_ERR_INVALID_ARGUMENT;
        }
        if (X.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) ||
            Y.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) ||
            n_blocks > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
            ctx.set_error("MB-PLS dimensions exceed ABI limits");
            return P4A_ERR_INVALID_ARGUMENT;
        }
        if (cfg.n_components < 1) {
            ctx.set_error("MB-PLS requires at least one PLS component");
            return P4A_ERR_INVALID_ARGUMENT;
        }

        status = validate_block_sizes(ctx, X, block_sizes, n_blocks);
        if (status != P4A_OK) {
            out = MbPlsResult{};
            return status;
        }

        std::vector<double> x_values;
        status = copy_float_matrix(ctx, X, "X", x_values);
        if (status != P4A_OK) {
            out = MbPlsResult{};
            return status;
        }
        std::vector<double> y_values;
        status = copy_float_matrix(ctx, Y, "Y", y_values);
        if (status != P4A_OK) {
            out = MbPlsResult{};
            return status;
        }

        const auto rows = static_cast<std::size_t>(X.rows);
        const auto cols = static_cast<std::size_t>(X.cols);
        const auto targets = static_cast<std::size_t>(Y.cols);
        const auto components = static_cast<std::size_t>(
            std::min<std::int32_t>(cfg.n_components,
                                   static_cast<std::int32_t>(
                                       std::min<std::size_t>(rows - 1U, cols))));
        std::size_t pred_size = 0;
        std::size_t coef_size = 0;
        if (!checked_mul_size(rows, targets, pred_size) ||
            !checked_mul_size(cols, targets, coef_size)) {
            ctx.set_error("MB-PLS output size overflows size_t");
            out = MbPlsResult{};
            return P4A_ERR_INVALID_ARGUMENT;
        }

        std::size_t latent_size = 0;
        if (!checked_mul_size(cols, components, latent_size)) {
            ctx.set_error("MB-PLS latent matrix size overflows size_t");
            out = MbPlsResult{};
            return P4A_ERR_INVALID_ARGUMENT;
        }

        column_means(x_values, rows, cols, out.x_mean);
        out.x_scale.assign(cols, 1.0);
        out.block_weights.assign(n_blocks, 1.0);
        std::vector<double> y_mean;
        column_means(y_values, rows, targets, y_mean);

        std::vector<std::vector<double>> x_res_blocks;
        std::vector<std::size_t> starts;
        x_res_blocks.reserve(n_blocks);
        starts.reserve(n_blocks);
        std::size_t start = 0;
        for (std::size_t block = 0; block < n_blocks; ++block) {
            const auto width = static_cast<std::size_t>(block_sizes[block]);
            starts.push_back(start);
            std::vector<double> block_values(rows * width, 0.0);
            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t local_col = 0; local_col < width; ++local_col) {
                    const std::size_t feature = start + local_col;
                    block_values[idx(row, width, local_col)] =
                        x_values[idx(row, cols, feature)] - out.x_mean[feature];
                }
            }
            x_res_blocks.push_back(std::move(block_values));
            start += width;
        }

        std::vector<double> y_res(rows * targets, 0.0);
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t target = 0; target < targets; ++target) {
                y_res[idx(row, targets, target)] =
                    y_values[idx(row, targets, target)] - y_mean[target];
            }
        }

        std::vector<double> W(cols * components, 0.0);
        std::vector<double> P(cols * components, 0.0);
        std::vector<double> Q(targets * components, 0.0);
        std::vector<double> score(rows, 0.0);
        std::vector<double> super_score(rows, 0.0);
        constexpr double kEps = 1e-10;

        for (std::size_t comp = 0; comp < components; ++comp) {
            std::fill(super_score.begin(), super_score.end(), 0.0);
            for (std::size_t block = 0; block < n_blocks; ++block) {
                const std::size_t width = static_cast<std::size_t>(block_sizes[block]);
                const std::size_t offset = starts[block];
                const auto& Xb = x_res_blocks[block];
                std::vector<double> w(width, 0.0);
                for (std::size_t local_col = 0; local_col < width; ++local_col) {
                    double value = 0.0;
                    for (std::size_t row = 0; row < rows; ++row) {
                        value += Xb[idx(row, width, local_col)] *
                                 y_res[idx(row, targets, 0)];
                    }
                    w[local_col] = value;
                }
                double w_norm = 0.0;
                for (double value : w) {
                    w_norm += value * value;
                }
                w_norm = std::sqrt(w_norm);
                if (w_norm > kEps) {
                    for (double& value : w) {
                        value /= w_norm;
                    }
                }
                std::fill(score.begin(), score.end(), 0.0);
                for (std::size_t row = 0; row < rows; ++row) {
                    for (std::size_t local_col = 0; local_col < width; ++local_col) {
                        score[row] += Xb[idx(row, width, local_col)] * w[local_col];
                    }
                    super_score[row] += score[row];
                }
                for (std::size_t local_col = 0; local_col < width; ++local_col) {
                    W[idx(offset + local_col, components, comp)] = w[local_col];
                }
            }

            for (double& value : super_score) {
                value /= static_cast<double>(n_blocks);
            }
            double t_norm = 0.0;
            for (double value : super_score) {
                t_norm += value * value;
            }
            t_norm = std::sqrt(t_norm);
            if (t_norm > kEps) {
                for (double& value : super_score) {
                    value /= t_norm;
                }
            }
            double t_dot = 1e-10;
            for (double value : super_score) {
                t_dot += value * value;
            }

            std::vector<std::vector<double>> p_blocks;
            p_blocks.reserve(n_blocks);
            for (std::size_t block = 0; block < n_blocks; ++block) {
                const std::size_t width = static_cast<std::size_t>(block_sizes[block]);
                const std::size_t offset = starts[block];
                const auto& Xb = x_res_blocks[block];
                std::vector<double> p_block(width, 0.0);
                for (std::size_t local_col = 0; local_col < width; ++local_col) {
                    double value = 0.0;
                    for (std::size_t row = 0; row < rows; ++row) {
                        value += Xb[idx(row, width, local_col)] * super_score[row];
                    }
                    value /= t_dot;
                    p_block[local_col] = value;
                    P[idx(offset + local_col, components, comp)] = value;
                }
                p_blocks.push_back(std::move(p_block));
            }

            for (std::size_t target = 0; target < targets; ++target) {
                double value = 0.0;
                for (std::size_t row = 0; row < rows; ++row) {
                    value += y_res[idx(row, targets, target)] * super_score[row];
                }
                Q[idx(target, components, comp)] = value / t_dot;
            }

            for (std::size_t block = 0; block < n_blocks; ++block) {
                const std::size_t width = static_cast<std::size_t>(block_sizes[block]);
                auto& Xb = x_res_blocks[block];
                const auto& p_block = p_blocks[block];
                for (std::size_t row = 0; row < rows; ++row) {
                    for (std::size_t local_col = 0; local_col < width; ++local_col) {
                        Xb[idx(row, width, local_col)] -=
                            super_score[row] * p_block[local_col];
                    }
                }
            }
            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t target = 0; target < targets; ++target) {
                    y_res[idx(row, targets, target)] -=
                        super_score[row] * Q[idx(target, components, comp)];
                }
            }
        }

        std::vector<double> ptw(components * components, 0.0);
        for (std::size_t row_comp = 0; row_comp < components; ++row_comp) {
            for (std::size_t col_comp = 0; col_comp < components; ++col_comp) {
                double value = 0.0;
                for (std::size_t feature = 0; feature < cols; ++feature) {
                    value += P[idx(feature, components, row_comp)] *
                             W[idx(feature, components, col_comp)];
                }
                if (row_comp == col_comp) {
                    value += 1e-10;
                }
                ptw[idx(row_comp, components, col_comp)] = value;
            }
        }
        std::vector<double> ptw_inv;
        if (!invert_square_matrix(ptw, components, ptw_inv)) {
            ctx.set_error("MB-PLS regularized latent system is singular");
            out = MbPlsResult{};
            return P4A_ERR_NUMERICAL_FAILURE;
        }

        std::vector<double> rotations(cols * components, 0.0);
        for (std::size_t feature = 0; feature < cols; ++feature) {
            for (std::size_t comp = 0; comp < components; ++comp) {
                double value = 0.0;
                for (std::size_t inner = 0; inner < components; ++inner) {
                    value += W[idx(feature, components, inner)] *
                             ptw_inv[idx(inner, components, comp)];
                }
                rotations[idx(feature, components, comp)] = value;
            }
        }

        out.coefficients.assign(coef_size, 0.0);
        for (std::size_t feature = 0; feature < cols; ++feature) {
            for (std::size_t target = 0; target < targets; ++target) {
                double value = 0.0;
                for (std::size_t comp = 0; comp < components; ++comp) {
                    value += rotations[idx(feature, components, comp)] *
                             Q[idx(target, components, comp)];
                }
                out.coefficients[idx(feature, targets, target)] = value;
            }
        }

        out.predictions.assign(pred_size, 0.0);
        out.intercept.assign(targets, 0.0);
        for (std::size_t target = 0; target < targets; ++target) {
            double value = y_mean[target];
            for (std::size_t feature = 0; feature < cols; ++feature) {
                value -= out.x_mean[feature] *
                         out.coefficients[idx(feature, targets, target)];
            }
            out.intercept[target] = value;
        }
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t target = 0; target < targets; ++target) {
                double value = out.intercept[target];
                for (std::size_t feature = 0; feature < cols; ++feature) {
                    value += x_values[idx(row, cols, feature)] *
                             out.coefficients[idx(feature, targets, target)];
                }
                out.predictions[idx(row, targets, target)] = value;
            }
        }

        out.n_samples = X.rows;
        out.n_features = static_cast<std::int32_t>(X.cols);
        out.n_targets = static_cast<std::int32_t>(Y.cols);
        out.n_components = static_cast<std::int32_t>(components);
        out.n_blocks = static_cast<std::int32_t>(n_blocks);
        ctx.clear_error();
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while fitting MB-PLS");
        out = MbPlsResult{};
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while fitting MB-PLS");
        out = MbPlsResult{};
        return P4A_ERR_INTERNAL;
    }
}

}  // namespace pls4all::core
