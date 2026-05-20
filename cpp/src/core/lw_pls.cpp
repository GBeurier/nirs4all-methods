// SPDX-License-Identifier: CECILL-2.1

#include "core/lw_pls.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <utility>
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

[[nodiscard]] double dot(const std::vector<double>& a,
                         const std::vector<double>& b) noexcept {
    double out = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        out += a[i] * b[i];
    }
    return out;
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
        out.predictions.assign(pred_size, 0.0);
        out.neighbor_indices.assign(neighbor_size, 0);
        std::vector<std::pair<double, std::int64_t>> order(rows);
        std::vector<double> distances(rows, 0.0);
        std::vector<double> weights(rows, 0.0);
        std::vector<double> x_weighted_mean(cols, 0.0);
        std::vector<double> centered_x(rows * cols, 0.0);
        std::vector<double> centered_query(cols, 0.0);
        std::vector<double> centered_y(rows, 0.0);
        std::vector<double> loading_weight(cols, 0.0);
        std::vector<double> score(rows, 0.0);
        std::vector<double> loading_p(cols, 0.0);
        const double lambda = std::max(1.0, 0.5 * static_cast<double>(n_neighbors));
        constexpr double kEps = 1e-10;

        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t other = 0; other < rows; ++other) {
                double distance = 0.0;
                for (std::size_t col = 0; col < cols; ++col) {
                    const double delta =
                        x_values[idx(other, cols, col)] -
                        x_values[idx(row, cols, col)];
                    distance += delta * delta;
                }
                distance = std::sqrt(distance);
                distances[other] = distance;
                order[other] = {distance, static_cast<std::int64_t>(other)};
            }
            std::sort(order.begin(), order.end());
            for (std::size_t nidx = 0; nidx < neighbors; ++nidx) {
                out.neighbor_indices[idx(row, neighbors, nidx)] = order[nidx].second;
            }

            double distance_mean = 0.0;
            for (double distance : distances) {
                distance_mean += distance;
            }
            distance_mean /= static_cast<double>(rows);
            double distance_var = 0.0;
            for (double distance : distances) {
                const double delta = distance - distance_mean;
                distance_var += delta * delta;
            }
            double distance_std = std::sqrt(distance_var / static_cast<double>(rows - 1U));
            if (distance_std <= 0.0 || !std::isfinite(distance_std)) {
                distance_std = 1.0;
            }
            double weight_sum = 0.0;
            for (std::size_t other = 0; other < rows; ++other) {
                const double weight = std::exp(-distances[other] / distance_std / lambda);
                weights[other] = weight;
                weight_sum += weight;
            }
            if (weight_sum < kEps) {
                const double uniform = 1.0 / static_cast<double>(rows);
                std::fill(weights.begin(), weights.end(), uniform);
                weight_sum = 1.0;
            }

            std::fill(x_weighted_mean.begin(), x_weighted_mean.end(), 0.0);
            for (std::size_t other = 0; other < rows; ++other) {
                for (std::size_t col = 0; col < cols; ++col) {
                    x_weighted_mean[col] += weights[other] * x_values[idx(other, cols, col)];
                }
            }
            for (double& value : x_weighted_mean) {
                value /= weight_sum;
            }
            for (std::size_t other = 0; other < rows; ++other) {
                for (std::size_t col = 0; col < cols; ++col) {
                    centered_x[idx(other, cols, col)] =
                        x_values[idx(other, cols, col)] - x_weighted_mean[col];
                }
            }
            for (std::size_t col = 0; col < cols; ++col) {
                centered_query[col] = x_values[idx(row, cols, col)] - x_weighted_mean[col];
            }

            for (std::size_t target = 0; target < targets; ++target) {
                double y_weighted_mean = 0.0;
                for (std::size_t other = 0; other < rows; ++other) {
                    y_weighted_mean += weights[other] * y_values[idx(other, targets, target)];
                }
                y_weighted_mean /= weight_sum;
                for (std::size_t other = 0; other < rows; ++other) {
                    centered_y[other] =
                        y_values[idx(other, targets, target)] - y_weighted_mean;
                }

                std::vector<double> local_x = centered_x;
                std::vector<double> local_query = centered_query;
                double prediction = y_weighted_mean;
                for (std::int32_t comp = 0; comp < cfg.n_components; ++comp) {
                    std::fill(loading_weight.begin(), loading_weight.end(), 0.0);
                    for (std::size_t col = 0; col < cols; ++col) {
                        for (std::size_t other = 0; other < rows; ++other) {
                            loading_weight[col] +=
                                local_x[idx(other, cols, col)] *
                                weights[other] * centered_y[other];
                        }
                    }
                    double norm = std::sqrt(dot(loading_weight, loading_weight));
                    if (norm < kEps) {
                        break;
                    }
                    for (double& value : loading_weight) {
                        value /= norm;
                    }

                    std::fill(score.begin(), score.end(), 0.0);
                    for (std::size_t other = 0; other < rows; ++other) {
                        for (std::size_t col = 0; col < cols; ++col) {
                            score[other] +=
                                local_x[idx(other, cols, col)] * loading_weight[col];
                        }
                    }
                    double denom = 0.0;
                    for (std::size_t other = 0; other < rows; ++other) {
                        denom += weights[other] * score[other] * score[other];
                    }
                    if (denom < kEps) {
                        break;
                    }

                    std::fill(loading_p.begin(), loading_p.end(), 0.0);
                    for (std::size_t col = 0; col < cols; ++col) {
                        for (std::size_t other = 0; other < rows; ++other) {
                            loading_p[col] +=
                                local_x[idx(other, cols, col)] *
                                weights[other] * score[other];
                        }
                        loading_p[col] /= denom;
                    }
                    double loading_q = 0.0;
                    for (std::size_t other = 0; other < rows; ++other) {
                        loading_q += centered_y[other] * weights[other] * score[other];
                    }
                    loading_q /= denom;

                    const double query_score = dot(local_query, loading_weight);
                    prediction += query_score * loading_q;

                    if (comp < cfg.n_components - 1) {
                        for (std::size_t other = 0; other < rows; ++other) {
                            for (std::size_t col = 0; col < cols; ++col) {
                                local_x[idx(other, cols, col)] -=
                                    score[other] * loading_p[col];
                            }
                            centered_y[other] -= score[other] * loading_q;
                        }
                        for (std::size_t col = 0; col < cols; ++col) {
                            local_query[col] -= query_score * loading_p[col];
                        }
                    }
                }
                out.predictions[idx(row, targets, target)] = prediction;
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
