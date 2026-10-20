// SPDX-License-Identifier: CECILL-2.1

#include "core/t2_selection.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <utility>
#include <vector>

#include "core/common/beta_quantile.h"
#include "core/cross_validation.hpp"
#include "core/common/matrix_view.hpp"
#include "core/model.hpp"
#include "core/common/status.hpp"

namespace {

[[nodiscard]] std::size_t idx(std::size_t row, std::size_t cols, std::size_t col) noexcept {
    return row * cols + col;
}

[[nodiscard]] double round2(double value) noexcept {
    return std::nearbyint(value * 100.0) / 100.0;
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

[[nodiscard]] n4m_status_t validate_request(::n4m::core::Context& ctx,
                                            const ::n4m::core::Config& cfg,
                                            const n4m_matrix_view_t& X,
                                            const n4m_matrix_view_t& Y,
                                            const ::n4m::core::ValidationPlan& plan,
                                            const std::vector<double>& alpha,
                                            std::int32_t min_selected) {
    n4m_status_t status = validate_float_view(ctx, X, "X");
    if (status != N4M_OK) {
        return status;
    }
    status = validate_float_view(ctx, Y, "Y");
    if (status != N4M_OK) {
        return status;
    }
    if (X.rows == 0 || X.cols == 0 || Y.cols == 0) {
        ctx.set_error("T2-PLS matrices must be non-empty");
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
        ctx.set_error("T2-PLS requires at least one validation fold");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (alpha.empty()) {
        ctx.set_error("T2-PLS requires at least one alpha value");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    for (const double value : alpha) {
        if (!std::isfinite(value) || value <= 0.0 || value >= 1.0) {
            ctx.set_error("T2-PLS alpha values must be finite and in (0, 1)");
            return N4M_ERR_INVALID_ARGUMENT;
        }
    }
    if (cfg.n_components < 1 || static_cast<std::int64_t>(cfg.n_components) >= X.cols - 1) {
        ctx.set_errorf("n_components must be in [1, %lld) for T2-PLS; got %d",
                       static_cast<long long>(X.cols - 1),
                       static_cast<int>(cfg.n_components));
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (min_selected < cfg.n_components || static_cast<std::int64_t>(min_selected) > X.cols) {
        ctx.set_errorf("min_selected must be in [n_components, %lld]; got %d",
                       static_cast<long long>(X.cols),
                       static_cast<int>(min_selected));
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (X.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) ||
        Y.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) ||
        alpha.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        ctx.set_error("T2-PLS dimensions exceed int32 result storage");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    return N4M_OK;
}

void resize_fill(std::vector<double>& values, std::size_t n, double fill) {
    values.clear();
    values.resize(n);
    std::fill(values.begin(), values.end(), fill);
}

// Gauss-Jordan inverse of the symmetric weight covariance. A pivot not above
// 1e-14 times the largest diagonal entry is refused as singular: the
// weights' scale follows the solver (unit W for NIPALS, W ~ 1 / ||X|| for
// SIMPLS or kernel), so the test is relative.
[[nodiscard]] bool invert_square(std::vector<double> a,
                                 std::size_t n,
                                 std::vector<double>& inverse) {
    resize_fill(inverse, n * n, 0.0);
    double largest_diagonal = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        inverse[idx(i, n, i)] = 1.0;
        largest_diagonal = std::max(largest_diagonal, std::fabs(a[idx(i, n, i)]));
    }
    const double pivot_floor = 1e-14 * largest_diagonal;
    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot = col;
        double pivot_abs = std::fabs(a[idx(col, n, col)]);
        for (std::size_t row = col + 1; row < n; ++row) {
            const double candidate = std::fabs(a[idx(row, n, col)]);
            if (candidate > pivot_abs) {
                pivot_abs = candidate;
                pivot = row;
            }
        }
        if (!(pivot_abs > pivot_floor)) {
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

[[nodiscard]] n4m_status_t copy_columns(::n4m::core::Context& ctx,
                                        const n4m_matrix_view_t& X,
                                        const std::vector<std::int64_t>& columns,
                                        std::vector<double>& out) {
    if (columns.empty()) {
        ctx.set_error("T2-PLS column selection must not be empty");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    std::size_t n_values = 0;
    if (!checked_matrix_size(X.rows, static_cast<std::int64_t>(columns.size()), n_values)) {
        ctx.set_error("T2-PLS subset matrix shape is too large");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    out.assign(n_values, 0.0);
    const auto rows = static_cast<std::size_t>(X.rows);
    const auto cols = columns.size();
    for (std::size_t local_col = 0; local_col < cols; ++local_col) {
        const std::int64_t original_col = columns[local_col];
        if (original_col < 0 || original_col >= X.cols) {
            ctx.set_error("T2-PLS column index out of range");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        const auto src_col = static_cast<std::size_t>(original_col);
        for (std::size_t row = 0; row < rows; ++row) {
            const double value = read_value(X, row, src_col);
            if (!std::isfinite(value)) {
                ctx.set_error("T2-PLS X contains NaN or Inf");
                return N4M_ERR_INVALID_ARGUMENT;
            }
            out[idx(row, cols, local_col)] = value;
        }
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t subset_rmse(::n4m::core::Context& ctx,
                                       const ::n4m::core::Config& cfg,
                                       const n4m_matrix_view_t& X,
                                       const n4m_matrix_view_t& Y,
                                       const ::n4m::core::ValidationPlan& plan,
                                       const std::vector<std::int64_t>& columns,
                                       double& out) {
    std::vector<double> subset_x;
    n4m_status_t status = copy_columns(ctx, X, columns, subset_x);
    if (status != N4M_OK) {
        return status;
    }
    n4m_matrix_view_t subset_view =
        rowmajor_f64_view(subset_x, X.rows, static_cast<std::int64_t>(columns.size()));
    ::n4m::core::CrossValidationResult cv;
    status = ::n4m::core::cross_validate_regression(ctx, cfg, subset_view, Y, plan, cv);
    if (status != N4M_OK) {
        return status;
    }
    out = cv.metrics.rmse;
    return N4M_OK;
}

}  // namespace

namespace n4m::core {

bool t2_beta_upper_quantile(double alpha, double a, double b, double& out) noexcept {
    double odds = 0.0;
    if (n4m_util_beta_small_tail_root(alpha, a, b, 1.0, &odds) == N4M_BETA_ROOT_TAIL_FAILURE) {
        return false;
    }
    out = odds / (1.0 + odds);
    return true;
}

n4m_status_t select_by_t2(Context& ctx,
                          const Config& cfg,
                          const n4m_matrix_view_t& X,
                          const n4m_matrix_view_t& Y,
                          const ValidationPlan& plan,
                          const std::vector<double>& alpha,
                          std::int32_t min_selected,
                          T2SelectionResult& out) {
    try {
        out = T2SelectionResult{};
        n4m_status_t status = validate_request(ctx, cfg, X, Y, plan, alpha, min_selected);
        if (status != N4M_OK) {
            return status;
        }

        std::unique_ptr<Model> model;
        status = fit_model(ctx, cfg, X, Y, model);
        if (status != N4M_OK) {
            return status;
        }
        const auto p = static_cast<std::size_t>(X.cols);
        const auto a = static_cast<std::size_t>(cfg.n_components);
        if (!model || model->weights_w.size() != p * a) {
            ctx.set_error("T2-PLS fitted model returned inconsistent weights");
            return N4M_ERR_INTERNAL;
        }

        // The weight T2 uses the components with a nonzero weight column: a
        // null PLS component (flagged in N4M_MODEL_UNRESOLVED_COMPONENTS) has
        // zero W and would make the weight covariance singular. With m kept
        // directions T2 and its limit are those of the truncated model.
        std::vector<std::size_t> kept;
        for (std::size_t comp = 0; comp < a; ++comp) {
            for (std::size_t feature = 0; feature < p; ++feature) {
                if (model->weights_w[idx(feature, a, comp)] != 0.0) {
                    kept.push_back(comp);
                    break;
                }
            }
        }
        const std::size_t m = kept.size();

        std::vector<double> mean(m, 0.0);
        for (std::size_t feature = 0; feature < p; ++feature) {
            for (std::size_t k = 0; k < m; ++k) {
                mean[k] += model->weights_w[idx(feature, a, kept[k])];
            }
        }
        for (double& value : mean) {
            value /= static_cast<double>(p);
        }

        std::vector<double> covariance(m * m, 0.0);
        for (std::size_t feature = 0; feature < p; ++feature) {
            for (std::size_t lhs = 0; lhs < m; ++lhs) {
                const double left = model->weights_w[idx(feature, a, kept[lhs])] - mean[lhs];
                for (std::size_t rhs = 0; rhs < m; ++rhs) {
                    const double right = model->weights_w[idx(feature, a, kept[rhs])] - mean[rhs];
                    covariance[idx(lhs, m, rhs)] += left * right;
                }
            }
        }
        for (double& value : covariance) {
            value /= static_cast<double>(p - 1U);
        }

        std::vector<double> inverse;
        if (!invert_square(covariance, m, inverse)) {
            ctx.set_error("T2-PLS loading-weight covariance is singular");
            return N4M_ERR_INVALID_ARGUMENT;
        }

        out.t2_scores.assign(p, 0.0);
        for (std::size_t feature = 0; feature < p; ++feature) {
            double t2 = 0.0;
            for (std::size_t lhs = 0; lhs < m; ++lhs) {
                const double left = model->weights_w[idx(feature, a, kept[lhs])] - mean[lhs];
                for (std::size_t rhs = 0; rhs < m; ++rhs) {
                    const double right = model->weights_w[idx(feature, a, kept[rhs])] - mean[rhs];
                    t2 += left * inverse[idx(lhs, m, rhs)] * right;
                }
            }
            out.t2_scores[feature] = round2(t2);
        }

        const std::size_t n_alpha = alpha.size();
        out.ucl_by_alpha.reserve(n_alpha);
        out.rmse_by_alpha.reserve(n_alpha);
        out.selected_counts.reserve(n_alpha);
        out.selected_mask.assign(n_alpha * p, 0);
        std::vector<std::vector<std::int64_t>> selected_by_alpha;
        selected_by_alpha.reserve(n_alpha);

        // T2 of p weight rows over m directions: (p - 1)^2 / p
        // Beta(m / 2, (p - m - 1) / 2).
        const double beta_a = static_cast<double>(m) / 2.0;
        const double beta_b = static_cast<double>(p - m - 1U) / 2.0;
        const double ucl_factor = static_cast<double>((p - 1U) * (p - 1U)) /
                                  static_cast<double>(p);
        double best_rmse = std::numeric_limits<double>::infinity();
        std::int32_t best_error_index = -1;
        std::int64_t min_count = std::numeric_limits<std::int64_t>::max();
        double min_set_rmse = std::numeric_limits<double>::infinity();
        std::int32_t min_set_index = -1;

        for (std::size_t alpha_idx = 0; alpha_idx < n_alpha; ++alpha_idx) {
            // No latent direction: every weight T2 is 0 and so is its limit
            // (no variable exceeds it; the min_selected fallback below keeps
            // the lowest indices).
            double quantile = 0.0;
            if (m > 0U && !t2_beta_upper_quantile(alpha[alpha_idx], beta_a, beta_b, quantile)) {
                ctx.set_error("T2-PLS beta quantile did not converge");
                out = T2SelectionResult{};
                return N4M_ERR_NUMERICAL_FAILURE;
            }
            const double ucl = round2(ucl_factor * quantile);
            out.ucl_by_alpha.push_back(ucl);

            std::vector<std::int64_t> selected;
            for (std::size_t feature = 0; feature < p; ++feature) {
                if (out.t2_scores[feature] > ucl) {
                    selected.push_back(static_cast<std::int64_t>(feature));
                }
            }
            if (selected.size() < static_cast<std::size_t>(min_selected)) {
                std::vector<std::int64_t> order(p, 0);
                std::iota(order.begin(), order.end(), static_cast<std::int64_t>(0));
                std::stable_sort(order.begin(), order.end(), [&out](std::int64_t lhs, std::int64_t rhs) {
                    const double left = out.t2_scores[static_cast<std::size_t>(lhs)];
                    const double right = out.t2_scores[static_cast<std::size_t>(rhs)];
                    if (left == right) {
                        return lhs < rhs;
                    }
                    return left > right;
                });
                selected.assign(order.begin(), order.begin() + min_selected);
            }
            for (const std::int64_t feature : selected) {
                out.selected_mask[idx(alpha_idx, p, static_cast<std::size_t>(feature))] = 1;
            }

            double rmse = 0.0;
            status = subset_rmse(ctx, cfg, X, Y, plan, selected, rmse);
            if (status != N4M_OK) {
                out = T2SelectionResult{};
                return status;
            }
            out.rmse_by_alpha.push_back(rmse);
            out.selected_counts.push_back(static_cast<std::int64_t>(selected.size()));
            selected_by_alpha.push_back(std::move(selected));

            const auto current_index = static_cast<std::int32_t>(alpha_idx);
            if (rmse < best_rmse) {
                best_rmse = rmse;
                best_error_index = current_index;
            }
            const std::int64_t count = out.selected_counts.back();
            if (count < min_count || (count == min_count && rmse < min_set_rmse)) {
                min_count = count;
                min_set_rmse = rmse;
                min_set_index = current_index;
            }
        }

        if (best_error_index < 0 || min_set_index < 0) {
            ctx.set_error("T2-PLS did not produce a selected subset");
            out = T2SelectionResult{};
            return N4M_ERR_INTERNAL;
        }
        out.n_features = static_cast<std::int32_t>(X.cols);
        out.n_targets = static_cast<std::int32_t>(Y.cols);
        out.n_components = cfg.n_components;
        out.n_latent_directions = static_cast<std::int32_t>(m);
        out.n_alphas = static_cast<std::int32_t>(n_alpha);
        out.min_selected = min_selected;
        out.best_error_index = best_error_index;
        out.min_set_index = min_set_index;
        out.best_rmse = best_rmse;
        out.min_set_rmse = min_set_rmse;
        out.selected_indices_best_error = selected_by_alpha[static_cast<std::size_t>(best_error_index)];
        out.selected_indices_min_set = selected_by_alpha[static_cast<std::size_t>(min_set_index)];
        ctx.clear_error();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while running T2-PLS selection");
        out = T2SelectionResult{};
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while running T2-PLS selection");
        out = T2SelectionResult{};
        return N4M_ERR_INTERNAL;
    }
}

}  // namespace n4m::core
