// SPDX-License-Identifier: CECILL-2.1

#include "core/rep_selection.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <vector>

#include "core/cross_validation.hpp"
#include "core/common/matrix_view.hpp"
#include "core/model.hpp"
#include "core/common/status.hpp"

namespace {

[[nodiscard]] std::size_t idx(std::size_t row, std::size_t cols, std::size_t col) noexcept {
    return row * cols + col;
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
                                            std::int32_t n_steps,
                                            std::int32_t min_features,
                                            std::int32_t remove_count) {
    n4m_status_t status = validate_float_view(ctx, X, "X");
    if (status != N4M_OK) {
        return status;
    }
    status = validate_float_view(ctx, Y, "Y");
    if (status != N4M_OK) {
        return status;
    }
    if (X.rows <= 0 || X.cols <= 0 || Y.cols <= 0) {
        ctx.set_error("REP matrices must be non-empty");
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
        ctx.set_error("REP requires at least one validation fold");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (n_steps <= 0) {
        ctx.set_errorf("n_steps must be >= 1; got %d", static_cast<int>(n_steps));
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (cfg.n_components < 1 || static_cast<std::int64_t>(cfg.n_components) > X.cols) {
        ctx.set_errorf("n_components must be in [1, %lld]; got %d",
                       static_cast<long long>(X.cols),
                       static_cast<int>(cfg.n_components));
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (min_features < cfg.n_components || min_features > X.cols) {
        ctx.set_errorf("min_features must be in [n_components, %lld]; got %d",
                       static_cast<long long>(X.cols),
                       static_cast<int>(min_features));
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (remove_count < 1) {
        ctx.set_errorf("remove_count must be >= 1; got %d", static_cast<int>(remove_count));
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (X.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) ||
        Y.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
        ctx.set_error("REP matrix dimensions exceed int32 result storage");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t copy_columns(::n4m::core::Context& ctx,
                                        const n4m_matrix_view_t& X,
                                        const std::vector<std::int64_t>& columns,
                                        std::vector<double>& out) {
    if (columns.empty()) {
        ctx.set_error("REP column selection must not be empty");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    std::size_t n_values = 0;
    if (!checked_matrix_size(X.rows, static_cast<std::int64_t>(columns.size()), n_values)) {
        ctx.set_error("REP subset matrix shape is too large");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    out.assign(n_values, 0.0);
    const auto rows = static_cast<std::size_t>(X.rows);
    const auto cols = columns.size();
    for (std::size_t local_col = 0; local_col < cols; ++local_col) {
        const std::int64_t original_col = columns[local_col];
        if (original_col < 0 || original_col >= X.cols) {
            ctx.set_error("REP column index out of range");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        const auto src_col = static_cast<std::size_t>(original_col);
        for (std::size_t row = 0; row < rows; ++row) {
            const double value = read_value(X, row, src_col);
            if (!std::isfinite(value)) {
                ctx.set_error("REP X contains NaN or Inf");
                return N4M_ERR_INVALID_ARGUMENT;
            }
            out[idx(row, cols, local_col)] = value;
        }
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t compute_subset_scores(::n4m::core::Context& ctx,
                                                 const ::n4m::core::Config& cfg,
                                                 const n4m_matrix_view_t& subset_x,
                                                 const n4m_matrix_view_t& Y,
                                                 std::vector<double>& out) {
    std::unique_ptr<::n4m::core::Model> model;
    n4m_status_t status = ::n4m::core::fit_model(ctx, cfg, subset_x, Y, model);
    if (status != N4M_OK) {
        return status;
    }
    const auto p = static_cast<std::size_t>(subset_x.cols);
    const auto q = static_cast<std::size_t>(Y.cols);
    if (!model || model->coefficients.size() != p * q) {
        ctx.set_error("REP fitted model returned inconsistent coefficients");
        return N4M_ERR_INTERNAL;
    }

    out.assign(p, 0.0);
    for (std::size_t feature = 0; feature < p; ++feature) {
        double best = 0.0;
        for (std::size_t target = 0; target < q; ++target) {
            best = std::max(best, std::fabs(model->coefficients[idx(feature, q, target)]));
        }
        out[feature] = best;
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

[[nodiscard]] std::vector<std::int64_t> remove_weakest(
    const std::vector<std::int64_t>& active,
    const std::vector<double>& full_scores,
    std::int32_t min_features,
    std::int32_t remove_count,
    std::vector<std::int64_t>& removed) {
    removed.clear();
    if (active.size() <= static_cast<std::size_t>(min_features)) {
        return active;
    }
    std::int32_t actual_remove =
        std::min(remove_count, static_cast<std::int32_t>(active.size()) - min_features);
    if (actual_remove <= 0) {
        return active;
    }
    std::vector<std::int64_t> order = active;
    std::stable_sort(order.begin(), order.end(), [&full_scores](std::int64_t lhs, std::int64_t rhs) {
        const double left = full_scores[static_cast<std::size_t>(lhs)];
        const double right = full_scores[static_cast<std::size_t>(rhs)];
        if (left == right) {
            return lhs > rhs;
        }
        return left < right;
    });
    removed.assign(order.begin(), order.begin() + actual_remove);
    std::sort(removed.begin(), removed.end());
    std::vector<unsigned char> remove(full_scores.size(), 0U);
    for (const std::int64_t feature : removed) {
        remove[static_cast<std::size_t>(feature)] = 1U;
    }
    std::vector<std::int64_t> next;
    next.reserve(active.size() - static_cast<std::size_t>(actual_remove));
    for (const std::int64_t feature : active) {
        if (remove[static_cast<std::size_t>(feature)] == 0U) {
            next.push_back(feature);
        }
    }
    std::sort(next.begin(), next.end());
    return next;
}

}  // namespace

namespace n4m::core {

n4m_status_t select_by_rep(Context& ctx,
                           const Config& cfg,
                           const n4m_matrix_view_t& X,
                           const n4m_matrix_view_t& Y,
                           const ValidationPlan& plan,
                           std::int32_t n_steps,
                           std::int32_t min_features,
                           std::int32_t remove_count,
                           RepSelectionResult& out) {
    try {
        out = RepSelectionResult{};
        n4m_status_t status = validate_request(ctx,
                                               cfg,
                                               X,
                                               Y,
                                               plan,
                                               n_steps,
                                               min_features,
                                               remove_count);
        if (status != N4M_OK) {
            return status;
        }

        const auto p = static_cast<std::size_t>(X.cols);
        std::vector<std::int64_t> active(p, 0);
        for (std::size_t feature = 0; feature < p; ++feature) {
            active[feature] = static_cast<std::int64_t>(feature);
        }

        std::vector<std::vector<std::int64_t>> candidates;
        candidates.reserve(static_cast<std::size_t>(n_steps));
        out.coefficient_scores.assign(static_cast<std::size_t>(n_steps) * p, 0.0);
        out.rmse_by_step.reserve(static_cast<std::size_t>(n_steps));
        out.retained_counts.reserve(static_cast<std::size_t>(n_steps));
        out.removed_counts.reserve(static_cast<std::size_t>(n_steps));

        double best_rmse = std::numeric_limits<double>::infinity();
        std::int32_t best_step = -1;

        for (std::int32_t step = 0; step < n_steps; ++step) {
            std::vector<double> active_x;
            status = copy_columns(ctx, X, active, active_x);
            if (status != N4M_OK) {
                out = RepSelectionResult{};
                return status;
            }
            n4m_matrix_view_t active_x_view =
                rowmajor_f64_view(active_x,
                                  X.rows,
                                  static_cast<std::int64_t>(active.size()));

            std::vector<double> active_scores;
            status = compute_subset_scores(ctx, cfg, active_x_view, Y, active_scores);
            if (status != N4M_OK) {
                out = RepSelectionResult{};
                return status;
            }
            if (active_scores.size() != active.size()) {
                ctx.set_error("REP active score count is inconsistent");
                out = RepSelectionResult{};
                return N4M_ERR_INTERNAL;
            }

            std::vector<double> full_scores(p, 0.0);
            for (std::size_t local = 0; local < active.size(); ++local) {
                const auto feature = static_cast<std::size_t>(active[local]);
                full_scores[feature] = active_scores[local];
                out.coefficient_scores[static_cast<std::size_t>(step) * p + feature] =
                    active_scores[local];
            }

            double rmse = 0.0;
            status = subset_rmse(ctx, cfg, X, Y, plan, active, rmse);
            if (status != N4M_OK) {
                out = RepSelectionResult{};
                return status;
            }
            out.retained_counts.push_back(static_cast<std::int64_t>(active.size()));
            out.rmse_by_step.push_back(rmse);
            candidates.push_back(active);
            if (rmse < best_rmse) {
                best_rmse = rmse;
                best_step = step;
            }

            std::vector<std::int64_t> removed;
            if (step != n_steps - 1) {
                active = remove_weakest(active,
                                        full_scores,
                                        min_features,
                                        remove_count,
                                        removed);
            }
            out.removed_counts.push_back(static_cast<std::int64_t>(removed.size()));
            out.removed_indices.insert(out.removed_indices.end(), removed.begin(), removed.end());
        }

        if (best_step < 0) {
            ctx.set_error("REP did not produce a candidate subset");
            out = RepSelectionResult{};
            return N4M_ERR_INTERNAL;
        }
        out.n_features = static_cast<std::int32_t>(X.cols);
        out.n_targets = static_cast<std::int32_t>(Y.cols);
        out.n_components = cfg.n_components;
        out.n_steps = n_steps;
        out.min_features = min_features;
        out.remove_count = remove_count;
        out.best_step = best_step;
        out.best_rmse = best_rmse;
        out.selected_indices = candidates[static_cast<std::size_t>(best_step)];
        ctx.clear_error();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while running REP selection");
        out = RepSelectionResult{};
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while running REP selection");
        out = RepSelectionResult{};
        return N4M_ERR_INTERNAL;
    }
}

}  // namespace n4m::core
