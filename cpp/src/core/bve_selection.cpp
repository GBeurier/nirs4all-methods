// SPDX-License-Identifier: CeCILL-2.1

#include "core/bve_selection.hpp"

#include <algorithm>
#include <cmath>
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

[[nodiscard]] p4a_status_t validate_request(::pls4all::core::Context& ctx,
                                            const ::pls4all::core::Config& cfg,
                                            const p4a_matrix_view_t& X,
                                            const p4a_matrix_view_t& Y,
                                            const ::pls4all::core::ValidationPlan& plan,
                                            std::int32_t n_steps,
                                            std::int32_t min_features) {
    p4a_status_t status = validate_float_view(ctx, X, "X");
    if (status != P4A_OK) {
        return status;
    }
    status = validate_float_view(ctx, Y, "Y");
    if (status != P4A_OK) {
        return status;
    }
    if (X.rows == 0 || X.cols == 0 || Y.cols == 0) {
        ctx.set_error("BVE matrices must be non-empty");
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
        ctx.set_error("BVE requires at least one validation fold");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (n_steps <= 0) {
        ctx.set_errorf("n_steps must be >= 1; got %d", static_cast<int>(n_steps));
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (cfg.n_components < 1 || static_cast<std::int64_t>(cfg.n_components) > X.cols) {
        ctx.set_errorf("n_components must be in [1, %lld]; got %d",
                       static_cast<long long>(X.cols),
                       static_cast<int>(cfg.n_components));
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (min_features < cfg.n_components || min_features >= X.cols) {
        ctx.set_errorf("min_features must be in [n_components, %lld); got %d",
                       static_cast<long long>(X.cols),
                       static_cast<int>(min_features));
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (X.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) ||
        Y.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
        ctx.set_error("BVE matrix dimensions exceed int32 result storage");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (static_cast<std::int64_t>(n_steps) > X.cols - min_features) {
        ctx.set_error("n_steps would remove more variables than min_features allows");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    return P4A_OK;
}

[[nodiscard]] p4a_status_t copy_columns(::pls4all::core::Context& ctx,
                                        const p4a_matrix_view_t& X,
                                        const std::vector<std::int64_t>& columns,
                                        std::vector<double>& out) {
    if (columns.empty()) {
        ctx.set_error("BVE column selection must not be empty");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    std::size_t n_values = 0;
    if (!checked_matrix_size(X.rows, static_cast<std::int64_t>(columns.size()), n_values)) {
        ctx.set_error("BVE subset matrix shape is too large");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    out.assign(n_values, 0.0);
    const auto rows = static_cast<std::size_t>(X.rows);
    const auto cols = columns.size();
    for (std::size_t local_col = 0; local_col < cols; ++local_col) {
        const std::int64_t original_col = columns[local_col];
        if (original_col < 0 || original_col >= X.cols) {
            ctx.set_error("BVE column index out of range");
            return P4A_ERR_INVALID_ARGUMENT;
        }
        const auto src_col = static_cast<std::size_t>(original_col);
        for (std::size_t row = 0; row < rows; ++row) {
            const double value = read_value(X, row, src_col);
            if (!std::isfinite(value)) {
                ctx.set_error("BVE X contains NaN or Inf");
                return P4A_ERR_INVALID_ARGUMENT;
            }
            out[idx(row, cols, local_col)] = value;
        }
    }
    return P4A_OK;
}

[[nodiscard]] p4a_status_t subset_rmse(::pls4all::core::Context& ctx,
                                       const ::pls4all::core::Config& cfg,
                                       const p4a_matrix_view_t& X,
                                       const p4a_matrix_view_t& Y,
                                       const ::pls4all::core::ValidationPlan& plan,
                                       const std::vector<std::int64_t>& columns,
                                       double& out) {
    std::vector<double> subset_x;
    p4a_status_t status = copy_columns(ctx, X, columns, subset_x);
    if (status != P4A_OK) {
        return status;
    }
    p4a_matrix_view_t subset_view =
        rowmajor_f64_view(subset_x, X.rows, static_cast<std::int64_t>(columns.size()));
    ::pls4all::core::CrossValidationResult cv;
    status = ::pls4all::core::cross_validate_regression(ctx, cfg, subset_view, Y, plan, cv);
    if (status != P4A_OK) {
        return status;
    }
    out = cv.metrics.rmse;
    return P4A_OK;
}

[[nodiscard]] std::vector<std::int64_t> without_feature(const std::vector<std::int64_t>& active,
                                                        std::int64_t feature) {
    std::vector<std::int64_t> out;
    out.reserve(active.size() - 1U);
    for (const std::int64_t candidate : active) {
        if (candidate != feature) {
            out.push_back(candidate);
        }
    }
    return out;
}

}  // namespace

namespace pls4all::core {

p4a_status_t select_by_bve(Context& ctx,
                           const Config& cfg,
                           const p4a_matrix_view_t& X,
                           const p4a_matrix_view_t& Y,
                           const ValidationPlan& plan,
                           std::int32_t n_steps,
                           std::int32_t min_features,
                           BveSelectionResult& out) {
    try {
        out = BveSelectionResult{};
        p4a_status_t status = validate_request(ctx,
                                               cfg,
                                               X,
                                               Y,
                                               plan,
                                               n_steps,
                                               min_features);
        if (status != P4A_OK) {
            return status;
        }

        const auto p = static_cast<std::size_t>(X.cols);
        std::vector<std::int64_t> active(p, 0);
        for (std::size_t feature = 0; feature < p; ++feature) {
            active[feature] = static_cast<std::int64_t>(feature);
        }

        std::vector<std::vector<std::int64_t>> candidates;
        candidates.reserve(static_cast<std::size_t>(n_steps));
        out.candidate_rmse.assign(static_cast<std::size_t>(n_steps) * p, 0.0);
        out.rmse_by_step.reserve(static_cast<std::size_t>(n_steps));
        out.retained_counts.reserve(static_cast<std::size_t>(n_steps));
        out.removed_indices.reserve(static_cast<std::size_t>(n_steps));

        double best_rmse = std::numeric_limits<double>::infinity();
        std::int32_t best_step = -1;

        for (std::int32_t step = 0; step < n_steps; ++step) {
            double best_candidate_rmse = std::numeric_limits<double>::infinity();
            std::int64_t best_removed = -1;
            std::vector<std::int64_t> best_candidate;

            for (const std::int64_t feature : active) {
                std::vector<std::int64_t> candidate = without_feature(active, feature);
                if (candidate.size() < static_cast<std::size_t>(min_features)) {
                    continue;
                }
                double rmse = 0.0;
                status = subset_rmse(ctx, cfg, X, Y, plan, candidate, rmse);
                if (status != P4A_OK) {
                    out = BveSelectionResult{};
                    return status;
                }
                out.candidate_rmse[static_cast<std::size_t>(step) * p +
                                   static_cast<std::size_t>(feature)] = rmse;
                if (rmse < best_candidate_rmse ||
                    (rmse == best_candidate_rmse &&
                     (best_removed < 0 || feature < best_removed))) {
                    best_candidate_rmse = rmse;
                    best_removed = feature;
                    best_candidate = std::move(candidate);
                }
            }
            if (best_removed < 0) {
                ctx.set_error("BVE did not produce a removable candidate");
                out = BveSelectionResult{};
                return P4A_ERR_INTERNAL;
            }

            out.rmse_by_step.push_back(best_candidate_rmse);
            out.retained_counts.push_back(static_cast<std::int64_t>(best_candidate.size()));
            out.removed_indices.push_back(best_removed);
            candidates.push_back(best_candidate);
            if (best_candidate_rmse < best_rmse) {
                best_rmse = best_candidate_rmse;
                best_step = step;
            }
            active = std::move(best_candidate);
        }

        if (best_step < 0) {
            ctx.set_error("BVE did not produce a candidate subset");
            out = BveSelectionResult{};
            return P4A_ERR_INTERNAL;
        }
        out.n_features = static_cast<std::int32_t>(X.cols);
        out.n_targets = static_cast<std::int32_t>(Y.cols);
        out.n_components = cfg.n_components;
        out.n_steps = n_steps;
        out.min_features = min_features;
        out.best_step = best_step;
        out.best_rmse = best_rmse;
        out.selected_indices = candidates[static_cast<std::size_t>(best_step)];
        ctx.clear_error();
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while running BVE selection");
        out = BveSelectionResult{};
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while running BVE selection");
        out = BveSelectionResult{};
        return P4A_ERR_INTERNAL;
    }
}

}  // namespace pls4all::core
