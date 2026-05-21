// SPDX-License-Identifier: CECILL-2.1

#include "core/st_selection.hpp"

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
                                            const std::vector<double>& thresholds,
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
        ctx.set_error("ST-PLS matrices must be non-empty");
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
        ctx.set_error("ST-PLS requires at least one validation fold");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (thresholds.empty()) {
        ctx.set_error("ST-PLS requires at least one threshold");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    for (double threshold : thresholds) {
        if (!std::isfinite(threshold) || threshold < 0.0 || threshold > 1.0) {
            ctx.set_error("ST-PLS thresholds must be finite and in [0, 1]");
            return N4M_ERR_INVALID_ARGUMENT;
        }
    }
    if (cfg.n_components < 1 || static_cast<std::int64_t>(cfg.n_components) > X.cols) {
        ctx.set_errorf("n_components must be in [1, %lld]; got %d",
                       static_cast<long long>(X.cols),
                       static_cast<int>(cfg.n_components));
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (min_selected < cfg.n_components || min_selected > X.cols) {
        ctx.set_errorf("min_selected must be in [n_components, %lld]; got %d",
                       static_cast<long long>(X.cols),
                       static_cast<int>(min_selected));
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (X.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) ||
        Y.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) ||
        thresholds.size() >
            static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        ctx.set_error("ST-PLS dimensions exceed int32 result storage");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t copy_columns(::n4m::core::Context& ctx,
                                        const n4m_matrix_view_t& X,
                                        const std::vector<std::int64_t>& columns,
                                        std::vector<double>& out) {
    if (columns.empty()) {
        ctx.set_error("ST-PLS column selection must not be empty");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    std::size_t n_values = 0;
    if (!checked_matrix_size(X.rows, static_cast<std::int64_t>(columns.size()), n_values)) {
        ctx.set_error("ST-PLS subset matrix shape is too large");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    out.assign(n_values, 0.0);
    const auto rows = static_cast<std::size_t>(X.rows);
    const auto cols = columns.size();
    for (std::size_t local_col = 0; local_col < cols; ++local_col) {
        const std::int64_t original_col = columns[local_col];
        if (original_col < 0 || original_col >= X.cols) {
            ctx.set_error("ST-PLS column index out of range");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        const auto src_col = static_cast<std::size_t>(original_col);
        for (std::size_t row = 0; row < rows; ++row) {
            const double value = read_value(X, row, src_col);
            if (!std::isfinite(value)) {
                ctx.set_error("ST-PLS X contains NaN or Inf");
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

[[nodiscard]] n4m_status_t coefficient_scores(::n4m::core::Context& ctx,
                                               const ::n4m::core::Config& cfg,
                                               const n4m_matrix_view_t& X,
                                               const n4m_matrix_view_t& Y,
                                               std::vector<double>& out) {
    std::unique_ptr<::n4m::core::Model> model;
    n4m_status_t status = ::n4m::core::fit_model(ctx, cfg, X, Y, model);
    if (status != N4M_OK) {
        return status;
    }
    const auto p = static_cast<std::size_t>(X.cols);
    const auto q = static_cast<std::size_t>(Y.cols);
    if (!model || model->coefficients.size() != p * q) {
        ctx.set_error("ST-PLS fitted model returned inconsistent coefficients");
        return N4M_ERR_INTERNAL;
    }

    out.assign(p, 0.0);
    double max_score = 0.0;
    for (std::size_t feature = 0; feature < p; ++feature) {
        double best = 0.0;
        for (std::size_t target = 0; target < q; ++target) {
            best = std::max(best, std::fabs(model->coefficients[idx(feature, q, target)]));
        }
        if (!std::isfinite(best)) {
            ctx.set_error("ST-PLS coefficient score contains NaN or Inf");
            return N4M_ERR_INTERNAL;
        }
        out[feature] = best;
        max_score = std::max(max_score, best);
    }
    if (!(max_score > std::numeric_limits<double>::epsilon())) {
        ctx.set_error("ST-PLS coefficient scores collapsed");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    for (double& value : out) {
        value /= max_score;
    }
    return N4M_OK;
}

[[nodiscard]] std::vector<std::int64_t> rank_indices(const std::vector<double>& scores) {
    std::vector<std::int64_t> ranking(scores.size(), 0);
    for (std::size_t feature = 0; feature < scores.size(); ++feature) {
        ranking[feature] = static_cast<std::int64_t>(feature);
    }
    std::stable_sort(ranking.begin(), ranking.end(), [&scores](std::int64_t lhs, std::int64_t rhs) {
        const double left = scores[static_cast<std::size_t>(lhs)];
        const double right = scores[static_cast<std::size_t>(rhs)];
        if (left == right) {
            return lhs < rhs;
        }
        return left > right;
    });
    return ranking;
}

[[nodiscard]] std::vector<std::int64_t> threshold_subset(
    const std::vector<double>& scores,
    const std::vector<std::int64_t>& ranking,
    double threshold,
    std::int32_t min_selected) {
    std::vector<std::int64_t> selected;
    selected.reserve(scores.size());
    for (std::size_t feature = 0; feature < scores.size(); ++feature) {
        if (scores[feature] >= threshold) {
            selected.push_back(static_cast<std::int64_t>(feature));
        }
    }
    if (selected.size() < static_cast<std::size_t>(min_selected)) {
        selected.clear();
        for (std::int32_t i = 0; i < min_selected; ++i) {
            selected.push_back(ranking[static_cast<std::size_t>(i)]);
        }
    }
    std::sort(selected.begin(), selected.end());
    return selected;
}

}  // namespace

namespace n4m::core {

n4m_status_t select_by_st(Context& ctx,
                          const Config& cfg,
                          const n4m_matrix_view_t& X,
                          const n4m_matrix_view_t& Y,
                          const ValidationPlan& plan,
                          const std::vector<double>& thresholds,
                          std::int32_t min_selected,
                          StSelectionResult& out) {
    try {
        out = StSelectionResult{};
        n4m_status_t status =
            validate_request(ctx, cfg, X, Y, plan, thresholds, min_selected);
        if (status != N4M_OK) {
            return status;
        }

        const auto p = static_cast<std::size_t>(X.cols);
        const auto n_thresholds = thresholds.size();
        std::vector<double> scores;
        status = coefficient_scores(ctx, cfg, X, Y, scores);
        if (status != N4M_OK) {
            out = StSelectionResult{};
            return status;
        }
        std::vector<std::int64_t> ranking = rank_indices(scores);

        out.coefficient_scores = scores;
        out.thresholds = thresholds;
        out.rmse_by_threshold.reserve(n_thresholds);
        out.selected_counts.reserve(n_thresholds);
        out.selected_masks.assign(n_thresholds * p, 0);

        std::vector<std::vector<std::int64_t>> candidates;
        candidates.reserve(n_thresholds);
        double best_rmse = std::numeric_limits<double>::infinity();
        std::int32_t best_index = -1;

        for (std::size_t threshold_index = 0; threshold_index < n_thresholds; ++threshold_index) {
            std::vector<std::int64_t> selected =
                threshold_subset(scores, ranking, thresholds[threshold_index], min_selected);
            for (std::int64_t feature : selected) {
                out.selected_masks[threshold_index * p + static_cast<std::size_t>(feature)] = 1;
            }

            double rmse = 0.0;
            status = subset_rmse(ctx, cfg, X, Y, plan, selected, rmse);
            if (status != N4M_OK) {
                out = StSelectionResult{};
                return status;
            }
            out.rmse_by_threshold.push_back(rmse);
            out.selected_counts.push_back(static_cast<std::int64_t>(selected.size()));
            candidates.push_back(selected);
            if (rmse < best_rmse) {
                best_rmse = rmse;
                best_index = static_cast<std::int32_t>(threshold_index);
            }
        }

        if (best_index < 0) {
            ctx.set_error("ST-PLS did not produce a candidate subset");
            out = StSelectionResult{};
            return N4M_ERR_INTERNAL;
        }
        out.n_features = static_cast<std::int32_t>(X.cols);
        out.n_targets = static_cast<std::int32_t>(Y.cols);
        out.n_components = cfg.n_components;
        out.n_thresholds = static_cast<std::int32_t>(n_thresholds);
        out.min_selected = min_selected;
        out.best_threshold_index = best_index;
        out.best_threshold = thresholds[static_cast<std::size_t>(best_index)];
        out.best_rmse = best_rmse;
        out.ranking_indices = ranking;
        out.selected_indices = candidates[static_cast<std::size_t>(best_index)];
        ctx.clear_error();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while running ST-PLS selection");
        out = StSelectionResult{};
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while running ST-PLS selection");
        out = StSelectionResult{};
        return N4M_ERR_INTERNAL;
    }
}

}  // namespace n4m::core
