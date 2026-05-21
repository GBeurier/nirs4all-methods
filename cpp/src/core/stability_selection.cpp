// SPDX-License-Identifier: CECILL-2.1

#include "core/stability_selection.hpp"

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

#include "core/common/matrix_view.hpp"
#include "core/model.hpp"
#include "core/common/status.hpp"

namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();

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
                                            std::int32_t top_k) {
    n4m_status_t status = validate_float_view(ctx, X, "X");
    if (status != N4M_OK) {
        return status;
    }
    status = validate_float_view(ctx, Y, "Y");
    if (status != N4M_OK) {
        return status;
    }
    if (X.rows == 0 || X.cols == 0 || Y.cols == 0) {
        ctx.set_error("stability-selection matrices must be non-empty");
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
    if (plan.folds.size() < 2U) {
        ctx.set_error("coefficient stability requires at least two Monte-Carlo folds");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (top_k <= 0 || top_k > X.cols) {
        ctx.set_errorf("top_k must be in [1, %lld]; got %d",
                       static_cast<long long>(X.cols),
                       static_cast<int>(top_k));
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (cfg.n_components < 1 || static_cast<std::int64_t>(cfg.n_components) > X.cols) {
        ctx.set_errorf("n_components must be in [1, %lld]; got %d",
                       static_cast<long long>(X.cols),
                       static_cast<int>(cfg.n_components));
        return N4M_ERR_INVALID_ARGUMENT;
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t copy_rows(::n4m::core::Context& ctx,
                                     const n4m_matrix_view_t& view,
                                     const std::vector<std::int64_t>& rows,
                                     const char* name,
                                     std::vector<double>& out) {
    if (rows.empty()) {
        ctx.set_errorf("%s row selection must not be empty", name);
        return N4M_ERR_INVALID_ARGUMENT;
    }
    std::size_t n_values = 0;
    if (!checked_matrix_size(static_cast<std::int64_t>(rows.size()),
                             view.cols,
                             n_values)) {
        ctx.set_errorf("%s row selection shape is too large", name);
        return N4M_ERR_INVALID_ARGUMENT;
    }
    out.assign(n_values, 0.0);
    const auto cols = static_cast<std::size_t>(view.cols);
    const auto n_rows = static_cast<std::uint64_t>(view.rows);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const std::int64_t row = rows[i];
        if (row < 0 || static_cast<std::uint64_t>(row) >= n_rows) {
            ctx.set_errorf("%s row index out of range: %lld",
                           name,
                           static_cast<long long>(row));
            return N4M_ERR_INVALID_ARGUMENT;
        }
        const auto src_row = static_cast<std::size_t>(row);
        for (std::size_t col = 0; col < cols; ++col) {
            out[i * cols + col] = read_value(view, src_row, col);
        }
    }
    return N4M_OK;
}

[[nodiscard]] std::vector<std::int64_t> rank_descending(const std::vector<double>& scores) {
    std::vector<std::int64_t> order(scores.size(), 0);
    std::iota(order.begin(), order.end(), static_cast<std::int64_t>(0));
    std::stable_sort(order.begin(), order.end(), [&scores](std::int64_t lhs, std::int64_t rhs) {
        const double left = scores[static_cast<std::size_t>(lhs)];
        const double right = scores[static_cast<std::size_t>(rhs)];
        if (left == right) {
            return lhs < rhs;
        }
        return left > right;
    });
    return order;
}

}  // namespace

namespace n4m::core {

n4m_status_t select_by_coefficient_stability(Context& ctx,
                                             const Config& cfg,
                                             const n4m_matrix_view_t& X,
                                             const n4m_matrix_view_t& Y,
                                             const ValidationPlan& plan,
                                             std::int32_t top_k,
                                             StabilitySelectionResult& out) {
    try {
        out = StabilitySelectionResult{};
        n4m_status_t status = validate_request(ctx, cfg, X, Y, plan, top_k);
        if (status != N4M_OK) {
            return status;
        }

        const auto n_features = static_cast<std::size_t>(X.cols);
        const auto n_targets = static_cast<std::size_t>(Y.cols);
        const auto n_repeats = plan.folds.size();
        const std::size_t coef_count = n_features * n_targets;
        std::vector<double> coefficient_samples(n_repeats * coef_count, 0.0);

        for (std::size_t repeat = 0; repeat < n_repeats; ++repeat) {
            const ValidationFold& fold = plan.folds[repeat];
            std::vector<double> train_x;
            std::vector<double> train_y;
            status = copy_rows(ctx, X, fold.train_indices, "X train", train_x);
            if (status != N4M_OK) {
                out = StabilitySelectionResult{};
                return status;
            }
            status = copy_rows(ctx, Y, fold.train_indices, "Y train", train_y);
            if (status != N4M_OK) {
                out = StabilitySelectionResult{};
                return status;
            }

            n4m_matrix_view_t train_x_view =
                rowmajor_f64_view(train_x,
                                  static_cast<std::int64_t>(fold.train_indices.size()),
                                  X.cols);
            n4m_matrix_view_t train_y_view =
                rowmajor_f64_view(train_y,
                                  static_cast<std::int64_t>(fold.train_indices.size()),
                                  Y.cols);

            std::unique_ptr<Model> model;
            status = fit_model(ctx, cfg, train_x_view, train_y_view, model);
            if (status != N4M_OK) {
                out = StabilitySelectionResult{};
                return status;
            }
            if (!model || model->coefficients.size() != coef_count) {
                ctx.set_error("Monte-Carlo fold fit returned inconsistent coefficients");
                out = StabilitySelectionResult{};
                return N4M_ERR_INTERNAL;
            }

            const std::size_t offset = repeat * coef_count;
            std::copy(model->coefficients.begin(),
                      model->coefficients.end(),
                      coefficient_samples.begin() + static_cast<std::ptrdiff_t>(offset));
        }

        out.n_features = static_cast<std::int32_t>(n_features);
        out.n_targets = static_cast<std::int32_t>(n_targets);
        out.n_repeats = static_cast<std::int32_t>(n_repeats);
        out.top_k = top_k;
        out.mean_coefficients.assign(coef_count, 0.0);
        out.std_coefficients.assign(coef_count, 0.0);
        out.stability_scores.assign(n_features, 0.0);

        const double inv_repeats = 1.0 / static_cast<double>(n_repeats);
        for (std::size_t coef = 0; coef < coef_count; ++coef) {
            double sum = 0.0;
            for (std::size_t repeat = 0; repeat < n_repeats; ++repeat) {
                sum += coefficient_samples[repeat * coef_count + coef];
            }
            out.mean_coefficients[coef] = sum * inv_repeats;
        }

        const double inv_dof = 1.0 / static_cast<double>(n_repeats - 1U);
        for (std::size_t coef = 0; coef < coef_count; ++coef) {
            double ss = 0.0;
            for (std::size_t repeat = 0; repeat < n_repeats; ++repeat) {
                const double delta =
                    coefficient_samples[repeat * coef_count + coef] - out.mean_coefficients[coef];
                ss += delta * delta;
            }
            out.std_coefficients[coef] = std::sqrt(ss * inv_dof);
        }

        for (std::size_t feature = 0; feature < n_features; ++feature) {
            double best = 0.0;
            for (std::size_t target = 0; target < n_targets; ++target) {
                const std::size_t coef = idx(feature, n_targets, target);
                const double denominator = std::max(out.std_coefficients[coef], kEps);
                best = std::max(best, std::fabs(out.mean_coefficients[coef]) / denominator);
            }
            out.stability_scores[feature] = best;
        }

        out.selected_indices = rank_descending(out.stability_scores);
        out.selected_indices.resize(static_cast<std::size_t>(top_k));
        ctx.clear_error();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while running coefficient stability selection");
        out = StabilitySelectionResult{};
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while running coefficient stability selection");
        out = StabilitySelectionResult{};
        return N4M_ERR_INTERNAL;
    }
}

}  // namespace n4m::core
