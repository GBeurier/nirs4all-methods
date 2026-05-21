// SPDX-License-Identifier: CECILL-2.1

#include "core/aom_selection.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <vector>

#include "core/aom_operators.hpp"
#include "core/component_coefficients.hpp"
#include "core/common/matrix_view.hpp"
#include "core/model.hpp"
#include "core/operator_entry.hpp"
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
    if (view.rows <= 0 || view.cols <= 0) {
        ctx.set_errorf("%s matrix must be non-empty", name);
        return N4M_ERR_INVALID_ARGUMENT;
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t copy_rows(::n4m::core::Context& ctx,
                                     const n4m_matrix_view_t& view,
                                     const std::vector<std::int64_t>& rows,
                                     const char* name,
                                     std::vector<double>& out) {
    std::size_t n_values = 0;
    if (!checked_matrix_size(static_cast<std::int64_t>(rows.size()),
                             view.cols,
                             n_values)) {
        ctx.set_errorf("%s row selection shape is too large", name);
        return N4M_ERR_INVALID_ARGUMENT;
    }
    out.assign(n_values, 0.0);
    const auto n_rows = static_cast<std::uint64_t>(view.rows);
    const auto cols = static_cast<std::size_t>(view.cols);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const std::int64_t source = rows[i];
        if (source < 0 || static_cast<std::uint64_t>(source) >= n_rows) {
            ctx.set_errorf("%s row index out of range: %lld",
                           name,
                           static_cast<long long>(source));
            return N4M_ERR_INVALID_ARGUMENT;
        }
        const auto src_row = static_cast<std::size_t>(source);
        for (std::size_t col = 0; col < cols; ++col) {
            const double value = read_value(view, src_row, col);
            if (!std::isfinite(value)) {
                ctx.set_errorf("%s contains NaN or Inf", name);
                return N4M_ERR_INVALID_ARGUMENT;
            }
            out[idx(i, cols, col)] = value;
        }
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t validate_plan(::n4m::core::Context& ctx,
                                         const ::n4m::core::ValidationPlan& plan,
                                         std::int64_t n_samples) {
    if (plan.folds.empty()) {
        ctx.set_error("AOM selection requires at least one validation fold");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (plan.n_samples != 0 && plan.n_samples != n_samples) {
        ctx.set_errorf("validation plan sample count (%lld) must match X rows (%lld)",
                       static_cast<long long>(plan.n_samples),
                       static_cast<long long>(n_samples));
        return N4M_ERR_SHAPE_MISMATCH;
    }
    if (n_samples <= 0) {
        ctx.set_error("AOM selection requires a positive sample count");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    std::vector<std::uint8_t> mask;
    try {
        mask.assign(static_cast<std::size_t>(n_samples), 0U);
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory validating AOM selection plan");
        return N4M_ERR_OUT_OF_MEMORY;
    }
    for (std::size_t fold_idx = 0; fold_idx < plan.folds.size(); ++fold_idx) {
        const auto& fold = plan.folds[fold_idx];
        if (fold.train_indices.empty() || fold.test_indices.empty()) {
            ctx.set_errorf("AOM selection fold %lu has empty train or test split",
                           static_cast<unsigned long>(fold_idx));
            return N4M_ERR_INVALID_ARGUMENT;
        }
        std::fill(mask.begin(), mask.end(), static_cast<std::uint8_t>(0));
        for (const auto sample : fold.train_indices) {
            if (sample < 0 || sample >= n_samples) {
                ctx.set_errorf("AOM selection fold %lu train index %lld out of [0, %lld)",
                               static_cast<unsigned long>(fold_idx),
                               static_cast<long long>(sample),
                               static_cast<long long>(n_samples));
                return N4M_ERR_INVALID_ARGUMENT;
            }
            auto& slot = mask[static_cast<std::size_t>(sample)];
            if (slot != 0U) {
                ctx.set_errorf("AOM selection fold %lu train index %lld is duplicated",
                               static_cast<unsigned long>(fold_idx),
                               static_cast<long long>(sample));
                return N4M_ERR_INVALID_ARGUMENT;
            }
            slot = 1U;
        }
        for (const auto sample : fold.test_indices) {
            if (sample < 0 || sample >= n_samples) {
                ctx.set_errorf("AOM selection fold %lu test index %lld out of [0, %lld)",
                               static_cast<unsigned long>(fold_idx),
                               static_cast<long long>(sample),
                               static_cast<long long>(n_samples));
                return N4M_ERR_INVALID_ARGUMENT;
            }
            auto& slot = mask[static_cast<std::size_t>(sample)];
            if (slot == 1U) {
                ctx.set_errorf(
                    "AOM selection fold %lu has test index %lld overlapping train set",
                    static_cast<unsigned long>(fold_idx),
                    static_cast<long long>(sample));
                return N4M_ERR_INVALID_ARGUMENT;
            }
            if (slot == 2U) {
                ctx.set_errorf("AOM selection fold %lu test index %lld is duplicated",
                               static_cast<unsigned long>(fold_idx),
                               static_cast<long long>(sample));
                return N4M_ERR_INVALID_ARGUMENT;
            }
            slot = 2U;
        }
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t validate_request(::n4m::core::Context& ctx,
                                            const ::n4m::core::Config& cfg,
                                            const ::n4m::core::OperatorBank& bank,
                                            const n4m_matrix_view_t& X,
                                            const n4m_matrix_view_t& Y,
                                            const ::n4m::core::ValidationPlan& plan,
                                            std::int32_t max_components) {
    n4m_status_t status = validate_float_view(ctx, X, "X");
    if (status != N4M_OK) {
        return status;
    }
    status = validate_float_view(ctx, Y, "Y");
    if (status != N4M_OK) {
        return status;
    }
    if (X.rows != Y.rows) {
        ctx.set_error("X and Y must have the same number of rows");
        return N4M_ERR_SHAPE_MISMATCH;
    }
    if (bank.entries().empty()) {
        ctx.set_error("AOM selection requires at least one operator");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (bank.entries().size() >
        static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        ctx.set_error("AOM operator count is too large");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (max_components < 1) {
        ctx.set_error("max_components must be >= 1");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (max_components > X.cols || max_components >= X.rows) {
        ctx.set_error("max_components must be <= X cols and < X rows");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (cfg.algorithm != N4M_ALGO_PLS_REGRESSION ||
        cfg.solver != N4M_SOLVER_SIMPLS ||
        cfg.deflation != N4M_DEFLATION_REGRESSION) {
        ctx.set_error("AOM selection currently supports SIMPLS regression only");
        return N4M_ERR_UNSUPPORTED;
    }
    return validate_plan(ctx, plan, X.rows);
}

[[nodiscard]] n4m_status_t transform_with_operator(::n4m::core::Context& ctx,
                                                   const ::n4m::core::OperatorEntry& entry,
                                                   const n4m_matrix_view_t& apply_x,
                                                   std::vector<double>& transformed) {
    return ::n4m::core::transform_aom_strict_operator(ctx, entry, apply_x, transformed);
}

[[nodiscard]] double prefix_rmse(const ::n4m::core::Model& model,
                                 const std::vector<double>& coefficients_by_component,
                                 std::int32_t prefix,
                                 const std::vector<double>& X,
                                 const std::vector<double>& Y,
                                 std::size_t rows) {
    const auto p = static_cast<std::size_t>(model.n_features);
    const auto q = static_cast<std::size_t>(model.n_targets);
    const std::size_t block = static_cast<std::size_t>(prefix - 1) * p * q;
    double sumsq = 0.0;
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t target = 0; target < q; ++target) {
            double prediction = model.y_mean[target];
            for (std::size_t feature = 0; feature < p; ++feature) {
                const double centered = X[idx(row, p, feature)] - model.x_mean[feature];
                prediction += centered *
                    coefficients_by_component[block + idx(feature, q, target)];
            }
            const double residual = prediction - Y[idx(row, q, target)];
            sumsq += residual * residual;
        }
    }
    return std::sqrt(sumsq / static_cast<double>(rows * q));
}

[[nodiscard]] n4m_status_t score_operator(::n4m::core::Context& ctx,
                                          const ::n4m::core::Config& cfg,
                                          const ::n4m::core::OperatorEntry& entry,
                                          const n4m_matrix_view_t& X,
                                          const n4m_matrix_view_t& Y,
                                          const ::n4m::core::ValidationPlan& plan,
                                          std::int32_t max_components,
                                          std::vector<double>& curve) {
    const double inf = std::numeric_limits<double>::infinity();
    curve.assign(static_cast<std::size_t>(max_components), 0.0);
    std::vector<std::int32_t> counts(static_cast<std::size_t>(max_components), 0);
    std::vector<unsigned char> invalid(static_cast<std::size_t>(max_components), 0U);

    for (const auto& fold : plan.folds) {
        if (fold.train_indices.empty() || fold.test_indices.empty()) {
            ctx.set_error("AOM global selection received an empty validation fold");
            return N4M_ERR_INVALID_ARGUMENT;
        }

        std::vector<double> train_x;
        std::vector<double> train_y;
        std::vector<double> test_x;
        std::vector<double> test_y;
        n4m_status_t status = copy_rows(ctx, X, fold.train_indices, "X train", train_x);
        if (status != N4M_OK) {
            return status;
        }
        status = copy_rows(ctx, Y, fold.train_indices, "Y train", train_y);
        if (status != N4M_OK) {
            return status;
        }
        status = copy_rows(ctx, X, fold.test_indices, "X test", test_x);
        if (status != N4M_OK) {
            return status;
        }
        status = copy_rows(ctx, Y, fold.test_indices, "Y test", test_y);
        if (status != N4M_OK) {
            return status;
        }

        n4m_matrix_view_t train_x_view = rowmajor_f64_view(train_x,
                                                           static_cast<std::int64_t>(fold.train_indices.size()),
                                                           X.cols);
        n4m_matrix_view_t train_y_view = rowmajor_f64_view(train_y,
                                                           static_cast<std::int64_t>(fold.train_indices.size()),
                                                           Y.cols);
        n4m_matrix_view_t test_x_view = rowmajor_f64_view(test_x,
                                                          static_cast<std::int64_t>(fold.test_indices.size()),
                                                          X.cols);

        std::vector<double> train_xt;
        std::vector<double> test_xt;
        status = transform_with_operator(ctx, entry, train_x_view, train_xt);
        if (status != N4M_OK) {
            return status;
        }
        status = transform_with_operator(ctx, entry, test_x_view, test_xt);
        if (status != N4M_OK) {
            return status;
        }

        n4m_matrix_view_t train_xt_view = rowmajor_f64_view(train_xt,
                                                            static_cast<std::int64_t>(fold.train_indices.size()),
                                                            X.cols);
        ::n4m::core::Config local_cfg = cfg;
        local_cfg.n_components = max_components;
        local_cfg.scale_x = 0;
        local_cfg.scale_y = 0;
        std::unique_ptr<::n4m::core::Model> model;
        status = ::n4m::core::fit_model(ctx, local_cfg, train_xt_view, train_y_view, model);
        if (status != N4M_OK || !model) {
            for (std::int32_t k = 0; k < max_components; ++k) {
                invalid[static_cast<std::size_t>(k)] = 1U;
            }
            continue;
        }

        std::vector<double> coefficients_by_component;
        status = ::n4m::core::compute_regression_coefficients_by_component(ctx,
                                                                               *model,
                                                                               coefficients_by_component);
        if (status != N4M_OK) {
            for (std::int32_t k = 0; k < max_components; ++k) {
                invalid[static_cast<std::size_t>(k)] = 1U;
            }
            continue;
        }

        const std::int32_t actual_components = std::min(model->n_components, max_components);
        for (std::int32_t k = 1; k <= actual_components; ++k) {
            const double rmse = prefix_rmse(*model,
                                            coefficients_by_component,
                                            k,
                                            test_xt,
                                            test_y,
                                            fold.test_indices.size());
            if (!std::isfinite(rmse)) {
                invalid[static_cast<std::size_t>(k - 1)] = 1U;
                continue;
            }
            curve[static_cast<std::size_t>(k - 1)] += rmse;
            counts[static_cast<std::size_t>(k - 1)] += 1;
        }
        for (std::int32_t k = actual_components + 1; k <= max_components; ++k) {
            invalid[static_cast<std::size_t>(k - 1)] = 1U;
        }
    }

    for (std::int32_t k = 0; k < max_components; ++k) {
        const auto pos = static_cast<std::size_t>(k);
        if (invalid[pos] != 0U ||
            counts[pos] != static_cast<std::int32_t>(plan.folds.size())) {
            curve[pos] = inf;
        } else {
            curve[pos] /= static_cast<double>(counts[pos]);
        }
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t fit_selected_predictions(::n4m::core::Context& ctx,
                                                    const ::n4m::core::Config& cfg,
                                                    const ::n4m::core::OperatorEntry& entry,
                                                    const n4m_matrix_view_t& X,
                                                    const n4m_matrix_view_t& Y,
                                                    std::int32_t n_components,
                                                    std::vector<double>& predictions) {
    std::vector<double> xt;
    n4m_status_t status = transform_with_operator(ctx, entry, X, xt);
    if (status != N4M_OK) {
        return status;
    }
    n4m_matrix_view_t xt_view = rowmajor_f64_view(xt, X.rows, X.cols);
    ::n4m::core::Config local_cfg = cfg;
    local_cfg.n_components = n_components;
    local_cfg.scale_x = 0;
    local_cfg.scale_y = 0;
    std::unique_ptr<::n4m::core::Model> model;
    status = ::n4m::core::fit_model(ctx, local_cfg, xt_view, Y, model);
    if (status != N4M_OK || !model) {
        return status == N4M_OK ? N4M_ERR_INTERNAL : status;
    }
    std::size_t pred_values = 0;
    if (!checked_matrix_size(X.rows, Y.cols, pred_values)) {
        ctx.set_error("AOM prediction matrix shape is too large");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    predictions.assign(pred_values, 0.0);
    n4m_matrix_view_t pred_view = rowmajor_f64_view(predictions, X.rows, Y.cols);
    return ::n4m::core::predict_into(ctx, *model, xt_view, pred_view);
}

struct StrictOperatorMatrix {
    std::vector<double> adjoint;  // row-major A^T
};

struct PopFit {
    std::int32_t n_components{0};
    std::vector<double> weights_z;   // p x n_components
    std::vector<double> loadings_p;  // p x n_components
    std::vector<double> loadings_q;  // q x n_components
};

[[nodiscard]] n4m_status_t copy_all_rows(::n4m::core::Context& ctx,
                                         const n4m_matrix_view_t& view,
                                         const char* name,
                                         std::vector<double>& out) {
    std::vector<std::int64_t> rows(static_cast<std::size_t>(view.rows), 0);
    for (std::int64_t row = 0; row < view.rows; ++row) {
        rows[static_cast<std::size_t>(row)] = row;
    }
    return copy_rows(ctx, view, rows, name, out);
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
    for (double& value : means) {
        value /= denom;
    }
}

void center_by_means(const std::vector<double>& values,
                     std::size_t rows,
                     std::size_t cols,
                     const std::vector<double>& means,
                     std::vector<double>& centered) {
    centered.assign(rows * cols, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            centered[idx(row, cols, col)] = values[idx(row, cols, col)] - means[col];
        }
    }
}

[[nodiscard]] double squared_norm(const std::vector<double>& values) noexcept {
    double out = 0.0;
    for (double value : values) {
        out += value * value;
    }
    return out;
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

[[nodiscard]] n4m_status_t build_operator_matrices(
    ::n4m::core::Context& ctx,
    const ::n4m::core::OperatorBank& bank,
    std::int64_t cols,
    std::vector<StrictOperatorMatrix>& matrices) {
    const auto p = static_cast<std::size_t>(cols);
    std::vector<double> identity(p * p, 0.0);
    for (std::size_t i = 0; i < p; ++i) {
        identity[idx(i, p, i)] = 1.0;
    }
    n4m_matrix_view_t identity_view = rowmajor_f64_view(identity, cols, cols);
    matrices.clear();
    matrices.reserve(bank.entries().size());
    for (const auto& entry : bank.entries()) {
        StrictOperatorMatrix matrix;
        n4m_status_t status =
            ::n4m::core::transform_aom_strict_operator(ctx, entry, identity_view, matrix.adjoint);
        if (status != N4M_OK) {
            matrices.clear();
            return status;
        }
        matrices.push_back(std::move(matrix));
    }
    return N4M_OK;
}

void cross_covariance(const std::vector<double>& X,
                      const std::vector<double>& Y,
                      std::size_t rows,
                      std::size_t p,
                      std::size_t q,
                      std::vector<double>& covariance) {
    covariance.assign(p * q, 0.0);
    for (std::size_t feature = 0; feature < p; ++feature) {
        for (std::size_t target = 0; target < q; ++target) {
            double sum = 0.0;
            for (std::size_t row = 0; row < rows; ++row) {
                sum += X[idx(row, p, feature)] * Y[idx(row, q, target)];
            }
            covariance[idx(feature, q, target)] = sum;
        }
    }
}

void apply_covariance(const StrictOperatorMatrix& matrix,
                      const std::vector<double>& covariance,
                      std::size_t p,
                      std::size_t q,
                      std::vector<double>& out) {
    out.assign(p * q, 0.0);
    for (std::size_t row = 0; row < p; ++row) {
        for (std::size_t target = 0; target < q; ++target) {
            double sum = 0.0;
            for (std::size_t inner = 0; inner < p; ++inner) {
                sum += matrix.adjoint[idx(inner, p, row)] * covariance[idx(inner, q, target)];
            }
            out[idx(row, q, target)] = sum;
        }
    }
}

void apply_adjoint(const StrictOperatorMatrix& matrix,
                   const std::vector<double>& direction,
                   std::size_t p,
                   std::vector<double>& out) {
    out.assign(p, 0.0);
    for (std::size_t row = 0; row < p; ++row) {
        double sum = 0.0;
        for (std::size_t col = 0; col < p; ++col) {
            sum += matrix.adjoint[idx(row, p, col)] * direction[col];
        }
        out[row] = sum;
    }
}

void dominant_left_direction(const std::vector<double>& covariance,
                             std::size_t p,
                             std::size_t q,
                             std::vector<double>& direction) {
    direction.assign(p, 0.0);
    if (q == 1U) {
        for (std::size_t feature = 0; feature < p; ++feature) {
            direction[feature] = covariance[idx(feature, q, 0)];
        }
        return;
    }

    std::vector<double> gram(p * p, 0.0);
    for (std::size_t row = 0; row < p; ++row) {
        for (std::size_t col = 0; col < p; ++col) {
            double sum = 0.0;
            for (std::size_t target = 0; target < q; ++target) {
                sum += covariance[idx(row, q, target)] * covariance[idx(col, q, target)];
            }
            gram[idx(row, p, col)] = sum;
        }
    }
    direction[0] = 1.0;
    std::vector<double> next(p, 0.0);
    for (int iter = 0; iter < 200; ++iter) {
        std::fill(next.begin(), next.end(), 0.0);
        for (std::size_t row = 0; row < p; ++row) {
            for (std::size_t col = 0; col < p; ++col) {
                next[row] += gram[idx(row, p, col)] * direction[col];
            }
        }
        const double norm = std::sqrt(squared_norm(next));
        if (norm <= std::numeric_limits<double>::epsilon()) {
            return;
        }
        for (double& value : next) {
            value /= norm;
        }
        double delta = 0.0;
        for (std::size_t i = 0; i < p; ++i) {
            const double diff = next[i] - direction[i];
            delta += diff * diff;
        }
        direction = next;
        if (delta < 1e-24) {
            break;
        }
    }
}

[[nodiscard]] n4m_status_t fit_pop_sequence(::n4m::core::Context& ctx,
                                            const std::vector<StrictOperatorMatrix>& matrices,
                                            const std::vector<std::int32_t>& op_indices,
                                            const std::vector<double>& X,
                                            const std::vector<double>& Y,
                                            std::size_t rows,
                                            std::size_t p,
                                            std::size_t q,
                                            PopFit& fit) {
    const auto components = static_cast<std::size_t>(op_indices.size());
    fit = PopFit{};
    fit.n_components = static_cast<std::int32_t>(components);
    fit.weights_z.assign(p * components, 0.0);
    fit.loadings_p.assign(p * components, 0.0);
    fit.loadings_q.assign(q * components, 0.0);
    std::vector<double> basis_v(p * components, 0.0);
    std::vector<double> covariance;
    cross_covariance(X, Y, rows, p, q, covariance);

    constexpr double kEps = 1e-14;
    for (std::size_t comp = 0; comp < components; ++comp) {
        const auto op_index = static_cast<std::size_t>(op_indices[comp]);
        if (op_index >= matrices.size()) {
            ctx.set_error("POP operator index is out of range");
            return N4M_ERR_INVALID_ARGUMENT;
        }

        std::vector<double> transformed_covariance;
        apply_covariance(matrices[op_index], covariance, p, q, transformed_covariance);
        std::vector<double> direction;
        dominant_left_direction(transformed_covariance, p, q, direction);
        const double direction_norm = std::sqrt(squared_norm(direction));
        if (direction_norm < kEps) {
            continue;
        }
        for (double& value : direction) {
            value /= direction_norm;
        }

        std::vector<double> z;
        apply_adjoint(matrices[op_index], direction, p, z);
        std::vector<double> score(rows, 0.0);
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t feature = 0; feature < p; ++feature) {
                score[row] += X[idx(row, p, feature)] * z[feature];
            }
        }
        const double score_norm = std::sqrt(squared_norm(score));
        if (score_norm < kEps) {
            continue;
        }
        for (double& value : score) {
            value /= score_norm;
        }
        for (double& value : z) {
            value /= score_norm;
        }

        std::vector<double> p_load(p, 0.0);
        for (std::size_t feature = 0; feature < p; ++feature) {
            for (std::size_t row = 0; row < rows; ++row) {
                p_load[feature] += X[idx(row, p, feature)] * score[row];
            }
        }
        std::vector<double> q_load(q, 0.0);
        for (std::size_t target = 0; target < q; ++target) {
            for (std::size_t row = 0; row < rows; ++row) {
                q_load[target] += Y[idx(row, q, target)] * score[row];
            }
        }

        std::vector<double> v = p_load;
        if (comp > 0U) {
            for (std::size_t prev = 0; prev < comp; ++prev) {
                double projection = 0.0;
                for (std::size_t feature = 0; feature < p; ++feature) {
                    projection += basis_v[idx(feature, components, prev)] * v[feature];
                }
                for (std::size_t feature = 0; feature < p; ++feature) {
                    v[feature] -= projection * basis_v[idx(feature, components, prev)];
                }
            }
        }
        const double v_norm = std::sqrt(squared_norm(v));
        if (v_norm < kEps) {
            continue;
        }
        for (double& value : v) {
            value /= v_norm;
        }

        for (std::size_t feature = 0; feature < p; ++feature) {
            basis_v[idx(feature, components, comp)] = v[feature];
            fit.weights_z[idx(feature, components, comp)] = z[feature];
            fit.loadings_p[idx(feature, components, comp)] = p_load[feature];
        }
        for (std::size_t target = 0; target < q; ++target) {
            fit.loadings_q[idx(target, components, comp)] = q_load[target];
        }

        std::vector<double> v_cov(q, 0.0);
        for (std::size_t target = 0; target < q; ++target) {
            for (std::size_t feature = 0; feature < p; ++feature) {
                v_cov[target] += v[feature] * covariance[idx(feature, q, target)];
            }
        }
        for (std::size_t feature = 0; feature < p; ++feature) {
            for (std::size_t target = 0; target < q; ++target) {
                covariance[idx(feature, q, target)] -= v[feature] * v_cov[target];
            }
        }
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t coefficients_for_prefix(::n4m::core::Context& ctx,
                                                   const PopFit& fit,
                                                   std::size_t p,
                                                   std::size_t q,
                                                   std::size_t prefix,
                                                   std::vector<double>& coefficients) {
    coefficients.assign(p * q, 0.0);
    if (prefix == 0U || prefix > static_cast<std::size_t>(fit.n_components)) {
        ctx.set_error("POP coefficient prefix is out of range");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const auto components = static_cast<std::size_t>(fit.n_components);
    std::vector<double> ptz(prefix * prefix, 0.0);
    for (std::size_t row_comp = 0; row_comp < prefix; ++row_comp) {
        for (std::size_t col_comp = 0; col_comp < prefix; ++col_comp) {
            double sum = 0.0;
            for (std::size_t feature = 0; feature < p; ++feature) {
                sum += fit.loadings_p[idx(feature, components, row_comp)] *
                       fit.weights_z[idx(feature, components, col_comp)];
            }
            ptz[idx(row_comp, prefix, col_comp)] = sum;
        }
    }
    std::vector<double> ptz_inv;
    if (!invert_square_matrix(ptz, prefix, ptz_inv)) {
        coefficients.clear();
        return N4M_ERR_NUMERICAL_FAILURE;
    }
    for (std::size_t feature = 0; feature < p; ++feature) {
        for (std::size_t target = 0; target < q; ++target) {
            double coefficient = 0.0;
            for (std::size_t comp = 0; comp < prefix; ++comp) {
                double rotation = 0.0;
                for (std::size_t inner = 0; inner < prefix; ++inner) {
                    rotation += fit.weights_z[idx(feature, components, inner)] *
                                ptz_inv[idx(inner, prefix, comp)];
                }
                coefficient += rotation * fit.loadings_q[idx(target, components, comp)];
            }
            coefficients[idx(feature, q, target)] = coefficient;
        }
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t score_pop_sequence_cv(::n4m::core::Context& ctx,
                                                 const std::vector<StrictOperatorMatrix>& matrices,
                                                 const std::vector<std::int32_t>& op_indices,
                                                 const n4m_matrix_view_t& X,
                                                 const n4m_matrix_view_t& Y,
                                                 const ::n4m::core::ValidationPlan& plan,
                                                 double& score) {
    score = std::numeric_limits<double>::infinity();
    if (op_indices.empty()) {
        return N4M_OK;
    }
    const auto p = static_cast<std::size_t>(X.cols);
    const auto q = static_cast<std::size_t>(Y.cols);
    double total_rmse = 0.0;
    std::int32_t valid_folds = 0;
    for (const auto& fold : plan.folds) {
        if (fold.train_indices.empty() || fold.test_indices.empty()) {
            ctx.set_error("POP selection received an empty validation fold");
            return N4M_ERR_INVALID_ARGUMENT;
        }

        std::vector<double> train_x;
        std::vector<double> train_y;
        std::vector<double> test_x;
        std::vector<double> test_y;
        n4m_status_t status = copy_rows(ctx, X, fold.train_indices, "X train", train_x);
        if (status != N4M_OK) {
            return status;
        }
        status = copy_rows(ctx, Y, fold.train_indices, "Y train", train_y);
        if (status != N4M_OK) {
            return status;
        }
        status = copy_rows(ctx, X, fold.test_indices, "X test", test_x);
        if (status != N4M_OK) {
            return status;
        }
        status = copy_rows(ctx, Y, fold.test_indices, "Y test", test_y);
        if (status != N4M_OK) {
            return status;
        }

        const std::size_t train_rows = fold.train_indices.size();
        const std::size_t test_rows = fold.test_indices.size();
        std::vector<double> x_mean;
        std::vector<double> y_mean;
        std::vector<double> train_x_centered;
        std::vector<double> train_y_centered;
        column_means(train_x, train_rows, p, x_mean);
        column_means(train_y, train_rows, q, y_mean);
        center_by_means(train_x, train_rows, p, x_mean, train_x_centered);
        center_by_means(train_y, train_rows, q, y_mean, train_y_centered);

        PopFit fit;
        status = fit_pop_sequence(ctx,
                                  matrices,
                                  op_indices,
                                  train_x_centered,
                                  train_y_centered,
                                  train_rows,
                                  p,
                                  q,
                                  fit);
        if (status != N4M_OK) {
            return status;
        }
        std::vector<double> coefficients;
        status = coefficients_for_prefix(ctx, fit, p, q, op_indices.size(), coefficients);
        if (status != N4M_OK) {
            continue;
        }

        std::vector<double> fold_predictions(test_rows * q, 0.0);
        for (std::size_t row = 0; row < test_rows; ++row) {
            for (std::size_t target = 0; target < q; ++target) {
                double prediction = y_mean[target];
                for (std::size_t feature = 0; feature < p; ++feature) {
                    prediction += (test_x[idx(row, p, feature)] - x_mean[feature]) *
                                  coefficients[idx(feature, q, target)];
                }
                fold_predictions[idx(row, q, target)] = prediction;
            }
        }
        double sumsq = 0.0;
        std::size_t residual_count = test_rows * q;
        if (q == 1U) {
            // Match bench AOM_v0 POP's cv_score_regression helper exactly:
            // y_val is (n, 1) while PLS1 predictions are 1D, so NumPy
            // broadcasts to an n x n residual matrix before RMSE.
            residual_count = test_rows * test_rows;
            for (std::size_t actual_row = 0; actual_row < test_rows; ++actual_row) {
                for (std::size_t pred_row = 0; pred_row < test_rows; ++pred_row) {
                    const double residual = test_y[idx(actual_row, q, 0)] -
                                            fold_predictions[idx(pred_row, q, 0)];
                    sumsq += residual * residual;
                }
            }
        } else {
            for (std::size_t row = 0; row < test_rows; ++row) {
                for (std::size_t target = 0; target < q; ++target) {
                    const double residual = test_y[idx(row, q, target)] -
                                            fold_predictions[idx(row, q, target)];
                    sumsq += residual * residual;
                }
            }
        }
        total_rmse += std::sqrt(sumsq / static_cast<double>(residual_count));
        valid_folds += 1;
    }
    if (valid_folds == static_cast<std::int32_t>(plan.folds.size())) {
        score = total_rmse / static_cast<double>(valid_folds);
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t fit_pop_predictions(::n4m::core::Context& ctx,
                                               const std::vector<StrictOperatorMatrix>& matrices,
                                               const std::vector<std::int32_t>& op_indices,
                                               const n4m_matrix_view_t& X,
                                               const n4m_matrix_view_t& Y,
                                               std::vector<double>& predictions) {
    const auto rows = static_cast<std::size_t>(X.rows);
    const auto p = static_cast<std::size_t>(X.cols);
    const auto q = static_cast<std::size_t>(Y.cols);
    std::vector<double> x_values;
    std::vector<double> y_values;
    n4m_status_t status = copy_all_rows(ctx, X, "X", x_values);
    if (status != N4M_OK) {
        return status;
    }
    status = copy_all_rows(ctx, Y, "Y", y_values);
    if (status != N4M_OK) {
        return status;
    }

    std::vector<double> x_mean;
    std::vector<double> y_mean;
    std::vector<double> x_centered;
    std::vector<double> y_centered;
    column_means(x_values, rows, p, x_mean);
    column_means(y_values, rows, q, y_mean);
    center_by_means(x_values, rows, p, x_mean, x_centered);
    center_by_means(y_values, rows, q, y_mean, y_centered);

    PopFit fit;
    status = fit_pop_sequence(ctx, matrices, op_indices, x_centered, y_centered, rows, p, q, fit);
    if (status != N4M_OK) {
        return status;
    }
    std::vector<double> coefficients;
    status = coefficients_for_prefix(ctx, fit, p, q, op_indices.size(), coefficients);
    if (status != N4M_OK) {
        return status;
    }

    predictions.assign(rows * q, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t target = 0; target < q; ++target) {
            double prediction = y_mean[target];
            for (std::size_t feature = 0; feature < p; ++feature) {
                prediction += (x_values[idx(row, p, feature)] - x_mean[feature]) *
                              coefficients[idx(feature, q, target)];
            }
            predictions[idx(row, q, target)] = prediction;
        }
    }
    return N4M_OK;
}

}  // namespace

namespace n4m::core {

n4m_status_t select_aom_global(Context& ctx,
                               const Config& cfg,
                               const OperatorBank& bank,
                               const n4m_matrix_view_t& X,
                               const n4m_matrix_view_t& Y,
                               const ValidationPlan& plan,
                               std::int32_t max_components,
                               AomGlobalSelectionResult& out) {
    try {
        out = AomGlobalSelectionResult{};
        n4m_status_t status = validate_request(ctx, cfg, bank, X, Y, plan, max_components);
        if (status != N4M_OK) {
            return status;
        }

        const auto n_ops = bank.entries().size();
        out.n_operators = static_cast<std::int32_t>(n_ops);
        out.max_components = max_components;
        out.operator_scores.assign(n_ops, std::numeric_limits<double>::infinity());
        out.rmse_curves.assign(n_ops * static_cast<std::size_t>(max_components),
                               std::numeric_limits<double>::infinity());
        out.operator_kinds.reserve(n_ops);

        double best_score = std::numeric_limits<double>::infinity();
        std::int32_t best_op = 0;
        std::int32_t best_k = 1;

        for (std::size_t op = 0; op < n_ops; ++op) {
            const OperatorEntry& entry = bank.entries()[op];
            out.operator_kinds.push_back(static_cast<std::int64_t>(entry.kind));
            std::vector<double> curve;
            status = score_operator(ctx, cfg, entry, X, Y, plan, max_components, curve);
            if (status != N4M_OK) {
                out = AomGlobalSelectionResult{};
                return status;
            }
            for (std::int32_t k = 1; k <= max_components; ++k) {
                const double score = curve[static_cast<std::size_t>(k - 1)];
                out.rmse_curves[idx(op,
                                    static_cast<std::size_t>(max_components),
                                    static_cast<std::size_t>(k - 1))] = score;
                if (score < out.operator_scores[op]) {
                    out.operator_scores[op] = score;
                }
                if (score < best_score) {
                    best_score = score;
                    best_op = static_cast<std::int32_t>(op);
                    best_k = k;
                }
            }
        }

        if (!std::isfinite(best_score)) {
            ctx.set_error("AOM global selection found no finite candidate score");
            out = AomGlobalSelectionResult{};
            return N4M_ERR_NUMERICAL_FAILURE;
        }

        out.selected_operator_index = best_op;
        out.selected_n_components = best_k;
        out.best_score = best_score;

        status = fit_selected_predictions(ctx,
                                          cfg,
                                          bank.entries()[static_cast<std::size_t>(best_op)],
                                          X,
                                          Y,
                                          best_k,
                                          out.predictions);
        if (status != N4M_OK) {
            out = AomGlobalSelectionResult{};
            return status;
        }

        ctx.clear_error();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while running AOM global selection");
        out = AomGlobalSelectionResult{};
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while running AOM global selection");
        out = AomGlobalSelectionResult{};
        return N4M_ERR_INTERNAL;
    }
}

n4m_status_t select_aom_per_component(Context& ctx,
                                      const Config& cfg,
                                      const OperatorBank& bank,
                                      const n4m_matrix_view_t& X,
                                      const n4m_matrix_view_t& Y,
                                      const ValidationPlan& plan,
                                      std::int32_t max_components,
                                      AomPerComponentSelectionResult& out) {
    try {
        out = AomPerComponentSelectionResult{};
        n4m_status_t status = validate_request(ctx, cfg, bank, X, Y, plan, max_components);
        if (status != N4M_OK) {
            return status;
        }

        std::vector<StrictOperatorMatrix> matrices;
        status = build_operator_matrices(ctx, bank, X.cols, matrices);
        if (status != N4M_OK) {
            out = AomPerComponentSelectionResult{};
            return status;
        }

        const auto n_ops = bank.entries().size();
        out.n_operators = static_cast<std::int32_t>(n_ops);
        out.max_components = max_components;
        out.operator_kinds.reserve(n_ops);
        for (const OperatorEntry& entry : bank.entries()) {
            out.operator_kinds.push_back(static_cast<std::int64_t>(entry.kind));
        }
        out.component_scores.assign(static_cast<std::size_t>(max_components) * n_ops,
                                    std::numeric_limits<double>::infinity());
        out.prefix_scores.assign(static_cast<std::size_t>(max_components),
                                 std::numeric_limits<double>::infinity());

        std::vector<std::int32_t> selected;
        selected.reserve(static_cast<std::size_t>(max_components));
        for (std::int32_t component = 0; component < max_components; ++component) {
            double best_score = std::numeric_limits<double>::infinity();
            std::int32_t best_op = 0;
            for (std::size_t op = 0; op < n_ops; ++op) {
                std::vector<std::int32_t> candidate = selected;
                candidate.push_back(static_cast<std::int32_t>(op));
                double score = std::numeric_limits<double>::infinity();
                status = score_pop_sequence_cv(ctx, matrices, candidate, X, Y, plan, score);
                if (status != N4M_OK) {
                    out = AomPerComponentSelectionResult{};
                    return status;
                }
                out.component_scores[idx(static_cast<std::size_t>(component), n_ops, op)] = score;
                if (score < best_score) {
                    best_score = score;
                    best_op = static_cast<std::int32_t>(op);
                }
            }
            if (!std::isfinite(best_score)) {
                ctx.set_error("POP selection found no finite candidate score");
                out = AomPerComponentSelectionResult{};
                return N4M_ERR_NUMERICAL_FAILURE;
            }
            selected.push_back(best_op);
        }

        double best_prefix_score = std::numeric_limits<double>::infinity();
        std::int32_t best_k = 1;
        for (std::int32_t k = 1; k <= max_components; ++k) {
            std::vector<std::int32_t> prefix(selected.begin(), selected.begin() + k);
            double score = std::numeric_limits<double>::infinity();
            status = score_pop_sequence_cv(ctx, matrices, prefix, X, Y, plan, score);
            if (status != N4M_OK) {
                out = AomPerComponentSelectionResult{};
                return status;
            }
            out.prefix_scores[static_cast<std::size_t>(k - 1)] = score;
            if (score < best_prefix_score) {
                best_prefix_score = score;
                best_k = k;
            }
        }
        if (!std::isfinite(best_prefix_score)) {
            ctx.set_error("POP selection found no finite prefix score");
            out = AomPerComponentSelectionResult{};
            return N4M_ERR_NUMERICAL_FAILURE;
        }

        out.selected_n_components = best_k;
        out.best_score = best_prefix_score;
        std::vector<std::int32_t> final_selected(selected.begin(), selected.begin() + best_k);
        out.selected_operator_indices.assign(final_selected.begin(), final_selected.end());

        status = fit_pop_predictions(ctx,
                                     matrices,
                                     final_selected,
                                     X,
                                     Y,
                                     out.predictions);
        if (status != N4M_OK) {
            out = AomPerComponentSelectionResult{};
            return status;
        }

        ctx.clear_error();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while running POP selection");
        out = AomPerComponentSelectionResult{};
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while running POP selection");
        out = AomPerComponentSelectionResult{};
        return N4M_ERR_INTERNAL;
    }
}

}  // namespace n4m::core
