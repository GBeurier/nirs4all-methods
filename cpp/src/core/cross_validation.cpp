// SPDX-License-Identifier: CECILL-2.1

#include "core/cross_validation.hpp"

#include <algorithm>
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

[[nodiscard]] unsigned long long ull(std::size_t value) noexcept {
    return static_cast<unsigned long long>(value);
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
    out.clear();
    out.resize(n_values);
    const auto cols = static_cast<std::size_t>(view.cols);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto src_row = static_cast<std::size_t>(rows[i]);
        for (std::size_t col = 0; col < cols; ++col) {
            out[i * cols + col] = read_value(view, src_row, col);
        }
    }
    return N4M_OK;
}

[[nodiscard]] n4m_status_t validate_plan_fold(
    ::n4m::core::Context& ctx,
    const ::n4m::core::ValidationFold& fold,
    std::size_t n_samples,
    std::size_t fold_idx,
    std::vector<std::int32_t>& prediction_counts) {
    if (fold.train_indices.empty()) {
        ctx.set_errorf("CV fold %llu has no training samples", ull(fold_idx));
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (fold.test_indices.empty()) {
        ctx.set_errorf("CV fold %llu has no test samples", ull(fold_idx));
        return N4M_ERR_INVALID_ARGUMENT;
    }

    std::vector<unsigned char> role(n_samples, 0U);
    for (const std::int64_t sample : fold.train_indices) {
        if (sample < 0 ||
            static_cast<std::uint64_t>(sample) >= static_cast<std::uint64_t>(n_samples)) {
            ctx.set_errorf("CV fold %llu train index out of range: %lld",
                           ull(fold_idx),
                           static_cast<long long>(sample));
            return N4M_ERR_INVALID_ARGUMENT;
        }
        const auto u = static_cast<std::size_t>(sample);
        if (role[u] != 0U) {
            ctx.set_errorf("CV fold %llu contains duplicate train index %lld",
                           ull(fold_idx),
                           static_cast<long long>(sample));
            return N4M_ERR_INVALID_ARGUMENT;
        }
        role[u] = 1U;
    }
    for (const std::int64_t sample : fold.test_indices) {
        if (sample < 0 ||
            static_cast<std::uint64_t>(sample) >= static_cast<std::uint64_t>(n_samples)) {
            ctx.set_errorf("CV fold %llu test index out of range: %lld",
                           ull(fold_idx),
                           static_cast<long long>(sample));
            return N4M_ERR_INVALID_ARGUMENT;
        }
        const auto u = static_cast<std::size_t>(sample);
        if (role[u] == 1U) {
            ctx.set_errorf("CV fold %llu test index %lld also appears in train",
                           ull(fold_idx),
                           static_cast<long long>(sample));
            return N4M_ERR_INVALID_ARGUMENT;
        }
        if (role[u] == 2U) {
            ctx.set_errorf("CV fold %llu contains duplicate test index %lld",
                           ull(fold_idx),
                           static_cast<long long>(sample));
            return N4M_ERR_INVALID_ARGUMENT;
        }
        if (prediction_counts[u] != 0) {
            ctx.set_errorf("sample %lld appears in more than one CV test fold",
                           static_cast<long long>(sample));
            return N4M_ERR_INVALID_ARGUMENT;
        }
        role[u] = 2U;
        prediction_counts[u] += 1;
    }
    return N4M_OK;
}

}  // namespace

namespace n4m::core {

n4m_status_t cross_validate_regression(Context& ctx,
                                       const Config& cfg,
                                       const n4m_matrix_view_t& X,
                                       const n4m_matrix_view_t& Y,
                                       const ValidationPlan& plan,
                                       CrossValidationResult& out) {
    try {
        out = CrossValidationResult{};

        n4m_status_t status = validate_float_view(ctx, X, "X");
        if (status != N4M_OK) {
            return status;
        }
        status = validate_float_view(ctx, Y, "Y");
        if (status != N4M_OK) {
            return status;
        }
        if (X.rows == 0 || X.cols == 0 || Y.cols == 0) {
            ctx.set_error("cross-validation matrices must be non-empty");
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
            ctx.set_error("validation plan must contain at least one fold");
            return N4M_ERR_INVALID_ARGUMENT;
        }

        const auto n = static_cast<std::size_t>(X.rows);
        const auto p = static_cast<std::size_t>(X.cols);
        const auto q = static_cast<std::size_t>(Y.cols);
        std::size_t prediction_values = 0;
        if (!checked_matrix_size(X.rows, Y.cols, prediction_values)) {
            ctx.set_error("cross-validation prediction matrix is too large");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        out.n_samples = X.rows;
        out.n_targets = Y.cols;
        out.n_folds = static_cast<std::int64_t>(plan.folds.size());
        out.predictions.assign(prediction_values, 0.0);
        out.test_offsets.clear();
        out.test_indices.clear();
        out.test_offsets.reserve(plan.folds.size() + 1U);
        out.test_offsets.push_back(0);

        std::vector<std::int32_t> prediction_counts(n, 0);
        std::size_t max_train = 0;
        std::size_t max_test = 0;
        for (std::size_t fold_idx = 0; fold_idx < plan.folds.size(); ++fold_idx) {
            const ValidationFold& fold = plan.folds[fold_idx];
            status = validate_plan_fold(ctx, fold, n, fold_idx, prediction_counts);
            if (status != N4M_OK) {
                out = CrossValidationResult{};
                return status;
            }
            max_train = std::max(max_train, fold.train_indices.size());
            max_test = std::max(max_test, fold.test_indices.size());
        }

        std::size_t train_x_capacity = 0;
        std::size_t train_y_capacity = 0;
        std::size_t test_x_capacity = 0;
        std::size_t fold_pred_capacity = 0;
        if (!checked_matrix_size(static_cast<std::int64_t>(max_train), X.cols,
                                 train_x_capacity) ||
            !checked_matrix_size(static_cast<std::int64_t>(max_train), Y.cols,
                                 train_y_capacity) ||
            !checked_matrix_size(static_cast<std::int64_t>(max_test), X.cols,
                                 test_x_capacity) ||
            !checked_matrix_size(static_cast<std::int64_t>(max_test), Y.cols,
                                 fold_pred_capacity)) {
            ctx.set_error("cross-validation fold buffer shape is too large");
            out = CrossValidationResult{};
            return N4M_ERR_INVALID_ARGUMENT;
        }

        std::vector<double> train_x;
        std::vector<double> train_y;
        std::vector<double> test_x;
        std::vector<double> fold_predictions;
        train_x.reserve(train_x_capacity);
        train_y.reserve(train_y_capacity);
        test_x.reserve(test_x_capacity);
        fold_predictions.reserve(fold_pred_capacity);

        for (std::size_t fold_idx = 0; fold_idx < plan.folds.size(); ++fold_idx) {
            const ValidationFold& fold = plan.folds[fold_idx];
            status = copy_rows(ctx, X, fold.train_indices, "X train", train_x);
            if (status != N4M_OK) {
                out = CrossValidationResult{};
                return status;
            }
            status = copy_rows(ctx, Y, fold.train_indices, "Y train", train_y);
            if (status != N4M_OK) {
                out = CrossValidationResult{};
                return status;
            }
            status = copy_rows(ctx, X, fold.test_indices, "X test", test_x);
            if (status != N4M_OK) {
                out = CrossValidationResult{};
                return status;
            }

            n4m_matrix_view_t train_x_view =
                rowmajor_f64_view(train_x,
                                  static_cast<std::int64_t>(fold.train_indices.size()),
                                  static_cast<std::int64_t>(p));
            n4m_matrix_view_t train_y_view =
                rowmajor_f64_view(train_y,
                                  static_cast<std::int64_t>(fold.train_indices.size()),
                                  static_cast<std::int64_t>(q));
            n4m_matrix_view_t test_x_view =
                rowmajor_f64_view(test_x,
                                  static_cast<std::int64_t>(fold.test_indices.size()),
                                  static_cast<std::int64_t>(p));

            std::unique_ptr<Model> model;
            status = fit_model(ctx, cfg, train_x_view, train_y_view, model);
            if (status != N4M_OK) {
                out = CrossValidationResult{};
                return status;
            }
            if (!model) {
                ctx.set_errorf("CV fold %llu fit returned no model", ull(fold_idx));
                out = CrossValidationResult{};
                return N4M_ERR_INTERNAL;
            }

            fold_predictions.assign(fold.test_indices.size() * q, 0.0);
            n4m_matrix_view_t pred_view =
                rowmajor_f64_view(fold_predictions,
                                  static_cast<std::int64_t>(fold.test_indices.size()),
                                  static_cast<std::int64_t>(q));
            status = predict_into(ctx, *model, test_x_view, pred_view);
            if (status != N4M_OK) {
                out = CrossValidationResult{};
                return status;
            }

            for (std::size_t i = 0; i < fold.test_indices.size(); ++i) {
                const auto sample = static_cast<std::size_t>(fold.test_indices[i]);
                for (std::size_t target = 0; target < q; ++target) {
                    out.predictions[sample * q + target] =
                        fold_predictions[i * q + target];
                }
                out.test_indices.push_back(fold.test_indices[i]);
            }
            out.test_offsets.push_back(static_cast<std::int64_t>(out.test_indices.size()));
        }

        for (std::size_t sample = 0; sample < n; ++sample) {
            if (prediction_counts[sample] != 1) {
                ctx.set_errorf("sample %llu is not covered by exactly one CV test fold",
                               ull(sample));
                out = CrossValidationResult{};
                return N4M_ERR_INVALID_ARGUMENT;
            }
        }

        n4m_matrix_view_t all_predictions =
            rowmajor_f64_view(out.predictions, X.rows, Y.cols);
        status = compute_regression_metrics(ctx, Y, all_predictions, out.metrics);
        if (status != N4M_OK) {
            out = CrossValidationResult{};
            return status;
        }

        ctx.clear_error();
        return N4M_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while running cross-validation");
        out = CrossValidationResult{};
        return N4M_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while running cross-validation");
        out = CrossValidationResult{};
        return N4M_ERR_INTERNAL;
    }
}

}  // namespace n4m::core
