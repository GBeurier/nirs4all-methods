// SPDX-License-Identifier: CeCILL-2.1

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
#include "core/matrix_view.hpp"
#include "core/model.hpp"
#include "core/operator_entry.hpp"
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
    if (view.rows <= 0 || view.cols <= 0) {
        ctx.set_errorf("%s matrix must be non-empty", name);
        return P4A_ERR_INVALID_ARGUMENT;
    }
    return P4A_OK;
}

[[nodiscard]] p4a_status_t copy_rows(::pls4all::core::Context& ctx,
                                     const p4a_matrix_view_t& view,
                                     const std::vector<std::int64_t>& rows,
                                     const char* name,
                                     std::vector<double>& out) {
    std::size_t n_values = 0;
    if (!checked_matrix_size(static_cast<std::int64_t>(rows.size()),
                             view.cols,
                             n_values)) {
        ctx.set_errorf("%s row selection shape is too large", name);
        return P4A_ERR_INVALID_ARGUMENT;
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
            return P4A_ERR_INVALID_ARGUMENT;
        }
        const auto src_row = static_cast<std::size_t>(source);
        for (std::size_t col = 0; col < cols; ++col) {
            const double value = read_value(view, src_row, col);
            if (!std::isfinite(value)) {
                ctx.set_errorf("%s contains NaN or Inf", name);
                return P4A_ERR_INVALID_ARGUMENT;
            }
            out[idx(i, cols, col)] = value;
        }
    }
    return P4A_OK;
}

[[nodiscard]] p4a_status_t validate_plan(::pls4all::core::Context& ctx,
                                         const ::pls4all::core::ValidationPlan& plan,
                                         std::int64_t n_samples) {
    if (plan.folds.empty()) {
        ctx.set_error("AOM global selection requires at least one validation fold");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (plan.n_samples != 0 && plan.n_samples != n_samples) {
        ctx.set_errorf("validation plan sample count (%lld) must match X rows (%lld)",
                       static_cast<long long>(plan.n_samples),
                       static_cast<long long>(n_samples));
        return P4A_ERR_SHAPE_MISMATCH;
    }
    return P4A_OK;
}

[[nodiscard]] p4a_status_t validate_request(::pls4all::core::Context& ctx,
                                            const ::pls4all::core::Config& cfg,
                                            const ::pls4all::core::OperatorBank& bank,
                                            const p4a_matrix_view_t& X,
                                            const p4a_matrix_view_t& Y,
                                            const ::pls4all::core::ValidationPlan& plan,
                                            std::int32_t max_components) {
    p4a_status_t status = validate_float_view(ctx, X, "X");
    if (status != P4A_OK) {
        return status;
    }
    status = validate_float_view(ctx, Y, "Y");
    if (status != P4A_OK) {
        return status;
    }
    if (X.rows != Y.rows) {
        ctx.set_error("X and Y must have the same number of rows");
        return P4A_ERR_SHAPE_MISMATCH;
    }
    if (bank.entries().empty()) {
        ctx.set_error("AOM global selection requires at least one operator");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (bank.entries().size() >
        static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        ctx.set_error("AOM global operator count is too large");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (max_components < 1) {
        ctx.set_error("max_components must be >= 1");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (max_components > X.cols || max_components >= X.rows) {
        ctx.set_error("max_components must be <= X cols and < X rows");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (cfg.algorithm != P4A_ALGO_PLS_REGRESSION ||
        cfg.solver != P4A_SOLVER_SIMPLS ||
        cfg.deflation != P4A_DEFLATION_REGRESSION) {
        ctx.set_error("AOM global selection currently supports SIMPLS regression only");
        return P4A_ERR_UNSUPPORTED;
    }
    return validate_plan(ctx, plan, X.rows);
}

[[nodiscard]] p4a_status_t transform_with_operator(::pls4all::core::Context& ctx,
                                                   const ::pls4all::core::OperatorEntry& entry,
                                                   const p4a_matrix_view_t& apply_x,
                                                   std::vector<double>& transformed) {
    return ::pls4all::core::transform_aom_strict_operator(ctx, entry, apply_x, transformed);
}

[[nodiscard]] double prefix_rmse(const ::pls4all::core::Model& model,
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

[[nodiscard]] p4a_status_t score_operator(::pls4all::core::Context& ctx,
                                          const ::pls4all::core::Config& cfg,
                                          const ::pls4all::core::OperatorEntry& entry,
                                          const p4a_matrix_view_t& X,
                                          const p4a_matrix_view_t& Y,
                                          const ::pls4all::core::ValidationPlan& plan,
                                          std::int32_t max_components,
                                          std::vector<double>& curve) {
    const double inf = std::numeric_limits<double>::infinity();
    curve.assign(static_cast<std::size_t>(max_components), 0.0);
    std::vector<std::int32_t> counts(static_cast<std::size_t>(max_components), 0);
    std::vector<unsigned char> invalid(static_cast<std::size_t>(max_components), 0U);

    for (const auto& fold : plan.folds) {
        if (fold.train_indices.empty() || fold.test_indices.empty()) {
            ctx.set_error("AOM global selection received an empty validation fold");
            return P4A_ERR_INVALID_ARGUMENT;
        }

        std::vector<double> train_x;
        std::vector<double> train_y;
        std::vector<double> test_x;
        std::vector<double> test_y;
        p4a_status_t status = copy_rows(ctx, X, fold.train_indices, "X train", train_x);
        if (status != P4A_OK) {
            return status;
        }
        status = copy_rows(ctx, Y, fold.train_indices, "Y train", train_y);
        if (status != P4A_OK) {
            return status;
        }
        status = copy_rows(ctx, X, fold.test_indices, "X test", test_x);
        if (status != P4A_OK) {
            return status;
        }
        status = copy_rows(ctx, Y, fold.test_indices, "Y test", test_y);
        if (status != P4A_OK) {
            return status;
        }

        p4a_matrix_view_t train_x_view = rowmajor_f64_view(train_x,
                                                           static_cast<std::int64_t>(fold.train_indices.size()),
                                                           X.cols);
        p4a_matrix_view_t train_y_view = rowmajor_f64_view(train_y,
                                                           static_cast<std::int64_t>(fold.train_indices.size()),
                                                           Y.cols);
        p4a_matrix_view_t test_x_view = rowmajor_f64_view(test_x,
                                                          static_cast<std::int64_t>(fold.test_indices.size()),
                                                          X.cols);

        std::vector<double> train_xt;
        std::vector<double> test_xt;
        status = transform_with_operator(ctx, entry, train_x_view, train_xt);
        if (status != P4A_OK) {
            return status;
        }
        status = transform_with_operator(ctx, entry, test_x_view, test_xt);
        if (status != P4A_OK) {
            return status;
        }

        p4a_matrix_view_t train_xt_view = rowmajor_f64_view(train_xt,
                                                            static_cast<std::int64_t>(fold.train_indices.size()),
                                                            X.cols);
        ::pls4all::core::Config local_cfg = cfg;
        local_cfg.n_components = max_components;
        local_cfg.scale_x = 0;
        local_cfg.scale_y = 0;
        std::unique_ptr<::pls4all::core::Model> model;
        status = ::pls4all::core::fit_model(ctx, local_cfg, train_xt_view, train_y_view, model);
        if (status != P4A_OK || !model) {
            for (std::int32_t k = 0; k < max_components; ++k) {
                invalid[static_cast<std::size_t>(k)] = 1U;
            }
            continue;
        }

        std::vector<double> coefficients_by_component;
        status = ::pls4all::core::compute_regression_coefficients_by_component(ctx,
                                                                               *model,
                                                                               coefficients_by_component);
        if (status != P4A_OK) {
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
    return P4A_OK;
}

[[nodiscard]] p4a_status_t fit_selected_predictions(::pls4all::core::Context& ctx,
                                                    const ::pls4all::core::Config& cfg,
                                                    const ::pls4all::core::OperatorEntry& entry,
                                                    const p4a_matrix_view_t& X,
                                                    const p4a_matrix_view_t& Y,
                                                    std::int32_t n_components,
                                                    std::vector<double>& predictions) {
    std::vector<double> xt;
    p4a_status_t status = transform_with_operator(ctx, entry, X, xt);
    if (status != P4A_OK) {
        return status;
    }
    p4a_matrix_view_t xt_view = rowmajor_f64_view(xt, X.rows, X.cols);
    ::pls4all::core::Config local_cfg = cfg;
    local_cfg.n_components = n_components;
    local_cfg.scale_x = 0;
    local_cfg.scale_y = 0;
    std::unique_ptr<::pls4all::core::Model> model;
    status = ::pls4all::core::fit_model(ctx, local_cfg, xt_view, Y, model);
    if (status != P4A_OK || !model) {
        return status == P4A_OK ? P4A_ERR_INTERNAL : status;
    }
    std::size_t pred_values = 0;
    if (!checked_matrix_size(X.rows, Y.cols, pred_values)) {
        ctx.set_error("AOM prediction matrix shape is too large");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    predictions.assign(pred_values, 0.0);
    p4a_matrix_view_t pred_view = rowmajor_f64_view(predictions, X.rows, Y.cols);
    return ::pls4all::core::predict_into(ctx, *model, xt_view, pred_view);
}

}  // namespace

namespace pls4all::core {

p4a_status_t select_aom_global(Context& ctx,
                               const Config& cfg,
                               const OperatorBank& bank,
                               const p4a_matrix_view_t& X,
                               const p4a_matrix_view_t& Y,
                               const ValidationPlan& plan,
                               std::int32_t max_components,
                               AomGlobalSelectionResult& out) {
    try {
        out = AomGlobalSelectionResult{};
        p4a_status_t status = validate_request(ctx, cfg, bank, X, Y, plan, max_components);
        if (status != P4A_OK) {
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
            if (status != P4A_OK) {
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
            return P4A_ERR_NUMERICAL_FAILURE;
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
        if (status != P4A_OK) {
            out = AomGlobalSelectionResult{};
            return status;
        }

        ctx.clear_error();
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while running AOM global selection");
        out = AomGlobalSelectionResult{};
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while running AOM global selection");
        out = AomGlobalSelectionResult{};
        return P4A_ERR_INTERNAL;
    }
}

}  // namespace pls4all::core
