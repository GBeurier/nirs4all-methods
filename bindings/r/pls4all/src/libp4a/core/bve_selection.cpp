// SPDX-License-Identifier: CECILL-2.1

#include "core/bve_selection.hpp"

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

#include "core/cross_validation.hpp"
#include "core/matrix_view.hpp"
#include "core/model.hpp"
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
    if (min_features < 1 || min_features >= X.cols) {
        ctx.set_errorf("min_features must be in [1, %lld); got %d",
                       static_cast<long long>(X.cols),
                       static_cast<int>(min_features));
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (X.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) ||
        Y.cols > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
        ctx.set_error("BVE matrix dimensions exceed int32 result storage");
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
                                       double& out,
                                       std::int32_t& best_n_components) {
    std::vector<double> subset_x;
    p4a_status_t status = copy_columns(ctx, X, columns, subset_x);
    if (status != P4A_OK) {
        return status;
    }
    p4a_matrix_view_t subset_view =
        rowmajor_f64_view(subset_x, X.rows, static_cast<std::int64_t>(columns.size()));
    const std::int32_t max_components =
        std::min<std::int32_t>(cfg.n_components,
                               static_cast<std::int32_t>(columns.size()));
    out = std::numeric_limits<double>::infinity();
    best_n_components = 1;
    for (std::int32_t n_components = 1; n_components <= max_components; ++n_components) {
        ::pls4all::core::Config local_cfg = cfg;
        local_cfg.n_components = n_components;
        ::pls4all::core::CrossValidationResult cv;
        status = ::pls4all::core::cross_validate_regression(ctx,
                                                            local_cfg,
                                                            subset_view,
                                                            Y,
                                                            plan,
                                                            cv);
        if (status != P4A_OK) {
            return status;
        }
        if (cv.metrics.rmse < out) {
            out = cv.metrics.rmse;
            best_n_components = n_components;
        }
    }
    return P4A_OK;
}

[[nodiscard]] p4a_status_t compute_subset_vip(::pls4all::core::Context& ctx,
                                              const ::pls4all::core::Config& cfg,
                                              const p4a_matrix_view_t& subset_x,
                                              const p4a_matrix_view_t& Y,
                                              std::vector<double>& out) {
    ::pls4all::core::Config local_cfg = cfg;
    local_cfg.store_scores = 1;
    std::unique_ptr<::pls4all::core::Model> model;
    p4a_status_t status = ::pls4all::core::fit_model(ctx, local_cfg, subset_x, Y, model);
    if (status != P4A_OK) {
        return status;
    }
    if (!model) {
        ctx.set_error("BVE fitted model is null");
        return P4A_ERR_INTERNAL;
    }
    const auto n = static_cast<std::size_t>(model->n_samples);
    const auto p = static_cast<std::size_t>(model->n_features);
    const auto q = static_cast<std::size_t>(model->n_targets);
    const auto a = static_cast<std::size_t>(model->n_components);
    if (model->weights_w.size() != p * a ||
        model->y_loadings_q.size() != q * a ||
        model->scores_t.size() != n * a) {
        ctx.set_error("BVE fitted model arrays are inconsistent for VIP");
        return P4A_ERR_INTERNAL;
    }

    std::vector<double> component_ssy(a, 0.0);
    double total_ssy = 0.0;
    for (std::size_t comp = 0; comp < a; ++comp) {
        double tss = 0.0;
        for (std::size_t row = 0; row < n; ++row) {
            const double value = model->scores_t[idx(row, a, comp)];
            tss += value * value;
        }
        double qss = 0.0;
        for (std::size_t target = 0; target < q; ++target) {
            const double value = model->y_loadings_q[idx(target, a, comp)];
            qss += value * value;
        }
        component_ssy[comp] = tss * qss;
        total_ssy += component_ssy[comp];
    }
    if (!std::isfinite(total_ssy) || total_ssy <= std::numeric_limits<double>::epsilon()) {
        ctx.set_error("BVE VIP total explained variance collapsed");
        return P4A_ERR_NUMERICAL_FAILURE;
    }

    std::vector<double> weight_norm_sq(a, 0.0);
    for (std::size_t comp = 0; comp < a; ++comp) {
        for (std::size_t feature = 0; feature < p; ++feature) {
            const double value = model->weights_w[idx(feature, a, comp)];
            weight_norm_sq[comp] += value * value;
        }
        if (!std::isfinite(weight_norm_sq[comp]) ||
            weight_norm_sq[comp] <= std::numeric_limits<double>::epsilon()) {
            ctx.set_error("BVE VIP loading-weight norm collapsed");
            return P4A_ERR_NUMERICAL_FAILURE;
        }
    }

    out.assign(p, 0.0);
    for (std::size_t feature = 0; feature < p; ++feature) {
        double weighted = 0.0;
        for (std::size_t comp = 0; comp < a; ++comp) {
            const double weight = model->weights_w[idx(feature, a, comp)];
            weighted += component_ssy[comp] * weight * weight / weight_norm_sq[comp];
        }
        out[feature] = std::sqrt(static_cast<double>(p) * weighted / total_ssy);
    }
    return P4A_OK;
}

[[nodiscard]] std::vector<std::size_t> bve_removal_order(const std::vector<double>& vip,
                                                         std::int32_t n_components) {
    std::vector<std::size_t> remove;
    remove.reserve(vip.size());
    for (std::size_t i = 0; i < vip.size(); ++i) {
        if (vip[i] < 1.0) {
            remove.push_back(i);
        }
    }
    if (remove.size() <= static_cast<std::size_t>(n_components + 1)) {
        remove.resize(vip.size());
        std::iota(remove.begin(), remove.end(), std::size_t{0});
        std::stable_sort(remove.begin(), remove.end(), [&vip](std::size_t lhs, std::size_t rhs) {
            const double left = vip[lhs];
            const double right = vip[rhs];
            if (left == right) {
                return lhs < rhs;
            }
            return left < right;
        });
        const auto keep = std::min(remove.size(), static_cast<std::size_t>(n_components));
        remove.resize(keep);
    } else {
        std::stable_sort(remove.begin(), remove.end());
    }
    if (remove.size() >= vip.size()) {
        remove.resize(vip.size() - 1U);
    }
    return remove;
}

[[nodiscard]] std::vector<std::int64_t> remove_local_indices(
    const std::vector<std::int64_t>& active,
    const std::vector<std::size_t>& local_remove) {
    std::vector<unsigned char> remove(active.size(), 0U);
    for (const std::size_t local : local_remove) {
        if (local < remove.size()) {
            remove[local] = 1U;
        }
    }
    std::vector<std::int64_t> next;
    next.reserve(active.size() - local_remove.size());
    for (std::size_t local = 0; local < active.size(); ++local) {
        if (remove[local] == 0U) {
            next.push_back(active[local]);
        }
    }
    return next;
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
        out.rmse_by_step.reserve(static_cast<std::size_t>(n_steps));
        out.retained_counts.reserve(static_cast<std::size_t>(n_steps));
        out.removed_indices.reserve(static_cast<std::size_t>(n_steps) *
                                    static_cast<std::size_t>(std::max(1, cfg.n_components)));

        double best_rmse = std::numeric_limits<double>::infinity();
        std::int32_t best_step = -1;
        std::int32_t actual_steps = 0;

        while (actual_steps < n_steps &&
               active.size() > static_cast<std::size_t>(cfg.n_components + 1)) {
            std::vector<double> active_x;
            status = copy_columns(ctx, X, active, active_x);
            if (status != P4A_OK) {
                out = BveSelectionResult{};
                return status;
            }
            p4a_matrix_view_t active_x_view =
                rowmajor_f64_view(active_x,
                                  X.rows,
                                  static_cast<std::int64_t>(active.size()));

            double rmse = 0.0;
            std::int32_t best_n_components = cfg.n_components;
            status = subset_rmse(ctx, cfg, X, Y, plan, active, rmse, best_n_components);
            if (status != P4A_OK) {
                out = BveSelectionResult{};
                return status;
            }

            std::vector<double> vip;
            ::pls4all::core::Config vip_cfg = cfg;
            vip_cfg.n_components = best_n_components;
            status = compute_subset_vip(ctx, vip_cfg, active_x_view, Y, vip);
            if (status != P4A_OK) {
                out = BveSelectionResult{};
                return status;
            }
            if (vip.size() != active.size()) {
                ctx.set_error("BVE VIP score count is inconsistent");
                out = BveSelectionResult{};
                return P4A_ERR_INTERNAL;
            }

            std::vector<double> row(p, 0.0);
            for (std::size_t local = 0; local < active.size(); ++local) {
                row[static_cast<std::size_t>(active[local])] = vip[local];
            }
            out.candidate_rmse.insert(out.candidate_rmse.end(), row.begin(), row.end());

            std::vector<std::size_t> local_remove = bve_removal_order(vip, cfg.n_components);
            if (local_remove.empty()) {
                ctx.set_error("BVE did not produce removable VIP candidates");
                out = BveSelectionResult{};
                return P4A_ERR_INTERNAL;
            }
            std::vector<std::int64_t> next = remove_local_indices(active, local_remove);
            if (next.empty()) {
                ctx.set_error("BVE removed all variables");
                out = BveSelectionResult{};
                return P4A_ERR_INTERNAL;
            }
            for (const std::size_t local : local_remove) {
                out.removed_indices.push_back(active[local]);
            }

            out.rmse_by_step.push_back(rmse);
            out.retained_counts.push_back(static_cast<std::int64_t>(next.size()));
            candidates.push_back(next);
            if (rmse < best_rmse) {
                best_rmse = rmse;
                best_step = actual_steps;
            }
            active = std::move(next);
            ++actual_steps;
        }

        if (best_step < 0) {
            ctx.set_error("BVE did not produce a candidate subset");
            out = BveSelectionResult{};
            return P4A_ERR_INTERNAL;
        }
        out.n_features = static_cast<std::int32_t>(X.cols);
        out.n_targets = static_cast<std::int32_t>(Y.cols);
        out.n_components = cfg.n_components;
        out.n_steps = actual_steps;
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
