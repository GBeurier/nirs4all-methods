// SPDX-License-Identifier: CECILL-2.1

#include "core/model_selection.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

#include "core/cross_validation.hpp"

namespace {

void append_metrics_row(const ::pls4all::core::RegressionMetrics& metrics,
                        std::vector<double>& out) {
    out.push_back(metrics.rmse);
    out.push_back(metrics.mae);
    out.push_back(metrics.bias);
    out.push_back(metrics.r2);
    out.push_back(metrics.q2);
    out.push_back(metrics.slope);
    out.push_back(metrics.intercept);
    out.push_back(metrics.rpd);
    out.push_back(metrics.rpiq);
}

}  // namespace

namespace pls4all::core {

p4a_status_t cross_validate_component_prefixes(
    Context& ctx,
    const Config& cfg,
    const p4a_matrix_view_t& X,
    const p4a_matrix_view_t& Y,
    const ValidationPlan& plan,
    std::int32_t max_components,
    ComponentCvResult& out) {
    try {
        out = ComponentCvResult{};
        if (max_components < 1) {
            ctx.set_errorf("max_components must be >= 1; got %d",
                           static_cast<int>(max_components));
            return P4A_ERR_INVALID_ARGUMENT;
        }
        if (plan.folds.empty()) {
            ctx.set_error("component CV requires at least one validation fold");
            return P4A_ERR_INVALID_ARGUMENT;
        }
        if (static_cast<std::int64_t>(max_components) > X.cols) {
            ctx.set_errorf("max_components (%d) must be <= X cols (%lld)",
                           static_cast<int>(max_components),
                           static_cast<long long>(X.cols));
            return P4A_ERR_INVALID_ARGUMENT;
        }

        out.max_components = max_components;
        out.metrics_by_component.reserve(static_cast<std::size_t>(max_components));
        out.metrics_matrix.reserve(static_cast<std::size_t>(max_components) * 9U);
        out.n_folds = static_cast<std::int32_t>(plan.folds.size());
        out.fold_rmse_matrix.assign(
            static_cast<std::size_t>(max_components) *
                static_cast<std::size_t>(out.n_folds),
            0.0);

        double best_rmse = std::numeric_limits<double>::infinity();
        for (std::int32_t k = 1; k <= max_components; ++k) {
            Config local_cfg = cfg;
            local_cfg.n_components = k;
            CrossValidationResult cv{};
            const p4a_status_t status = cross_validate_regression(ctx,
                                                                  local_cfg,
                                                                  X,
                                                                  Y,
                                                                  plan,
                                                                  cv);
            if (status != P4A_OK) {
                out = ComponentCvResult{};
                return status;
            }
            out.metrics_by_component.push_back(cv.metrics);
            append_metrics_row(cv.metrics, out.metrics_matrix);
            if (cv.metrics.rmse < best_rmse) {
                best_rmse = cv.metrics.rmse;
                out.best_n_components = k;
            }
            // Compute per-fold RMSE from the OOF predictions matrix. cv
            // stores predictions in original sample order; test_offsets +
            // test_indices identify which row belongs to which fold. Y is
            // row-major (n_samples x n_targets); RMSE here averages over
            // all targets within each fold.
            const std::size_t n_samples = static_cast<std::size_t>(cv.n_samples);
            const std::size_t n_targets = static_cast<std::size_t>(cv.n_targets);
            const auto* y_ptr = static_cast<const double*>(Y.data);
            for (std::int32_t fold = 0; fold < out.n_folds; ++fold) {
                const std::int64_t start = cv.test_offsets[
                    static_cast<std::size_t>(fold)];
                const std::int64_t stop = cv.test_offsets[
                    static_cast<std::size_t>(fold) + 1U];
                double sumsq = 0.0;
                std::size_t count = 0;
                for (std::int64_t i = start; i < stop; ++i) {
                    const std::int64_t sample = cv.test_indices[
                        static_cast<std::size_t>(i)];
                    if (sample < 0 ||
                        static_cast<std::size_t>(sample) >= n_samples) {
                        out = ComponentCvResult{};
                        ctx.set_error("test index out of bounds in CV result");
                        return P4A_ERR_INTERNAL;
                    }
                    for (std::size_t t = 0; t < n_targets; ++t) {
                        const double pred = cv.predictions[
                            static_cast<std::size_t>(sample) * n_targets + t];
                        const double truth = y_ptr[
                            static_cast<std::size_t>(sample *
                                static_cast<std::int64_t>(Y.row_stride) +
                                static_cast<std::int64_t>(t) * Y.col_stride)];
                        const double diff = pred - truth;
                        sumsq += diff * diff;
                        ++count;
                    }
                }
                const double rmse = count > 0
                    ? std::sqrt(sumsq / static_cast<double>(count))
                    : 0.0;
                out.fold_rmse_matrix[
                    static_cast<std::size_t>(k - 1) *
                        static_cast<std::size_t>(out.n_folds) +
                    static_cast<std::size_t>(fold)] = rmse;
            }
        }

        out.one_se_n_components = out.best_n_components;
        ctx.clear_error();
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while running component-count CV");
        out = ComponentCvResult{};
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while running component-count CV");
        out = ComponentCvResult{};
        return P4A_ERR_INTERNAL;
    }
}

p4a_status_t select_one_se_components(Context& ctx,
                                       ComponentCvResult& result) {
    if (result.max_components < 1 || result.n_folds < 2) {
        ctx.set_error(
            "one-SE rule requires max_components >= 1 and n_folds >= 2");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (result.fold_rmse_matrix.size() !=
        static_cast<std::size_t>(result.max_components) *
            static_cast<std::size_t>(result.n_folds)) {
        ctx.set_error("fold_rmse_matrix is not populated; run "
                      "cross_validate_component_prefixes first");
        return P4A_ERR_INVALID_ARGUMENT;
    }

    // Mean RMSE per component (averaging across folds).
    std::vector<double> mean_rmse(static_cast<std::size_t>(result.max_components),
                                  0.0);
    for (std::int32_t k = 0; k < result.max_components; ++k) {
        double sum = 0.0;
        for (std::int32_t fold = 0; fold < result.n_folds; ++fold) {
            sum += result.fold_rmse_matrix[
                static_cast<std::size_t>(k) *
                    static_cast<std::size_t>(result.n_folds) +
                static_cast<std::size_t>(fold)];
        }
        mean_rmse[static_cast<std::size_t>(k)] =
            sum / static_cast<double>(result.n_folds);
    }

    // Standard error at the best component count.
    std::int32_t best_k = result.best_n_components;
    if (best_k < 1) best_k = 1;
    const std::size_t best_idx = static_cast<std::size_t>(best_k - 1);
    double mean_at_best = mean_rmse[best_idx];
    double sumsq = 0.0;
    for (std::int32_t fold = 0; fold < result.n_folds; ++fold) {
        const double fold_rmse = result.fold_rmse_matrix[
            best_idx * static_cast<std::size_t>(result.n_folds) +
            static_cast<std::size_t>(fold)];
        const double diff = fold_rmse - mean_at_best;
        sumsq += diff * diff;
    }
    const double variance =
        sumsq / static_cast<double>(result.n_folds - 1);
    const double standard_error =
        std::sqrt(variance / static_cast<double>(result.n_folds));
    result.one_se_standard_error = standard_error;
    result.one_se_threshold = mean_at_best + standard_error;

    // Pick the smallest k with mean RMSE <= threshold.
    std::int32_t selected = best_k;
    for (std::int32_t k = 1; k <= result.max_components; ++k) {
        if (mean_rmse[static_cast<std::size_t>(k - 1)] <=
            result.one_se_threshold) {
            selected = k;
            break;
        }
    }
    result.one_se_n_components = selected;
    ctx.clear_error();
    return P4A_OK;
}

}  // namespace pls4all::core
