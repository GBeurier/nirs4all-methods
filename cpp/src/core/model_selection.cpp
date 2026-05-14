// SPDX-License-Identifier: CeCILL-2.1

#include "core/model_selection.hpp"

#include <cstdint>
#include <limits>
#include <new>

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
        }

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

}  // namespace pls4all::core
