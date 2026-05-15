// SPDX-License-Identifier: CeCILL-2.1

#include "core/pls_monitoring.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "core/pls_diagnostics.hpp"

namespace pls4all::core {

namespace {

[[nodiscard]] double percentile(std::vector<double> sorted_values,
                                double quantile) {
    // Sorted values assumed; this overload sorts a copy.
    std::sort(sorted_values.begin(), sorted_values.end());
    if (sorted_values.empty()) return 0.0;
    if (sorted_values.size() == 1U) return sorted_values.front();
    const double position = quantile *
        static_cast<double>(sorted_values.size() - 1U);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    if (lower == upper) return sorted_values[lower];
    const double fraction = position - static_cast<double>(lower);
    return sorted_values[lower] +
           fraction * (sorted_values[upper] - sorted_values[lower]);
}

}  // namespace

p4a_status_t pls_monitoring_fit(Context& ctx,
                                 const Model& model,
                                 const p4a_matrix_view_t& X_reference,
                                 double alpha,
                                 MonitoringThresholds& out) {
    out = MonitoringThresholds{};
    if (!(alpha > 0.0 && alpha < 1.0)) {
        ctx.set_errorf("alpha must be in (0, 1); got %.6g", alpha);
        return P4A_ERR_INVALID_ARGUMENT;
    }
    std::vector<double> t2;
    p4a_status_t status = pls_hotelling_t2(ctx, model, X_reference,
                                            &X_reference, t2);
    if (status != P4A_OK) return status;
    std::vector<double> q;
    status = pls_q_residuals(ctx, model, X_reference, q);
    if (status != P4A_OK) return status;

    if (t2.size() != q.size() || t2.empty()) {
        ctx.set_error("reference statistics produced inconsistent sizes");
        return P4A_ERR_INTERNAL;
    }

    out.alpha = alpha;
    out.n_reference = static_cast<std::int64_t>(t2.size());
    out.t2_reference = t2;
    out.q_reference = q;
    const double quantile = 1.0 - alpha;
    out.t2_threshold = percentile(t2, quantile);
    out.q_threshold = percentile(q, quantile);
    ctx.clear_error();
    return P4A_OK;
}

p4a_status_t pls_monitoring_evaluate(Context& ctx,
                                      const Model& model,
                                      const MonitoringThresholds& thresholds,
                                      const p4a_matrix_view_t& X,
                                      MonitoringResult& out) {
    out = MonitoringResult{};
    if (thresholds.n_reference <= 0 || thresholds.t2_reference.empty()) {
        ctx.set_error("monitoring thresholds are not fitted");
        return P4A_ERR_NOT_FITTED;
    }
    // Score variances are baked into the model.scores_t when the model
    // was fitted with store_scores=1. The fit step in pls_monitoring_fit
    // also produced t2_reference using those variances; the new samples
    // here re-use that same reference.
    std::vector<double> t2;
    p4a_status_t status = pls_hotelling_t2(ctx, model, X, nullptr, t2);
    if (status != P4A_OK) return status;
    std::vector<double> q;
    status = pls_q_residuals(ctx, model, X, q);
    if (status != P4A_OK) return status;
    if (t2.size() != q.size()) {
        ctx.set_error("monitoring statistics produced inconsistent sizes");
        return P4A_ERR_INTERNAL;
    }
    out.t2 = std::move(t2);
    out.q = std::move(q);
    out.t2_alarms.resize(out.t2.size());
    out.q_alarms.resize(out.q.size());
    out.any_alarms.resize(out.t2.size());
    for (std::size_t i = 0; i < out.t2.size(); ++i) {
        out.t2_alarms[i] =
            (out.t2[i] > thresholds.t2_threshold) ? 1 : 0;
        out.q_alarms[i] = (out.q[i] > thresholds.q_threshold) ? 1 : 0;
        out.any_alarms[i] =
            (out.t2_alarms[i] != 0 || out.q_alarms[i] != 0) ? 1 : 0;
    }
    ctx.clear_error();
    return P4A_OK;
}

}  // namespace pls4all::core
