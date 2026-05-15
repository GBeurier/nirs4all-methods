// SPDX-License-Identifier: CeCILL-2.1
//
// Hotelling T2 / Q-residual process monitoring (§19 of Overview.md).
//
// Empirical thresholds are derived from a fitted Model and a reference
// (phase-1) design matrix. Each new sample evaluated by
// `pls_monitoring_evaluate` is flagged as an alarm when T2 or Q exceeds
// the threshold at the requested confidence level. The thresholds are
// percentile-based — no distributional assumption is made.

#pragma once

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

#include "core/context.hpp"
#include "core/model.hpp"

namespace pls4all::core {

struct MonitoringThresholds {
    double alpha{0.05};                  // tail probability (e.g. 0.05 → 95%)
    double t2_threshold{0.0};
    double q_threshold{0.0};
    std::int64_t n_reference{0};
    std::vector<double> t2_reference;    // training T2 values
    std::vector<double> q_reference;     // training Q-residuals
};

[[nodiscard]] p4a_status_t pls_monitoring_fit(
    Context& ctx,
    const Model& model,
    const p4a_matrix_view_t& X_reference,
    double alpha,
    MonitoringThresholds& out);

struct MonitoringResult {
    std::vector<double> t2;
    std::vector<double> q;
    // 1 if the corresponding statistic exceeds its threshold, 0 otherwise.
    std::vector<std::int32_t> t2_alarms;
    std::vector<std::int32_t> q_alarms;
    // Combined: 1 if either T2 or Q triggers an alarm.
    std::vector<std::int32_t> any_alarms;
};

[[nodiscard]] p4a_status_t pls_monitoring_evaluate(
    Context& ctx,
    const Model& model,
    const MonitoringThresholds& thresholds,
    const p4a_matrix_view_t& X,
    MonitoringResult& out);

}  // namespace pls4all::core
