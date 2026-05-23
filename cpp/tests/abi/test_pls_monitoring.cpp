// SPDX-License-Identifier: CECILL-2.1
//
// Hotelling T2 / Q-residual process monitoring (§19 of Overview.md).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "n4m/n4m.h"

#include "core/config.hpp"
#include "core/context.hpp"
#include "core/model.hpp"
#include "core/pls_monitoring.hpp"

#include "harness.hpp"

namespace {

n4m_matrix_view_t make_view(const std::vector<double>& data,
                            std::int64_t rows,
                            std::int64_t cols) {
    n4m_matrix_view_t view{};
    view.data = const_cast<double*>(data.data());
    view.rows = rows;
    view.cols = cols;
    view.row_stride = cols;
    view.col_stride = 1;
    view.dtype = N4M_DTYPE_F64;
    return view;
}

void fit_simpls(int& failures,
                ::n4m::core::Context& ctx,
                std::unique_ptr<::n4m::core::Model>& out_model) {
    static const std::vector<double> Xv = {
        0.10, 0.50, 0.90, 0.20,
        0.30, 0.40, 0.80, 0.60,
        0.60, 0.10, 0.20, 0.70,
        0.40, 0.30, 0.60, 0.90,
        0.80, 0.20, 0.10, 0.50,
        0.20, 0.90, 0.40, 0.10,
        0.70, 0.60, 0.30, 0.20,
        0.50, 0.80, 0.70, 0.80,
    };
    static const std::vector<double> Yv = {
        1.1, 0.7, 0.4, 1.0, 0.9, 0.6, 1.2, 1.3,
    };
    n4m_matrix_view_t X = make_view(Xv, 8, 4);
    n4m_matrix_view_t Y = make_view(Yv, 8, 1);

    ::n4m::core::Config cfg;
    cfg.algorithm = N4M_ALGO_PLS_REGRESSION;
    cfg.solver = N4M_SOLVER_SIMPLS;
    cfg.deflation = N4M_DEFLATION_REGRESSION;
    cfg.n_components = 2;
    cfg.center_x = 1;
    cfg.scale_x = 0;
    cfg.center_y = 1;
    cfg.scale_y = 0;
    cfg.store_scores = 1;
    CHECK_EQ(::n4m::core::fit_pls_regression_simpls(ctx, cfg, X, Y, out_model),
             N4M_OK);
}

}  // namespace

TEST(pls_monitoring_phase11, thresholds_match_quantile_of_reference) {
    ::n4m::core::Context ctx;
    std::unique_ptr<::n4m::core::Model> model;
    fit_simpls(failures, ctx, model);

    std::vector<double> Xv = {
        0.10, 0.50, 0.90, 0.20,
        0.30, 0.40, 0.80, 0.60,
        0.60, 0.10, 0.20, 0.70,
        0.40, 0.30, 0.60, 0.90,
        0.80, 0.20, 0.10, 0.50,
        0.20, 0.90, 0.40, 0.10,
        0.70, 0.60, 0.30, 0.20,
        0.50, 0.80, 0.70, 0.80,
    };
    n4m_matrix_view_t X = make_view(Xv, 8, 4);

    ::n4m::core::MonitoringThresholds thresholds;
    CHECK_EQ(::n4m::core::pls_monitoring_fit(ctx, *model, X, 0.10,
                                                 thresholds),
             N4M_OK);
    CHECK_EQ(thresholds.n_reference, 8);
    CHECK(thresholds.t2_threshold > 0.0);
    CHECK(thresholds.q_threshold > 0.0);
    CHECK(thresholds.alpha == 0.10);

    // 90th percentile of 8 reference T2 values: the threshold must be the
    // second-largest value (linear interpolation between indices 6 and 7
    // since position = 0.9 * 7 = 6.3).
    std::vector<double> sorted_t2 = thresholds.t2_reference;
    std::sort(sorted_t2.begin(), sorted_t2.end());
    if (sorted_t2.size() < 8U) {
        ++failures;
        std::fprintf(stderr, "  FAIL expected at least 8 T2 references\n");
        return;
    }
    const double expected =
        sorted_t2.at(6) + 0.3 * (sorted_t2.at(7) - sorted_t2.at(6));
    if (std::fabs(thresholds.t2_threshold - expected) > 1e-12) {
        ++failures;
        std::fprintf(stderr,
                     "  FAIL t2_threshold expected %.17g got %.17g\n",
                     expected, thresholds.t2_threshold);
    }
}

TEST(pls_monitoring_phase11, evaluate_flags_outliers) {
    ::n4m::core::Context ctx;
    std::unique_ptr<::n4m::core::Model> model;
    fit_simpls(failures, ctx, model);

    // Reference set = training data.
    std::vector<double> Xv = {
        0.10, 0.50, 0.90, 0.20,
        0.30, 0.40, 0.80, 0.60,
        0.60, 0.10, 0.20, 0.70,
        0.40, 0.30, 0.60, 0.90,
        0.80, 0.20, 0.10, 0.50,
        0.20, 0.90, 0.40, 0.10,
        0.70, 0.60, 0.30, 0.20,
        0.50, 0.80, 0.70, 0.80,
    };
    n4m_matrix_view_t X_ref = make_view(Xv, 8, 4);

    ::n4m::core::MonitoringThresholds thresholds;
    CHECK_EQ(::n4m::core::pls_monitoring_fit(ctx, *model, X_ref, 0.10,
                                                 thresholds),
             N4M_OK);

    // New data: one sample matches training distribution (should not
    // alarm), one sample is an extreme outlier (should alarm).
    std::vector<double> Xn = {
        0.30, 0.40, 0.80, 0.60,  // ~training sample 2
        5.00, 5.00, 5.00, 5.00,  // outlier
    };
    n4m_matrix_view_t X_new = make_view(Xn, 2, 4);
    ::n4m::core::MonitoringResult result;
    CHECK_EQ(::n4m::core::pls_monitoring_evaluate(ctx, *model, thresholds,
                                                       X_new, result),
             N4M_OK);
    CHECK_EQ(result.t2.size(), static_cast<std::size_t>(2));
    CHECK_EQ(result.q.size(), static_cast<std::size_t>(2));
    CHECK(result.any_alarms[1] == 1);  // outlier must trigger an alarm
}

TEST(pls_monitoring_phase11, rejects_invalid_alpha) {
    ::n4m::core::Context ctx;
    std::unique_ptr<::n4m::core::Model> model;
    fit_simpls(failures, ctx, model);
    std::vector<double> Xv = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    n4m_matrix_view_t X = make_view(Xv, 2, 4);

    ::n4m::core::MonitoringThresholds thresholds;
    CHECK_EQ(::n4m::core::pls_monitoring_fit(ctx, *model, X, 0.0,
                                                 thresholds),
             N4M_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::n4m::core::pls_monitoring_fit(ctx, *model, X, 1.0,
                                                 thresholds),
             N4M_ERR_INVALID_ARGUMENT);
}
