// SPDX-License-Identifier: CECILL-2.1
//
// Recursive (moving-window) PLS (§12 of Overview.md).

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

#include "core/config.hpp"
#include "core/context.hpp"
#include "core/recursive_pls.hpp"

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

}  // namespace

TEST(recursive_pls_phase15, moving_window_predicts_in_streaming_order) {
    // 12 samples × 3 features.
    std::vector<double> Xv = {
        0.10, 0.50, 0.90,
        0.30, 0.40, 0.80,
        0.60, 0.10, 0.20,
        0.40, 0.30, 0.60,
        0.80, 0.20, 0.10,
        0.20, 0.90, 0.40,
        0.70, 0.60, 0.30,
        0.50, 0.80, 0.70,
        0.15, 0.55, 0.85,
        0.35, 0.45, 0.75,
        0.65, 0.15, 0.25,
        0.45, 0.35, 0.65,
    };
    std::vector<double> Yv = {
        1.1, 0.7, 0.4, 1.0, 0.9, 0.6, 1.2, 1.3, 1.05, 0.72, 0.45, 1.02
    };
    n4m_matrix_view_t X = make_view(Xv, 12, 3);
    n4m_matrix_view_t Y = make_view(Yv, 12, 1);

    ::n4m::core::Context ctx;
    ::n4m::core::Config cfg;
    cfg.algorithm = N4M_ALGO_PLS_REGRESSION;
    cfg.solver = N4M_SOLVER_SIMPLS;
    cfg.deflation = N4M_DEFLATION_REGRESSION;
    cfg.n_components = 2;
    cfg.center_x = 1;
    cfg.scale_x = 0;
    cfg.center_y = 1;
    cfg.scale_y = 0;

    ::n4m::core::RecursivePlsResult result;
    CHECK_EQ(::n4m::core::fit_predict_recursive_pls(ctx, cfg, X, Y, 6,
                                                        result),
             N4M_OK);
    CHECK_EQ(result.n_samples, 12);
    CHECK_EQ(result.window_size, 6);
    CHECK_EQ(result.predictions.size(), static_cast<std::size_t>(12));
    CHECK_EQ(result.in_window.size(), static_cast<std::size_t>(12));

    // Samples 0..5 are warmup (no model fitted yet).
    for (std::size_t i = 0; i < 6; ++i) {
        CHECK_EQ(result.in_window[i], 0);
        CHECK_EQ(result.predictions[i], 0.0);
    }
    // Samples 6..11 must each have a prediction from a fitted window.
    for (std::size_t i = 6; i < 12; ++i) {
        CHECK_EQ(result.in_window[i], 1);
        CHECK(std::isfinite(result.predictions[i]));
    }
}

TEST(recursive_pls_phase15, rejects_invalid_window) {
    std::vector<double> Xv = {0.1, 0.2, 0.3, 0.4};
    std::vector<double> Yv = {1.0, 0.0};
    n4m_matrix_view_t X = make_view(Xv, 2, 2);
    n4m_matrix_view_t Y = make_view(Yv, 2, 1);

    ::n4m::core::Context ctx;
    ::n4m::core::Config cfg;
    cfg.algorithm = N4M_ALGO_PLS_REGRESSION;
    cfg.solver = N4M_SOLVER_SIMPLS;
    cfg.deflation = N4M_DEFLATION_REGRESSION;
    cfg.n_components = 1;
    ::n4m::core::RecursivePlsResult result;

    // window_size < 2.
    CHECK_EQ(::n4m::core::fit_predict_recursive_pls(ctx, cfg, X, Y, 1,
                                                        result),
             N4M_ERR_INVALID_ARGUMENT);
    // window_size > n_rows.
    CHECK_EQ(::n4m::core::fit_predict_recursive_pls(ctx, cfg, X, Y, 3,
                                                        result),
             N4M_ERR_INVALID_ARGUMENT);
    // n_components >= window_size.
    cfg.n_components = 2;
    CHECK_EQ(::n4m::core::fit_predict_recursive_pls(ctx, cfg, X, Y, 2,
                                                        result),
             N4M_ERR_INVALID_ARGUMENT);
}
