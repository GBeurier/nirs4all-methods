// SPDX-License-Identifier: CECILL-2.1
//
// 3-way N-PLS (§9). Verifies fit/predict on a 3-way synthetic dataset.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "n4m/n4m.h"

#include "core/config.hpp"
#include "core/context.hpp"
#include "core/tensor_pls.hpp"

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

TEST(n_pls_phase22, fits_simple_rank_one_tensor) {
    // 8 samples × J=3 × K=2 tensor. The signal is a known rank-1 product
    // u_j ⊗ u_k scaled by t_i; the predictor should recover that.
    const std::vector<double> u_j = {0.6, 0.5, 0.62};
    const std::vector<double> u_k = {0.7, 0.71};
    const std::vector<double> t_vals = {1.0, 2.0, 0.5, 1.5, -1.0, -0.5, 2.5, 0.0};
    std::vector<double> Xv(8 * 6, 0.0);
    for (std::size_t i = 0; i < 8; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            for (std::size_t k = 0; k < 2; ++k) {
                Xv[i * 6 + j * 2 + k] = t_vals[i] * u_j[j] * u_k[k] +
                                         0.01 * static_cast<double>(j + k);
            }
        }
    }
    // y = 2 * t_vals
    std::vector<double> Yv(8, 0.0);
    for (std::size_t i = 0; i < 8; ++i) Yv[i] = 2.0 * t_vals[i];

    n4m_matrix_view_t X = make_view(Xv, 8, 6);
    n4m_matrix_view_t Y = make_view(Yv, 8, 1);

    ::n4m::core::Context ctx;
    ::n4m::core::Config cfg;
    cfg.n_components = 1;
    cfg.center_x = 1;
    cfg.center_y = 1;
    ::n4m::core::NPlsResult model;
    CHECK_EQ(::n4m::core::fit_n_pls(ctx, cfg, X, 3, 2, Y, model),
             N4M_OK);
    CHECK_EQ(model.mode_j, 3);
    CHECK_EQ(model.mode_k, 2);
    CHECK_EQ(model.n_components, 1);
    CHECK_EQ(model.coefficients.size(), static_cast<std::size_t>(6));
    for (double v : model.coefficients) CHECK(std::isfinite(v));

    std::vector<double> preds;
    CHECK_EQ(::n4m::core::predict_n_pls(ctx, model, X, preds), N4M_OK);
    CHECK_EQ(preds.size(), static_cast<std::size_t>(8));

    double rmse = 0.0;
    for (std::size_t i = 0; i < 8; ++i) {
        const double d = preds[i] - Yv[i];
        rmse += d * d;
    }
    rmse = std::sqrt(rmse / 8.0);
    if (!(rmse < 0.5)) {
        ++failures;
        std::fprintf(stderr, "  FAIL N-PLS RMSE %.6f >= 0.5\n", rmse);
    }
}

TEST(n_pls_phase22, rejects_invalid_inputs) {
    std::vector<double> Xv = {0.1, 0.2, 0.3, 0.4};
    std::vector<double> Yv = {1.0, 0.0};
    n4m_matrix_view_t X = make_view(Xv, 2, 4);
    n4m_matrix_view_t Y = make_view(Yv, 2, 1);

    ::n4m::core::Context ctx;
    ::n4m::core::Config cfg;
    cfg.n_components = 1;
    ::n4m::core::NPlsResult model;
    // J * K != cols
    CHECK_EQ(::n4m::core::fit_n_pls(ctx, cfg, X, 3, 2, Y, model),
             N4M_ERR_SHAPE_MISMATCH);
    // negative mode
    CHECK_EQ(::n4m::core::fit_n_pls(ctx, cfg, X, -1, 2, Y, model),
             N4M_ERR_INVALID_ARGUMENT);
    // n_components < 1
    cfg.n_components = 0;
    CHECK_EQ(::n4m::core::fit_n_pls(ctx, cfg, X, 2, 2, Y, model),
             N4M_ERR_INVALID_ARGUMENT);
}
