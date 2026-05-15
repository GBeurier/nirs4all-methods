// SPDX-License-Identifier: CeCILL-2.1
//
// Domain-invariant PLS (§13 of Overview.md). Verifies (a) di_lambda=0
// matches plain SIMPLS exactly, (b) di_lambda>0 reduces the projection
// of the weight vector onto the source-target difference direction.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "pls4all/p4a.h"

#include "core/config.hpp"
#include "core/context.hpp"
#include "core/model.hpp"

#include "harness.hpp"

namespace {

p4a_matrix_view_t make_view(const std::vector<double>& data,
                            std::int64_t rows,
                            std::int64_t cols) {
    p4a_matrix_view_t view{};
    view.data = const_cast<double*>(data.data());
    view.rows = rows;
    view.cols = cols;
    view.row_stride = cols;
    view.col_stride = 1;
    view.dtype = P4A_DTYPE_F64;
    return view;
}

}  // namespace

TEST(di_pls_phase13, lambda_zero_matches_plain_simpls) {
    std::vector<double> Xs = {
        0.10, 0.50, 0.90, 0.20,
        0.30, 0.40, 0.80, 0.60,
        0.60, 0.10, 0.20, 0.70,
        0.40, 0.30, 0.60, 0.90,
        0.80, 0.20, 0.10, 0.50,
        0.20, 0.90, 0.40, 0.10,
        0.70, 0.60, 0.30, 0.20,
        0.50, 0.80, 0.70, 0.80,
    };
    std::vector<double> Ys = {1.1, 0.7, 0.4, 1.0, 0.9, 0.6, 1.2, 1.3};
    // X_target shifted slightly along feature 0.
    std::vector<double> Xt = Xs;
    for (std::size_t row = 0; row < 8; ++row) Xt[row * 4] += 0.5;
    p4a_matrix_view_t Xs_view = make_view(Xs, 8, 4);
    p4a_matrix_view_t Ys_view = make_view(Ys, 8, 1);
    p4a_matrix_view_t Xt_view = make_view(Xt, 8, 4);

    ::pls4all::core::Context ctx;
    ::pls4all::core::Config cfg;
    cfg.algorithm = P4A_ALGO_PLS_REGRESSION;
    cfg.solver = P4A_SOLVER_SIMPLS;
    cfg.deflation = P4A_DEFLATION_REGRESSION;
    cfg.n_components = 2;
    cfg.center_x = 1;
    cfg.scale_x = 0;
    cfg.center_y = 1;
    cfg.scale_y = 0;
    cfg.store_scores = 0;
    cfg.di_lambda = 0.0;

    std::unique_ptr<::pls4all::core::Model> m_di;
    std::unique_ptr<::pls4all::core::Model> m_plain;
    CHECK_EQ(::pls4all::core::fit_di_pls(ctx, cfg, Xs_view, Ys_view, Xt_view, m_di),
             P4A_OK);
    CHECK_EQ(::pls4all::core::fit_pls_regression_simpls(ctx, cfg, Xs_view, Ys_view,
                                                        m_plain),
             P4A_OK);
    CHECK_EQ(m_di->coefficients.size(), m_plain->coefficients.size());
    for (std::size_t i = 0; i < m_di->coefficients.size(); ++i) {
        const double diff = std::fabs(m_di->coefficients[i] -
                                      m_plain->coefficients[i]);
        if (diff > 1e-10) {
            ++failures;
            std::fprintf(stderr,
                         "  FAIL coeff[%lu] di=%.17g plain=%.17g\n",
                         static_cast<unsigned long>(i),
                         m_di->coefficients[i], m_plain->coefficients[i]);
        }
    }
}

TEST(di_pls_phase13, lambda_positive_reduces_diff_projection) {
    std::vector<double> Xs = {
        0.10, 0.50, 0.90, 0.20,
        0.30, 0.40, 0.80, 0.60,
        0.60, 0.10, 0.20, 0.70,
        0.40, 0.30, 0.60, 0.90,
        0.80, 0.20, 0.10, 0.50,
        0.20, 0.90, 0.40, 0.10,
        0.70, 0.60, 0.30, 0.20,
        0.50, 0.80, 0.70, 0.80,
    };
    std::vector<double> Ys = {1.1, 0.7, 0.4, 1.0, 0.9, 0.6, 1.2, 1.3};
    std::vector<double> Xt = Xs;
    for (std::size_t row = 0; row < 8; ++row) Xt[row * 4] += 0.5;
    p4a_matrix_view_t Xs_view = make_view(Xs, 8, 4);
    p4a_matrix_view_t Ys_view = make_view(Ys, 8, 1);
    p4a_matrix_view_t Xt_view = make_view(Xt, 8, 4);

    ::pls4all::core::Context ctx;
    ::pls4all::core::Config cfg;
    cfg.algorithm = P4A_ALGO_PLS_REGRESSION;
    cfg.solver = P4A_SOLVER_SIMPLS;
    cfg.deflation = P4A_DEFLATION_REGRESSION;
    cfg.n_components = 1;
    cfg.center_x = 1;
    cfg.scale_x = 0;
    cfg.center_y = 1;
    cfg.scale_y = 0;

    std::unique_ptr<::pls4all::core::Model> m_low;
    std::unique_ptr<::pls4all::core::Model> m_high;
    cfg.di_lambda = 0.0;
    CHECK_EQ(::pls4all::core::fit_di_pls(ctx, cfg, Xs_view, Ys_view, Xt_view, m_low),
             P4A_OK);
    cfg.di_lambda = 100.0;
    CHECK_EQ(::pls4all::core::fit_di_pls(ctx, cfg, Xs_view, Ys_view, Xt_view, m_high),
             P4A_OK);

    // Compute source-target diff in the centered space.
    std::vector<double> diff(4, 0.0);
    for (std::size_t feature = 0; feature < 4; ++feature) {
        double mean_s = 0.0, mean_t = 0.0;
        for (std::size_t row = 0; row < 8; ++row) {
            mean_s += Xs[row * 4 + feature];
            mean_t += Xt[row * 4 + feature];
        }
        diff[feature] = (mean_s - mean_t) / 8.0;
    }

    auto projection = [&](const std::vector<double>& w) {
        double sum = 0.0;
        for (std::size_t f = 0; f < 4; ++f) sum += w[f] * diff[f];
        return sum;
    };
    std::vector<double> w_low(4), w_high(4);
    for (std::size_t f = 0; f < 4; ++f) {
        w_low[f] = m_low->weights_w[f];
        w_high[f] = m_high->weights_w[f];
    }
    const double proj_low = std::fabs(projection(w_low));
    const double proj_high = std::fabs(projection(w_high));
    if (!(proj_high < proj_low)) {
        ++failures;
        std::fprintf(stderr,
                     "  FAIL DI lambda=100 should reduce diff projection: "
                     "low=%.17g high=%.17g\n",
                     proj_low, proj_high);
    }
}

TEST(di_pls_phase13, rejects_invalid_inputs) {
    std::vector<double> Xs = {0.1, 0.2, 0.3, 0.4};
    std::vector<double> Ys = {0.5, 0.6};
    std::vector<double> Xt_bad = {0.1};  // wrong cols
    p4a_matrix_view_t Xs_view = make_view(Xs, 2, 2);
    p4a_matrix_view_t Ys_view = make_view(Ys, 2, 1);
    p4a_matrix_view_t Xt_view = make_view(Xt_bad, 1, 1);

    ::pls4all::core::Context ctx;
    ::pls4all::core::Config cfg;
    cfg.algorithm = P4A_ALGO_PLS_REGRESSION;
    cfg.solver = P4A_SOLVER_SIMPLS;
    cfg.deflation = P4A_DEFLATION_REGRESSION;
    cfg.n_components = 1;

    std::unique_ptr<::pls4all::core::Model> m;
    CHECK_EQ(::pls4all::core::fit_di_pls(ctx, cfg, Xs_view, Ys_view, Xt_view, m),
             P4A_ERR_SHAPE_MISMATCH);

    cfg.di_lambda = -1.0;
    p4a_matrix_view_t Xt_ok = make_view(Xs, 2, 2);
    CHECK_EQ(::pls4all::core::fit_di_pls(ctx, cfg, Xs_view, Ys_view, Xt_ok, m),
             P4A_ERR_INVALID_ARGUMENT);
}
