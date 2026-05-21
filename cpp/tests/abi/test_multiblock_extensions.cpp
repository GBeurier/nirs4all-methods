// SPDX-License-Identifier: CECILL-2.1
//
// Multi-block PLS extensions (§8): O2PLS, SO-PLS, OnPLS, ROSA.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

#include "core/config.hpp"
#include "core/context.hpp"
#include "core/multiblock_extensions.hpp"

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

TEST(o2pls_phase16, fits_predictive_orthogonal_components) {
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
    std::vector<double> Yv = {
        1.1, 0.5,
        0.7, 0.4,
        0.4, 0.9,
        1.0, 0.6,
        0.9, 0.5,
        0.6, 0.7,
        1.2, 0.4,
        1.3, 0.5,
    };
    n4m_matrix_view_t X = make_view(Xv, 8, 4);
    n4m_matrix_view_t Y = make_view(Yv, 8, 2);
    ::n4m::core::Context ctx;
    ::n4m::core::Config cfg;
    ::n4m::core::O2PlsResult result;
    CHECK_EQ(::n4m::core::fit_o2pls(ctx, cfg, X, Y, 1, 1, 1, result),
             N4M_OK);
    CHECK_EQ(result.n_predictive, 1);
    CHECK_EQ(result.n_x_orthogonal, 1);
    CHECK_EQ(result.n_y_orthogonal, 1);
    CHECK_EQ(result.coefficients.size(), static_cast<std::size_t>(4 * 2));
    for (double v : result.coefficients) CHECK(std::isfinite(v));
}

TEST(o2pls_phase16, rejects_invalid_inputs) {
    std::vector<double> Xv = {0.1, 0.2};
    std::vector<double> Yv = {1.0};
    n4m_matrix_view_t X = make_view(Xv, 2, 1);
    n4m_matrix_view_t Y = make_view(Yv, 1, 1);
    ::n4m::core::Context ctx;
    ::n4m::core::Config cfg;
    ::n4m::core::O2PlsResult result;
    CHECK_EQ(::n4m::core::fit_o2pls(ctx, cfg, X, Y, 1, 0, 0, result),
             N4M_ERR_SHAPE_MISMATCH);

    Y = make_view(Yv, 2, 1);
    // n_predictive = 0 → invalid
    Yv = {1.0, 0.5};
    Y = make_view(Yv, 2, 1);
    CHECK_EQ(::n4m::core::fit_o2pls(ctx, cfg, X, Y, 0, 0, 0, result),
             N4M_ERR_INVALID_ARGUMENT);
}

TEST(so_pls_phase17, sequential_predicts_finite_values) {
    std::vector<double> Xa = {
        0.1, 0.5, 0.9, 0.2,
        0.3, 0.4, 0.8, 0.6,
        0.6, 0.1, 0.2, 0.7,
        0.4, 0.3, 0.6, 0.9,
        0.8, 0.2, 0.1, 0.5,
        0.2, 0.9, 0.4, 0.1,
        0.7, 0.6, 0.3, 0.2,
        0.5, 0.8, 0.7, 0.8,
    };
    std::vector<double> Xb = {
        0.7, 0.3,
        0.5, 0.4,
        0.4, 0.6,
        0.9, 0.2,
        0.1, 0.7,
        0.8, 0.3,
        0.4, 0.9,
        0.6, 0.5,
    };
    std::vector<double> Yv = {1.1, 0.7, 0.4, 1.0, 0.9, 0.6, 1.2, 1.3};
    n4m_matrix_view_t Xa_view = make_view(Xa, 8, 4);
    n4m_matrix_view_t Xb_view = make_view(Xb, 8, 2);
    n4m_matrix_view_t Y_view = make_view(Yv, 8, 1);

    ::n4m::core::Context ctx;
    ::n4m::core::Config cfg;
    ::n4m::core::SoPlsResult result;
    std::vector<n4m_matrix_view_t> blocks = {Xa_view, Xb_view};
    std::vector<std::int32_t> ncomps = {2, 1};
    CHECK_EQ(::n4m::core::fit_so_pls(ctx, cfg, blocks, Y_view, ncomps,
                                          result),
             N4M_OK);
    CHECK_EQ(result.n_blocks, 2);
    CHECK_EQ(result.predictions.size(), static_cast<std::size_t>(8));
    for (double v : result.predictions) CHECK(std::isfinite(v));
}

TEST(on_pls_phase18, extracts_joint_and_unique_components) {
    std::vector<double> Xa = {
        0.10, 0.50, 0.90, 0.20,
        0.30, 0.40, 0.80, 0.60,
        0.60, 0.10, 0.20, 0.70,
        0.40, 0.30, 0.60, 0.90,
        0.80, 0.20, 0.10, 0.50,
        0.20, 0.90, 0.40, 0.10,
    };
    std::vector<double> Xb = {
        0.7, 0.3, 0.5,
        0.5, 0.4, 0.9,
        0.4, 0.6, 0.2,
        0.9, 0.2, 0.7,
        0.1, 0.7, 0.4,
        0.8, 0.3, 0.6,
    };
    n4m_matrix_view_t Xa_view = make_view(Xa, 6, 4);
    n4m_matrix_view_t Xb_view = make_view(Xb, 6, 3);

    ::n4m::core::Context ctx;
    ::n4m::core::Config cfg;
    ::n4m::core::OnPlsResult result;
    std::vector<n4m_matrix_view_t> blocks = {Xa_view, Xb_view};
    std::vector<std::int32_t> unique = {1, 1};
    CHECK_EQ(::n4m::core::fit_on_pls(ctx, cfg, blocks, 1, unique, result),
             N4M_OK);
    CHECK_EQ(result.n_blocks, 2);
    CHECK_EQ(result.n_joint, 1);
    CHECK_EQ(result.joint_loadings_per_block.size(),
             static_cast<std::size_t>(2));
    CHECK_EQ(result.joint_loadings_per_block[0].size(),
             static_cast<std::size_t>(4));
    CHECK_EQ(result.joint_loadings_per_block[1].size(),
             static_cast<std::size_t>(3));
}

TEST(rosa_phase19, selects_best_block_per_component) {
    std::vector<double> Xa = {
        0.10, 0.50, 0.90, 0.20,
        0.30, 0.40, 0.80, 0.60,
        0.60, 0.10, 0.20, 0.70,
        0.40, 0.30, 0.60, 0.90,
        0.80, 0.20, 0.10, 0.50,
        0.20, 0.90, 0.40, 0.10,
        0.70, 0.60, 0.30, 0.20,
        0.50, 0.80, 0.70, 0.80,
    };
    std::vector<double> Xb = {
        0.7, 0.3,
        0.5, 0.4,
        0.4, 0.6,
        0.9, 0.2,
        0.1, 0.7,
        0.8, 0.3,
        0.4, 0.9,
        0.6, 0.5,
    };
    std::vector<double> Yv = {1.1, 0.7, 0.4, 1.0, 0.9, 0.6, 1.2, 1.3};
    n4m_matrix_view_t Xa_view = make_view(Xa, 8, 4);
    n4m_matrix_view_t Xb_view = make_view(Xb, 8, 2);
    n4m_matrix_view_t Y_view = make_view(Yv, 8, 1);

    ::n4m::core::Context ctx;
    ::n4m::core::Config cfg;
    ::n4m::core::RosaResult result;
    std::vector<n4m_matrix_view_t> blocks = {Xa_view, Xb_view};
    CHECK_EQ(::n4m::core::fit_rosa(ctx, cfg, blocks, Y_view, 3, result),
             N4M_OK);
    CHECK_EQ(result.n_components, 3);
    CHECK_EQ(result.selected_block_per_component.size(),
             static_cast<std::size_t>(3));
    CHECK_EQ(result.predictions.size(), static_cast<std::size_t>(8));
    for (double v : result.predictions) CHECK(std::isfinite(v));
}
