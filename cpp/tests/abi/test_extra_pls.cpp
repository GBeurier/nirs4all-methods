// SPDX-License-Identifier: CECILL-2.1
//
// Catalogue smoke tests for the §7/§13/§20 extras shipped in
// cpp/src/core/extra_pls.cpp. Each method is exercised on a small
// synthetic dataset; outputs must be finite and shapes correct.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

#include "core/config.hpp"
#include "core/context.hpp"
#include "core/extra_pls.hpp"

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

std::vector<double> small_X() {
    return {
        0.10, 0.50, 0.90, 0.20,
        0.30, 0.40, 0.80, 0.60,
        0.60, 0.10, 0.20, 0.70,
        0.40, 0.30, 0.60, 0.90,
        0.80, 0.20, 0.10, 0.50,
        0.20, 0.90, 0.40, 0.10,
        0.70, 0.60, 0.30, 0.20,
        0.50, 0.80, 0.70, 0.80,
    };
}

std::vector<double> small_Y() {
    return {1.1, 0.7, 0.4, 1.0, 0.9, 0.6, 1.2, 1.3};
}

::pls4all::core::Config base_cfg(std::int32_t n_components = 2) {
    ::pls4all::core::Config cfg;
    cfg.algorithm = P4A_ALGO_PLS_REGRESSION;
    cfg.solver = P4A_SOLVER_SIMPLS;
    cfg.deflation = P4A_DEFLATION_REGRESSION;
    cfg.n_components = n_components;
    cfg.center_x = 1;
    cfg.center_y = 1;
    return cfg;
}

}  // namespace

TEST(extra_pls_phase20_sparse_da, fits_and_returns_class_priors) {
    auto Xv = small_X();
    std::vector<std::int32_t> labels = {0, 1, 0, 1, 2, 1, 0, 2};
    p4a_matrix_view_t X = make_view(Xv, 8, 4);
    ::pls4all::core::Context ctx;
    ::pls4all::core::Config cfg = base_cfg(2);
    cfg.sparsity_lambda = 0.05;
    ::pls4all::core::SparsePlsDaResult res;
    CHECK_EQ(::pls4all::core::fit_sparse_pls_da(ctx, cfg, X, labels, res),
             P4A_OK);
    CHECK_EQ(res.n_classes, 3);
    CHECK_EQ(res.class_priors.size(), static_cast<std::size_t>(3));
    for (double v : res.class_priors) CHECK(v > 0.0);
    for (double v : res.coefficients) CHECK(std::isfinite(v));
}

TEST(extra_pls_phase21_group_sparse, fits_with_groups) {
    auto Xv = small_X();
    auto Yv = small_Y();
    p4a_matrix_view_t X = make_view(Xv, 8, 4);
    p4a_matrix_view_t Y = make_view(Yv, 8, 1);
    std::vector<std::int32_t> groups = {0, 0, 1, 1};
    ::pls4all::core::Context ctx;
    ::pls4all::core::Config cfg = base_cfg(2);
    ::pls4all::core::GroupSparsePlsResult res;
    CHECK_EQ(::pls4all::core::fit_group_sparse_pls(ctx, cfg, X, Y, groups,
                                                    0.1, res),
             P4A_OK);
    CHECK_EQ(res.n_groups, 2);
    for (double v : res.coefficients) CHECK(std::isfinite(v));
}

TEST(extra_pls_phase21_fused_sparse, smooths_coefficients) {
    auto Xv = small_X();
    auto Yv = small_Y();
    p4a_matrix_view_t X = make_view(Xv, 8, 4);
    p4a_matrix_view_t Y = make_view(Yv, 8, 1);
    ::pls4all::core::Context ctx;
    ::pls4all::core::Config cfg = base_cfg(2);
    ::pls4all::core::GroupSparsePlsResult res;
    CHECK_EQ(::pls4all::core::fit_fused_sparse_pls(ctx, cfg, X, Y, 0.05, 1.0,
                                                    res),
             P4A_OK);
    for (double v : res.coefficients) CHECK(std::isfinite(v));
}

TEST(extra_pls_phase25_cppls, interpolates_pls_and_ols) {
    auto Xv = small_X();
    auto Yv = small_Y();
    p4a_matrix_view_t X = make_view(Xv, 8, 4);
    p4a_matrix_view_t Y = make_view(Yv, 8, 1);
    ::pls4all::core::Context ctx;
    ::pls4all::core::Config cfg = base_cfg(2);
    ::pls4all::core::CpplsResult res;
    CHECK_EQ(::pls4all::core::fit_cppls(ctx, cfg, X, Y, 0.5, res), P4A_OK);
    CHECK_EQ(res.gamma, 0.5);
    for (double v : res.coefficients) CHECK(std::isfinite(v));
}

TEST(extra_pls_phase26_weighted, accepts_per_sample_weights) {
    auto Xv = small_X();
    auto Yv = small_Y();
    p4a_matrix_view_t X = make_view(Xv, 8, 4);
    p4a_matrix_view_t Y = make_view(Yv, 8, 1);
    std::vector<double> w = {1.0, 2.0, 0.5, 1.5, 1.0, 0.8, 1.2, 1.0};
    ::pls4all::core::Context ctx;
    ::pls4all::core::Config cfg = base_cfg(2);
    ::pls4all::core::WeightedPlsResult res;
    CHECK_EQ(::pls4all::core::fit_weighted_pls(ctx, cfg, X, Y, w, res),
             P4A_OK);
    for (double v : res.coefficients) CHECK(std::isfinite(v));
}

TEST(extra_pls_phase26_robust, runs_irls) {
    auto Xv = small_X();
    auto Yv = small_Y();
    p4a_matrix_view_t X = make_view(Xv, 8, 4);
    p4a_matrix_view_t Y = make_view(Yv, 8, 1);
    ::pls4all::core::Context ctx;
    ::pls4all::core::Config cfg = base_cfg(2);
    ::pls4all::core::WeightedPlsResult res;
    CHECK_EQ(::pls4all::core::fit_robust_pls(ctx, cfg, X, Y, 1.345, 3, res),
             P4A_OK);
    for (double v : res.coefficients) CHECK(std::isfinite(v));
}

TEST(extra_pls_phase26_ridge, accepts_ridge_lambda) {
    auto Xv = small_X();
    auto Yv = small_Y();
    p4a_matrix_view_t X = make_view(Xv, 8, 4);
    p4a_matrix_view_t Y = make_view(Yv, 8, 1);
    ::pls4all::core::Context ctx;
    ::pls4all::core::Config cfg = base_cfg(2);
    ::pls4all::core::WeightedPlsResult res;
    CHECK_EQ(::pls4all::core::fit_ridge_pls(ctx, cfg, X, Y, 0.1, res), P4A_OK);
    for (double v : res.coefficients) CHECK(std::isfinite(v));
}

TEST(extra_pls_phase26_continuum, interpolates_with_tau) {
    auto Xv = small_X();
    auto Yv = small_Y();
    p4a_matrix_view_t X = make_view(Xv, 8, 4);
    p4a_matrix_view_t Y = make_view(Yv, 8, 1);
    ::pls4all::core::Context ctx;
    ::pls4all::core::Config cfg = base_cfg(2);
    ::pls4all::core::WeightedPlsResult res;
    CHECK_EQ(::pls4all::core::fit_continuum_regression(ctx, cfg, X, Y, 0.5,
                                                       res),
             P4A_OK);
    for (double v : res.coefficients) CHECK(std::isfinite(v));
}

TEST(extra_pls_phase27_glm, fits_logistic_target) {
    auto Xv = small_X();
    // 3-class one-hot
    std::vector<double> Yv = {
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        1, 0, 0,
        0, 1, 0,
        1, 0, 0,
        0, 0, 1,
        0, 1, 0,
    };
    p4a_matrix_view_t X = make_view(Xv, 8, 4);
    p4a_matrix_view_t Y = make_view(Yv, 8, 3);
    ::pls4all::core::Context ctx;
    ::pls4all::core::Config cfg = base_cfg(2);
    ::pls4all::core::PlsGlmResult res;
    CHECK_EQ(::pls4all::core::fit_pls_glm(ctx, cfg, X, Y, false, res),
             P4A_OK);
    CHECK_EQ(res.n_classes, 3);
    for (double v : res.coefficients) CHECK(std::isfinite(v));
}

TEST(extra_pls_phase27_qda, fits_class_covariances) {
    auto Xv = small_X();
    std::vector<std::int32_t> labels = {0, 1, 0, 1, 2, 1, 0, 2};
    p4a_matrix_view_t X = make_view(Xv, 8, 4);
    ::pls4all::core::Context ctx;
    ::pls4all::core::Config cfg = base_cfg(2);
    ::pls4all::core::PlsQdaResult res;
    CHECK_EQ(::pls4all::core::fit_pls_qda(ctx, cfg, X, labels, res), P4A_OK);
    CHECK_EQ(res.n_classes, 3);
    CHECK_EQ(res.class_covariances.size(),
             static_cast<std::size_t>(3 * 2 * 2));
}

TEST(extra_pls_phase27_cox, computes_breslow_baseline) {
    auto Xv = small_X();
    std::vector<double> times = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    std::vector<std::int32_t> events = {1, 0, 1, 1, 0, 1, 0, 1};
    p4a_matrix_view_t X = make_view(Xv, 8, 4);
    ::pls4all::core::Context ctx;
    ::pls4all::core::Config cfg = base_cfg(2);
    ::pls4all::core::PlsCoxResult res;
    CHECK_EQ(::pls4all::core::fit_pls_cox(ctx, cfg, X, times, events, res),
             P4A_OK);
    CHECK_EQ(res.n_features, 4);
    CHECK(res.baseline_hazard.size() > 0);
}

TEST(extra_pls_phase28_pds_ds, learns_transformation) {
    auto Xs = small_X();
    auto Xt = small_X();
    // Shift target spectra to simulate an instrument drift.
    for (std::size_t i = 0; i < Xt.size(); i += 4) {
        Xt[i] *= 0.95;
        Xt[i + 1] *= 1.05;
    }
    p4a_matrix_view_t Xs_view = make_view(Xs, 8, 4);
    p4a_matrix_view_t Xt_view = make_view(Xt, 8, 4);
    ::pls4all::core::Context ctx;
    ::pls4all::core::PdsResult pds;
    CHECK_EQ(::pls4all::core::fit_pds(ctx, Xs_view, Xt_view, 1, pds), P4A_OK);
    for (double v : pds.transformation) CHECK(std::isfinite(v));
    ::pls4all::core::DsResult ds;
    CHECK_EQ(::pls4all::core::fit_ds(ctx, Xs_view, Xt_view, ds), P4A_OK);
    for (double v : ds.transformation) CHECK(std::isfinite(v));
}

TEST(extra_pls_phase28_mir_pls, predicts_inverse_regression) {
    auto Xv = small_X();
    auto Yv = small_Y();
    p4a_matrix_view_t X = make_view(Xv, 8, 4);
    p4a_matrix_view_t Y = make_view(Yv, 8, 1);
    ::pls4all::core::Context ctx;
    ::pls4all::core::Config cfg = base_cfg(1);
    ::pls4all::core::MirPlsResult res;
    CHECK_EQ(::pls4all::core::fit_mir_pls(ctx, cfg, X, Y, res), P4A_OK);
    for (double v : res.coefficients) CHECK(std::isfinite(v));
}

TEST(extra_pls_phase28_missing, imputes_nans_with_means) {
    std::vector<double> Xv = {
        0.10, std::nan(""), 0.90, 0.20,
        0.30, 0.40, 0.80, 0.60,
        std::nan(""), 0.10, 0.20, 0.70,
        0.40, 0.30, 0.60, 0.90,
        0.80, 0.20, 0.10, 0.50,
        0.20, 0.90, 0.40, std::nan(""),
        0.70, 0.60, 0.30, 0.20,
        0.50, 0.80, 0.70, 0.80,
    };
    auto Yv = small_Y();
    p4a_matrix_view_t X = make_view(Xv, 8, 4);
    p4a_matrix_view_t Y = make_view(Yv, 8, 1);
    ::pls4all::core::Context ctx;
    ::pls4all::core::Config cfg = base_cfg(2);
    ::pls4all::core::WeightedPlsResult res;
    CHECK_EQ(::pls4all::core::fit_missing_aware_nipals(ctx, cfg, X, Y, res),
             P4A_OK);
    for (double v : res.coefficients) CHECK(std::isfinite(v));
}

TEST(extra_pls_phase29_press, selects_component_count) {
    auto Xv = small_X();
    auto Yv = small_Y();
    p4a_matrix_view_t X = make_view(Xv, 8, 4);
    p4a_matrix_view_t Y = make_view(Yv, 8, 1);
    ::pls4all::core::Context ctx;
    ::pls4all::core::Config cfg = base_cfg(1);
    ::pls4all::core::ApproximatePressResult res;
    CHECK_EQ(::pls4all::core::approximate_press(ctx, cfg, X, Y, 3, res),
             P4A_OK);
    CHECK_EQ(res.press_per_component.size(), static_cast<std::size_t>(3));
    CHECK(res.selected_n_components >= 1);
    CHECK(res.selected_n_components <= 3);
}

TEST(extra_pls_phase30_bagging, averages_base_learners) {
    auto Xv = small_X();
    auto Yv = small_Y();
    p4a_matrix_view_t X = make_view(Xv, 8, 4);
    p4a_matrix_view_t Y = make_view(Yv, 8, 1);
    ::pls4all::core::Context ctx;
    ::pls4all::core::Config cfg = base_cfg(2);
    ::pls4all::core::EnsemblePlsResult res;
    CHECK_EQ(::pls4all::core::fit_bagging_pls(ctx, cfg, X, Y, 5, 7, res),
             P4A_OK);
    CHECK_EQ(res.n_estimators, 5);
    for (double v : res.coefficients) CHECK(std::isfinite(v));
}

TEST(extra_pls_phase30_boosting, fits_sequential_residuals) {
    auto Xv = small_X();
    auto Yv = small_Y();
    p4a_matrix_view_t X = make_view(Xv, 8, 4);
    p4a_matrix_view_t Y = make_view(Yv, 8, 1);
    ::pls4all::core::Context ctx;
    ::pls4all::core::Config cfg = base_cfg(2);
    ::pls4all::core::EnsemblePlsResult res;
    CHECK_EQ(::pls4all::core::fit_boosting_pls(ctx, cfg, X, Y, 4, 0.5, res),
             P4A_OK);
    CHECK_EQ(res.n_estimators, 4);
    for (double v : res.coefficients) CHECK(std::isfinite(v));
}

TEST(extra_pls_phase30_random_subspace, fits_feature_subsets) {
    auto Xv = small_X();
    auto Yv = small_Y();
    p4a_matrix_view_t X = make_view(Xv, 8, 4);
    p4a_matrix_view_t Y = make_view(Yv, 8, 1);
    ::pls4all::core::Context ctx;
    ::pls4all::core::Config cfg = base_cfg(2);
    ::pls4all::core::EnsemblePlsResult res;
    CHECK_EQ(::pls4all::core::fit_random_subspace_pls(ctx, cfg, X, Y, 5, 2,
                                                       11, res),
             P4A_OK);
    CHECK_EQ(res.n_estimators, 5);
    for (double v : res.coefficients) CHECK(std::isfinite(v));
}
