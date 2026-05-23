// SPDX-License-Identifier: CECILL-2.1
//
// Sparse SIMPLS (§7 of Overview.md). Covers two algorithm modes:
//   * default Chun & Keles 2010 (matches R `spls::spls`) — variable
//     selection then ordinary SIMPLS on the active set.
//   * legacy per-component absolute soft-threshold of the SIMPLS direction
//     (`cfg.sparse_simpls_legacy = 1`), retained for older bundles.
// The `lambda == 0` path bypasses sparsity in both modes and matches plain
// SIMPLS bit-for-bit.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "n4m/n4m.h"

#include "core/config.hpp"
#include "core/context.hpp"
#include "core/matrix_view.hpp"
#include "core/model.hpp"

#include "harness.hpp"

namespace {

constexpr double kAbsTol = 1e-10;

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

::n4m::core::Config make_config(double lambda) {
    ::n4m::core::Config cfg;
    cfg.algorithm = N4M_ALGO_SPARSE_PLS;
    cfg.solver = N4M_SOLVER_SIMPLS;
    cfg.deflation = N4M_DEFLATION_REGRESSION;
    cfg.n_components = 2;
    cfg.center_x = 1;
    cfg.scale_x = 0;
    cfg.center_y = 1;
    cfg.scale_y = 0;
    cfg.store_scores = 0;
    cfg.tol = 1e-8;
    cfg.max_iter = 500;
    cfg.sparsity_lambda = lambda;
    return cfg;
}

}  // namespace

TEST(sparse_pls_phase8, lambda_zero_matches_plain_simpls) {
    // 8-sample, 5-feature dataset.
    std::vector<double> Xv = {
        0.10, 0.50, 0.90, 0.20, 0.70,
        0.30, 0.40, 0.80, 0.60, 0.10,
        0.60, 0.10, 0.20, 0.70, 0.50,
        0.40, 0.30, 0.60, 0.90, 0.20,
        0.80, 0.20, 0.10, 0.50, 0.90,
        0.20, 0.90, 0.40, 0.10, 0.30,
        0.70, 0.60, 0.30, 0.20, 0.80,
        0.50, 0.80, 0.70, 0.80, 0.40,
    };
    std::vector<double> Yv = {1.1, 0.7, 0.4, 1.0, 0.9, 0.6, 1.2, 1.3};
    n4m_matrix_view_t X = make_view(Xv, 8, 5);
    n4m_matrix_view_t Y = make_view(Yv, 8, 1);

    ::n4m::core::Context ctx;
    ::n4m::core::Config cfg_sparse = make_config(0.0);
    ::n4m::core::Config cfg_plain = cfg_sparse;
    cfg_plain.algorithm = N4M_ALGO_PLS_REGRESSION;

    std::unique_ptr<::n4m::core::Model> m_sparse;
    std::unique_ptr<::n4m::core::Model> m_plain;
    CHECK_EQ(::n4m::core::fit_pls_sparse_simpls(ctx, cfg_sparse, X, Y,
                                                    m_sparse),
             N4M_OK);
    CHECK_EQ(::n4m::core::fit_pls_regression_simpls(ctx, cfg_plain, X, Y,
                                                        m_plain),
             N4M_OK);
    CHECK(m_sparse != nullptr && m_plain != nullptr);
    CHECK_EQ(m_sparse->coefficients.size(), m_plain->coefficients.size());
    for (std::size_t i = 0; i < m_sparse->coefficients.size(); ++i) {
        const double diff = std::fabs(m_sparse->coefficients[i] -
                                      m_plain->coefficients[i]);
        if (diff > kAbsTol) {
            ++failures;
            std::fprintf(stderr,
                         "  FAIL coeff[%lu]: sparse %.17g plain %.17g\n",
                         static_cast<unsigned long>(i),
                         m_sparse->coefficients[i],
                         m_plain->coefficients[i]);
        }
    }
}

TEST(sparse_pls_phase8, legacy_lambda_positive_zeroes_small_weights) {
    std::vector<double> Xv = {
        0.10, 0.50, 0.90, 0.20, 0.70,
        0.30, 0.40, 0.80, 0.60, 0.10,
        0.60, 0.10, 0.20, 0.70, 0.50,
        0.40, 0.30, 0.60, 0.90, 0.20,
        0.80, 0.20, 0.10, 0.50, 0.90,
        0.20, 0.90, 0.40, 0.10, 0.30,
        0.70, 0.60, 0.30, 0.20, 0.80,
        0.50, 0.80, 0.70, 0.80, 0.40,
    };
    std::vector<double> Yv = {1.1, 0.7, 0.4, 1.0, 0.9, 0.6, 1.2, 1.3};
    n4m_matrix_view_t X = make_view(Xv, 8, 5);
    n4m_matrix_view_t Y = make_view(Yv, 8, 1);

    ::n4m::core::Context ctx;
    ::n4m::core::Config cfg = make_config(0.15);  // moderate lambda
    cfg.n_components = 1;  // first component has enough mass to survive 0.15
    cfg.sparse_simpls_legacy = 1;  // legacy populates weights_w
    std::unique_ptr<::n4m::core::Model> model;
    CHECK_EQ(::n4m::core::fit_pls_sparse_simpls(ctx, cfg, X, Y, model),
             N4M_OK);
    CHECK(model != nullptr);
    const auto& w = model->weights_w;  // p x a row-major
    int nonzero = 0;
    for (double v : w) {
        if (std::fabs(v) > 1e-12) ++nonzero;
    }
    // Some weights should be zero with lambda=0.15; not all should be.
    CHECK(nonzero > 0);
    CHECK(nonzero < static_cast<int>(w.size()));

    // Predict: must be finite and shape-correct.
    std::vector<double> pred(8, 0.0);
    n4m_matrix_view_t pred_view = make_view(pred, 8, 1);
    CHECK_EQ(::n4m::core::predict_into(ctx, *model, X, pred_view), N4M_OK);
    for (double v : pred) {
        CHECK(std::isfinite(v));
    }
}

TEST(sparse_pls_phase8, chun_keles_lambda_positive_zeroes_inactive_coeffs) {
    std::vector<double> Xv = {
        0.10, 0.50, 0.90, 0.20, 0.70,
        0.30, 0.40, 0.80, 0.60, 0.10,
        0.60, 0.10, 0.20, 0.70, 0.50,
        0.40, 0.30, 0.60, 0.90, 0.20,
        0.80, 0.20, 0.10, 0.50, 0.90,
        0.20, 0.90, 0.40, 0.10, 0.30,
        0.70, 0.60, 0.30, 0.20, 0.80,
        0.50, 0.80, 0.70, 0.80, 0.40,
    };
    std::vector<double> Yv = {1.1, 0.7, 0.4, 1.0, 0.9, 0.6, 1.2, 1.3};
    n4m_matrix_view_t X = make_view(Xv, 8, 5);
    n4m_matrix_view_t Y = make_view(Yv, 8, 1);

    ::n4m::core::Context ctx;
    // High `eta` close to 1 keeps only the largest absolute Z entry on the
    // first iteration; for this dataset that drops at least one feature.
    ::n4m::core::Config cfg = make_config(0.95);
    cfg.n_components = 1;
    std::unique_ptr<::n4m::core::Model> model;
    CHECK_EQ(::n4m::core::fit_pls_sparse_simpls(ctx, cfg, X, Y, model),
             N4M_OK);
    CHECK(model != nullptr);

    int zero_coeffs = 0;
    for (double v : model->coefficients) {
        if (std::fabs(v) < 1e-12) ++zero_coeffs;
    }
    CHECK(zero_coeffs > 0);
    CHECK(zero_coeffs < static_cast<int>(model->coefficients.size()));

    // Predict: must be finite and shape-correct.
    std::vector<double> pred(8, 0.0);
    n4m_matrix_view_t pred_view = make_view(pred, 8, 1);
    CHECK_EQ(::n4m::core::predict_into(ctx, *model, X, pred_view), N4M_OK);
    for (double v : pred) {
        CHECK(std::isfinite(v));
    }
}

TEST(sparse_pls_phase8, legacy_lambda_too_large_returns_numerical_failure) {
    std::vector<double> Xv(40, 0.5);
    std::vector<double> Yv(8, 1.0);
    // Make sure X has non-zero variance so SIMPLS would otherwise succeed.
    for (std::size_t i = 0; i < 40; ++i) Xv[i] += 0.01 * static_cast<double>(i);
    for (std::size_t i = 0; i < 8;  ++i) Yv[i] += 0.05 * static_cast<double>(i);
    n4m_matrix_view_t X = make_view(Xv, 8, 5);
    n4m_matrix_view_t Y = make_view(Yv, 8, 1);

    ::n4m::core::Context ctx;
    ::n4m::core::Config cfg = make_config(1e9);  // wipes every weight
    cfg.sparse_simpls_legacy = 1;
    std::unique_ptr<::n4m::core::Model> model;
    const n4m_status_t status =
        ::n4m::core::fit_pls_sparse_simpls(ctx, cfg, X, Y, model);
    CHECK_EQ(status, N4M_ERR_NUMERICAL_FAILURE);
}

TEST(sparse_pls_phase8, chun_keles_rejects_eta_out_of_range) {
    std::vector<double> Xv(40, 0.5);
    std::vector<double> Yv(8, 1.0);
    for (std::size_t i = 0; i < 40; ++i) Xv[i] += 0.01 * static_cast<double>(i);
    for (std::size_t i = 0; i < 8;  ++i) Yv[i] += 0.05 * static_cast<double>(i);
    n4m_matrix_view_t X = make_view(Xv, 8, 5);
    n4m_matrix_view_t Y = make_view(Yv, 8, 1);

    ::n4m::core::Context ctx;
    ::n4m::core::Config cfg = make_config(1.5);  // > 1, illegal for Chun & Keles
    std::unique_ptr<::n4m::core::Model> model;
    CHECK_EQ(::n4m::core::fit_pls_sparse_simpls(ctx, cfg, X, Y, model),
             N4M_ERR_INVALID_ARGUMENT);
}

TEST(sparse_pls_phase8, rejects_non_simpls_solver) {
    std::vector<double> Xv = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> Yv = {0.5, 0.6};
    n4m_matrix_view_t X = make_view(Xv, 2, 2);
    n4m_matrix_view_t Y = make_view(Yv, 2, 1);

    ::n4m::core::Context ctx;
    ::n4m::core::Config cfg = make_config(0.1);
    cfg.solver = N4M_SOLVER_NIPALS;
    std::unique_ptr<::n4m::core::Model> model;
    CHECK_EQ(::n4m::core::fit_pls_sparse_simpls(ctx, cfg, X, Y, model),
             N4M_ERR_UNSUPPORTED);

    cfg.solver = N4M_SOLVER_SIMPLS;
    cfg.sparsity_lambda = -1.0;
    CHECK_EQ(::n4m::core::fit_pls_sparse_simpls(ctx, cfg, X, Y, model),
             N4M_ERR_INVALID_ARGUMENT);
}
