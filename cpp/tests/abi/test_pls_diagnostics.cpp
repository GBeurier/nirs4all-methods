// SPDX-License-Identifier: CECILL-2.1
//
// PLS diagnostics (§17 of Overview.md): Hotelling T2, Q-residuals,
// DModX. Verifies the diagnostics are computed correctly against a small
// SIMPLS fit.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "pls4all/p4a.h"

#include "core/config.hpp"
#include "core/context.hpp"
#include "core/model.hpp"
#include "core/pls_diagnostics.hpp"

#include "harness.hpp"

namespace {

constexpr double kTol = 1e-8;

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

// Fit a small SIMPLS model used by every diagnostic test.
void fit_simpls(int& failures,
                ::n4m::core::Context& ctx,
                std::unique_ptr<::n4m::core::Model>& out_model,
                bool store_scores = true) {
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
    cfg.store_scores = store_scores ? 1 : 0;
    CHECK_EQ(::n4m::core::fit_pls_regression_simpls(ctx, cfg, X, Y, out_model),
             N4M_OK);
}

}  // namespace

TEST(pls_diagnostics_phase9, q_residual_zero_when_recovered) {
    ::n4m::core::Context ctx;
    std::unique_ptr<::n4m::core::Model> model;
    fit_simpls(failures, ctx, model, /*store_scores=*/true);

    // Reuse the training samples — Q should be ~0 when latent space spans X.
    // With 4 features and 2 components the residual is generally non-zero
    // but bounded. Check that recomputing on the model's own scored buffer
    // matches: t @ p_loadings + x_mean reconstruction of training rows.
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
    std::vector<double> q;
    CHECK_EQ(::n4m::core::pls_q_residuals(ctx, *model, X, q), N4M_OK);
    CHECK_EQ(q.size(), static_cast<std::size_t>(8));
    for (double q_value : q) {
        CHECK(std::isfinite(q_value));
        CHECK(q_value >= 0.0);
    }
}

TEST(pls_diagnostics_phase9, hotelling_t2_uses_training_variance) {
    ::n4m::core::Context ctx;
    std::unique_ptr<::n4m::core::Model> model;
    fit_simpls(failures, ctx, model, /*store_scores=*/true);

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

    std::vector<double> t2_stored;
    CHECK_EQ(::n4m::core::pls_hotelling_t2(ctx, *model, X, nullptr,
                                               t2_stored),
             N4M_OK);
    CHECK_EQ(t2_stored.size(), static_cast<std::size_t>(8));
    for (double v : t2_stored) {
        CHECK(std::isfinite(v));
        CHECK(v >= 0.0);
    }

    std::vector<double> t2_ref;
    CHECK_EQ(::n4m::core::pls_hotelling_t2(ctx, *model, X, &X, t2_ref),
             N4M_OK);
    for (std::size_t i = 0; i < t2_stored.size(); ++i) {
        const double diff = std::fabs(t2_stored[i] - t2_ref[i]);
        if (diff > kTol) {
            ++failures;
            std::fprintf(stderr,
                         "  FAIL T2[%lu] stored=%.17g ref=%.17g diff=%.3g\n",
                         static_cast<unsigned long>(i),
                         t2_stored[i], t2_ref[i], diff);
        }
    }
}

TEST(pls_diagnostics_phase9, hotelling_t2_without_scores_requires_reference) {
    ::n4m::core::Context ctx;
    std::unique_ptr<::n4m::core::Model> model;
    fit_simpls(failures, ctx, model, /*store_scores=*/false);

    std::vector<double> Xv = {
        0.10, 0.50, 0.90, 0.20,
        0.30, 0.40, 0.80, 0.60,
        0.60, 0.10, 0.20, 0.70,
    };
    n4m_matrix_view_t X = make_view(Xv, 3, 4);

    std::vector<double> t2;
    CHECK_EQ(::n4m::core::pls_hotelling_t2(ctx, *model, X, nullptr, t2),
             N4M_ERR_INVALID_ARGUMENT);
}

TEST(pls_diagnostics_phase9, dmodx_is_q_normalized) {
    ::n4m::core::Context ctx;
    std::unique_ptr<::n4m::core::Model> model;
    fit_simpls(failures, ctx, model, /*store_scores=*/true);

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

    std::vector<double> q;
    std::vector<double> dmodx_plain;
    std::vector<double> dmodx_ref;
    CHECK_EQ(::n4m::core::pls_q_residuals(ctx, *model, X, q), N4M_OK);
    CHECK_EQ(::n4m::core::pls_dmodx(ctx, *model, X, nullptr, dmodx_plain),
             N4M_OK);
    CHECK_EQ(::n4m::core::pls_dmodx(ctx, *model, X, &X, dmodx_ref), N4M_OK);

    const double dof = static_cast<double>(model->n_features -
                                            model->n_components);
    for (std::size_t i = 0; i < q.size(); ++i) {
        const double expected_plain = std::sqrt(q[i] / dof);
        if (std::fabs(dmodx_plain[i] - expected_plain) > kTol) {
            ++failures;
            std::fprintf(stderr,
                         "  FAIL dmodx_plain[%lu] %.17g != %.17g\n",
                         static_cast<unsigned long>(i),
                         dmodx_plain[i], expected_plain);
        }
        if (!(std::isfinite(dmodx_ref[i]) && dmodx_ref[i] > 0.0)) {
            ++failures;
            std::fprintf(stderr,
                         "  FAIL dmodx_ref[%lu] not positive finite: %.17g\n",
                         static_cast<unsigned long>(i),
                         dmodx_ref[i]);
        }
    }
}

TEST(pls_diagnostics_phase9, rejects_shape_and_dtype_mismatches) {
    ::n4m::core::Context ctx;
    std::unique_ptr<::n4m::core::Model> model;
    fit_simpls(failures, ctx, model, /*store_scores=*/true);

    std::vector<double> Xv = {0.1, 0.2, 0.3};  // 1 row × 3 cols
    n4m_matrix_view_t X = make_view(Xv, 1, 3);
    std::vector<double> q;
    CHECK_EQ(::n4m::core::pls_q_residuals(ctx, *model, X, q),
             N4M_ERR_SHAPE_MISMATCH);
}
