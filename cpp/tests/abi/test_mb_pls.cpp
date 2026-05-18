// SPDX-License-Identifier: CECILL-2.1

#include "core/mb_pls.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "fixtures/mb_pls_fixtures.hpp"
#include "harness.hpp"

namespace {

constexpr double kAbsTol = 1e-8;
constexpr double kRelTol = 1e-8;

p4a_matrix_view_t matrix_view(const ::pls4all::test::fixtures::MatrixRef& ref) {
    p4a_matrix_view_t view{};
    view.data = const_cast<double*>(ref.values);
    view.rows = ref.rows;
    view.cols = ref.cols;
    view.row_stride = ref.cols > 0 ? ref.cols : 1;
    view.col_stride = 1;
    view.dtype = P4A_DTYPE_F64;
    return view;
}

void check_close_values(int& failures,
                        const char* label,
                        const std::vector<double>& actual,
                        const ::pls4all::test::fixtures::MatrixRef& expected) {
    if (actual.size() != expected.size) {
        ++failures;
        std::fprintf(stderr,
                     "  FAIL %s size: actual %lu expected %lu\n",
                     label,
                     static_cast<unsigned long>(actual.size()),
                     static_cast<unsigned long>(expected.size));
        return;
    }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double diff = std::fabs(actual[i] - expected.values[i]);
        const double scale = std::max(std::max(std::fabs(actual[i]), std::fabs(expected.values[i])), 1.0);
        if (diff > kAbsTol && diff > kRelTol * scale) {
            ++failures;
            std::fprintf(stderr,
                         "  FAIL %s[%lu]: actual %.17g expected %.17g diff %.3g\n",
                         label,
                         static_cast<unsigned long>(i),
                         actual[i],
                         expected.values[i],
                         diff);
            return;
        }
    }
}

void check_fixture(int& failures,
                   const ::pls4all::test::fixtures::MbPlsFixture& fixture) {
    ::pls4all::core::Context ctx;
    ::pls4all::core::Config cfg;
    cfg.algorithm = P4A_ALGO_PLS_REGRESSION;
    cfg.solver = P4A_SOLVER_NIPALS;
    cfg.deflation = P4A_DEFLATION_REGRESSION;
    cfg.n_components = fixture.n_components;

    p4a_matrix_view_t X = matrix_view(fixture.X);
    p4a_matrix_view_t Y = matrix_view(fixture.Y);
    ::pls4all::core::MbPlsResult result;
    CHECK_EQ(::pls4all::core::fit_predict_mb_pls(ctx,
                                                 cfg,
                                                 X,
                                                 Y,
                                                 fixture.block_sizes.values,
                                                 fixture.block_sizes.size,
                                                 result),
             P4A_OK);
    CHECK_EQ(result.n_samples, fixture.X.rows);
    CHECK_EQ(result.n_features, fixture.X.cols);
    CHECK_EQ(result.n_targets, fixture.Y.cols);
    CHECK_EQ(result.n_components, fixture.n_components);
    CHECK_EQ(result.n_blocks, static_cast<std::int32_t>(fixture.block_sizes.size));
    check_close_values(failures, "mb_pls_predictions", result.predictions, fixture.predictions);
    check_close_values(failures, "mb_pls_coefficients", result.coefficients, fixture.coefficients);
    check_close_values(failures, "mb_pls_intercept", result.intercept, fixture.intercept);
    check_close_values(failures, "mb_pls_x_mean", result.x_mean, fixture.x_mean);
    check_close_values(failures, "mb_pls_x_scale", result.x_scale, fixture.x_scale);
    check_close_values(failures, "mb_pls_block_weights", result.block_weights, fixture.block_weights);
}

}  // namespace

TEST(mb_pls_phase4r, generated_fixture_matches_python_reference) {
    for (const auto& fixture : ::pls4all::test::fixtures::kMbPlsFixtures) {
        check_fixture(failures, fixture);
    }
}

TEST(mb_pls_phase4r, rejects_invalid_block_layouts) {
    ::pls4all::core::Context ctx;
    ::pls4all::core::Config cfg;
    cfg.n_components = 1;
    double x_values[] = {
        0.0, 0.1, 0.2,
        0.3, 0.4, 0.5,
        0.6, 0.7, 0.8,
    };
    double y_values[] = {
        0.0,
        0.3,
        0.6,
    };
    p4a_matrix_view_t X{};
    p4a_matrix_view_t Y{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X, x_values, 3, 3, P4A_DTYPE_F64), P4A_OK);
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&Y, y_values, 3, 1, P4A_DTYPE_F64), P4A_OK);

    ::pls4all::core::MbPlsResult result;
    CHECK_EQ(::pls4all::core::fit_predict_mb_pls(ctx, cfg, X, Y, nullptr, 0, result),
             P4A_ERR_INVALID_ARGUMENT);

    const std::int64_t zero_block[] = {1, 0, 2};
    CHECK_EQ(::pls4all::core::fit_predict_mb_pls(ctx, cfg, X, Y, zero_block, 3, result),
             P4A_ERR_INVALID_ARGUMENT);

    const std::int64_t wrong_sum[] = {1, 1};
    CHECK_EQ(::pls4all::core::fit_predict_mb_pls(ctx, cfg, X, Y, wrong_sum, 2, result),
             P4A_ERR_SHAPE_MISMATCH);

    p4a_matrix_view_t mismatched_y = Y;
    mismatched_y.rows = 2;
    const std::int64_t valid_blocks[] = {1, 2};
    CHECK_EQ(::pls4all::core::fit_predict_mb_pls(ctx, cfg, X, mismatched_y, valid_blocks, 2, result),
             P4A_ERR_SHAPE_MISMATCH);
}
