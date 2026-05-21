// SPDX-License-Identifier: CECILL-2.1

#include "core/cross_validation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "fixtures/cross_validation_fixtures.hpp"
#include "harness.hpp"

namespace {

constexpr double kAbsTol = 1e-8;
constexpr double kRelTol = 1e-8;

void check_close_values(int& failures,
                        const char* label,
                        const std::vector<double>& actual,
                        const ::n4m::test::fixtures::MatrixRef& expected) {
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

void check_close_metrics(int& failures,
                         const ::n4m::core::RegressionMetrics& actual,
                         const ::n4m::test::fixtures::MatrixRef& expected) {
    CHECK_EQ(expected.size, static_cast<std::size_t>(9U));
    const double values[] = {
        actual.rmse,
        actual.mae,
        actual.bias,
        actual.r2,
        actual.q2,
        actual.slope,
        actual.intercept,
        actual.rpd,
        actual.rpiq,
    };
    const std::vector<double> actual_values(values, values + 9);
    check_close_values(failures, "cv_metrics", actual_values, expected);
}

void check_index_ref(int& failures,
                     const char* label,
                     const std::vector<std::int64_t>& actual,
                     const ::n4m::test::fixtures::CvIndexRef& expected) {
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
        if (actual[i] != expected.values[i]) {
            ++failures;
            std::fprintf(stderr,
                         "  FAIL %s[%lu]: actual %lld expected %lld\n",
                         label,
                         static_cast<unsigned long>(i),
                         static_cast<long long>(actual[i]),
                         static_cast<long long>(expected.values[i]));
            return;
        }
    }
}

void check_fixture(int& failures,
                   const ::n4m::test::fixtures::CrossValidationFixture& fixture) {
    ::n4m::core::Context ctx;
    ::n4m::core::Config cfg;
    cfg.algorithm = N4M_ALGO_PLS_REGRESSION;
    cfg.solver = N4M_SOLVER_NIPALS;
    cfg.deflation = N4M_DEFLATION_REGRESSION;
    cfg.n_components = fixture.n_components;

    n4m_matrix_view_t X{};
    n4m_matrix_view_t Y{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&Y,
                                           const_cast<double*>(fixture.Y.values),
                                           fixture.Y.rows,
                                           fixture.Y.cols,
                                           N4M_DTYPE_F64),
             N4M_OK);

    ::n4m::core::ValidationPlan plan;
    CHECK_EQ(::n4m::core::make_kfold_validation_plan(ctx,
                                                         fixture.X.rows,
                                                         fixture.n_splits,
                                                         plan),
             N4M_OK);
    ::n4m::core::CrossValidationResult result;
    CHECK_EQ(::n4m::core::cross_validate_regression(ctx, cfg, X, Y, plan, result), N4M_OK);
    CHECK_EQ(result.n_samples, fixture.X.rows);
    CHECK_EQ(result.n_targets, fixture.Y.cols);
    CHECK_EQ(result.n_folds, static_cast<std::int64_t>(fixture.n_splits));
    check_close_values(failures, "cv_predictions", result.predictions, fixture.predictions);
    check_close_metrics(failures, result.metrics, fixture.metrics);
    check_index_ref(failures, "cv_test_offsets", result.test_offsets, fixture.test_offsets);
    check_index_ref(failures, "cv_test_indices", result.test_indices, fixture.test_indices);
}

}  // namespace

TEST(cross_validation_phase3m, generated_kfold_fixtures_match_sklearn_reference) {
    for (const auto& fixture : ::n4m::test::fixtures::kCrossValidationFixtures) {
        check_fixture(failures, fixture);
    }
}

TEST(cross_validation_phase3m, rejects_malformed_validation_plans) {
    ::n4m::core::Context ctx;
    ::n4m::core::Config cfg;
    cfg.n_components = 1;

    double x_values[] = {
        0.0, 0.2,
        0.4, 0.7,
        0.9, 1.1,
        1.3, 1.4,
        1.8, 1.7,
        2.0, 2.1,
    };
    double y_values[] = {0.1, 0.5, 1.0, 1.2, 1.7, 2.2};
    n4m_matrix_view_t X{};
    n4m_matrix_view_t Y{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&X, x_values, 6, 2, N4M_DTYPE_F64), N4M_OK);
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&Y, y_values, 6, 1, N4M_DTYPE_F64), N4M_OK);

    ::n4m::core::CrossValidationResult result;
    ::n4m::core::ValidationPlan empty;
    empty.n_samples = 6;
    CHECK_EQ(::n4m::core::cross_validate_regression(ctx, cfg, X, Y, empty, result),
             N4M_ERR_INVALID_ARGUMENT);

    ::n4m::core::ValidationPlan mismatch;
    mismatch.n_samples = 5;
    mismatch.folds.push_back({{2, 3, 4, 5}, {0, 1}});
    CHECK_EQ(::n4m::core::cross_validate_regression(ctx, cfg, X, Y, mismatch, result),
             N4M_ERR_SHAPE_MISMATCH);

    ::n4m::core::ValidationPlan overlap;
    overlap.n_samples = 6;
    overlap.folds.push_back({{0, 1, 2, 4}, {2, 3}});
    CHECK_EQ(::n4m::core::cross_validate_regression(ctx, cfg, X, Y, overlap, result),
             N4M_ERR_INVALID_ARGUMENT);

    ::n4m::core::ValidationPlan duplicate_test;
    duplicate_test.n_samples = 6;
    duplicate_test.folds.push_back({{2, 3, 4, 5}, {0, 1}});
    duplicate_test.folds.push_back({{0, 3, 4, 5}, {1, 2}});
    CHECK_EQ(::n4m::core::cross_validate_regression(ctx, cfg, X, Y, duplicate_test, result),
             N4M_ERR_INVALID_ARGUMENT);

    ::n4m::core::ValidationPlan missing_test;
    missing_test.n_samples = 6;
    missing_test.folds.push_back({{2, 3, 4, 5}, {0, 1}});
    missing_test.folds.push_back({{0, 1, 4, 5}, {2, 3}});
    CHECK_EQ(::n4m::core::cross_validate_regression(ctx, cfg, X, Y, missing_test, result),
             N4M_ERR_INVALID_ARGUMENT);
}
