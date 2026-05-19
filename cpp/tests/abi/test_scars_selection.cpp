// SPDX-License-Identifier: CECILL-2.1

#include "core/scars_selection.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

#include "fixtures/scars_selection_fixtures.hpp"
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

void check_close_scalar(int& failures, const char* label, double actual, double expected) {
    const double diff = std::fabs(actual - expected);
    const double scale = std::max(std::max(std::fabs(actual), std::fabs(expected)), 1.0);
    if (diff > kAbsTol && diff > kRelTol * scale) {
        ++failures;
        std::fprintf(stderr,
                     "  FAIL %s: actual %.17g expected %.17g diff %.3g\n",
                     label,
                     actual,
                     expected,
                     diff);
    }
}

void check_indices(int& failures,
                   const char* label,
                   const std::vector<std::int64_t>& actual,
                   const ::pls4all::test::fixtures::ScarsSelectionIndexRef& expected) {
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
                   const ::pls4all::test::fixtures::ScarsSelectionFixture& fixture) {
    ::pls4all::core::Context ctx;
    ::pls4all::core::Config cfg;
    cfg.algorithm = P4A_ALGO_PLS_REGRESSION;
    cfg.solver = P4A_SOLVER_NIPALS;
    cfg.deflation = P4A_DEFLATION_REGRESSION;
    cfg.n_components = fixture.n_components;

    p4a_matrix_view_t X = matrix_view(fixture.X);
    p4a_matrix_view_t Y = matrix_view(fixture.Y);
    ::pls4all::core::ValidationPlan plan;
    CHECK_EQ(::pls4all::core::make_kfold_validation_plan(ctx,
                                                         fixture.X.rows,
                                                         fixture.n_splits,
                                                         plan),
             P4A_OK);

    ::pls4all::core::ScarsSelectionResult result;
    CHECK_EQ(::pls4all::core::select_by_scars(ctx,
                                              cfg,
                                              X,
                                              Y,
                                              plan,
                                              fixture.n_iterations,
                                              fixture.min_features,
                                              fixture.sample_fraction,
                                              fixture.seed,
                                              result),
             P4A_OK);
    CHECK_EQ(result.n_features, static_cast<std::int32_t>(fixture.X.cols));
    CHECK_EQ(result.n_targets, static_cast<std::int32_t>(fixture.Y.cols));
    CHECK_EQ(result.n_components, fixture.n_components);
    CHECK_EQ(result.n_iterations, fixture.n_iterations);
    CHECK_EQ(result.min_features, fixture.min_features);
    CHECK_EQ(result.sample_count, fixture.sample_count);
    CHECK_EQ(result.best_iteration, fixture.best_iteration);
    CHECK_EQ(result.seed, fixture.seed);
    check_close_scalar(failures, "sample_fraction", result.sample_fraction, fixture.sample_fraction);
    check_close_scalar(failures, "best_rmse", result.best_rmse, fixture.best_rmse);
    check_close_values(failures, "coefficient_scores", result.coefficient_scores, fixture.coefficient_scores);
    check_close_values(failures, "stability_scores", result.stability_scores, fixture.stability_scores);
    check_close_values(failures, "rmse_by_iteration", result.rmse_by_iteration, fixture.rmse_by_iteration);
    check_indices(failures, "retained_counts", result.retained_counts, fixture.retained_counts);
    check_indices(failures, "selected_indices", result.selected_indices, fixture.selected_indices);
}

}  // namespace

TEST(scars_selection_phase5h, generated_fixture_matches_python_reference) {
    for (const auto& fixture : ::pls4all::test::fixtures::kScarsSelectionFixtures) {
        check_fixture(failures, fixture);
    }
}

TEST(scars_selection_phase5h, rejects_invalid_scars_requests) {
    const auto& fixture = ::pls4all::test::fixtures::kScarsSelectionFixtures[0];
    ::pls4all::core::Context ctx;
    ::pls4all::core::Config cfg;
    cfg.n_components = fixture.n_components;

    p4a_matrix_view_t X = matrix_view(fixture.X);
    p4a_matrix_view_t Y = matrix_view(fixture.Y);
    ::pls4all::core::ValidationPlan plan;
    CHECK_EQ(::pls4all::core::make_kfold_validation_plan(ctx,
                                                         fixture.X.rows,
                                                         fixture.n_splits,
                                                         plan),
             P4A_OK);

    ::pls4all::core::ScarsSelectionResult result;
    CHECK_EQ(::pls4all::core::select_by_scars(ctx,
                                              cfg,
                                              X,
                                              Y,
                                              plan,
                                              0,
                                              fixture.min_features,
                                              fixture.sample_fraction,
                                              fixture.seed,
                                              result),
             P4A_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::pls4all::core::select_by_scars(ctx,
                                              cfg,
                                              X,
                                              Y,
                                              plan,
                                              fixture.n_iterations,
                                              cfg.n_components - 1,
                                              fixture.sample_fraction,
                                              fixture.seed,
                                              result),
             P4A_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::pls4all::core::select_by_scars(ctx,
                                              cfg,
                                              X,
                                              Y,
                                              plan,
                                              fixture.n_iterations,
                                              fixture.min_features,
                                              0.0,
                                              fixture.seed,
                                              result),
             P4A_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::pls4all::core::select_by_scars(ctx,
                                              cfg,
                                              X,
                                              Y,
                                              plan,
                                              fixture.n_iterations,
                                              static_cast<std::int32_t>(fixture.X.cols + 1),
                                              fixture.sample_fraction,
                                              fixture.seed,
                                              result),
             P4A_ERR_INVALID_ARGUMENT);

    p4a_matrix_view_t mismatched = Y;
    mismatched.rows = Y.rows - 1;
    CHECK_EQ(::pls4all::core::select_by_scars(ctx,
                                              cfg,
                                              X,
                                              mismatched,
                                              plan,
                                              fixture.n_iterations,
                                              fixture.min_features,
                                              fixture.sample_fraction,
                                              fixture.seed,
                                              result),
             P4A_ERR_SHAPE_MISMATCH);
}
