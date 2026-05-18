// SPDX-License-Identifier: CECILL-2.1

#include "core/sipls_selection.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "fixtures/sipls_selection_fixtures.hpp"
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

void check_indices(int& failures,
                   const char* label,
                   const std::vector<std::int64_t>& actual,
                   const ::pls4all::test::fixtures::SiplsSelectionIndexRef& expected) {
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
                   const ::pls4all::test::fixtures::SiplsSelectionFixture& fixture) {
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

    ::pls4all::core::SiplsSelectionResult result;
    CHECK_EQ(::pls4all::core::select_by_sipls(ctx,
                                              cfg,
                                              X,
                                              Y,
                                              plan,
                                              fixture.interval_width,
                                              fixture.combination_size,
                                              result),
             P4A_OK);
    CHECK_EQ(result.n_intervals, static_cast<std::int32_t>(fixture.intervals.size / 2U));
    CHECK_EQ(result.interval_width, fixture.interval_width);
    CHECK_EQ(result.combination_size, fixture.combination_size);
    CHECK_EQ(result.n_combinations, static_cast<std::int32_t>(fixture.rmse_by_combination.size));
    CHECK_EQ(result.best_combination_index, fixture.best_combination_index);
    check_indices(failures, "intervals", result.intervals, fixture.intervals);
    check_indices(failures, "candidate_interval_indices", result.candidate_interval_indices, fixture.candidate_interval_indices);
    check_close_values(failures, "rmse_by_combination", result.rmse_by_combination, fixture.rmse_by_combination);
    check_indices(failures, "selected_interval_indices", result.selected_interval_indices, fixture.selected_interval_indices);
    check_indices(failures, "selected_feature_indices", result.selected_feature_indices, fixture.selected_feature_indices);
}

}  // namespace

TEST(sipls_selection_phase5q, generated_fixture_matches_python_reference) {
    for (const auto& fixture : ::pls4all::test::fixtures::kSiplsSelectionFixtures) {
        check_fixture(failures, fixture);
    }
}

TEST(sipls_selection_phase5q, rejects_invalid_sipls_requests) {
    const auto& fixture = ::pls4all::test::fixtures::kSiplsSelectionFixtures[0];
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

    ::pls4all::core::SiplsSelectionResult result;
    CHECK_EQ(::pls4all::core::select_by_sipls(ctx,
                                              cfg,
                                              X,
                                              Y,
                                              plan,
                                              0,
                                              fixture.combination_size,
                                              result),
             P4A_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::pls4all::core::select_by_sipls(ctx,
                                              cfg,
                                              X,
                                              Y,
                                              plan,
                                              fixture.interval_width,
                                              0,
                                              result),
             P4A_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::pls4all::core::select_by_sipls(ctx,
                                              cfg,
                                              X,
                                              Y,
                                              plan,
                                              fixture.interval_width,
                                              static_cast<std::int32_t>(fixture.intervals.size / 2U + 1U),
                                              result),
             P4A_ERR_INVALID_ARGUMENT);

    p4a_matrix_view_t mismatched = Y;
    mismatched.rows = Y.rows - 1;
    CHECK_EQ(::pls4all::core::select_by_sipls(ctx,
                                              cfg,
                                              X,
                                              mismatched,
                                              plan,
                                              fixture.interval_width,
                                              fixture.combination_size,
                                              result),
             P4A_ERR_SHAPE_MISMATCH);

    cfg.n_components = static_cast<std::int32_t>(fixture.X.cols + 1);
    CHECK_EQ(::pls4all::core::select_by_sipls(ctx,
                                              cfg,
                                              X,
                                              Y,
                                              plan,
                                              fixture.interval_width,
                                              fixture.combination_size,
                                              result),
             P4A_ERR_INVALID_ARGUMENT);
}
