// SPDX-License-Identifier: CECILL-2.1

#include "core/t2_selection.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

#include "fixtures/t2_selection_fixtures.hpp"
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
                   const ::pls4all::test::fixtures::T2SelectionIndexRef& expected) {
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

std::vector<double> alpha_values(const ::pls4all::test::fixtures::T2SelectionFixture& fixture) {
    return std::vector<double>(fixture.alpha.values, fixture.alpha.values + fixture.alpha.size);
}

void check_fixture(int& failures,
                   const ::pls4all::test::fixtures::T2SelectionFixture& fixture) {
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

    ::pls4all::core::T2SelectionResult result;
    CHECK_EQ(::pls4all::core::select_by_t2(ctx,
                                           cfg,
                                           X,
                                           Y,
                                           plan,
                                           alpha_values(fixture),
                                           fixture.min_selected,
                                           result),
             P4A_OK);
    CHECK_EQ(result.n_features, static_cast<std::int32_t>(fixture.X.cols));
    CHECK_EQ(result.n_targets, static_cast<std::int32_t>(fixture.Y.cols));
    CHECK_EQ(result.n_components, fixture.n_components);
    CHECK_EQ(result.n_alphas, static_cast<std::int32_t>(fixture.alpha.size));
    CHECK_EQ(result.min_selected, fixture.min_selected);
    CHECK_EQ(result.best_error_index, fixture.best_error_index);
    CHECK_EQ(result.min_set_index, fixture.min_set_index);
    check_close_scalar(failures, "best_rmse", result.best_rmse, fixture.best_rmse);
    check_close_scalar(failures, "min_set_rmse", result.min_set_rmse, fixture.min_set_rmse);
    check_close_values(failures, "t2_scores", result.t2_scores, fixture.t2_scores);
    check_close_values(failures, "ucl_by_alpha", result.ucl_by_alpha, fixture.ucl_by_alpha);
    check_close_values(failures, "rmse_by_alpha", result.rmse_by_alpha, fixture.rmse_by_alpha);
    check_indices(failures, "selected_counts", result.selected_counts, fixture.selected_counts);
    check_indices(failures, "selected_mask", result.selected_mask, fixture.selected_mask);
    check_indices(failures,
                  "selected_indices_best_error",
                  result.selected_indices_best_error,
                  fixture.selected_indices_best_error);
    check_indices(failures,
                  "selected_indices_min_set",
                  result.selected_indices_min_set,
                  fixture.selected_indices_min_set);
}

}  // namespace

TEST(t2_selection_phase5l, generated_fixture_matches_python_reference) {
    for (const auto& fixture : ::pls4all::test::fixtures::kT2SelectionFixtures) {
        check_fixture(failures, fixture);
    }
}

TEST(t2_selection_phase5l, rejects_invalid_t2_requests) {
    const auto& fixture = ::pls4all::test::fixtures::kT2SelectionFixtures[0];
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

    ::pls4all::core::T2SelectionResult result;
    std::vector<double> alpha = alpha_values(fixture);
    CHECK_EQ(::pls4all::core::select_by_t2(ctx,
                                           cfg,
                                           X,
                                           Y,
                                           plan,
                                           {},
                                           fixture.min_selected,
                                           result),
             P4A_ERR_INVALID_ARGUMENT);
    std::vector<double> invalid_alpha = alpha;
    invalid_alpha[0] = 1.0;
    CHECK_EQ(::pls4all::core::select_by_t2(ctx,
                                           cfg,
                                           X,
                                           Y,
                                           plan,
                                           invalid_alpha,
                                           fixture.min_selected,
                                           result),
             P4A_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::pls4all::core::select_by_t2(ctx,
                                           cfg,
                                           X,
                                           Y,
                                           plan,
                                           alpha,
                                           cfg.n_components - 1,
                                           result),
             P4A_ERR_INVALID_ARGUMENT);
    cfg.n_components = static_cast<std::int32_t>(fixture.X.cols - 1);
    CHECK_EQ(::pls4all::core::select_by_t2(ctx,
                                           cfg,
                                           X,
                                           Y,
                                           plan,
                                           alpha,
                                           fixture.min_selected,
                                           result),
             P4A_ERR_INVALID_ARGUMENT);

    cfg.n_components = fixture.n_components;
    p4a_matrix_view_t mismatched = Y;
    mismatched.rows = Y.rows - 1;
    CHECK_EQ(::pls4all::core::select_by_t2(ctx,
                                           cfg,
                                           X,
                                           mismatched,
                                           plan,
                                           alpha,
                                           fixture.min_selected,
                                           result),
             P4A_ERR_SHAPE_MISMATCH);
}
