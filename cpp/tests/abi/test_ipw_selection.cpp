// SPDX-License-Identifier: CeCILL-2.1

#include "core/ipw_selection.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

#include "fixtures/ipw_selection_fixtures.hpp"
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
                   const ::pls4all::test::fixtures::IpwSelectionIndexRef& expected) {
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
                   const ::pls4all::test::fixtures::IpwSelectionFixture& fixture) {
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

    ::pls4all::core::IpwSelectionResult result;
    CHECK_EQ(::pls4all::core::select_by_ipw(ctx,
                                            cfg,
                                            X,
                                            Y,
                                            plan,
                                            fixture.n_iterations,
                                            fixture.top_k,
                                            fixture.damping,
                                            fixture.weight_floor,
                                            result),
             P4A_OK);
    CHECK_EQ(result.n_features, static_cast<std::int32_t>(fixture.X.cols));
    CHECK_EQ(result.n_targets, static_cast<std::int32_t>(fixture.Y.cols));
    CHECK_EQ(result.n_components, fixture.n_components);
    CHECK_EQ(result.n_iterations, fixture.n_iterations);
    CHECK_EQ(result.top_k, fixture.top_k);
    CHECK_EQ(result.best_iteration, fixture.best_iteration);
    check_close_scalar(failures, "damping", result.damping, fixture.damping);
    check_close_scalar(failures, "weight_floor", result.weight_floor, fixture.weight_floor);
    check_close_scalar(failures, "best_rmse", result.best_rmse, fixture.best_rmse);
    check_close_values(failures, "score_path", result.score_path, fixture.score_path);
    check_close_values(failures, "weight_path", result.weight_path, fixture.weight_path);
    check_close_values(failures, "rmse_by_iteration", result.rmse_by_iteration, fixture.rmse_by_iteration);
    check_indices(failures, "selected_by_iteration", result.selected_by_iteration, fixture.selected_by_iteration);
    check_indices(failures, "ranking_indices", result.ranking_indices, fixture.ranking_indices);
    check_indices(failures, "selected_indices", result.selected_indices, fixture.selected_indices);
}

}  // namespace

TEST(ipw_selection_phase5t, generated_fixture_matches_python_reference) {
    for (const auto& fixture : ::pls4all::test::fixtures::kIpwSelectionFixtures) {
        check_fixture(failures, fixture);
    }
}

TEST(ipw_selection_phase5t, rejects_invalid_ipw_requests) {
    const auto& fixture = ::pls4all::test::fixtures::kIpwSelectionFixtures[0];
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

    ::pls4all::core::IpwSelectionResult result;
    CHECK_EQ(::pls4all::core::select_by_ipw(ctx,
                                            cfg,
                                            X,
                                            Y,
                                            plan,
                                            0,
                                            fixture.top_k,
                                            fixture.damping,
                                            fixture.weight_floor,
                                            result),
             P4A_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::pls4all::core::select_by_ipw(ctx,
                                            cfg,
                                            X,
                                            Y,
                                            plan,
                                            fixture.n_iterations,
                                            cfg.n_components - 1,
                                            fixture.damping,
                                            fixture.weight_floor,
                                            result),
             P4A_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::pls4all::core::select_by_ipw(ctx,
                                            cfg,
                                            X,
                                            Y,
                                            plan,
                                            fixture.n_iterations,
                                            fixture.top_k,
                                            1.0,
                                            fixture.weight_floor,
                                            result),
             P4A_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::pls4all::core::select_by_ipw(ctx,
                                            cfg,
                                            X,
                                            Y,
                                            plan,
                                            fixture.n_iterations,
                                            fixture.top_k,
                                            fixture.damping,
                                            0.0,
                                            result),
             P4A_ERR_INVALID_ARGUMENT);

    p4a_matrix_view_t mismatched = Y;
    mismatched.rows = Y.rows - 1;
    CHECK_EQ(::pls4all::core::select_by_ipw(ctx,
                                            cfg,
                                            X,
                                            mismatched,
                                            plan,
                                            fixture.n_iterations,
                                            fixture.top_k,
                                            fixture.damping,
                                            fixture.weight_floor,
                                            result),
             P4A_ERR_SHAPE_MISMATCH);
}
