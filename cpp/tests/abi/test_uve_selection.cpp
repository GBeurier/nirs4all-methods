// SPDX-License-Identifier: CECILL-2.1

#include "core/uve_selection.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

#include "fixtures/uve_selection_fixtures.hpp"
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

double fixture_value(const ::pls4all::test::fixtures::MatrixRef& matrix,
                     std::size_t row,
                     std::size_t col) {
    return matrix.values[row * static_cast<std::size_t>(matrix.cols) + col];
}

double fixture_feature_response_score(const ::pls4all::test::fixtures::UveSelectionFixture& fixture,
                                      std::size_t feature) {
    const auto rows = static_cast<std::size_t>(fixture.X.rows);
    const auto targets = static_cast<std::size_t>(fixture.Y.cols);
    double x_sum = 0.0;
    for (std::size_t row = 0; row < rows; ++row) {
        x_sum += fixture_value(fixture.X, row, feature);
    }
    const double x_mean = x_sum / static_cast<double>(rows);
    double x_ss = 0.0;
    for (std::size_t row = 0; row < rows; ++row) {
        const double dx = fixture_value(fixture.X, row, feature) - x_mean;
        x_ss += dx * dx;
    }
    double best = 0.0;
    for (std::size_t target = 0; target < targets; ++target) {
        double y_sum = 0.0;
        for (std::size_t row = 0; row < rows; ++row) {
            y_sum += fixture_value(fixture.Y, row, target);
        }
        const double y_mean = y_sum / static_cast<double>(rows);
        double y_ss = 0.0;
        double xy = 0.0;
        for (std::size_t row = 0; row < rows; ++row) {
            const double dx = fixture_value(fixture.X, row, feature) - x_mean;
            const double dy = fixture_value(fixture.Y, row, target) - y_mean;
            xy += dx * dy;
            y_ss += dy * dy;
        }
        if (x_ss > 0.0 && y_ss > 0.0) {
            best = std::max(best, std::fabs(xy) / std::sqrt(x_ss * y_ss));
        }
    }
    return best;
}

void check_fixture(int& failures,
                   const ::pls4all::test::fixtures::UveSelectionFixture& fixture) {
    ::pls4all::core::Context ctx;
    ::pls4all::core::Config cfg;
    cfg.algorithm = P4A_ALGO_PLS_REGRESSION;
    cfg.solver = P4A_SOLVER_NIPALS;
    cfg.deflation = P4A_DEFLATION_REGRESSION;
    cfg.n_components = fixture.n_components;

    p4a_matrix_view_t X = matrix_view(fixture.X);
    p4a_matrix_view_t Y = matrix_view(fixture.Y);
    ::pls4all::core::ValidationPlan plan;
    CHECK_EQ(::pls4all::core::make_monte_carlo_validation_plan(ctx,
                                                               fixture.X.rows,
                                                               fixture.test_count,
                                                               fixture.n_repeats,
                                                               fixture.validation_seed,
                                                               plan),
             P4A_OK);

    ::pls4all::core::UveSelectionResult result;
    CHECK_EQ(::pls4all::core::select_by_uve(ctx,
                                            cfg,
                                            X,
                                            Y,
                                            plan,
                                            fixture.noise_features,
                                            fixture.noise_seed,
                                            result),
             P4A_OK);
    CHECK_EQ(result.n_features, static_cast<std::int32_t>(fixture.X.cols));
    CHECK_EQ(result.n_targets, static_cast<std::int32_t>(fixture.Y.cols));
    CHECK_EQ(result.n_repeats, fixture.n_repeats);
    CHECK_EQ(result.n_noise_features,
             std::max(fixture.noise_features, static_cast<std::int32_t>(fixture.X.cols)));
    CHECK_EQ(result.real_stability_scores.size(), static_cast<std::size_t>(fixture.X.cols));
    CHECK_EQ(result.noise_stability_scores.size(),
             static_cast<std::size_t>(result.n_noise_features));
    CHECK_EQ(result.selected_count, static_cast<std::int32_t>(result.selected_indices.size()));
    CHECK_EQ(result.noise_seed, fixture.noise_seed);

    const double expected_threshold = *std::max_element(result.noise_stability_scores.begin(),
                                                       result.noise_stability_scores.end());
    check_close_scalar(failures, "noise_threshold", result.noise_threshold, expected_threshold);

    std::vector<std::int64_t> expected_selected;
    for (std::size_t feature = 0; feature < result.real_stability_scores.size(); ++feature) {
        if (result.real_stability_scores[feature] > result.noise_threshold) {
            expected_selected.push_back(static_cast<std::int64_t>(feature));
        }
    }
    if (expected_selected.size() <=
        static_cast<std::size_t>(fixture.n_components + 1)) {
        expected_selected.resize(result.real_stability_scores.size());
        std::iota(expected_selected.begin(), expected_selected.end(), std::int64_t{0});
        std::stable_sort(expected_selected.begin(),
                         expected_selected.end(),
                         [&result](std::int64_t lhs, std::int64_t rhs) {
                             const double left =
                                 result.real_stability_scores[static_cast<std::size_t>(lhs)];
                             const double right =
                                 result.real_stability_scores[static_cast<std::size_t>(rhs)];
                             if (left == right) {
                                 return lhs < rhs;
                             }
                             return left > right;
        });
        expected_selected.resize(static_cast<std::size_t>(fixture.n_components));
    }
    if (expected_selected.size() > 1U &&
        expected_selected.size() <= static_cast<std::size_t>(fixture.n_components)) {
        const auto best = std::max_element(
            expected_selected.begin(),
            expected_selected.end(),
            [&fixture](std::int64_t lhs, std::int64_t rhs) {
                const double left =
                    fixture_feature_response_score(fixture, static_cast<std::size_t>(lhs));
                const double right =
                    fixture_feature_response_score(fixture, static_cast<std::size_t>(rhs));
                if (left == right) {
                    return lhs > rhs;
                }
                return left < right;
            });
        expected_selected.assign(1U, *best);
    }
    if (result.selected_indices != expected_selected) {
        ++failures;
        std::fprintf(stderr, "  FAIL selected_indices do not match UVE threshold rule\n");
    }
}

}  // namespace

TEST(uve_selection_phase5d, fixture_uses_plsvarsel_style_noise_threshold) {
    for (const auto& fixture : ::pls4all::test::fixtures::kUveSelectionFixtures) {
        check_fixture(failures, fixture);
    }
}

TEST(uve_selection_phase5d, rejects_invalid_uve_requests) {
    const auto& fixture = ::pls4all::test::fixtures::kUveSelectionFixtures[0];
    ::pls4all::core::Context ctx;
    ::pls4all::core::Config cfg;
    cfg.n_components = fixture.n_components;

    p4a_matrix_view_t X = matrix_view(fixture.X);
    p4a_matrix_view_t Y = matrix_view(fixture.Y);
    ::pls4all::core::ValidationPlan plan;
    CHECK_EQ(::pls4all::core::make_monte_carlo_validation_plan(ctx,
                                                               fixture.X.rows,
                                                               fixture.test_count,
                                                               fixture.n_repeats,
                                                               fixture.validation_seed,
                                                               plan),
             P4A_OK);

    ::pls4all::core::UveSelectionResult result;
    CHECK_EQ(::pls4all::core::select_by_uve(ctx,
                                            cfg,
                                            X,
                                            Y,
                                            plan,
                                            0,
                                            fixture.noise_seed,
                                            result),
             P4A_ERR_INVALID_ARGUMENT);

    ::pls4all::core::ValidationPlan one_fold;
    CHECK_EQ(::pls4all::core::make_monte_carlo_validation_plan(ctx,
                                                               fixture.X.rows,
                                                               fixture.test_count,
                                                               1,
                                                               fixture.validation_seed,
                                                               one_fold),
             P4A_OK);
    CHECK_EQ(::pls4all::core::select_by_uve(ctx,
                                            cfg,
                                            X,
                                            Y,
                                            one_fold,
                                            fixture.noise_features,
                                            fixture.noise_seed,
                                            result),
             P4A_ERR_INVALID_ARGUMENT);

    p4a_matrix_view_t mismatched = Y;
    mismatched.rows = Y.rows - 1;
    CHECK_EQ(::pls4all::core::select_by_uve(ctx,
                                            cfg,
                                            X,
                                            mismatched,
                                            plan,
                                            fixture.noise_features,
                                            fixture.noise_seed,
                                            result),
             P4A_ERR_SHAPE_MISMATCH);
}
