// SPDX-License-Identifier: CeCILL-2.1

#include "core/emcuve_selection.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "fixtures/emcuve_selection_fixtures.hpp"
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
                   const ::pls4all::test::fixtures::EmcuveSelectionIndexRef& expected) {
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
                   const ::pls4all::test::fixtures::EmcuveSelectionFixture& fixture) {
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

    ::pls4all::core::EmcuveSelectionResult result;
    CHECK_EQ(::pls4all::core::select_by_emcuve(ctx,
                                               cfg,
                                               X,
                                               Y,
                                               plan,
                                               fixture.noise_features,
                                               fixture.noise_seed,
                                               fixture.n_ensembles,
                                               fixture.vote_threshold,
                                               result),
             P4A_OK);
    CHECK_EQ(result.n_features, static_cast<std::int32_t>(fixture.X.cols));
    CHECK_EQ(result.n_targets, static_cast<std::int32_t>(fixture.Y.cols));
    CHECK_EQ(result.n_repeats, fixture.n_repeats);
    CHECK_EQ(result.n_noise_features, fixture.noise_features);
    CHECK_EQ(result.n_ensembles, fixture.n_ensembles);
    CHECK_EQ(result.selected_count, static_cast<std::int32_t>(fixture.selected_indices.size));
    CHECK_EQ(result.noise_seed, fixture.noise_seed);
    check_close_values(failures, "mean_real_stability_scores", result.mean_real_stability_scores, fixture.mean_real_stability_scores);
    check_close_values(failures, "vote_frequencies", result.vote_frequencies, fixture.vote_frequencies);
    check_close_values(failures, "noise_thresholds", result.noise_thresholds, fixture.noise_thresholds);
    check_indices(failures, "selected_indices", result.selected_indices, fixture.selected_indices);
}

}  // namespace

TEST(emcuve_selection_phase5n, generated_fixture_matches_python_reference) {
    for (const auto& fixture : ::pls4all::test::fixtures::kEmcuveSelectionFixtures) {
        check_fixture(failures, fixture);
    }
}

TEST(emcuve_selection_phase5n, rejects_invalid_emcuve_requests) {
    const auto& fixture = ::pls4all::test::fixtures::kEmcuveSelectionFixtures[0];
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

    ::pls4all::core::EmcuveSelectionResult result;
    CHECK_EQ(::pls4all::core::select_by_emcuve(ctx,
                                               cfg,
                                               X,
                                               Y,
                                               plan,
                                               fixture.noise_features,
                                               fixture.noise_seed,
                                               0,
                                               fixture.vote_threshold,
                                               result),
             P4A_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::pls4all::core::select_by_emcuve(ctx,
                                               cfg,
                                               X,
                                               Y,
                                               plan,
                                               fixture.noise_features,
                                               fixture.noise_seed,
                                               fixture.n_ensembles,
                                               0.0,
                                               result),
             P4A_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::pls4all::core::select_by_emcuve(ctx,
                                               cfg,
                                               X,
                                               Y,
                                               plan,
                                               0,
                                               fixture.noise_seed,
                                               fixture.n_ensembles,
                                               fixture.vote_threshold,
                                               result),
             P4A_ERR_INVALID_ARGUMENT);

    p4a_matrix_view_t mismatched = Y;
    mismatched.rows = Y.rows - 1;
    CHECK_EQ(::pls4all::core::select_by_emcuve(ctx,
                                               cfg,
                                               X,
                                               mismatched,
                                               plan,
                                               fixture.noise_features,
                                               fixture.noise_seed,
                                               fixture.n_ensembles,
                                               fixture.vote_threshold,
                                               result),
             P4A_ERR_SHAPE_MISMATCH);
}

