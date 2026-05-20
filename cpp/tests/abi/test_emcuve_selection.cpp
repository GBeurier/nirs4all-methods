// SPDX-License-Identifier: CECILL-2.1

#include "core/emcuve_selection.hpp"

#include <algorithm>
#include <cstdio>
#include <vector>

#include "fixtures/emcuve_selection_fixtures.hpp"
#include "harness.hpp"

namespace {

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
    CHECK_EQ(result.n_noise_features,
             std::max(fixture.noise_features, static_cast<std::int32_t>(fixture.X.cols)));
    CHECK_EQ(result.n_ensembles, fixture.n_ensembles);
    CHECK_EQ(result.selected_count, static_cast<std::int32_t>(result.selected_indices.size()));
    CHECK_EQ(result.noise_seed, fixture.noise_seed);
    CHECK_EQ(result.mean_real_stability_scores.size(), static_cast<std::size_t>(fixture.X.cols));
    CHECK_EQ(result.vote_frequencies.size(), static_cast<std::size_t>(fixture.X.cols));
    CHECK_EQ(result.noise_thresholds.size(), static_cast<std::size_t>(fixture.n_ensembles));
    for (const double frequency : result.vote_frequencies) {
        CHECK(frequency >= 0.0 && frequency <= 1.0);
    }

    std::vector<std::int64_t> expected_selected;
    for (std::size_t feature = 0; feature < result.vote_frequencies.size(); ++feature) {
        if (result.vote_frequencies[feature] >= fixture.vote_threshold) {
            expected_selected.push_back(static_cast<std::int64_t>(feature));
        }
    }
    std::stable_sort(expected_selected.begin(),
                     expected_selected.end(),
                     [&result](std::int64_t lhs, std::int64_t rhs) {
                         const auto left = static_cast<std::size_t>(lhs);
                         const auto right = static_cast<std::size_t>(rhs);
                         if (result.vote_frequencies[left] == result.vote_frequencies[right]) {
                             if (result.mean_real_stability_scores[left] ==
                                 result.mean_real_stability_scores[right]) {
                                 return lhs < rhs;
                             }
                             return result.mean_real_stability_scores[left] >
                                    result.mean_real_stability_scores[right];
                         }
                         return result.vote_frequencies[left] > result.vote_frequencies[right];
                     });
    if (result.selected_indices != expected_selected) {
        ++failures;
        std::fprintf(stderr, "  FAIL selected_indices do not match EMCUVE vote rule\n");
    }
}

}  // namespace

TEST(emcuve_selection_phase5n, fixture_uses_plsvarsel_style_member_votes) {
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
