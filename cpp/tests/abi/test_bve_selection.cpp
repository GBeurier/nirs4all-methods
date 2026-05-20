// SPDX-License-Identifier: CECILL-2.1

#include "core/bve_selection.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

#include "fixtures/bve_selection_fixtures.hpp"
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

void check_finite_values(int& failures,
                         const char* label,
                         const std::vector<double>& actual,
                         std::size_t expected_size) {
    if (actual.size() != expected_size) {
        ++failures;
        std::fprintf(stderr,
                     "  FAIL %s size: actual %lu expected %lu\n",
                     label,
                     static_cast<unsigned long>(actual.size()),
                     static_cast<unsigned long>(expected_size));
        return;
    }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i])) {
            ++failures;
            std::fprintf(stderr,
                         "  FAIL %s[%lu] is not finite: %.17g\n",
                         label,
                         static_cast<unsigned long>(i),
                         actual[i]);
            return;
        }
    }
}

void check_indices_valid(int& failures,
                         const char* label,
                         const std::vector<std::int64_t>& actual,
                         std::int64_t n_features) {
    if (actual.empty()) {
        ++failures;
        std::fprintf(stderr,
                     "  FAIL %s must not be empty\n",
                     label);
        return;
    }
    std::int64_t previous = -1;
    for (const std::int64_t feature : actual) {
        if (feature < 0 || feature >= n_features || feature <= previous) {
            ++failures;
            std::fprintf(stderr,
                         "  FAIL %s invalid index sequence at %lld\n",
                         label,
                         static_cast<long long>(feature));
            return;
        }
        previous = feature;
    }
}

void check_fixture(int& failures,
                   const ::pls4all::test::fixtures::BveSelectionFixture& fixture) {
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

    ::pls4all::core::BveSelectionResult result;
    CHECK_EQ(::pls4all::core::select_by_bve(ctx,
                                            cfg,
                                            X,
                                            Y,
                                            plan,
                                            fixture.n_steps,
                                            fixture.min_features,
                                            result),
             P4A_OK);
    CHECK_EQ(result.n_features, static_cast<std::int32_t>(fixture.X.cols));
    CHECK_EQ(result.n_targets, static_cast<std::int32_t>(fixture.Y.cols));
    CHECK_EQ(result.n_components, fixture.n_components);
    CHECK(result.n_steps > 0 && result.n_steps <= fixture.n_steps);
    CHECK_EQ(result.min_features, fixture.min_features);
    CHECK(result.best_step >= 0 && result.best_step < result.n_steps);
    CHECK(std::isfinite(result.best_rmse));
    check_finite_values(failures,
                        "candidate_rmse",
                        result.candidate_rmse,
                        static_cast<std::size_t>(result.n_steps) *
                            static_cast<std::size_t>(fixture.X.cols));
    check_finite_values(failures,
                        "rmse_by_step",
                        result.rmse_by_step,
                        static_cast<std::size_t>(result.n_steps));
    CHECK_EQ(result.retained_counts.size(), static_cast<std::size_t>(result.n_steps));
    for (std::size_t i = 1; i < result.retained_counts.size(); ++i) {
        CHECK(result.retained_counts[i] < result.retained_counts[i - 1]);
    }
    check_indices_valid(failures,
                        "selected_indices",
                        result.selected_indices,
                        fixture.X.cols);
    for (const std::int64_t feature : result.removed_indices) {
        CHECK(feature >= 0 && feature < fixture.X.cols);
    }
}

}  // namespace

TEST(bve_selection_phase5k, generated_fixture_matches_python_reference) {
    for (const auto& fixture : ::pls4all::test::fixtures::kBveSelectionFixtures) {
        check_fixture(failures, fixture);
    }
}

TEST(bve_selection_phase5k, rejects_invalid_bve_requests) {
    const auto& fixture = ::pls4all::test::fixtures::kBveSelectionFixtures[0];
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

    ::pls4all::core::BveSelectionResult result;
    CHECK_EQ(::pls4all::core::select_by_bve(ctx,
                                            cfg,
                                            X,
                                            Y,
                                            plan,
                                            0,
                                            fixture.min_features,
                                            result),
             P4A_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::pls4all::core::select_by_bve(ctx,
                                            cfg,
                                            X,
                                            Y,
                                            plan,
                                            fixture.n_steps,
                                            0,
                                            result),
             P4A_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::pls4all::core::select_by_bve(ctx,
                                            cfg,
                                            X,
                                            Y,
                                            plan,
                                            fixture.n_steps,
                                            static_cast<std::int32_t>(fixture.X.cols),
                                            result),
             P4A_ERR_INVALID_ARGUMENT);

    p4a_matrix_view_t mismatched = Y;
    mismatched.rows = Y.rows - 1;
    CHECK_EQ(::pls4all::core::select_by_bve(ctx,
                                            cfg,
                                            X,
                                            mismatched,
                                            plan,
                                            fixture.n_steps,
                                            fixture.min_features,
                                            result),
             P4A_ERR_SHAPE_MISMATCH);
}
