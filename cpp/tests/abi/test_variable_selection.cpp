// SPDX-License-Identifier: CECILL-2.1

#include "core/variable_selection.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

#include "fixtures/variable_selection_fixtures.hpp"
#include "harness.hpp"

namespace {

constexpr double kAbsTol = 1e-8;
constexpr double kRelTol = 1e-8;

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
                   const ::pls4all::test::fixtures::VariableSelectionIndexRef& expected) {
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

void fit_fixture(int& failures,
                 const ::pls4all::test::fixtures::VariableSelectionFixture& fixture,
                 bool store_scores,
                 ::pls4all::core::Context& ctx,
                 std::unique_ptr<::pls4all::core::Model>& model,
                 p4a_matrix_view_t& X,
                 p4a_matrix_view_t& Y) {
    ::pls4all::core::Config cfg;
    cfg.n_components = fixture.n_components;
    cfg.store_scores = store_scores ? 1 : 0;

    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X,
                                           const_cast<double*>(fixture.X.values),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&Y,
                                           const_cast<double*>(fixture.Y.values),
                                           fixture.Y.rows,
                                           fixture.Y.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);
    CHECK_EQ(::pls4all::core::fit_model(ctx, cfg, X, Y, model), P4A_OK);
    CHECK_NE(model.get(), nullptr);
}

void check_method(int& failures,
                  const char* label,
                  ::pls4all::core::Context& ctx,
                  const ::pls4all::core::Model& model,
                  const p4a_matrix_view_t& X,
                  ::pls4all::core::VariableSelectionMethod method,
                  std::int32_t top_k,
                  const ::pls4all::test::fixtures::MatrixRef& expected_scores,
                  const ::pls4all::test::fixtures::VariableSelectionIndexRef& expected_indices) {
    ::pls4all::core::VariableSelectionResult result;
    CHECK_EQ(::pls4all::core::select_variables(ctx, model, X, method, top_k, result), P4A_OK);
    CHECK_EQ(result.n_features, static_cast<std::int32_t>(expected_scores.size));
    CHECK_EQ(result.top_k, top_k);
    check_close_values(failures, label, result.scores, expected_scores);
    check_indices(failures, label, result.selected_indices, expected_indices);
}

void check_fixture(int& failures,
                   const ::pls4all::test::fixtures::VariableSelectionFixture& fixture) {
    ::pls4all::core::Context ctx;
    std::unique_ptr<::pls4all::core::Model> model;
    p4a_matrix_view_t X{};
    p4a_matrix_view_t Y{};
    fit_fixture(failures, fixture, true, ctx, model, X, Y);

    check_method(failures,
                 "vip",
                 ctx,
                 *model,
                 X,
                 ::pls4all::core::VariableSelectionMethod::Vip,
                 fixture.top_k,
                 fixture.vip_scores,
                 fixture.vip_indices);
    check_method(failures,
                 "coefficient_magnitude",
                 ctx,
                 *model,
                 X,
                 ::pls4all::core::VariableSelectionMethod::CoefficientMagnitude,
                 fixture.top_k,
                 fixture.coefficient_scores,
                 fixture.coefficient_indices);
    check_method(failures,
                 "selectivity_ratio",
                 ctx,
                 *model,
                 X,
                 ::pls4all::core::VariableSelectionMethod::SelectivityRatio,
                 fixture.top_k,
                 fixture.selectivity_ratio_scores,
                 fixture.selectivity_ratio_indices);
}

}  // namespace

TEST(variable_selection_phase5a, generated_fixture_matches_sklearn_reference) {
    for (const auto& fixture : ::pls4all::test::fixtures::kVariableSelectionFixtures) {
        check_fixture(failures, fixture);
    }
}

TEST(variable_selection_phase5a, rejects_invalid_top_k_missing_scores_and_bad_x) {
    const auto& fixture = ::pls4all::test::fixtures::kVariableSelectionFixtures[0];
    ::pls4all::core::Context ctx;
    std::unique_ptr<::pls4all::core::Model> model;
    p4a_matrix_view_t X{};
    p4a_matrix_view_t Y{};
    fit_fixture(failures, fixture, false, ctx, model, X, Y);

    ::pls4all::core::VariableSelectionResult result;
    CHECK_EQ(::pls4all::core::select_variables(ctx,
                                               *model,
                                               X,
                                               ::pls4all::core::VariableSelectionMethod::CoefficientMagnitude,
                                               0,
                                               result),
             P4A_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::pls4all::core::select_variables(ctx,
                                               *model,
                                               X,
                                               ::pls4all::core::VariableSelectionMethod::CoefficientMagnitude,
                                               static_cast<std::int32_t>(fixture.X.cols + 1),
                                               result),
             P4A_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::pls4all::core::select_variables(ctx,
                                               *model,
                                               X,
                                               ::pls4all::core::VariableSelectionMethod::Vip,
                                               fixture.top_k,
                                               result),
             P4A_ERR_UNSUPPORTED);
    CHECK_EQ(result.scores.size(), static_cast<std::size_t>(0));
    CHECK_EQ(result.selected_indices.size(), static_cast<std::size_t>(0));

    model.reset();
    fit_fixture(failures, fixture, true, ctx, model, X, Y);

    p4a_matrix_view_t mismatched = X;
    mismatched.cols = X.cols - 1;
    CHECK_EQ(::pls4all::core::select_variables(ctx,
                                               *model,
                                               mismatched,
                                               ::pls4all::core::VariableSelectionMethod::SelectivityRatio,
                                               fixture.top_k,
                                               result),
             P4A_ERR_SHAPE_MISMATCH);

    std::vector<double> bad_x(fixture.X.values, fixture.X.values + fixture.X.size);
    bad_x[3] = std::numeric_limits<double>::quiet_NaN();
    p4a_matrix_view_t bad_view{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&bad_view,
                                           bad_x.data(),
                                           fixture.X.rows,
                                           fixture.X.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);
    CHECK_EQ(::pls4all::core::select_variables(ctx,
                                               *model,
                                               bad_view,
                                               ::pls4all::core::VariableSelectionMethod::SelectivityRatio,
                                               fixture.top_k,
                                               result),
             P4A_ERR_INVALID_ARGUMENT);
}
