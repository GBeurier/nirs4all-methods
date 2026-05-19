// SPDX-License-Identifier: CECILL-2.1

#include "core/aom_selection.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "fixtures/aom_pop_selection_fixtures.hpp"
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

::pls4all::core::OperatorBank operator_bank(
    const ::pls4all::test::fixtures::AomPopSelectionFixture& fixture) {
    ::pls4all::core::OperatorBank bank;
    for (std::size_t op = 0; op < fixture.operator_kinds.size; ++op) {
        const auto start = static_cast<std::size_t>(fixture.operator_param_offsets.values[op]);
        const auto stop = static_cast<std::size_t>(fixture.operator_param_offsets.values[op + 1U]);
        const double* params = stop > start ? fixture.operator_params.values + start : nullptr;
        bank.add(static_cast<p4a_operator_kind_t>(fixture.operator_kinds.values[op]),
                 params,
                 static_cast<std::int32_t>(stop - start));
    }
    return bank;
}

::pls4all::core::ValidationPlan validation_plan(
    const ::pls4all::test::fixtures::AomPopSelectionFixture& fixture) {
    ::pls4all::core::ValidationPlan plan;
    plan.n_samples = fixture.X.rows;
    for (std::int32_t fold = 0; fold < fixture.n_folds; ++fold) {
        ::pls4all::core::ValidationFold item;
        const auto train_start = static_cast<std::size_t>(fixture.fold_train_offsets.values[fold]);
        const auto train_stop = static_cast<std::size_t>(fixture.fold_train_offsets.values[fold + 1]);
        const auto test_start = static_cast<std::size_t>(fixture.fold_test_offsets.values[fold]);
        const auto test_stop = static_cast<std::size_t>(fixture.fold_test_offsets.values[fold + 1]);
        for (std::size_t i = train_start; i < train_stop; ++i) {
            item.train_indices.push_back(fixture.fold_train_indices.values[i]);
        }
        for (std::size_t i = test_start; i < test_stop; ++i) {
            item.test_indices.push_back(fixture.fold_test_indices.values[i]);
        }
        plan.folds.push_back(std::move(item));
    }
    return plan;
}

::pls4all::core::Config simpls_config() {
    ::pls4all::core::Config cfg;
    cfg.algorithm = P4A_ALGO_PLS_REGRESSION;
    cfg.solver = P4A_SOLVER_SIMPLS;
    cfg.deflation = P4A_DEFLATION_REGRESSION;
    cfg.center_x = 1;
    cfg.scale_x = 0;
    cfg.center_y = 1;
    cfg.scale_y = 0;
    cfg.store_scores = 0;
    return cfg;
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
                   const ::pls4all::test::fixtures::AomPopSelectionIndexRef& expected) {
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
                   const ::pls4all::test::fixtures::AomPopSelectionFixture& fixture) {
    ::pls4all::core::Context ctx;
    ::pls4all::core::OperatorBank bank = operator_bank(fixture);
    ::pls4all::core::ValidationPlan plan = validation_plan(fixture);
    p4a_matrix_view_t X = matrix_view(fixture.X);
    p4a_matrix_view_t Y = matrix_view(fixture.Y);
    ::pls4all::core::AomPerComponentSelectionResult result;
    CHECK_EQ(::pls4all::core::select_aom_per_component(ctx,
                                                       simpls_config(),
                                                       bank,
                                                       X,
                                                       Y,
                                                       plan,
                                                       fixture.max_components,
                                                       result),
             P4A_OK);
    CHECK_EQ(result.n_operators, fixture.n_operators);
    CHECK_EQ(result.max_components, fixture.max_components);
    CHECK_EQ(result.selected_n_components, fixture.selected_n_components);
    const double best_diff = std::fabs(result.best_score - fixture.best_score);
    if (best_diff > kAbsTol && best_diff > kRelTol * std::max(std::fabs(fixture.best_score), 1.0)) {
        ++failures;
        std::fprintf(stderr,
                     "  FAIL best_score: actual %.17g expected %.17g diff %.3g\n",
                     result.best_score,
                     fixture.best_score,
                     best_diff);
    }
    check_indices(failures, "operator_kinds", result.operator_kinds, fixture.operator_kinds);
    check_indices(failures,
                  "selected_operator_indices",
                  result.selected_operator_indices,
                  fixture.selected_operator_indices);
    check_close_values(failures, "component_scores", result.component_scores, fixture.component_scores);
    check_close_values(failures, "prefix_scores", result.prefix_scores, fixture.prefix_scores);
    check_close_values(failures, "predictions", result.predictions, fixture.predictions);
}

}  // namespace

TEST(aom_pop_selection_phase6e, generated_fixture_matches_bench_aom_v0_reference) {
    for (const auto& fixture : ::pls4all::test::fixtures::kAomPopSelectionFixtures) {
        check_fixture(failures, fixture);
    }
}

TEST(aom_pop_selection_phase6e, rejects_invalid_requests) {
    const auto& fixture = ::pls4all::test::fixtures::kAomPopSelectionFixtures[0];
    ::pls4all::core::Context ctx;
    p4a_matrix_view_t X = matrix_view(fixture.X);
    p4a_matrix_view_t Y = matrix_view(fixture.Y);
    ::pls4all::core::ValidationPlan plan = validation_plan(fixture);
    ::pls4all::core::AomPerComponentSelectionResult result;

    ::pls4all::core::OperatorBank empty_bank;
    CHECK_EQ(::pls4all::core::select_aom_per_component(ctx,
                                                       simpls_config(),
                                                       empty_bank,
                                                       X,
                                                       Y,
                                                       plan,
                                                       fixture.max_components,
                                                       result),
             P4A_ERR_INVALID_ARGUMENT);

    ::pls4all::core::OperatorBank non_strict_bank;
    non_strict_bank.add(P4A_OP_SNV, nullptr, 0);
    CHECK_EQ(::pls4all::core::select_aom_per_component(ctx,
                                                       simpls_config(),
                                                       non_strict_bank,
                                                       X,
                                                       Y,
                                                       plan,
                                                       fixture.max_components,
                                                       result),
             P4A_ERR_UNSUPPORTED);
}
