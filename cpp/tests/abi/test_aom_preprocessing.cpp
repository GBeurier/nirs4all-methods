// SPDX-License-Identifier: CECILL-2.1

#include "core/aom_preprocessing.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

#include "fixtures/aom_preprocessing_fixtures.hpp"
#include "harness.hpp"

namespace {

constexpr double kAbsTol = 1e-10;
constexpr double kRelTol = 1e-10;

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
                   const ::pls4all::test::fixtures::AomPreprocessingIndexRef& expected) {
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

::pls4all::core::OperatorBank operator_bank(
    const ::pls4all::test::fixtures::AomPreprocessingIndexRef& kinds) {
    ::pls4all::core::OperatorBank bank;
    for (std::size_t i = 0; i < kinds.size; ++i) {
        bank.add(static_cast<p4a_operator_kind_t>(kinds.values[i]), nullptr, 0);
    }
    return bank;
}

void check_fixture(int& failures,
                   const ::pls4all::test::fixtures::AomPreprocessingFixture& fixture) {
    ::pls4all::core::Context ctx;
    ::pls4all::core::OperatorBank bank = operator_bank(fixture.operator_kinds);
    ::pls4all::core::GatingStrategy gate(static_cast<p4a_gating_mode_t>(fixture.gating_mode));
    p4a_matrix_view_t X = matrix_view(fixture.X);

    ::pls4all::core::AomPreprocessingResult result;
    CHECK_EQ(::pls4all::core::apply_aom_preprocessing(ctx,
                                                      bank,
                                                      gate,
                                                      X,
                                                      nullptr,
                                                      result),
             P4A_OK);
    CHECK_EQ(result.n_samples, fixture.X.rows);
    CHECK_EQ(result.n_features, fixture.X.cols);
    CHECK_EQ(result.n_operators, fixture.n_operators);
    CHECK_EQ(static_cast<std::int32_t>(result.mode), fixture.gating_mode);
    check_close_values(failures, "weights", result.weights, fixture.weights);
    check_close_values(failures, "operator_outputs", result.operator_outputs, fixture.operator_outputs);
    check_close_values(failures, "transformed", result.transformed, fixture.transformed);
    check_indices(failures, "operator_kinds", result.operator_kinds, fixture.operator_kinds);
}

}  // namespace

TEST(aom_preprocessing_phase6a, generated_fixture_matches_python_reference) {
    for (const auto& fixture : ::pls4all::test::fixtures::kAomPreprocessingFixtures) {
        check_fixture(failures, fixture);
    }
}

TEST(aom_preprocessing_phase6a, rejects_invalid_aom_preprocessing_requests) {
    const auto& fixture = ::pls4all::test::fixtures::kAomPreprocessingFixtures[0];
    ::pls4all::core::Context ctx;
    p4a_matrix_view_t X = matrix_view(fixture.X);

    ::pls4all::core::OperatorBank empty_bank;
    ::pls4all::core::GatingStrategy soft_gate(P4A_GATING_SOFT);
    ::pls4all::core::AomPreprocessingResult result;
    CHECK_EQ(::pls4all::core::apply_aom_preprocessing(ctx,
                                                      empty_bank,
                                                      soft_gate,
                                                      X,
                                                      nullptr,
                                                      result),
             P4A_ERR_INVALID_ARGUMENT);

    ::pls4all::core::OperatorBank bank = operator_bank(fixture.operator_kinds);
    ::pls4all::core::GatingStrategy unsupported_gate(P4A_GATING_PER_COMPONENT);
    CHECK_EQ(::pls4all::core::apply_aom_preprocessing(ctx,
                                                      bank,
                                                      unsupported_gate,
                                                      X,
                                                      nullptr,
                                                      result),
             P4A_ERR_UNSUPPORTED);

    p4a_matrix_view_t empty_x = X;
    empty_x.rows = 0;
    CHECK_EQ(::pls4all::core::apply_aom_preprocessing(ctx,
                                                      bank,
                                                      soft_gate,
                                                      empty_x,
                                                      nullptr,
                                                      result),
             P4A_ERR_INVALID_ARGUMENT);
}
