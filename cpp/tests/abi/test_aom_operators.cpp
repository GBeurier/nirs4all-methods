// SPDX-License-Identifier: CECILL-2.1

#include "core/aom_operators.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "fixtures/aom_operator_fixtures.hpp"
#include "harness.hpp"

namespace {

constexpr double kAbsTol = 1e-10;
constexpr double kRelTol = 1e-10;

n4m_matrix_view_t matrix_view(const ::n4m::test::fixtures::MatrixRef& ref) {
    n4m_matrix_view_t view{};
    view.data = const_cast<double*>(ref.values);
    view.rows = ref.rows;
    view.cols = ref.cols;
    view.row_stride = ref.cols > 0 ? ref.cols : 1;
    view.col_stride = 1;
    view.dtype = N4M_DTYPE_F64;
    return view;
}

::n4m::core::OperatorEntry operator_entry(
    const ::n4m::test::fixtures::AomOperatorFixture& fixture,
    std::size_t op) {
    const auto start = static_cast<std::size_t>(fixture.operator_param_offsets.values[op]);
    const auto stop = static_cast<std::size_t>(fixture.operator_param_offsets.values[op + 1U]);
    const double* params = stop > start ? fixture.operator_params.values + start : nullptr;
    return ::n4m::core::OperatorEntry{
        static_cast<n4m_operator_kind_t>(fixture.operator_kinds.values[op]),
        params,
        static_cast<std::int32_t>(stop - start),
    };
}

void check_close_values(int& failures,
                        const char* label,
                        const std::vector<double>& actual,
                        const double* expected,
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
    for (std::size_t i = 0; i < expected_size; ++i) {
        const double diff = std::fabs(actual[i] - expected[i]);
        const double scale = std::max(std::max(std::fabs(actual[i]), std::fabs(expected[i])), 1.0);
        if (diff > kAbsTol && diff > kRelTol * scale) {
            ++failures;
            std::fprintf(stderr,
                         "  FAIL %s[%lu]: actual %.17g expected %.17g diff %.3g\n",
                         label,
                         static_cast<unsigned long>(i),
                         actual[i],
                         expected[i],
                         diff);
            return;
        }
    }
}

void check_fixture(int& failures,
                   const ::n4m::test::fixtures::AomOperatorFixture& fixture) {
    ::n4m::core::Context ctx;
    n4m_matrix_view_t X = matrix_view(fixture.X);
    const std::size_t block_size = fixture.X.size;
    CHECK_EQ(fixture.operator_outputs.size,
             block_size * static_cast<std::size_t>(fixture.n_operators));
    for (std::size_t op = 0; op < static_cast<std::size_t>(fixture.n_operators); ++op) {
        std::vector<double> actual;
        CHECK_EQ(::n4m::core::transform_aom_strict_operator(ctx,
                                                                operator_entry(fixture, op),
                                                                X,
                                                                actual),
                 N4M_OK);
        check_close_values(failures,
                           fixture.id,
                           actual,
                           fixture.operator_outputs.values + op * block_size,
                           block_size);
    }
}

}  // namespace

TEST(aom_operators_phase6c, generated_fixture_matches_bench_aom_v0_reference) {
    for (const auto& fixture : ::n4m::test::fixtures::kAomOperatorFixtures) {
        check_fixture(failures, fixture);
    }
}

TEST(aom_operators_phase6c, rejects_invalid_or_non_strict_requests) {
    const auto& fixture = ::n4m::test::fixtures::kAomOperatorFixtures[0];
    ::n4m::core::Context ctx;
    n4m_matrix_view_t X = matrix_view(fixture.X);
    std::vector<double> out;

    ::n4m::core::OperatorEntry snv{N4M_OP_SNV, nullptr, 0};
    CHECK_EQ(::n4m::core::transform_aom_strict_operator(ctx, snv, X, out),
             N4M_ERR_UNSUPPORTED);

    const double bad_order[] = {3.0};
    ::n4m::core::OperatorEntry finite_difference{
        N4M_OP_FINITE_DIFFERENCE,
        bad_order,
        1,
    };
    CHECK_EQ(::n4m::core::transform_aom_strict_operator(ctx,
                                                            finite_difference,
                                                            X,
                                                            out),
             N4M_ERR_INVALID_ARGUMENT);

    const double bad_lambda[] = {-1.0};
    ::n4m::core::OperatorEntry whittaker{
        N4M_OP_WHITTAKER,
        bad_lambda,
        1,
    };
    CHECK_EQ(::n4m::core::transform_aom_strict_operator(ctx,
                                                            whittaker,
                                                            X,
                                                            out),
             N4M_ERR_INVALID_ARGUMENT);

    const double even_kernel[] = {1.0, 1.0, 6.0, 3.0};
    ::n4m::core::OperatorEntry fck{
        N4M_OP_FCK,
        even_kernel,
        4,
    };
    CHECK_EQ(::n4m::core::transform_aom_strict_operator(ctx,
                                                            fck,
                                                            X,
                                                            out),
             N4M_ERR_INVALID_ARGUMENT);
}
