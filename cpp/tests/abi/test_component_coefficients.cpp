// SPDX-License-Identifier: CECILL-2.1

#include "core/component_coefficients.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include "fixtures/component_coefficients_fixtures.hpp"
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

void check_fixture(int& failures,
                   const ::pls4all::test::fixtures::ComponentCoefficientsFixture& fixture) {
    ::pls4all::core::Context ctx;
    ::pls4all::core::Config cfg;
    cfg.n_components = fixture.n_components;

    p4a_matrix_view_t X{};
    p4a_matrix_view_t Y{};
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

    std::unique_ptr<::pls4all::core::Model> model;
    CHECK_EQ(::pls4all::core::fit_model(ctx, cfg, X, Y, model), P4A_OK);
    CHECK_NE(model.get(), nullptr);

    std::vector<double> coefficients;
    CHECK_EQ(::pls4all::core::compute_regression_coefficients_by_component(ctx,
                                                                           *model,
                                                                           coefficients),
             P4A_OK);
    CHECK_EQ(fixture.coefficients_by_component.rows, fixture.n_components);
    CHECK_EQ(fixture.coefficients_by_component.cols, fixture.X.cols * fixture.Y.cols);
    check_close_values(failures,
                       "coefficients_by_component",
                       coefficients,
                       fixture.coefficients_by_component);

    model->weights_w.clear();
    CHECK_EQ(::pls4all::core::compute_regression_coefficients_by_component(ctx,
                                                                           *model,
                                                                           coefficients),
             P4A_ERR_INTERNAL);
}

}  // namespace

TEST(component_coefficients_phase3p, generated_fixture_matches_sklearn_reference) {
    for (const auto& fixture : ::pls4all::test::fixtures::kComponentCoefficientsFixtures) {
        check_fixture(failures, fixture);
    }
}
