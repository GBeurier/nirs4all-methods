// SPDX-License-Identifier: CECILL-2.1

#include "core/pls_logistic.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

#include "fixtures/pls_logistic_fixtures.hpp"
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

p4a_matrix_view_t label_view(const ::pls4all::test::fixtures::PlsLogisticLabelRef& ref) {
    p4a_matrix_view_t view{};
    view.data = const_cast<std::int32_t*>(ref.values);
    view.rows = ref.rows;
    view.cols = ref.cols;
    view.row_stride = ref.cols > 0 ? ref.cols : 1;
    view.col_stride = 1;
    view.dtype = P4A_DTYPE_I32;
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

void check_predictions(int& failures,
                       const std::vector<std::int32_t>& actual,
                       const ::pls4all::test::fixtures::PlsLogisticLabelRef& expected) {
    CHECK_EQ(actual.size(), expected.size);
    const std::size_t n = std::min(actual.size(), expected.size);
    for (std::size_t i = 0; i < n; ++i) {
        if (actual[i] != expected.values[i]) {
            ++failures;
            std::fprintf(stderr,
                         "  FAIL prediction[%lu]: actual %d expected %d\n",
                         static_cast<unsigned long>(i),
                         static_cast<int>(actual[i]),
                         static_cast<int>(expected.values[i]));
            return;
        }
    }
}

void check_probability_rows(int& failures,
                            const ::pls4all::core::PlsLogisticResult& result) {
    const auto rows = static_cast<std::size_t>(result.n_samples);
    const auto cols = static_cast<std::size_t>(result.n_classes);
    for (std::size_t row = 0; row < rows; ++row) {
        double sum = 0.0;
        for (std::size_t col = 0; col < cols; ++col) {
            const double value = result.probabilities[row * cols + col];
            if (value < -kAbsTol || value > 1.0 + kAbsTol) {
                ++failures;
                std::fprintf(stderr,
                             "  FAIL probability[%lu,%lu] out of range: %.17g\n",
                             static_cast<unsigned long>(row),
                             static_cast<unsigned long>(col),
                             value);
                return;
            }
            sum += value;
        }
        if (std::fabs(sum - 1.0) > 1e-10) {
            ++failures;
            std::fprintf(stderr,
                         "  FAIL probability row %lu sum: %.17g\n",
                         static_cast<unsigned long>(row),
                         sum);
            return;
        }
    }
}

void check_fixture(int& failures,
                   const ::pls4all::test::fixtures::PlsLogisticFixture& fixture) {
    ::pls4all::core::Context ctx;
    ::pls4all::core::Config cfg;
    cfg.algorithm = P4A_ALGO_PLS_DA;
    cfg.solver = P4A_SOLVER_NIPALS;
    cfg.deflation = P4A_DEFLATION_REGRESSION;
    cfg.n_components = fixture.n_components;

    p4a_matrix_view_t X = matrix_view(fixture.X);
    p4a_matrix_view_t labels = label_view(fixture.labels);
    ::pls4all::core::PlsLogisticResult result;
    CHECK_EQ(::pls4all::core::fit_predict_pls_logistic(ctx,
                                                       cfg,
                                                       X,
                                                       labels,
                                                       fixture.n_classes,
                                                       result),
             P4A_OK);
    CHECK_EQ(result.n_samples, fixture.X.rows);
    CHECK_EQ(result.n_classes, fixture.n_classes);
    CHECK_EQ(result.n_components, fixture.n_components);
    check_predictions(failures, result.predictions, fixture.predictions);
    check_close_values(failures, "pls_logistic_decision_scores", result.decision_scores, fixture.decision_scores);
    check_close_values(failures, "pls_logistic_probabilities", result.probabilities, fixture.probabilities);
    check_close_values(failures, "pls_logistic_intercepts", result.intercepts, fixture.intercepts);
    check_close_values(failures, "pls_logistic_coefficients", result.coefficients, fixture.coefficients);
    check_probability_rows(failures, result);
}

}  // namespace

TEST(pls_logistic_phase4q, generated_fixture_matches_python_reference) {
    for (const auto& fixture : ::pls4all::test::fixtures::kPlsLogisticFixtures) {
        check_fixture(failures, fixture);
    }
}

TEST(pls_logistic_phase4q, rejects_invalid_inputs) {
    ::pls4all::core::Context ctx;
    ::pls4all::core::Config cfg;
    cfg.n_components = 1;
    double x_values[] = {
        0.0, 0.1,
        0.2, 0.3,
        1.0, 1.1,
        1.2, 1.3,
    };
    std::int32_t labels_values[] = {0, 0, 1, 1};
    p4a_matrix_view_t X{};
    p4a_matrix_view_t labels{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&X, x_values, 4, 2, P4A_DTYPE_F64), P4A_OK);
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&labels, labels_values, 4, 1, P4A_DTYPE_I32), P4A_OK);

    ::pls4all::core::PlsLogisticResult result;
    CHECK_EQ(::pls4all::core::fit_predict_pls_logistic(ctx, cfg, X, labels, 1, result),
             P4A_ERR_INVALID_ARGUMENT);

    std::int32_t missing_class_values[] = {0, 0, 0, 1};
    p4a_matrix_view_t missing{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&missing, missing_class_values, 4, 1, P4A_DTYPE_I32), P4A_OK);
    CHECK_EQ(::pls4all::core::fit_predict_pls_logistic(ctx, cfg, X, missing, 3, result),
             P4A_ERR_INVALID_ARGUMENT);

    p4a_matrix_view_t mismatched = labels;
    mismatched.rows = 3;
    CHECK_EQ(::pls4all::core::fit_predict_pls_logistic(ctx, cfg, X, mismatched, 2, result),
             P4A_ERR_SHAPE_MISMATCH);
}
