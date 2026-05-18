// SPDX-License-Identifier: CECILL-2.1

#include "core/classification_metrics.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <vector>

#include "fixtures/classification_metrics_fixtures.hpp"
#include "harness.hpp"

namespace {

constexpr double kAbsTol = 1e-12;
constexpr double kRelTol = 1e-12;

void check_metric(int& failures,
                  const char* label,
                  double actual,
                  double expected) {
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

void check_fixture(int& failures,
                   const ::pls4all::test::fixtures::ClassificationMetricsFixture& fixture) {
    ::pls4all::core::Context ctx;
    p4a_matrix_view_t labels{};
    p4a_matrix_view_t scores{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&labels,
                                           const_cast<std::int32_t*>(fixture.labels.values),
                                           fixture.labels.rows,
                                           fixture.labels.cols,
                                           P4A_DTYPE_I32),
             P4A_OK);
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&scores,
                                           const_cast<double*>(fixture.scores.values),
                                           fixture.scores.rows,
                                           fixture.scores.cols,
                                           P4A_DTYPE_F64),
             P4A_OK);

    ::pls4all::core::BinaryClassificationMetrics metrics{};
    CHECK_EQ(::pls4all::core::compute_binary_classification_metrics(ctx,
                                                                    labels,
                                                                    scores,
                                                                    fixture.threshold,
                                                                    metrics),
             P4A_OK);
    CHECK_EQ(fixture.expected.size, static_cast<std::size_t>(16U));
    const double* expected = fixture.expected.values;
    check_metric(failures, "count", static_cast<double>(metrics.count), expected[0]);
    check_metric(failures, "positives", static_cast<double>(metrics.positives), expected[1]);
    check_metric(failures, "negatives", static_cast<double>(metrics.negatives), expected[2]);
    check_metric(failures, "tp", static_cast<double>(metrics.true_positive), expected[3]);
    check_metric(failures, "tn", static_cast<double>(metrics.true_negative), expected[4]);
    check_metric(failures, "fp", static_cast<double>(metrics.false_positive), expected[5]);
    check_metric(failures, "fn", static_cast<double>(metrics.false_negative), expected[6]);
    check_metric(failures, "threshold", metrics.threshold, expected[7]);
    check_metric(failures, "sensitivity", metrics.sensitivity, expected[8]);
    check_metric(failures, "specificity", metrics.specificity, expected[9]);
    check_metric(failures, "balanced_accuracy", metrics.balanced_accuracy, expected[10]);
    check_metric(failures, "accuracy", metrics.accuracy, expected[11]);
    check_metric(failures, "precision", metrics.precision, expected[12]);
    check_metric(failures, "f1", metrics.f1, expected[13]);
    check_metric(failures, "mcc", metrics.mcc, expected[14]);
    check_metric(failures, "auc", metrics.auc, expected[15]);
}

}  // namespace

TEST(classification_metrics_phase3n, generated_binary_fixture_matches_numpy_reference) {
    for (const auto& fixture : ::pls4all::test::fixtures::kClassificationMetricsFixtures) {
        check_fixture(failures, fixture);
    }
}

TEST(classification_metrics_phase3n, rejects_invalid_inputs) {
    ::pls4all::core::Context ctx;
    std::int32_t labels_values[] = {0, 1, 0, 1};
    double scores_values[] = {0.1, 0.8, 0.4, 0.7};
    p4a_matrix_view_t labels{};
    p4a_matrix_view_t scores{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&labels, labels_values, 4, 1, P4A_DTYPE_I32), P4A_OK);
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&scores, scores_values, 4, 1, P4A_DTYPE_F64), P4A_OK);

    ::pls4all::core::BinaryClassificationMetrics metrics{};
    p4a_matrix_view_t mismatched = scores;
    mismatched.rows = 3;
    CHECK_EQ(::pls4all::core::compute_binary_classification_metrics(ctx,
                                                                    labels,
                                                                    mismatched,
                                                                    0.5,
                                                                    metrics),
             P4A_ERR_SHAPE_MISMATCH);

    std::int32_t int_scores_values[] = {0, 1, 0, 1};
    p4a_matrix_view_t int_scores{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&int_scores, int_scores_values, 4, 1, P4A_DTYPE_I32),
             P4A_OK);
    CHECK_EQ(::pls4all::core::compute_binary_classification_metrics(ctx,
                                                                    labels,
                                                                    int_scores,
                                                                    0.5,
                                                                    metrics),
             P4A_ERR_DTYPE_MISMATCH);

    std::int32_t bad_labels_values[] = {0, 1, 2, 1};
    p4a_matrix_view_t bad_labels{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&bad_labels, bad_labels_values, 4, 1, P4A_DTYPE_I32),
             P4A_OK);
    CHECK_EQ(::pls4all::core::compute_binary_classification_metrics(ctx,
                                                                    bad_labels,
                                                                    scores,
                                                                    0.5,
                                                                    metrics),
             P4A_ERR_INVALID_ARGUMENT);

    double bad_scores_values[] = {0.1, 0.8, std::numeric_limits<double>::quiet_NaN(), 0.7};
    p4a_matrix_view_t bad_scores{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&bad_scores, bad_scores_values, 4, 1, P4A_DTYPE_F64),
             P4A_OK);
    CHECK_EQ(::pls4all::core::compute_binary_classification_metrics(ctx,
                                                                    labels,
                                                                    bad_scores,
                                                                    0.5,
                                                                    metrics),
             P4A_ERR_INVALID_ARGUMENT);

    std::int32_t one_class_values[] = {1, 1, 1, 1};
    p4a_matrix_view_t one_class{};
    CHECK_EQ(p4a_matrix_view_init_rowmajor(&one_class, one_class_values, 4, 1, P4A_DTYPE_I32),
             P4A_OK);
    CHECK_EQ(::pls4all::core::compute_binary_classification_metrics(ctx,
                                                                    one_class,
                                                                    scores,
                                                                    0.5,
                                                                    metrics),
             P4A_ERR_INVALID_ARGUMENT);

    CHECK_EQ(::pls4all::core::compute_binary_classification_metrics(
                 ctx,
                 labels,
                 scores,
                 std::numeric_limits<double>::quiet_NaN(),
                 metrics),
             P4A_ERR_INVALID_ARGUMENT);
}
