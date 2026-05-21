// SPDX-License-Identifier: CECILL-2.1

#include "core/classification_metrics.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "fixtures/classification_extension_fixtures.hpp"
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

n4m_matrix_view_t label_view(const ::n4m::test::fixtures::ClassificationExtensionLabelRef& ref) {
    n4m_matrix_view_t view{};
    view.data = const_cast<std::int32_t*>(ref.values);
    view.rows = ref.rows;
    view.cols = ref.cols;
    view.row_stride = ref.cols > 0 ? ref.cols : 1;
    view.col_stride = 1;
    view.dtype = N4M_DTYPE_I32;
    return view;
}

n4m_matrix_view_t score_view(const ::n4m::test::fixtures::MatrixRef& ref) {
    n4m_matrix_view_t view{};
    view.data = const_cast<double*>(ref.values);
    view.rows = ref.rows;
    view.cols = ref.cols;
    view.row_stride = ref.cols > 0 ? ref.cols : 1;
    view.col_stride = 1;
    view.dtype = N4M_DTYPE_F64;
    return view;
}

void check_vector_ref(int& failures,
                      const char* label,
                      const std::vector<double>& actual,
                      const ::n4m::test::fixtures::MatrixRef& expected) {
    CHECK_EQ(actual.size(), expected.size);
    const std::size_t n = std::min(actual.size(), expected.size);
    for (std::size_t i = 0; i < n; ++i) {
        char buffer[128];
        std::snprintf(buffer, sizeof(buffer), "%s[%lu]", label, static_cast<unsigned long>(i));
        check_metric(failures, buffer, actual[i], expected.values[i]);
    }
}

void check_multiclass_fixture(int& failures,
                              const ::n4m::test::fixtures::ClassificationExtensionFixture& fixture) {
    ::n4m::core::Context ctx;
    n4m_matrix_view_t labels = label_view(fixture.labels);
    n4m_matrix_view_t scores = score_view(fixture.scores);
    ::n4m::core::MulticlassClassificationMetrics metrics{};
    CHECK_EQ(::n4m::core::compute_multiclass_classification_metrics(ctx,
                                                                        labels,
                                                                        scores,
                                                                        fixture.n_classes,
                                                                        metrics),
             N4M_OK);

    CHECK_EQ(fixture.summary_metrics.size, static_cast<std::size_t>(11U));
    const double* expected = fixture.summary_metrics.values;
    check_metric(failures, "count", static_cast<double>(metrics.count), expected[0]);
    check_metric(failures, "n_classes", static_cast<double>(metrics.n_classes), expected[1]);
    check_metric(failures, "accuracy", metrics.accuracy, expected[2]);
    check_metric(failures, "macro_sensitivity", metrics.macro_sensitivity, expected[3]);
    check_metric(failures, "macro_specificity", metrics.macro_specificity, expected[4]);
    check_metric(failures, "macro_precision", metrics.macro_precision, expected[5]);
    check_metric(failures, "macro_f1", metrics.macro_f1, expected[6]);
    check_metric(failures, "macro_auc_ovr", metrics.macro_auc_ovr, expected[7]);
    check_metric(failures, "micro_precision", metrics.micro_precision, expected[8]);
    check_metric(failures, "micro_recall", metrics.micro_recall, expected[9]);
    check_metric(failures, "micro_f1", metrics.micro_f1, expected[10]);

    std::vector<double> per_class;
    per_class.reserve(metrics.sensitivity.size() * 5U);
    for (std::size_t cls = 0; cls < metrics.sensitivity.size(); ++cls) {
        per_class.push_back(metrics.sensitivity[cls]);
        per_class.push_back(metrics.specificity[cls]);
        per_class.push_back(metrics.precision[cls]);
        per_class.push_back(metrics.f1[cls]);
        per_class.push_back(metrics.auc_ovr[cls]);
    }
    check_vector_ref(failures, "per_class", per_class, fixture.per_class_metrics);

    std::vector<double> confusion;
    confusion.reserve(metrics.confusion_matrix.size());
    for (const std::int64_t value : metrics.confusion_matrix) {
        confusion.push_back(static_cast<double>(value));
    }
    check_vector_ref(failures, "confusion", confusion, fixture.confusion_matrix);
}

void check_calibration_fixture(int& failures,
                               const ::n4m::test::fixtures::ClassificationExtensionFixture& fixture) {
    ::n4m::core::Context ctx;
    n4m_matrix_view_t labels = label_view(fixture.labels);
    n4m_matrix_view_t scores = score_view(fixture.scores);
    ::n4m::core::BinaryCalibrationCurve curve{};
    CHECK_EQ(::n4m::core::compute_binary_calibration_curve(ctx,
                                                               labels,
                                                               scores,
                                                               fixture.n_bins,
                                                               curve),
             N4M_OK);

    std::vector<double> flat;
    flat.reserve(static_cast<std::size_t>(curve.n_bins) * 5U);
    for (std::size_t bin = 0; bin < curve.counts.size(); ++bin) {
        flat.push_back(curve.bin_lower[bin]);
        flat.push_back(curve.bin_upper[bin]);
        flat.push_back(static_cast<double>(curve.counts[bin]));
        flat.push_back(curve.mean_score[bin]);
        flat.push_back(curve.positive_rate[bin]);
    }
    check_vector_ref(failures, "calibration", flat, fixture.calibration_curve);
}

}  // namespace

TEST(classification_extensions_phase3r, generated_extension_fixtures_match_numpy_reference) {
    for (const auto& fixture : ::n4m::test::fixtures::kClassificationExtensionFixtures) {
        if (std::strcmp(fixture.kind, "multiclass") == 0) {
            check_multiclass_fixture(failures, fixture);
        } else if (std::strcmp(fixture.kind, "calibration") == 0) {
            check_calibration_fixture(failures, fixture);
        } else {
            CHECK(false);
        }
    }
}

TEST(classification_extensions_phase3r, rejects_invalid_extension_inputs) {
    ::n4m::core::Context ctx;
    std::int32_t labels_values[] = {0, 1, 2, 0};
    double scores_values[] = {
        0.7, 0.2, 0.1,
        0.2, 0.6, 0.2,
        0.1, 0.3, 0.6,
        0.4, 0.5, 0.1,
    };
    n4m_matrix_view_t labels{};
    n4m_matrix_view_t scores{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&labels, labels_values, 4, 1, N4M_DTYPE_I32), N4M_OK);
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&scores, scores_values, 4, 3, N4M_DTYPE_F64), N4M_OK);

    ::n4m::core::MulticlassClassificationMetrics metrics{};
    CHECK_EQ(::n4m::core::compute_multiclass_classification_metrics(ctx,
                                                                        labels,
                                                                        scores,
                                                                        1,
                                                                        metrics),
             N4M_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::n4m::core::compute_multiclass_classification_metrics(ctx,
                                                                        labels,
                                                                        scores,
                                                                        4,
                                                                        metrics),
             N4M_ERR_SHAPE_MISMATCH);

    std::int32_t missing_class_labels_values[] = {0, 1, 1, 0};
    n4m_matrix_view_t missing_class_labels{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&missing_class_labels,
                                           missing_class_labels_values,
                                           4,
                                           1,
                                           N4M_DTYPE_I32),
             N4M_OK);
    CHECK_EQ(::n4m::core::compute_multiclass_classification_metrics(ctx,
                                                                        missing_class_labels,
                                                                        scores,
                                                                        3,
                                                                        metrics),
             N4M_ERR_INVALID_ARGUMENT);

    ::n4m::core::BinaryCalibrationCurve curve{};
    double binary_scores_values[] = {0.1, 0.4, 0.8, 1.1};
    n4m_matrix_view_t binary_scores{};
    CHECK_EQ(n4m_matrix_view_init_rowmajor(&binary_scores,
                                           binary_scores_values,
                                           4,
                                           1,
                                           N4M_DTYPE_F64),
             N4M_OK);
    CHECK_EQ(::n4m::core::compute_binary_calibration_curve(ctx,
                                                               labels,
                                                               binary_scores,
                                                               0,
                                                               curve),
             N4M_ERR_INVALID_ARGUMENT);
    CHECK_EQ(::n4m::core::compute_binary_calibration_curve(ctx,
                                                               labels,
                                                               binary_scores,
                                                               4,
                                                               curve),
             N4M_ERR_INVALID_ARGUMENT);
}
