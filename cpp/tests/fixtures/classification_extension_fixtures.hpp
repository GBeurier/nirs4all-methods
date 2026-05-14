// SPDX-License-Identifier: CeCILL-2.1
// Generated mechanically from parity/fixtures/*classification-extensions*.json.
#pragma once

#include <cstddef>
#include <cstdint>

#include "phase1_fixtures.hpp"

namespace pls4all::test::fixtures {

struct ClassificationExtensionLabelRef {
    std::int64_t rows;
    std::int64_t cols;
    const std::int32_t* values;
    std::size_t size;
};

struct ClassificationExtensionFixture {
    const char* id;
    const char* kind;
    std::int32_t n_classes;
    std::int32_t n_bins;
    ClassificationExtensionLabelRef labels;
    MatrixRef scores;
    MatrixRef summary_metrics;
    MatrixRef per_class_metrics;
    MatrixRef confusion_matrix;
    MatrixRef calibration_curve;
};

inline const std::int32_t synthetic_classification_multiclass_metrics_v1_labels[] = {
    0, 1, 2, 1, 0, 2, 2, 0,
    1, 0, 2, 1, 0, 1, 2,
};

inline const double synthetic_classification_multiclass_metrics_v1_scores[] = {
    0.81999999999999995, 0.11, 0.070000000000000007, 0.20999999999999999,
    0.62, 0.17000000000000001, 0.16, 0.31,
    0.53000000000000003, 0.34000000000000002, 0.45000000000000001, 0.20999999999999999,
    0.46999999999999997, 0.41999999999999998, 0.11, 0.12,
    0.39000000000000001, 0.48999999999999999, 0.28000000000000003, 0.26000000000000001,
    0.46000000000000002, 0.51000000000000001, 0.17999999999999999, 0.31,
    0.22, 0.35999999999999999, 0.41999999999999998, 0.39000000000000001,
    0.44, 0.17000000000000001, 0.17000000000000001, 0.23000000000000001,
    0.59999999999999998, 0.27000000000000002, 0.54000000000000004, 0.19,
    0.57999999999999996, 0.23999999999999999, 0.17999999999999999, 0.29999999999999999,
    0.47999999999999998, 0.22, 0.23999999999999999, 0.29999999999999999,
    0.46000000000000002,
};

inline const double synthetic_classification_multiclass_metrics_v1_summary_metrics[] = {
    15, 3, 0.8666666666666667, 0.8666666666666667,
    0.93333333333333324, 0.87777777777777777, 0.86599326599326609, 0.97999999999999998,
    0.8666666666666667, 0.8666666666666667, 0.8666666666666667,
};

inline const double synthetic_classification_multiclass_metrics_v1_per_class_metrics[] = {
    0.80000000000000004, 1, 1, 0.88888888888888895,
    1, 0.80000000000000004, 0.90000000000000002, 0.80000000000000004,
    0.80000000000000016, 0.93999999999999995, 1, 0.90000000000000002,
    0.83333333333333337, 0.90909090909090906, 1,
};

inline const double synthetic_classification_multiclass_metrics_v1_confusion_matrix[] = {
    4, 1, 0, 0,
    4, 1, 0, 0,
    5,
};

inline const std::int32_t synthetic_classification_calibration_curve_v1_labels[] = {
    0, 0, 1, 0, 1, 1, 0, 1,
    0, 1, 0, 1, 1, 0, 1, 0,
};

inline const double synthetic_classification_calibration_curve_v1_scores[] = {
    0.029999999999999999, 0.17999999999999999, 0.23999999999999999, 0.31,
    0.37, 0.41999999999999998, 0.46000000000000002, 0.51000000000000001,
    0.57999999999999996, 0.63, 0.68000000000000005, 0.73999999999999999,
    0.79000000000000004, 0.82999999999999996, 0.91000000000000003, 1,
};

inline const double synthetic_classification_calibration_curve_v1_calibration_curve[] = {
    0, 0.20000000000000001, 2, 0.105,
    0, 0.20000000000000001, 0.40000000000000002, 3,
    0.3066666666666667, 0.66666666666666663, 0.40000000000000002, 0.59999999999999998,
    4, 0.49250000000000005, 0.5, 0.59999999999999998,
    0.80000000000000004, 4, 0.70999999999999996, 0.75,
    0.80000000000000004, 1, 3, 0.91333333333333344,
    0.33333333333333331,
};

inline const ClassificationExtensionFixture kClassificationExtensionFixtures[] = {
    {
        "synthetic_classification_multiclass_metrics_v1",
        "multiclass",
        3,
        0,
        ClassificationExtensionLabelRef{15, 1, synthetic_classification_multiclass_metrics_v1_labels, sizeof(synthetic_classification_multiclass_metrics_v1_labels) / sizeof(std::int32_t)},
        MatrixRef{15, 3, synthetic_classification_multiclass_metrics_v1_scores, sizeof(synthetic_classification_multiclass_metrics_v1_scores) / sizeof(double), false},
        MatrixRef{1, 11, synthetic_classification_multiclass_metrics_v1_summary_metrics, sizeof(synthetic_classification_multiclass_metrics_v1_summary_metrics) / sizeof(double), false},
        MatrixRef{3, 5, synthetic_classification_multiclass_metrics_v1_per_class_metrics, sizeof(synthetic_classification_multiclass_metrics_v1_per_class_metrics) / sizeof(double), false},
        MatrixRef{3, 3, synthetic_classification_multiclass_metrics_v1_confusion_matrix, sizeof(synthetic_classification_multiclass_metrics_v1_confusion_matrix) / sizeof(double), false},
        MatrixRef{0, 0, nullptr, 0, false}
    },
    {
        "synthetic_classification_calibration_curve_v1",
        "calibration",
        0,
        5,
        ClassificationExtensionLabelRef{16, 1, synthetic_classification_calibration_curve_v1_labels, sizeof(synthetic_classification_calibration_curve_v1_labels) / sizeof(std::int32_t)},
        MatrixRef{16, 1, synthetic_classification_calibration_curve_v1_scores, sizeof(synthetic_classification_calibration_curve_v1_scores) / sizeof(double), false},
        MatrixRef{0, 0, nullptr, 0, false},
        MatrixRef{0, 0, nullptr, 0, false},
        MatrixRef{0, 0, nullptr, 0, false},
        MatrixRef{5, 5, synthetic_classification_calibration_curve_v1_calibration_curve, sizeof(synthetic_classification_calibration_curve_v1_calibration_curve) / sizeof(double), false}
    }
};

}  // namespace pls4all::test::fixtures
