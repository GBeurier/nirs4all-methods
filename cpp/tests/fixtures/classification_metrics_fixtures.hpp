// SPDX-License-Identifier: CECILL-2.1
// Generated mechanically from parity/fixtures/*classification-metrics*.json.
#pragma once

#include <cstddef>
#include <cstdint>

#include "phase1_fixtures.hpp"

namespace pls4all::test::fixtures {

struct ClassLabelRef {
    std::int64_t rows;
    std::int64_t cols;
    const std::int32_t* values;
    std::size_t size;
};

struct ClassificationMetricsFixture {
    const char* id;
    double threshold;
    ClassLabelRef labels;
    MatrixRef scores;
    MatrixRef expected;
};

inline const std::int32_t synthetic_classification_binary_metrics_v1_labels[] = {
    0, 1, 0, 1, 1, 0, 0, 1,
    0, 1, 1, 0, 1, 0,
};

inline const double synthetic_classification_binary_metrics_v1_scores[] = {
    0.050000000000000003, 0.81000000000000005, 0.34000000000000002, 0.62,
    0.48999999999999999, 0.56999999999999995, 0.23999999999999999, 0.91000000000000003,
    0.45000000000000001, 0.71999999999999997, 0.56999999999999995, 0.12,
    0.66000000000000003, 0.39000000000000001,
};

inline const double synthetic_classification_binary_metrics_v1_expected[] = {
    14, 7, 7, 6,
    6, 1, 1, 0.5,
    0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571,
    0.8571428571428571, 0.8571428571428571, 0.7142857142857143, 0.96938775510204078,
};

inline const ClassificationMetricsFixture kClassificationMetricsFixtures[] = {
    {
        "synthetic_classification_binary_metrics_v1",
        0.5,
        ClassLabelRef{14, 1, synthetic_classification_binary_metrics_v1_labels, sizeof(synthetic_classification_binary_metrics_v1_labels) / sizeof(std::int32_t)},
        MatrixRef{14, 1, synthetic_classification_binary_metrics_v1_scores, sizeof(synthetic_classification_binary_metrics_v1_scores) / sizeof(double), false},
        MatrixRef{1, 16, synthetic_classification_binary_metrics_v1_expected, sizeof(synthetic_classification_binary_metrics_v1_expected) / sizeof(double), false}
    }
};

}  // namespace pls4all::test::fixtures
