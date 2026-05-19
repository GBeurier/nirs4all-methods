// SPDX-License-Identifier: CECILL-2.1
//
// Internal binary classification metric kernels for PLS-DA style scores.

#pragma once

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

#include "core/context.hpp"

namespace pls4all::core {

struct BinaryClassificationMetrics {
    std::int64_t count{0};
    std::int64_t positives{0};
    std::int64_t negatives{0};
    std::int64_t true_positive{0};
    std::int64_t true_negative{0};
    std::int64_t false_positive{0};
    std::int64_t false_negative{0};

    double threshold{0.5};
    double sensitivity{0.0};
    double specificity{0.0};
    double balanced_accuracy{0.0};
    double accuracy{0.0};
    double precision{0.0};
    double f1{0.0};
    double mcc{0.0};
    double auc{0.0};
};

struct MulticlassClassificationMetrics {
    std::int64_t count{0};
    std::int32_t n_classes{0};

    // Row-major n_classes x n_classes, rows are actual labels, columns predicted labels.
    std::vector<std::int64_t> confusion_matrix;

    std::vector<double> sensitivity;
    std::vector<double> specificity;
    std::vector<double> precision;
    std::vector<double> f1;
    std::vector<double> auc_ovr;

    double accuracy{0.0};
    double macro_sensitivity{0.0};
    double macro_specificity{0.0};
    double macro_precision{0.0};
    double macro_f1{0.0};
    double macro_auc_ovr{0.0};
    double micro_precision{0.0};
    double micro_recall{0.0};
    double micro_f1{0.0};
};

struct BinaryCalibrationCurve {
    std::int32_t n_bins{0};
    std::vector<std::int64_t> counts;
    std::vector<double> bin_lower;
    std::vector<double> bin_upper;
    std::vector<double> mean_score;
    std::vector<double> positive_rate;
};

[[nodiscard]] p4a_status_t compute_binary_classification_metrics(
    Context& ctx,
    const p4a_matrix_view_t& labels,
    const p4a_matrix_view_t& scores,
    double threshold,
    BinaryClassificationMetrics& out);

[[nodiscard]] p4a_status_t compute_multiclass_classification_metrics(
    Context& ctx,
    const p4a_matrix_view_t& labels,
    const p4a_matrix_view_t& scores,
    std::int32_t n_classes,
    MulticlassClassificationMetrics& out);

[[nodiscard]] p4a_status_t compute_binary_calibration_curve(
    Context& ctx,
    const p4a_matrix_view_t& labels,
    const p4a_matrix_view_t& scores,
    std::int32_t n_bins,
    BinaryCalibrationCurve& out);

}  // namespace pls4all::core
