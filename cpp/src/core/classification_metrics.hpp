// SPDX-License-Identifier: CeCILL-2.1
//
// Internal binary classification metric kernels for PLS-DA style scores.

#pragma once

#include <cstdint>

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

[[nodiscard]] p4a_status_t compute_binary_classification_metrics(
    Context& ctx,
    const p4a_matrix_view_t& labels,
    const p4a_matrix_view_t& scores,
    double threshold,
    BinaryClassificationMetrics& out);

}  // namespace pls4all::core
