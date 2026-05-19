// SPDX-License-Identifier: CECILL-2.1
//
// Non-linear kernel PLS (§10.2). Supports RBF, polynomial, sigmoid via
// the Gram-matrix dual path. The model stores the kernel parameters and
// the training X so predictions can recompute the test-vs-train Gram
// matrix.

#pragma once

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

#include "core/config.hpp"
#include "core/context.hpp"

namespace pls4all::core {

enum class KernelType : std::int32_t {
    LINEAR     = 0,
    RBF        = 1,
    POLYNOMIAL = 2,
    SIGMOID    = 3,
};

struct KernelPlsResult {
    std::int64_t n_train{0};
    std::int32_t n_features{0};
    std::int32_t n_targets{0};
    std::int32_t n_components{0};

    KernelType kernel_type{KernelType::LINEAR};
    double gamma{0.0};
    double coef0{1.0};
    std::int32_t degree{3};

    std::vector<double> x_train;          // n_train × n_features row-major
    std::vector<double> y_mean;           // q
    std::vector<double> alpha;            // n_train × q (dual coefficients)

    // Cached for prediction-time Gram centering.
    std::vector<double> K_train_row_means;   // n_train
    double               K_train_global_mean{0.0};
};

[[nodiscard]] p4a_status_t fit_kernel_pls(
    Context& ctx,
    const Config& cfg,
    KernelType kernel,
    double gamma,
    double coef0,
    std::int32_t degree,
    const p4a_matrix_view_t& X,
    const p4a_matrix_view_t& Y,
    KernelPlsResult& out);

[[nodiscard]] p4a_status_t predict_kernel_pls(
    Context& ctx,
    const KernelPlsResult& model,
    const p4a_matrix_view_t& X,
    std::vector<double>& out_predictions);

}  // namespace pls4all::core
