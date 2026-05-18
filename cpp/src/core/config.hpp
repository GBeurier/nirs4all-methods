// SPDX-License-Identifier: CECILL-2.1
//
// Internal Config — POD-style storage for every knob declared on the
// public p4a_config_t handle. Setter validation lives in the c_api_config
// wrappers; the Config class itself only holds values.

#pragma once

#include <cstdint>

#include "pls4all/p4a.h"

namespace pls4all::core {

class Config {
  public:
    Config() noexcept;

    Config(const Config&) noexcept = default;
    Config& operator=(const Config&) noexcept = default;
    Config(Config&&) noexcept = default;
    Config& operator=(Config&&) noexcept = default;

    // Algorithm / solver / deflation
    p4a_algorithm_t algorithm{P4A_ALGO_PLS_REGRESSION};
    p4a_solver_t    solver{P4A_SOLVER_NIPALS};
    p4a_deflation_t deflation{P4A_DEFLATION_REGRESSION};

    // Geometry / numerics
    std::int32_t n_components{2};
    std::int32_t center_x{1};
    std::int32_t scale_x{1};
    std::int32_t center_y{1};
    std::int32_t scale_y{1};
    double       tol{1e-6};
    std::int32_t max_iter{500};
    std::int32_t store_scores{0};
    std::int32_t store_diagnostics{0};
    p4a_dtype_t  dtype{P4A_DTYPE_F64};

    // Sparse PLS soft-threshold lambda. 0.0 means no sparsity; only
    // consumed by `fit_pls_sparse_simpls` (P4A_ALGO_SPARSE_PLS). Each
    // weight w[i] is replaced with sign(w[i]) * max(|w[i]| - lambda, 0)
    // before renormalization.
    double sparsity_lambda{0.0};

    // RBF / polynomial kernel parameters consumed by the kernel-algorithm
    // PLS solver when kernel_type != P4A_KERNEL_LINEAR.
    std::int32_t kernel_type{0};      // 0=linear (default), 1=RBF, 2=polynomial
    double       kernel_gamma{0.0};   // RBF: kernel(x,y) = exp(-gamma * |x-y|^2); 0 means 1/n_features.
    double       kernel_coef0{1.0};   // polynomial: (gamma * x.y + coef0)^degree.
    std::int32_t kernel_degree{3};    // polynomial degree.

    // Domain-invariant PLS (di-PLS) regularization weight. Penalizes the
    // squared norm of (W * (mean_source_X - mean_target_X)) at each
    // direction. 0.0 = no DI penalty.
    double di_lambda{0.0};

    // Recursive PLS forgetting factor in (0, 1]; 1.0 = full memory.
    double recursive_forgetting{1.0};

    // Composability hooks (non-owning).
    const p4a_pipeline_t*        pipeline{nullptr};
    const p4a_operator_bank_t*   operator_bank{nullptr};
    const p4a_gating_strategy_t* gating_strategy{nullptr};
};

}  // namespace pls4all::core

// Opaque alias for the C ABI.
struct p4a_config_s : public ::pls4all::core::Config {};
