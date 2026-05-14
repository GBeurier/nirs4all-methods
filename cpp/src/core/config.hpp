// SPDX-License-Identifier: CeCILL-2.1
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

    // Composability hooks (non-owning).
    const p4a_pipeline_t*        pipeline{nullptr};
    const p4a_operator_bank_t*   operator_bank{nullptr};
    const p4a_gating_strategy_t* gating_strategy{nullptr};
};

}  // namespace pls4all::core

// Opaque alias for the C ABI.
struct p4a_config_s : public ::pls4all::core::Config {};
