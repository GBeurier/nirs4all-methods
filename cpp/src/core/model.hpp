// SPDX-License-Identifier: CeCILL-2.1
//
// Internal fitted-model state and dependency-free PLS regression solvers.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "pls4all/p4a.h"

#include "core/config.hpp"
#include "core/context.hpp"

namespace pls4all::core {

struct Array {
    p4a_dtype_t dtype{P4A_DTYPE_F64};
    std::int64_t rows{0};
    std::int64_t cols{0};
    std::vector<double> values;
};

class Model {
  public:
    p4a_algorithm_t algorithm{P4A_ALGO_PLS_REGRESSION};
    p4a_solver_t solver{P4A_SOLVER_NIPALS};
    p4a_deflation_t deflation{P4A_DEFLATION_REGRESSION};

    std::int64_t n_samples{0};
    std::int32_t n_features{0};
    std::int32_t n_targets{0};
    std::int32_t n_components{0};

    std::int32_t center_x{1};
    std::int32_t scale_x{1};
    std::int32_t center_y{1};
    std::int32_t scale_y{1};
    std::int32_t store_scores{0};
    double tol{1e-6};
    std::int32_t max_iter{500};

    // Row-major arrays.
    std::vector<double> x_mean;       // n_features
    std::vector<double> x_scale;      // n_features
    std::vector<double> y_mean;       // n_targets
    std::vector<double> y_scale;      // n_targets
    std::vector<double> coefficients; // n_features x n_targets
    std::vector<double> weights_w;    // n_features x n_components
    std::vector<double> loadings_p;   // n_features x n_components
    std::vector<double> y_loadings_q; // n_targets x n_components
    std::vector<double> rotations_r;  // n_features x n_components
    std::vector<double> scores_t;     // n_samples x n_components, optional
    std::vector<double> y_scores_u;   // n_samples x n_components, optional
};

[[nodiscard]] p4a_status_t fit_pls_regression_nipals(
    Context& ctx,
    const Config& cfg,
    const p4a_matrix_view_t& X,
    const p4a_matrix_view_t& Y,
    std::unique_ptr<Model>& out_model);

[[nodiscard]] p4a_status_t fit_pls_regression_simpls(
    Context& ctx,
    const Config& cfg,
    const p4a_matrix_view_t& X,
    const p4a_matrix_view_t& Y,
    std::unique_ptr<Model>& out_model);

[[nodiscard]] p4a_status_t predict_into(
    Context& ctx,
    const Model& model,
    const p4a_matrix_view_t& X,
    p4a_matrix_view_t& out);

[[nodiscard]] p4a_status_t transform_into(
    Context& ctx,
    const Model& model,
    const p4a_matrix_view_t& X,
    p4a_matrix_view_t& out_scores);

[[nodiscard]] p4a_status_t model_array(
    Context& ctx,
    const Model& model,
    p4a_model_array_t which,
    std::unique_ptr<Array>& out);

}  // namespace pls4all::core

struct p4a_model_s : public ::pls4all::core::Model {};
struct p4a_array_s : public ::pls4all::core::Array {};
