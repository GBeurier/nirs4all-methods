// SPDX-License-Identifier: CECILL-2.1
//
// Internal fitted-model state and dependency-free PLS regression solvers.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "n4m/n4m.h"

#include "core/config.hpp"
#include "core/context.hpp"

namespace n4m::core {

struct Array {
    n4m_dtype_t dtype{N4M_DTYPE_F64};
    std::int64_t rows{0};
    std::int64_t cols{0};
    std::vector<double> values;
};

class Model {
  public:
    n4m_algorithm_t algorithm{N4M_ALGO_PLS_REGRESSION};
    n4m_solver_t solver{N4M_SOLVER_NIPALS};
    n4m_deflation_t deflation{N4M_DEFLATION_REGRESSION};

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

[[nodiscard]] n4m_status_t fit_pls_regression_nipals(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::unique_ptr<Model>& out_model);

[[nodiscard]] n4m_status_t fit_pls_regression_simpls(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::unique_ptr<Model>& out_model);

[[nodiscard]] n4m_status_t fit_pls_regression_orthogonal_scores(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::unique_ptr<Model>& out_model);

[[nodiscard]] n4m_status_t fit_pls_regression_svd(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::unique_ptr<Model>& out_model);

[[nodiscard]] n4m_status_t fit_pls_svd(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::unique_ptr<Model>& out_model);

[[nodiscard]] n4m_status_t fit_pls_regression_power(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::unique_ptr<Model>& out_model);

[[nodiscard]] n4m_status_t fit_pls_regression_randomized_svd(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::unique_ptr<Model>& out_model);

[[nodiscard]] n4m_status_t fit_pls_canonical_nipals(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::unique_ptr<Model>& out_model);

[[nodiscard]] n4m_status_t fit_pls_canonical_svd(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::unique_ptr<Model>& out_model);

[[nodiscard]] n4m_status_t fit_opls_nipals(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::unique_ptr<Model>& out_model);

[[nodiscard]] n4m_status_t fit_pls_regression_kernel(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::unique_ptr<Model>& out_model);

[[nodiscard]] n4m_status_t fit_pcr_svd(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::unique_ptr<Model>& out_model);

[[nodiscard]] n4m_status_t fit_pls_sparse_simpls(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::unique_ptr<Model>& out_model);

[[nodiscard]] n4m_status_t fit_pls_regression_nonlinear_kernel(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::unique_ptr<Model>& out_model);

[[nodiscard]] n4m_status_t fit_di_pls(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X_source,
    const n4m_matrix_view_t& Y_source,
    const n4m_matrix_view_t& X_target,
    std::unique_ptr<Model>& out_model);

[[nodiscard]] n4m_status_t fit_model(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::unique_ptr<Model>& out_model);

[[nodiscard]] n4m_status_t predict_into(
    Context& ctx,
    const Model& model,
    const n4m_matrix_view_t& X,
    n4m_matrix_view_t& out);

[[nodiscard]] n4m_status_t transform_into(
    Context& ctx,
    const Model& model,
    const n4m_matrix_view_t& X,
    n4m_matrix_view_t& out_scores);

[[nodiscard]] n4m_status_t model_array(
    Context& ctx,
    const Model& model,
    n4m_model_array_t which,
    std::unique_ptr<Array>& out);

}  // namespace n4m::core

struct n4m_model_s : public ::n4m::core::Model {};
struct n4m_array_s : public ::n4m::core::Array {};
