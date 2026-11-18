// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
//
// Model-intrinsic structure and importances of fitted estimators and role
// pipelines (n4m/estimator.h, ABI 2.24): affine maps, latent structure,
// VIP / target-projection selectivity ratio / standardized coefficients and
// the raw-input affine map of an affine pipeline. Generic over every adapter
// that exposes its N4MM model (Adapter::linear_model) or declares the
// transformer traits; no per-method lists.

#pragma once

#include <cstdint>
#include <vector>

#include "core/estimator/role_pipeline.hpp"
#include "core/estimator/spec.hpp"

namespace n4m::estimator {

// predict(X) = X B + b: B (n_features_in x n_outputs, row-major), b.
n4m_status_t affine_map(n4m_context_t* ctx, const n4m_estimator_s& est,
                        std::vector<double>& coefficients, std::vector<double>& intercept);

struct LatentStructure {
    std::int64_t n_features = 0;
    std::int64_t n_targets = 0;
    std::int64_t n_components = 0;
    const std::vector<double>* x_weights = nullptr;
    const std::vector<double>* x_rotations = nullptr;
    const std::vector<double>* x_loadings = nullptr;
    const std::vector<double>* y_loadings = nullptr;
    const std::vector<double>* x_mean = nullptr;
    const std::vector<double>* x_scale = nullptr;
};

// Borrowed views of the fitted latent arrays (valid while est is unchanged).
n4m_status_t latent_structure(n4m_context_t* ctx, const n4m_estimator_s& est,
                              LatentStructure& out);

// n4m_importance_kind_t profile on rows X of the estimator's input space.
n4m_status_t importance(n4m_context_t* ctx, const n4m_estimator_s& est,
                        const n4m_matrix_view_t* X, std::int32_t kind,
                        std::int64_t output_index, std::vector<double>& out);

// Raw-input affine map of an affine role pipeline.
n4m_status_t pipeline_input_affine(n4m_context_t* ctx, const n4m_role_pipeline_s& pipeline,
                                   std::vector<double>& coefficients,
                                   std::vector<double>& intercept);

}  // namespace n4m::estimator
