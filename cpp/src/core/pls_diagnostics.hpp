// SPDX-License-Identifier: CECILL-2.1
//
// PLS diagnostics: Hotelling T2, Q-residuals (SPE), DModX. Computed from a
// fitted Model and a reference / test design matrix.

#pragma once

#include <cstdint>
#include <vector>

#include "pls4all/p4a.h"

#include "core/context.hpp"
#include "core/model.hpp"

namespace n4m::core {

// Compute training-side Hotelling T2 statistics for each row of X.
//
// X must have the same number of features as the model and is reduced to
// scores via t = (X - x_mean) @ rotations_R. The score variances used to
// normalize the components come from the model's stored scores_t (which
// requires cfg.store_scores == 1 at fit time) when X_reference is empty,
// or from t_reference = (X_reference - x_mean) @ rotations_R otherwise.
//
// out_t2 is resized to X.rows.
[[nodiscard]] n4m_status_t pls_hotelling_t2(
    Context& ctx,
    const Model& model,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t* X_reference,
    std::vector<double>& out_t2);

// Squared prediction error of X reconstruction from latent scores.
//
// Q[i] = sum_j (X_centered[i,j] - x_hat[i,j])^2
// where x_hat = t @ loadings_P^T, t = (X - x_mean) @ rotations_R.
//
// out_q is resized to X.rows.
[[nodiscard]] n4m_status_t pls_q_residuals(
    Context& ctx,
    const Model& model,
    const n4m_matrix_view_t& X,
    std::vector<double>& out_q);

// Distance-to-model X (DModX).
//
// DModX[i] = sqrt(Q[i] / (n_features - n_components)) / sigma_train
// where sigma_train comes from the reference / training set when
// X_reference is provided. When X_reference is NULL the function reports
// the unnormalized residual standard deviation per sample, i.e.
// sqrt(Q[i] / max(1, n_features - n_components)).
[[nodiscard]] n4m_status_t pls_dmodx(
    Context& ctx,
    const Model& model,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t* X_reference,
    std::vector<double>& out_dmodx);

}  // namespace n4m::core
