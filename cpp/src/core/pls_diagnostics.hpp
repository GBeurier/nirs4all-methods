// SPDX-License-Identifier: CECILL-2.1
//
// PLS diagnostics: Hotelling T2, Q-residuals (SPE), DModX. Computed from a
// fitted Model and a reference / test design matrix.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "n4m/n4m.h"

#include "core/common/context.hpp"
#include "core/model.hpp"

namespace n4m::core {

// Every statistic works on rows mapped into the model's X space exactly as
// n4m_model_transform maps them (fitted SNV/Savitzky-Golay pipeline when the
// model owns one, then x_mean / x_scale), so they hold for autoscaled models.
//
// Latent directions: the components whose rotation column is nonzero. A PLS
// component flagged in N4M_MODEL_UNRESOLVED_COMPONENTS is a null component
// (zero W, P, R columns) and has no X direction; a PCR component flagged only
// for its Y side keeps its resolved X direction. A model with null components
// has the T2, Q and DModX of the model truncated to its resolved components.
[[nodiscard]] std::size_t pls_latent_directions(const Model& model);

// Hotelling T2 of each row of X over the k latent directions:
// T2[i] = sum_k t_ik^2 / var_k, t = x_model @ rotations_R. The variances come
// from the model's stored scores_t (cfg.store_scores == 1 at fit time) when
// X_reference is NULL, or from t_reference = x_reference_model @ rotations_R
// otherwise; a kept direction with zero reference variance is refused. With
// no latent direction (every component null) T2 is 0 for every row.
//
// out_t2 is resized to X.rows.
[[nodiscard]] n4m_status_t pls_hotelling_t2(
    Context& ctx,
    const Model& model,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t* X_reference,
    std::vector<double>& out_t2);

// Squared prediction error of the X reconstruction from latent scores, in the
// model's X space:
//
// Q[i] = sum_j (x_model[i,j] - x_hat[i,j])^2
// where x_hat = t @ loadings_P^T, t = x_model @ rotations_R.
//
// out_q is resized to X.rows.
[[nodiscard]] n4m_status_t pls_q_residuals(
    Context& ctx,
    const Model& model,
    const n4m_matrix_view_t& X,
    std::vector<double>& out_q);

// Distance-to-model X (DModX), with k the number of latent directions:
//
// DModX[i] = sqrt(Q[i] / (n_features - k)) / sigma_ref
// where sigma_ref = sqrt(sum Q_ref / ((n_features - k) (n_ref - k - 1)))
// comes from X_reference when it is provided (n_ref >= k + 2). When
// X_reference is NULL the function reports the unnormalized residual standard
// deviation per sample, sqrt(Q[i] / (n_features - k)). n_features <= k is
// refused.
[[nodiscard]] n4m_status_t pls_dmodx(
    Context& ctx,
    const Model& model,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t* X_reference,
    std::vector<double>& out_dmodx);

}  // namespace n4m::core
