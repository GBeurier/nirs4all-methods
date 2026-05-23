// SPDX-License-Identifier: CECILL-2.1
//
// Catalog of PLS variants required to close the Overview taxonomy.
// Internal kernels — public ABI exposure ships with the binding tranche.
//
// Each method exposes a `fit_*` entry point that returns its own result
// struct. Predict-time helpers are provided where the model is not the
// usual (coefficients, intercept) linear shape.

#pragma once

#include <cstdint>
#include <vector>

#include "n4m/n4m.h"

#include "core/config.hpp"
#include "core/common/context.hpp"

namespace n4m::core {

// ---- §7 Sparse variants ------------------------------------------------

// Sparse PLS-DA: dummy-encodes Y class labels then runs sparse SIMPLS.
struct SparsePlsDaResult {
    std::int32_t n_classes{0};
    std::vector<double> class_priors;       // n_classes
    std::vector<double> coefficients;       // n_features × n_classes
    std::vector<double> x_mean;
    std::vector<double> y_mean;             // n_classes
};

[[nodiscard]] n4m_status_t fit_sparse_pls_da(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const std::vector<std::int32_t>& y_labels,
    SparsePlsDaResult& out);

// Group sparse PLS: soft-thresholds groups of features together. Groups
// are described by a `group_assignment` vector of length n_features
// (0-based group ids).
struct GroupSparsePlsResult {
    std::int32_t n_features{0};
    std::int32_t n_components{0};
    std::int32_t n_targets{0};
    std::int32_t n_groups{0};
    std::vector<double> coefficients;       // p × q
    std::vector<double> x_mean;
    std::vector<double> y_mean;
};

[[nodiscard]] n4m_status_t fit_group_sparse_pls(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    const std::vector<std::int32_t>& group_assignment,
    double group_lambda,
    GroupSparsePlsResult& out);

// Fused sparse PLS: penalizes consecutive feature pairs (|w_i - w_{i+1}|)
// in addition to the elementwise L1. Returns the standard PLS shape.
[[nodiscard]] n4m_status_t fit_fused_sparse_pls(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    double l1_lambda,
    double fusion_lambda,
    GroupSparsePlsResult& out);

// ---- §1 CPPLS / powered PLS -------------------------------------------

struct CpplsResult {
    std::int32_t n_features{0};
    std::int32_t n_targets{0};
    std::int32_t n_components{0};
    double gamma{0.5};
    std::vector<double> coefficients;
    std::vector<double> x_mean;
    std::vector<double> y_mean;
};

[[nodiscard]] n4m_status_t fit_cppls(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    double gamma,
    CpplsResult& out);

// ---- Weighted / robust / ridge / continuum -----------------------------

struct WeightedPlsResult {
    std::int32_t n_features{0};
    std::int32_t n_targets{0};
    std::int32_t n_components{0};
    std::vector<double> coefficients;
    std::vector<double> x_mean;
    std::vector<double> y_mean;
};

// Weighted PLS: each sample has a positive weight; pre-multiplies (X, Y)
// rows by sqrt(weight) and runs SIMPLS.
[[nodiscard]] n4m_status_t fit_weighted_pls(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    const std::vector<double>& sample_weights,
    WeightedPlsResult& out);

// Robust PLS: IRLS with Huber weights re-derived from residuals.
[[nodiscard]] n4m_status_t fit_robust_pls(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    double huber_k,
    std::int32_t max_irls_iter,
    WeightedPlsResult& out);

// Ridge PLS: solves (X' X + lambda I) beta = X' Y via SVD; then orders
// directions by Y-explained variance and keeps n_components.
[[nodiscard]] n4m_status_t fit_ridge_pls(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    double ridge_lambda,
    WeightedPlsResult& out);

// Continuum regression: tau in [0, 1] interpolates between PCR (tau=0),
// PLS (tau=0.5) and OLS (tau=1).
[[nodiscard]] n4m_status_t fit_continuum_regression(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    double tau,
    WeightedPlsResult& out);

// ---- §13 classifier / GLM heads ----------------------------------------

struct PlsGlmResult {
    std::int32_t n_features{0};
    std::int32_t n_classes{0};
    std::vector<double> coefficients;       // p × n_classes
    std::vector<double> intercept;          // n_classes
    std::int32_t n_components{0};
    bool poisson{false};
};

// PLS-GLM: PLS-reduced design feeding a softmax / Poisson IRLS.
[[nodiscard]] n4m_status_t fit_pls_glm(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,           // n × q (count or one-hot)
    bool poisson,
    PlsGlmResult& out);

struct PlsQdaResult {
    std::int32_t n_classes{0};
    std::int32_t n_components{0};
    std::vector<double> class_means;        // n_classes × n_components
    std::vector<double> class_covariances;  // n_classes × n_components × n_components
    std::vector<double> log_class_priors;
    std::vector<double> rotations_r;        // p × n_components
    std::vector<double> x_mean;
};

[[nodiscard]] n4m_status_t fit_pls_qda(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const std::vector<std::int32_t>& y_labels,
    PlsQdaResult& out);

struct PlsCoxResult {
    std::int32_t n_features{0};
    std::int32_t n_components{0};
    std::vector<double> coefficients;       // p (linear predictor coefficient)
    std::vector<double> baseline_hazard;    // n_unique_event_times
    std::vector<double> event_times;
    std::vector<double> x_mean;
};

[[nodiscard]] n4m_status_t fit_pls_cox(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const std::vector<double>& survival_times,
    const std::vector<std::int32_t>& event_indicators,
    PlsCoxResult& out);

// ---- Calibration transfer + MIR-PLS + missing-aware NIPALS -------------

struct PdsResult {
    std::int32_t window_half_width{0};
    std::vector<double> transformation;     // (window × p_target) × p_source
};

[[nodiscard]] n4m_status_t fit_pds(
    Context& ctx,
    const n4m_matrix_view_t& X_source,
    const n4m_matrix_view_t& X_target,
    std::int32_t window_half_width,
    PdsResult& out);

struct DsResult {
    std::vector<double> transformation;     // p_source × p_target
    std::vector<double> bias;               // p_target
};

[[nodiscard]] n4m_status_t fit_ds(
    Context& ctx,
    const n4m_matrix_view_t& X_source,
    const n4m_matrix_view_t& X_target,
    DsResult& out);

struct MirPlsResult {
    std::vector<double> coefficients;       // p × q  (X → Y mapping)
    std::vector<double> x_mean;
    std::vector<double> y_mean;
};

[[nodiscard]] n4m_status_t fit_mir_pls(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    MirPlsResult& out);

// Missing-aware NIPALS: same shape as a regular PLS regression model but
// tolerates NaN entries in X (replaced with current iterate during NIPALS).
[[nodiscard]] n4m_status_t fit_missing_aware_nipals(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    WeightedPlsResult& out);

// ---- §18 approximate-PRESS / Bayesian rules ----------------------------

struct ApproximatePressResult {
    std::int32_t n_components{0};
    std::int32_t selected_n_components{0};
    std::vector<double> press_per_component;
    std::vector<double> rmse_per_component;
};

[[nodiscard]] n4m_status_t approximate_press(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::int32_t max_components,
    ApproximatePressResult& out);

// ---- §20 Ensembles ----------------------------------------------------

struct EnsemblePlsResult {
    std::int32_t n_estimators{0};
    std::int32_t n_features{0};
    std::int32_t n_targets{0};
    std::int32_t n_components{0};
    // Averaged coefficients across base learners.
    std::vector<double> coefficients;
    std::vector<double> x_mean;
    std::vector<double> y_mean;
};

[[nodiscard]] n4m_status_t fit_bagging_pls(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::int32_t n_estimators,
    std::uint64_t seed,
    EnsemblePlsResult& out);

[[nodiscard]] n4m_status_t fit_boosting_pls(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::int32_t n_estimators,
    double learning_rate,
    EnsemblePlsResult& out);

[[nodiscard]] n4m_status_t fit_random_subspace_pls(
    Context& ctx,
    const Config& cfg,
    const n4m_matrix_view_t& X,
    const n4m_matrix_view_t& Y,
    std::int32_t n_estimators,
    std::int32_t features_per_subspace,
    std::uint64_t seed,
    EnsemblePlsResult& out);

}  // namespace n4m::core
