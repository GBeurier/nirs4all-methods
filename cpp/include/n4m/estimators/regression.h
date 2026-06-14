/* SPDX-License-Identifier: CECILL-2.1 */
/* cpp/include/n4m/estimators/regression.h — estimators.regression methods (ABI 2.0). */
#ifndef N4M_ESTIMATORS_REGRESSION_H
#define N4M_ESTIMATORS_REGRESSION_H
#include "n4m/n4m.h"   /* shared infra: status, matrix view, context, N4M_API */

#ifdef __cplusplus
extern "C" {
#endif

/* Sparse SIMPLS (§7). Soft-thresholds the SIMPLS direction by
 * `sparsity_lambda` at each component. The result contains:
 *   "coefficients" (n_features x n_targets, row-major)
 *   "predictions"  (n_samples  x n_targets, row-major)
 *   "x_mean"       (1 x n_features)
 *   "y_mean"       (1 x n_targets)
 *   "weights_w"    (n_features x n_components)
 *   scalar "rmse"  in-sample RMSE
 */
N4M_API n4m_status_t n4m_estimators_sparse_simpls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    double sparsity_lambda,
    n4m_method_result_t** out_result);

/* Principal Components Regression (PCR). Forces Algorithm.PCR + Solver.SVD
 * on top of the caller's centering/scaling/n_components config. The result
 * contains:
 *   "coefficients" (n_features x n_targets, row-major)
 *   "predictions"  (n_samples  x n_targets, row-major)
 *   "x_mean", "x_scale", "y_mean", "y_scale"
 *   "weights_w", "loadings_p", "rotations_r" (n_features x n_components)
 *   scalar "rmse", scalar "n_components"
 */
N4M_API n4m_status_t n4m_estimators_pcr_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    n4m_method_result_t** out_result);

/* Recursive (moving-window) PLS (§12). Predicts each sample at index >=
 * window_size from a SIMPLS model fitted on the previous window_size rows.
 * The result contains:
 *   "predictions" (n_samples x n_targets) — zeros for warmup samples
 *   int "in_window" (n_samples)           — 1 when predicted, 0 warmup
 *   scalar "rmse_predictable" on samples i >= window_size
 */
N4M_API n4m_status_t n4m_estimators_recursive_pls_run(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t window_size,
    n4m_method_result_t** out_result);

/* CPPLS — Canonical Powered PLS (§1). Each X column is rescaled by its
 * std^gamma before SIMPLS, then coefficients are unscaled. gamma in [0, 1]
 * interpolates between PLS (gamma=0) and a power-rescaled flavour
 * (gamma=1). The result contains:
 *   "coefficients" (n_features x n_targets, row-major)
 *   "predictions"  (n_samples  x n_targets)
 *   "x_mean", "y_mean"
 *   scalar "rmse", scalar "gamma"
 */
N4M_API n4m_status_t n4m_estimators_cppls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    double gamma,
    n4m_method_result_t** out_result);

/* Weighted PLS (§26). Pre-multiplies the centered (X, Y) rows by
 * sqrt(sample_weights[i]) before SIMPLS. `weights` must point to
 * `n_samples` strictly positive finite doubles. The result contains:
 *   "coefficients", "predictions", "x_mean", "y_mean"
 *   scalar "rmse"
 */
N4M_API n4m_status_t n4m_estimators_weighted_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const double* sample_weights,
    int64_t sample_weights_size,
    n4m_method_result_t** out_result);

/* Robust PLS via IRLS with Huber weights (§26). max_irls_iter <= 0 falls
 * back to 5. The result contains:
 *   "coefficients", "predictions", "x_mean", "y_mean"
 *   "final_weights" (1 x n_samples) — Huber weights at convergence
 *   scalar "rmse", scalar "huber_k"
 */
N4M_API n4m_status_t n4m_estimators_robust_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    double huber_k,
    int32_t max_irls_iter,
    n4m_method_result_t** out_result);

/* Ridge-augmented PLS (§26). Augments (X, Y) with sqrt(ridge_lambda)*I and
 * 0 rows, then runs SIMPLS. ridge_lambda must be >= 0. The result contains:
 *   "coefficients", "predictions", "x_mean", "y_mean"
 *   scalar "rmse", scalar "ridge_lambda"
 */
N4M_API n4m_status_t n4m_estimators_ridge_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    double ridge_lambda,
    n4m_method_result_t** out_result);

/* Continuum regression (§26). tau in [0, 1] interpolates between PLS
 * (tau=0) and a whitened-X / OLS-like flavour (tau=1). The result contains:
 *   "coefficients", "predictions", "x_mean", "y_mean"
 *   scalar "rmse", scalar "tau"
 */
N4M_API n4m_status_t n4m_estimators_continuum_regression_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    double tau,
    n4m_method_result_t** out_result);

/* Direct (closed-form) multi-output Ridge regression — genuine
 *   beta = (Xc'Xc + lambda I)^-1 Xc'Yc
 * on column-centered X, Y, with intercept = y_mean - x_mean.beta (the penalty
 * is NOT applied to the intercept, for sklearn.linear_model.Ridge parity).
 * This is distinct from n4m_estimators_ridge_pls_fit (ridge-augmented SIMPLS).
 *
 * Solver is chosen automatically by shape: PRIMAL (augmented-QR) when
 * p <= n, DUAL (Gram-on-samples) when p > n; both give identical coefficients.
 * Centering follows cfg.center_x/center_y (gated by cfg.ridge_fit_intercept);
 * cfg.scale_x scales X columns (a zero-variance column gets scale 1.0, sklearn
 * convention -> 0 coefficient). cfg.scale_y is IGNORED: Y is never scaled, to
 * match sklearn.linear_model.Ridge. Coefficients/intercept are on the original
 * (de-scaled) X scale.
 *
 * `lambdas`/`n_lambdas` reserve a lambda-path/selection extension. v1 uses a
 * single lambda: pass NULL+0 to take cfg.ridge_lambda, or a 1-element array to
 * override it (lambdas[0]); n_lambdas > 1 is rejected (N4M_ERR_INVALID_ARGUMENT)
 * until the path ships. lambda must be finite and >= 0 (lambda=0 is OLS and
 * may be ill-posed for p >= n). The result contains:
 *   "coefficients" (p x q), "intercept" (1 x q), "x_mean" (1 x p),
 *   "x_scale" (1 x p), "y_mean" (1 x q), "predictions" (n x q)
 *   scalar "rmse", scalar "lambda"
 */
N4M_API n4m_status_t n4m_estimators_ridge_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const double* lambdas,
    int64_t n_lambdas,
    n4m_method_result_t** out_result);

/* N-PLS (§22). 3-way tensor regression via Bro's algorithm. X must be a
 * row-major (n_samples x (mode_j * mode_k)) flat view of the (n x J x K)
 * tensor. The result contains:
 *   "predictions"   (n_samples x n_targets)
 *   "coefficients"  ((mode_j*mode_k) x n_targets)
 *   "w_j"           (mode_j x n_components)
 *   "w_k"           (mode_k x n_components)
 *   "scores_t"      (n_samples x n_components)
 *   "x_mean", "y_mean"
 *   scalar "rmse"
 */
N4M_API n4m_status_t n4m_estimators_n_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X_flat,
    int32_t mode_j,
    int32_t mode_k,
    const n4m_matrix_view_t* Y,
    n4m_method_result_t** out_result);

/* Non-linear kernel PLS (§24). `kernel_type`:
 *   0 = linear, 1 = RBF, 2 = polynomial, 3 = sigmoid
 * `gamma`, `coef0`, `degree` are kernel parameters; ignored when not
 * applicable to the selected kernel. The result contains:
 *   "predictions"   (n_samples x n_targets)  in-sample predictions
 *   "alpha"         (n_samples x n_targets)  dual coefficients
 *   "y_mean"        (1 x n_targets)
 *   scalar "rmse", scalar "kernel_type" (as double)
 */
N4M_API n4m_status_t n4m_estimators_kernel_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    int32_t kernel_type,
    double gamma,
    double coef0,
    int32_t degree,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    n4m_method_result_t** out_result);

/* Group sparse PLS (§7). Soft-thresholds groups of features together.
 * `group_assignment` is a length-n_features buffer of 0-based group IDs.
 * Result contains coefficients, predictions, x_mean, y_mean, rmse, and
 * scalar `n_groups`.
 */
N4M_API n4m_status_t n4m_estimators_group_sparse_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const int32_t* group_assignment,
    int64_t group_assignment_size,
    double group_lambda,
    n4m_method_result_t** out_result);

/* Fused sparse PLS (§7). Combines L1 sparsity with fusion of
 * consecutive feature pairs. Result identical to group_sparse_pls_fit
 * plus scalars `l1_lambda` and `fusion_lambda`.
 */
N4M_API n4m_status_t n4m_estimators_fused_sparse_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    double l1_lambda,
    double fusion_lambda,
    n4m_method_result_t** out_result);

/* GPR-on-PLS (§47): two-stage regression. First fits a SIMPLS PLS with
 * cfg.n_components latent components and rotation R (p x k). Then fits a
 * Gaussian Process with RBF kernel
 *     K(t_i, t_j) = exp(-||t_i - t_j||^2 / (2 * length_scale^2))
 * and diagonal noise sigma^2 (the `noise_level` parameter is interpreted
 * as **variance**, matching sklearn `WhiteKernel(noise_level=...)`, not
 * standard deviation). The GP solve uses Cholesky on (K + noise_level * I)
 * on the training scores T = (X - x_mean) @ R and centred y.
 * Y must be single-target (n x 1) in Phase 47.
 *
 * `seed` is reserved for ABI symmetry with the ensemble methods; the fit
 * is fully deterministic.
 *
 * Result keys:
 *   "rotation_r"           (n_features x n_components)
 *   "x_mean"               (1 x n_features)
 *   "alpha"                (n_samples x 1)            — GP dual weights
 *   "L_lower"              (n_samples x n_samples)    — Cholesky factor
 *   "training_scores"      (n_samples x n_components)
 *   "predictions"          (n_samples x 1)
 *   "predictive_variance"  (n_samples x 1)            — noise-free posterior
 *   scalar "length_scale", "noise_level", "n_components", "y_mean", "rmse", "seed"
 *
 * Returns N4M_ERR_INVALID_ARGUMENT for non-positive length_scale or
 * noise_level, or for n_components outside [1, min(n,p)].
 * Returns N4M_ERR_SHAPE_MISMATCH if Y has more than one column.
 * Returns N4M_ERR_NUMERICAL_FAILURE if the Cholesky of (K + noise*I)
 * fails (typically at very low noise with redundant score directions).
 */
N4M_API n4m_status_t n4m_estimators_gpr_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    double length_scale,
    double noise_level,
    uint64_t seed,
    n4m_method_result_t** out_result);

/* Simplified SIMPLS PLS regression — raw-pointer signature for
 * language bindings whose FFI layer struggles with the
 * `n4m_matrix_view_t*` parameter pattern (notably Emscripten 5.0.7
 * and Julia 1.12 ccall with `Ref{MatrixView}`).
 *
 * X is row-major (n × p), Y is row-major (n × q). The function fits
 * SIMPLS with `center_x = center_y = scale_x = scale_y = 1` and writes
 * into caller-provided buffers:
 *
 *   coefficients_out: (p × q) row-major regression coefficients
 *   x_mean_out:       (1 × p)
 *   y_mean_out:       (1 × q)
 *   predictions_out:  (n × q) row-major in-sample predictions; pass
 *                     NULL to skip the in-sample predict step.
 *
 * Returns N4M_OK on success or a N4M_ERR_* status. This is a stable
 * additive helper — implementations may grow new variants but the
 * shape of this one is fixed at ABI minor 1.13.
 */
N4M_API n4m_status_t n4m_estimators_pls_fit(
    const double* x, const double* y,
    int32_t n, int32_t p, int32_t q, int32_t n_components,
    double* coefficients_out,
    double* x_mean_out,
    double* y_mean_out,
    double* predictions_out);

/* PLS-GLM (§5). PLS-reduced design feeding a softmax / Poisson IRLS.
 * `poisson` selects the Poisson-link path; otherwise a one-vs-rest
 * softmax-like fit on a continuous PLS regression on Y. The result
 * contains:
 *   "coefficients"  (n_features x n_classes)
 *   "intercept"     (1 x n_classes)
 *   "predictions"   (n_samples x n_classes)
 *   "x_mean"
 *   scalar "rmse", scalar "poisson" (0 or 1)
 */
N4M_API n4m_status_t n4m_estimators_pls_glm_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t poisson,
    n4m_method_result_t** out_result);

/* Missing-aware NIPALS (§13). Same shape as a regular PLS regression
 * model but tolerates NaN entries in X (replaced with the current
 * latent-space iterate during NIPALS). The result contains:
 *   "coefficients", "predictions", "x_mean", "y_mean"
 *   scalar "rmse"
 */
N4M_API n4m_status_t n4m_estimators_missing_aware_nipals_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    n4m_method_result_t** out_result);

/* LW-PLS — locally-weighted PLS with k-NN windows (Phase 4s). Predicts
 * each test row from a per-row PLS refit on its `n_neighbors` nearest
 * training rows. Currently the training set is X itself (in-sample). Result
 * keys:
 *   "predictions"            (n × n_targets) double matrix
 *   "neighbor_indices"       (n × n_neighbors) double matrix (cast from
 *                            int64 for unified matrix-shaped reads)
 *   int64 "neighbor_indices_i64" — same data as a row-major int64 vector
 *                            (preferred for index semantics)
 *   scalar "n_neighbors", scalar "n_components", scalar "rmse"
 */
N4M_API n4m_status_t n4m_estimators_lw_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t n_neighbors,
    n4m_method_result_t** out_result);

/* ECR — Elastic Component Regression (Phase 50). Liu 2009/2010.
 * `alpha` in [0, 1]: 0 = PCR-like, 1 = PLS-like. Values outside the
 * interval are silently clamped to [0, 1] (mirrors libPLS `ecr.m`).
 * cfg.n_components selects how many ECR components to extract. The
 * effective component count is clamped to min(n-1, p-1, cfg.n_components).
 * The result is shaped like the other "fit" methods:
 *   "coefficients"  (n_features x n_targets)
 *   "predictions"   (n_samples x n_targets)
 *   "x_mean", "y_mean", "x_scale", "y_scale"
 *   "weights_w"     (n_features x n_components)
 *   "loadings_p"    (n_features x n_components)
 *   "y_loadings"    (n_targets  x n_components)
 *   "wstar"         (n_features x n_components, so that X · wstar = T)
 *   "r2x"           (1 x n_components, % X variance per component)
 *   "r2y"           (1 x n_components, % Y variance per component)
 *   scalars: n_samples, n_features, n_targets, n_components, alpha, rmse
 */
N4M_API n4m_status_t n4m_estimators_ecr_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    double alpha,
    n4m_method_result_t** out_result);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* N4M_ESTIMATORS_REGRESSION_H */
