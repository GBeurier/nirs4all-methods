/* SPDX-License-Identifier: CECILL-2.1 */
/* cpp/include/n4m/estimators/multiblock.h — estimators.multiblock methods (ABI 2.0). */
#ifndef N4M_ESTIMATORS_MULTIBLOCK_H
#define N4M_ESTIMATORS_MULTIBLOCK_H
#include "n4m/n4m.h"   /* shared infra: status, matrix view, context, N4M_API */

#ifdef __cplusplus
extern "C" {
#endif

/* O2PLS (§16; Trygg & Wold 2003). Bi-directional OPLS with
 * `n_predictive` joint + `n_x_orthogonal` X-orthogonal +
 * `n_y_orthogonal` Y-orthogonal components. The result contains:
 *   "coefficients"    (n_features_x x n_features_y)
 *   "predictions"     (n_samples x n_features_y)
 *   "x_mean", "y_mean"
 *   "w_predictive"    (n_features_x x n_predictive)
 *   "c_predictive"    (n_features_y x n_predictive)
 *   "w_x_orthogonal"  (n_features_x x n_x_orthogonal)
 *   "c_y_orthogonal"  (n_features_y x n_y_orthogonal)
 *   "b_predictive"    (1 x n_predictive)
 *   scalar "rmse"
 */
N4M_API n4m_status_t n4m_estimators_o2pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t n_predictive,
    int32_t n_x_orthogonal,
    int32_t n_y_orthogonal,
    n4m_method_result_t** out_result);

/* SO-PLS (§17). Sequential and Orthogonalized PLS for B X-blocks
 * predicting one Y. `X_blocks` is an array of `n_blocks`
 * n4m_matrix_view_t structs (all sharing X.rows = Y.rows).
 * `n_components_per_block` is a length-n_blocks int32 array.
 * The result contains:
 *   "predictions" (n_samples x n_targets)
 *   "y_mean"      (1 x n_targets)
 *   For each block b: "block_coefficients_<b>" of shape (p_b x n_targets)
 *   scalar "n_blocks"
 */
N4M_API n4m_status_t n4m_estimators_so_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X_blocks,
    int32_t n_blocks,
    const n4m_matrix_view_t* Y,
    const int32_t* n_components_per_block,
    int64_t n_components_per_block_size,
    n4m_method_result_t** out_result);

/* OnPLS (§18). Generalized orthogonal projection for multi-block PLS.
 * Removes joint and per-block unique components.
 * Result contains scalars `n_blocks` and `n_joint`, plus per-block
 * loading matrices "joint_loadings_<b>", "unique_loadings_<b>" and
 * "joint_scores_<b>".
 */
N4M_API n4m_status_t n4m_estimators_on_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X_blocks,
    int32_t n_blocks,
    int32_t n_joint,
    const int32_t* n_unique_per_block,
    int64_t n_unique_per_block_size,
    n4m_method_result_t** out_result);

/* ROSA (§19). Response-Oriented Sequential Alternation: at each
 * component, picks the block whose latent direction yields the highest
 * correlation with the current Y residual.
 * Result contains:
 *   "predictions"                  (n_samples x n_targets)
 *   "y_mean"
 *   "selected_block_per_component" int vector
 *   For each block b: "block_coefficients_<b>"
 *   scalar "n_components"
 */
N4M_API n4m_status_t n4m_estimators_rosa_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X_blocks,
    int32_t n_blocks,
    const n4m_matrix_view_t* Y,
    int32_t n_components,
    n4m_method_result_t** out_result);

/* MIR-PLS — Multiple Inverse Regression PLS (§13). Inverts the X→Y
 * relationship by running SIMPLS on (Y, X) and pseudoinverting the
 * resulting Y→X coefficients to obtain X→Y prediction coefficients.
 * The result contains:
 *   "coefficients"  (n_features x n_targets)
 *   "predictions"   (n_samples x n_targets)
 *   "x_mean", "y_mean"
 *   scalar "rmse"
 */
N4M_API n4m_status_t n4m_estimators_mir_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    n4m_method_result_t** out_result);

/* MB-PLS — block-weighted multi-block PLS (Phase 4r). Predicts on the
 * concatenated feature matrix X (n × Σ block_sizes). Result keys:
 *   "predictions"    (n × n_targets)
 *   "coefficients"   (Σ block_sizes × n_targets, original X scale)
 *   "x_mean"         (1 × Σ block_sizes)
 *   "x_scale"        (1 × Σ block_sizes)
 *   "intercept"      (1 × n_targets)
 *   "block_weights"  (1 × n_blocks)
 *   scalar "n_blocks", scalar "rmse"
 */
N4M_API n4m_status_t n4m_estimators_mb_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const int64_t* block_sizes,
    int64_t n_blocks,
    n4m_method_result_t** out_result);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* N4M_ESTIMATORS_MULTIBLOCK_H */
