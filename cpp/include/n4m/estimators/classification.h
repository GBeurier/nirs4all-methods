/* SPDX-License-Identifier: CECILL-2.1 */
/* cpp/include/n4m/estimators/classification.h — estimators.classification methods (ABI 2.0). */
#ifndef N4M_ESTIMATORS_CLASSIFICATION_H
#define N4M_ESTIMATORS_CLASSIFICATION_H
#include "n4m/n4m.h"   /* shared infra: status, matrix view, context, N4M_API */

#ifdef __cplusplus
extern "C" {
#endif

/* Sparse PLS-DA (§7). Dummy-encodes integer class labels then runs
 * sparse SIMPLS with the configured `sparsity_lambda` from `cfg`.
 * `y_labels` must be a length-n_samples buffer of non-negative class
 * IDs (max value implies n_classes). The result contains:
 *   "coefficients"  (n_features x n_classes)
 *   "predictions"   (n_samples x n_classes) — soft assignments
 *   "x_mean", "y_mean"
 *   "class_priors"  (1 x n_classes)
 *   scalar "rmse"
 */
N4M_API n4m_status_t n4m_estimators_sparse_pls_da_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const int32_t* y_labels,
    int64_t y_labels_size,
    n4m_method_result_t** out_result);

/* PLS-QDA (§5). Quadratic discriminant analysis on PLS scores.
 * `y_labels` is a length-n_samples buffer of non-negative class IDs.
 * The result contains:
 *   "class_means"        (n_classes x n_components)
 *   "class_covariances"  (n_classes x (n_components * n_components))
 *   "log_class_priors"   (1 x n_classes)
 *   "rotations_r"        (n_features x n_components)
 *   "x_mean"
 *   "predictions"        (n_samples x n_classes)  log-likelihood scores
 *   scalar "rmse"
 */
N4M_API n4m_status_t n4m_estimators_pls_qda_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const int32_t* y_labels,
    int64_t y_labels_size,
    n4m_method_result_t** out_result);

/* PLS-LDA — Linear Discriminant Analysis on PLS scores (Phase 4p). Result
 * keys:
 *   "decision_scores"   (n × n_classes)
 *   int "predictions"   (n)
 *   scalar "n_classes", scalar "n_components"
 */
N4M_API n4m_status_t n4m_estimators_pls_lda_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const int32_t* y_labels,
    int64_t y_labels_size,
    int32_t n_classes,
    n4m_method_result_t** out_result);

/* PLS-Logistic — multinomial logistic regression on PLS scores (Phase 4q).
 * Result keys:
 *   "decision_scores"  (n × n_classes)
 *   "probabilities"    (n × n_classes)
 *   "intercepts"       (1 × (n_classes - 1))
 *   "coefficients"     ((n_classes - 1) × n_components)
 *   int "predictions"  (n)
 *   scalar "n_classes", scalar "n_components"
 */
N4M_API n4m_status_t n4m_estimators_pls_logistic_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const int32_t* y_labels,
    int64_t y_labels_size,
    int32_t n_classes,
    n4m_method_result_t** out_result);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* N4M_ESTIMATORS_CLASSIFICATION_H */
