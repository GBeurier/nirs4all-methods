/* SPDX-License-Identifier: CECILL-2.1 */
/* cpp/include/n4m/estimators/survival.h — estimators.survival methods (ABI 2.0). */
#ifndef N4M_ESTIMATORS_SURVIVAL_H
#define N4M_ESTIMATORS_SURVIVAL_H
#include "n4m/n4m.h"   /* shared infra: status, matrix view, context, N4M_API */

#ifdef __cplusplus
extern "C" {
#endif

/* PLS-Cox (§5). PLS-reduced Cox proportional hazards with Breslow
 * baseline hazard. `survival_times` is the observed time, and
 * `event_indicators` is 1 (event) or 0 (censored). The result contains:
 *   "coefficients"     (n_features x 1)  linear predictor coefficients
 *   "baseline_hazard"  (1 x n_unique_event_times)
 *   "event_times"      (1 x n_unique_event_times)
 *   "x_mean"
 *   "predictions"      (n_samples x 1)  linear-predictor scores
 */
N4M_API n4m_status_t n4m_estimators_pls_cox_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const double* survival_times,
    int64_t survival_times_size,
    const int32_t* event_indicators,
    int64_t event_indicators_size,
    n4m_method_result_t** out_result);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* N4M_ESTIMATORS_SURVIVAL_H */
