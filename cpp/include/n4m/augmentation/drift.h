/* SPDX-License-Identifier: CECILL-2.1 */
/* cpp/include/n4m/augmentation/drift.h — augmentation.drift methods (ABI 2.0). */
#ifndef N4M_AUGMENTATION_DRIFT_H
#define N4M_AUGMENTATION_DRIFT_H
#include "n4m/n4m.h"   /* shared infra: status, matrix view, context, N4M_API */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_aug_linear_drift_handle_t         n4m_aug_linear_drift_handle_t;
typedef struct n4m_aug_poly_drift_handle_t           n4m_aug_poly_drift_handle_t;
typedef struct n4m_aug_path_length_handle_t          n4m_aug_path_length_handle_t;

N4M_API n4m_status_t n4m_augmentation_linear_drift_create(
    n4m_aug_linear_drift_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double offset_min, double offset_max,
    double slope_min,  double slope_max);
N4M_API void n4m_augmentation_linear_drift_destroy(n4m_aug_linear_drift_handle_t* handle);
N4M_API n4m_status_t n4m_augmentation_linear_drift_apply(
    const n4m_aug_linear_drift_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

N4M_API n4m_status_t n4m_augmentation_poly_drift_create(
    n4m_aug_poly_drift_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    int32_t degree,
    const double* coeff_min, const double* coeff_max);
N4M_API void n4m_augmentation_poly_drift_destroy(n4m_aug_poly_drift_handle_t* handle);
N4M_API n4m_status_t n4m_augmentation_poly_drift_apply(
    const n4m_aug_poly_drift_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

N4M_API n4m_status_t n4m_augmentation_path_length_create(
    n4m_aug_path_length_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double path_length_std, double min_path_length);
N4M_API void n4m_augmentation_path_length_destroy(n4m_aug_path_length_handle_t* handle);
N4M_API n4m_status_t n4m_augmentation_path_length_apply(
    const n4m_aug_path_length_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* N4M_AUGMENTATION_DRIFT_H */
