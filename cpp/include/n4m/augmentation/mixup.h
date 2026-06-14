/* SPDX-License-Identifier: CECILL-2.1 */
/* cpp/include/n4m/augmentation/mixup.h — augmentation.mixup methods (ABI 2.0). */
#ifndef N4M_AUGMENTATION_MIXUP_H
#define N4M_AUGMENTATION_MIXUP_H
#include "n4m/n4m.h"   /* shared infra: status, matrix view, context, N4M_API */

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- MixupAugmenter ---------- */
typedef struct n4m_aug_mixup_handle_t n4m_aug_mixup_handle_t;
N4M_API n4m_status_t n4m_augmentation_mixup_create(n4m_aug_mixup_handle_t** out,
                                           n4m_rng_pcg64_state_t* rng,
                                           double alpha);
N4M_API n4m_status_t n4m_augmentation_mixup_apply(const n4m_aug_mixup_handle_t* handle,
                                          n4m_matrix_view_t X,
                                          n4m_matrix_view_t out);
N4M_API void         n4m_augmentation_mixup_destroy(n4m_aug_mixup_handle_t* handle);

/* ---------- LocalMixupAugmenter ---------- */
typedef struct n4m_aug_local_mixup_handle_t n4m_aug_local_mixup_handle_t;
N4M_API n4m_status_t n4m_augmentation_local_mixup_create(n4m_aug_local_mixup_handle_t** out,
                                                 n4m_rng_pcg64_state_t* rng,
                                                 double alpha,
                                                 int32_t k_neighbors);
N4M_API n4m_status_t n4m_augmentation_local_mixup_apply(const n4m_aug_local_mixup_handle_t* handle,
                                                n4m_matrix_view_t X,
                                                n4m_matrix_view_t out);
N4M_API void         n4m_augmentation_local_mixup_destroy(n4m_aug_local_mixup_handle_t* handle);

/* --- Rotate_Translate --------------------------------------------------- */
typedef struct n4m_aug_rotate_translate_handle_t
    n4m_aug_rotate_translate_handle_t;
N4M_API n4m_status_t n4m_augmentation_rotate_translate_create(
    n4m_aug_rotate_translate_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double p_range,
    double y_factor);
N4M_API n4m_status_t n4m_augmentation_rotate_translate_apply(
    const n4m_aug_rotate_translate_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void n4m_augmentation_rotate_translate_destroy(
    n4m_aug_rotate_translate_handle_t* handle);

/* --- Random_X_Operation ------------------------------------------------- */
typedef struct n4m_aug_random_x_op_handle_t n4m_aug_random_x_op_handle_t;
N4M_API n4m_status_t n4m_augmentation_random_x_op_create(
    n4m_aug_random_x_op_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    int32_t op_kind,
    double  operator_range_min,
    double  operator_range_max);
N4M_API n4m_status_t n4m_augmentation_random_x_op_apply(
    const n4m_aug_random_x_op_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void n4m_augmentation_random_x_op_destroy(
    n4m_aug_random_x_op_handle_t* handle);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* N4M_AUGMENTATION_MIXUP_H */
