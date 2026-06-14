/* SPDX-License-Identifier: CECILL-2.1 */
/* cpp/include/n4m/augmentation/spectral.h — augmentation.spectral methods (ABI 2.0). */
#ifndef N4M_AUGMENTATION_SPECTRAL_H
#define N4M_AUGMENTATION_SPECTRAL_H
#include "n4m/n4m.h"   /* shared infra: status, matrix view, context, N4M_API */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_aug_band_perturb_handle_t       n4m_aug_band_perturb_handle_t;
typedef struct n4m_aug_band_mask_handle_t          n4m_aug_band_mask_handle_t;
typedef struct n4m_aug_channel_dropout_handle_t    n4m_aug_channel_dropout_handle_t;
typedef struct n4m_aug_gauss_jitter_handle_t       n4m_aug_gauss_jitter_handle_t;
typedef struct n4m_aug_unsharp_mask_handle_t       n4m_aug_unsharp_mask_handle_t;
typedef struct n4m_aug_magnitude_warp_handle_t     n4m_aug_magnitude_warp_handle_t;
typedef struct n4m_aug_local_clip_handle_t         n4m_aug_local_clip_handle_t;

N4M_API n4m_status_t n4m_augmentation_band_perturb_create(
    n4m_aug_band_perturb_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    int32_t n_bands,
    int32_t bw_lo, int32_t bw_hi,
    double  gain_lo, double  gain_hi,
    double  offset_lo, double offset_hi);
N4M_API n4m_status_t n4m_augmentation_band_perturb_apply(
    const n4m_aug_band_perturb_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void n4m_augmentation_band_perturb_destroy(n4m_aug_band_perturb_handle_t* handle);

N4M_API n4m_status_t n4m_augmentation_band_mask_create(
    n4m_aug_band_mask_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    int32_t n_bands_lo, int32_t n_bands_hi,
    int32_t bw_lo, int32_t bw_hi,
    int32_t mode);
N4M_API n4m_status_t n4m_augmentation_band_mask_apply(
    const n4m_aug_band_mask_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void n4m_augmentation_band_mask_destroy(n4m_aug_band_mask_handle_t* handle);

N4M_API n4m_status_t n4m_augmentation_channel_dropout_create(
    n4m_aug_channel_dropout_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double  dropout_prob,
    int32_t mode);
N4M_API n4m_status_t n4m_augmentation_channel_dropout_apply(
    const n4m_aug_channel_dropout_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void n4m_augmentation_channel_dropout_destroy(n4m_aug_channel_dropout_handle_t* handle);

N4M_API n4m_status_t n4m_augmentation_gauss_jitter_create(
    n4m_aug_gauss_jitter_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double  sigma_lo, double sigma_hi,
    int32_t kernel_width);
N4M_API n4m_status_t n4m_augmentation_gauss_jitter_apply(
    const n4m_aug_gauss_jitter_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void n4m_augmentation_gauss_jitter_destroy(n4m_aug_gauss_jitter_handle_t* handle);

N4M_API n4m_status_t n4m_augmentation_unsharp_mask_create(
    n4m_aug_unsharp_mask_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double  amount_lo, double amount_hi,
    double  sigma, int32_t kernel_width);
N4M_API n4m_status_t n4m_augmentation_unsharp_mask_apply(
    const n4m_aug_unsharp_mask_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void n4m_augmentation_unsharp_mask_destroy(n4m_aug_unsharp_mask_handle_t* handle);

N4M_API n4m_status_t n4m_augmentation_magnitude_warp_create(
    n4m_aug_magnitude_warp_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    int32_t n_control_points,
    double  gain_lo, double gain_hi,
    const double* wavelengths, int64_t n_wavelengths);
N4M_API n4m_status_t n4m_augmentation_magnitude_warp_apply(
    const n4m_aug_magnitude_warp_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void n4m_augmentation_magnitude_warp_destroy(n4m_aug_magnitude_warp_handle_t* handle);

N4M_API n4m_status_t n4m_augmentation_local_clip_create(
    n4m_aug_local_clip_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    int32_t n_regions,
    int32_t width_lo, int32_t width_hi);
N4M_API n4m_status_t n4m_augmentation_local_clip_apply(
    const n4m_aug_local_clip_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void n4m_augmentation_local_clip_destroy(n4m_aug_local_clip_handle_t* handle);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* N4M_AUGMENTATION_SPECTRAL_H */
