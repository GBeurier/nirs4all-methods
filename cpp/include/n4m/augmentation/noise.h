/* SPDX-License-Identifier: CECILL-2.1 */
/* cpp/include/n4m/augmentation/noise.h — augmentation.noise methods (ABI 2.0). */
#ifndef N4M_AUGMENTATION_NOISE_H
#define N4M_AUGMENTATION_NOISE_H
#include "n4m/n4m.h"   /* shared infra: status, matrix view, context, N4M_API */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_aug_gaussian_noise_handle_t       n4m_aug_gaussian_noise_handle_t;
typedef struct n4m_aug_multiplicative_noise_handle_t n4m_aug_multiplicative_noise_handle_t;
typedef struct n4m_aug_spike_noise_handle_t          n4m_aug_spike_noise_handle_t;
typedef struct n4m_aug_hetero_noise_handle_t         n4m_aug_hetero_noise_handle_t;

N4M_API n4m_status_t n4m_augmentation_gaussian_noise_create(
    n4m_aug_gaussian_noise_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double sigma);
N4M_API void n4m_augmentation_gaussian_noise_destroy(n4m_aug_gaussian_noise_handle_t* handle);
N4M_API n4m_status_t n4m_augmentation_gaussian_noise_apply(
    const n4m_aug_gaussian_noise_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

N4M_API n4m_status_t n4m_augmentation_multiplicative_noise_create(
    n4m_aug_multiplicative_noise_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double sigma_gain);
N4M_API void n4m_augmentation_multiplicative_noise_destroy(
    n4m_aug_multiplicative_noise_handle_t* handle);
N4M_API n4m_status_t n4m_augmentation_multiplicative_noise_apply(
    const n4m_aug_multiplicative_noise_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

N4M_API n4m_status_t n4m_augmentation_spike_noise_create(
    n4m_aug_spike_noise_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    int32_t n_spikes_min, int32_t n_spikes_max,
    double amplitude_min, double amplitude_max);
N4M_API void n4m_augmentation_spike_noise_destroy(n4m_aug_spike_noise_handle_t* handle);
N4M_API n4m_status_t n4m_augmentation_spike_noise_apply(
    const n4m_aug_spike_noise_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

N4M_API n4m_status_t n4m_augmentation_hetero_noise_create(
    n4m_aug_hetero_noise_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double noise_base, double noise_signal_dep);
N4M_API void n4m_augmentation_hetero_noise_destroy(n4m_aug_hetero_noise_handle_t* handle);
N4M_API n4m_status_t n4m_augmentation_hetero_noise_apply(
    const n4m_aug_hetero_noise_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* N4M_AUGMENTATION_NOISE_H */
