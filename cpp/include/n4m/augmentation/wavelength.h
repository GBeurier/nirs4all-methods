/* SPDX-License-Identifier: CECILL-2.1 */
/* cpp/include/n4m/augmentation/wavelength.h — augmentation.wavelength methods (ABI 2.0). */
#ifndef N4M_AUGMENTATION_WAVELENGTH_H
#define N4M_AUGMENTATION_WAVELENGTH_H
#include "n4m/n4m.h"   /* shared infra: status, matrix view, context, N4M_API */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_aug_wavelength_shift_handle_t   n4m_aug_wavelength_shift_handle_t;
typedef struct n4m_aug_wavelength_stretch_handle_t n4m_aug_wavelength_stretch_handle_t;
typedef struct n4m_aug_local_warp_handle_t         n4m_aug_local_warp_handle_t;

N4M_API n4m_status_t n4m_augmentation_wavelength_shift_create(
    n4m_aug_wavelength_shift_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double shift_lo, double shift_hi,
    const double* wavelengths, int64_t n_wavelengths);
N4M_API n4m_status_t n4m_augmentation_wavelength_shift_apply(
    const n4m_aug_wavelength_shift_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void n4m_augmentation_wavelength_shift_destroy(n4m_aug_wavelength_shift_handle_t* handle);

N4M_API n4m_status_t n4m_augmentation_wavelength_stretch_create(
    n4m_aug_wavelength_stretch_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double stretch_lo, double stretch_hi,
    const double* wavelengths, int64_t n_wavelengths);
N4M_API n4m_status_t n4m_augmentation_wavelength_stretch_apply(
    const n4m_aug_wavelength_stretch_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void n4m_augmentation_wavelength_stretch_destroy(n4m_aug_wavelength_stretch_handle_t* handle);

N4M_API n4m_status_t n4m_augmentation_local_warp_create(
    n4m_aug_local_warp_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    int32_t n_control_points,
    double  max_shift,
    const double* wavelengths, int64_t n_wavelengths);
N4M_API n4m_status_t n4m_augmentation_local_warp_apply(
    const n4m_aug_local_warp_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void n4m_augmentation_local_warp_destroy(n4m_aug_local_warp_handle_t* handle);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* N4M_AUGMENTATION_WAVELENGTH_H */
