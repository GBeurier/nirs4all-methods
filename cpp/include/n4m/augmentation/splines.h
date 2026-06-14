/* SPDX-License-Identifier: CECILL-2.1 */
/* cpp/include/n4m/augmentation/splines.h — augmentation.splines methods (ABI 2.0). */
#ifndef N4M_AUGMENTATION_SPLINES_H
#define N4M_AUGMENTATION_SPLINES_H
#include "n4m/n4m.h"   /* shared infra: status, matrix view, context, N4M_API */

#ifdef __cplusplus
extern "C" {
#endif

/* --- Spline_Smoothing --------------------------------------------------- */
typedef struct n4m_aug_spline_smooth_handle_t
    n4m_aug_spline_smooth_handle_t;
N4M_API n4m_status_t n4m_augmentation_spline_smoothing_create(
    n4m_aug_spline_smooth_handle_t** out,
    n4m_rng_pcg64_state_t* rng);
N4M_API n4m_status_t n4m_augmentation_spline_smoothing_apply(
    const n4m_aug_spline_smooth_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void n4m_augmentation_spline_smoothing_destroy(
    n4m_aug_spline_smooth_handle_t* handle);

/* --- Spline_X_Perturbations -------------------------------------------- */
typedef struct n4m_aug_spline_x_perturb_handle_t
    n4m_aug_spline_x_perturb_handle_t;
N4M_API n4m_status_t n4m_augmentation_spline_x_perturbations_create(
    n4m_aug_spline_x_perturb_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    int32_t spline_degree,
    double  perturbation_density,
    double  perturbation_range_min,
    double  perturbation_range_max);
N4M_API n4m_status_t n4m_augmentation_spline_x_perturbations_apply(
    const n4m_aug_spline_x_perturb_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void n4m_augmentation_spline_x_perturbations_destroy(
    n4m_aug_spline_x_perturb_handle_t* handle);

/* --- Spline_Y_Perturbations -------------------------------------------- */
typedef struct n4m_aug_spline_y_perturb_handle_t
    n4m_aug_spline_y_perturb_handle_t;
/* spline_points <= 0 means "use n_features / 2". */
N4M_API n4m_status_t n4m_augmentation_spline_y_perturbations_create(
    n4m_aug_spline_y_perturb_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    int32_t spline_points,
    double  perturbation_intensity);
N4M_API n4m_status_t n4m_augmentation_spline_y_perturbations_apply(
    const n4m_aug_spline_y_perturb_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void n4m_augmentation_spline_y_perturbations_destroy(
    n4m_aug_spline_y_perturb_handle_t* handle);

/* --- Spline_X_Simplification (cubic refit through a random control subset;
 * spline_points <= 0 -> n_features / 4) ---------------------------------- */
typedef struct n4m_aug_spline_x_simplify_handle_t
    n4m_aug_spline_x_simplify_handle_t;
N4M_API n4m_status_t n4m_augmentation_spline_x_simplification_create(
    n4m_aug_spline_x_simplify_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    int32_t spline_points,
    int32_t uniform);
N4M_API n4m_status_t n4m_augmentation_spline_x_simplification_apply(
    const n4m_aug_spline_x_simplify_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void n4m_augmentation_spline_x_simplification_destroy(
    n4m_aug_spline_x_simplify_handle_t* handle);

/* --- Spline_Curve_Simplification (as above; differs only in the uniform
 * path's np.unique handling) --------------------------------------------- */
typedef struct n4m_aug_spline_curve_simplify_handle_t
    n4m_aug_spline_curve_simplify_handle_t;
N4M_API n4m_status_t n4m_augmentation_spline_curve_simplification_create(
    n4m_aug_spline_curve_simplify_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    int32_t spline_points,
    int32_t uniform);
N4M_API n4m_status_t n4m_augmentation_spline_curve_simplification_apply(
    const n4m_aug_spline_curve_simplify_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void n4m_augmentation_spline_curve_simplification_destroy(
    n4m_aug_spline_curve_simplify_handle_t* handle);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* N4M_AUGMENTATION_SPLINES_H */
