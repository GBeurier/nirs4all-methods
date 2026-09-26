/* SPDX-License-Identifier: CECILL-2.1 */
/* cpp/include/n4m/augmentation/instrument.h — augmentation.instrument methods (ABI 2.0). */
#ifndef N4M_AUGMENTATION_INSTRUMENT_H
#define N4M_AUGMENTATION_INSTRUMENT_H
#include "n4m/n4m.h"   /* shared infra: status, matrix view, context, N4M_API */

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- TemperatureAugmenter ---------- */
typedef struct n4m_aug_temperature_handle_t n4m_aug_temperature_handle_t;
N4M_API n4m_status_t n4m_augmentation_temperature_create(
    n4m_aug_temperature_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double temperature_delta,
    int    use_temp_range, double temp_low, double temp_high,
    int    enable_shift, int enable_intensity, int enable_broadening,
    int    region_specific,
    const double* wavelengths, int64_t n_wavelengths);
N4M_API n4m_status_t n4m_augmentation_temperature_apply(
    const n4m_aug_temperature_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void         n4m_augmentation_temperature_destroy(
    n4m_aug_temperature_handle_t* handle);

/* ---------- MoistureAugmenter ---------- */
typedef struct n4m_aug_moisture_handle_t n4m_aug_moisture_handle_t;
N4M_API n4m_status_t n4m_augmentation_moisture_create(
    n4m_aug_moisture_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double water_activity_delta,
    int    use_aw_range, double aw_low, double aw_high,
    double reference_water_activity,
    double free_water_fraction,
    double bound_water_shift,
    double moisture_content,
    int    enable_shift, int enable_intensity,
    const double* wavelengths, int64_t n_wavelengths);
N4M_API n4m_status_t n4m_augmentation_moisture_apply(
    const n4m_aug_moisture_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void         n4m_augmentation_moisture_destroy(
    n4m_aug_moisture_handle_t* handle);

/* --- DetectorRollOff ----------------------------------------------------- */
typedef struct n4m_aug_detector_rolloff_handle_t
    n4m_aug_detector_rolloff_handle_t;
N4M_API n4m_status_t n4m_augmentation_detector_rolloff_create(
    n4m_aug_detector_rolloff_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    int32_t detector_model,
    double  effect_strength,
    double  noise_amplification,
    int32_t include_baseline_distortion);
N4M_API n4m_status_t n4m_augmentation_detector_rolloff_apply(
    const n4m_aug_detector_rolloff_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t wavelengths,
    n4m_matrix_view_t out);
N4M_API void n4m_augmentation_detector_rolloff_destroy(
    n4m_aug_detector_rolloff_handle_t* handle);

/* --- StrayLight --------------------------------------------------------- */
typedef struct n4m_aug_stray_light_handle_t n4m_aug_stray_light_handle_t;
N4M_API n4m_status_t n4m_augmentation_stray_light_create(
    n4m_aug_stray_light_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double stray_light_fraction,
    double edge_enhancement,
    double edge_width,
    int32_t include_peak_truncation);
N4M_API n4m_status_t n4m_augmentation_stray_light_apply(
    const n4m_aug_stray_light_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t wavelengths,
    n4m_matrix_view_t out);
N4M_API void n4m_augmentation_stray_light_destroy(
    n4m_aug_stray_light_handle_t* handle);

/* --- EdgeCurvature ------------------------------------------------------ */
typedef struct n4m_aug_edge_curve_handle_t n4m_aug_edge_curve_handle_t;
N4M_API n4m_status_t n4m_augmentation_edge_curvature_create(
    n4m_aug_edge_curve_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double curvature_strength,
    int32_t curvature_type,
    double asymmetry,
    double edge_focus);
N4M_API n4m_status_t n4m_augmentation_edge_curvature_apply(
    const n4m_aug_edge_curve_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t wavelengths,
    n4m_matrix_view_t out);
N4M_API void n4m_augmentation_edge_curvature_destroy(n4m_aug_edge_curve_handle_t* handle);

/* --- TruncatedPeak ------------------------------------------------------ */
typedef struct n4m_aug_truncated_peak_handle_t
    n4m_aug_truncated_peak_handle_t;
N4M_API n4m_status_t n4m_augmentation_truncated_peak_create(
    n4m_aug_truncated_peak_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double peak_probability,
    double amplitude_min, double amplitude_max,
    double width_min,     double width_max,
    int32_t left_edge,
    int32_t right_edge);
N4M_API n4m_status_t n4m_augmentation_truncated_peak_apply(
    const n4m_aug_truncated_peak_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t wavelengths,
    n4m_matrix_view_t out);
N4M_API void n4m_augmentation_truncated_peak_destroy(
    n4m_aug_truncated_peak_handle_t* handle);

/* --- EdgeArtifacts (combined wrapper) ----------------------------------- */
/* Flags:
 *   bit 0 (0x1)  : enable DetectorRollOff
 *   bit 1 (0x2)  : enable StrayLight
 *   bit 2 (0x4)  : enable EdgeCurvature
 *   bit 3 (0x8)  : enable TruncatedPeak
 * Default sequence mirrors the Python reference (truncated -> curve ->
 * stray -> detector). The overall_strength scales every sub-augmenter's
 * default strength. */
#define N4M_AUG_EDGE_ARTIFACTS_DETECTOR_ROLL_OFF 0x1
#define N4M_AUG_EDGE_ARTIFACTS_STRAY_LIGHT       0x2
#define N4M_AUG_EDGE_ARTIFACTS_EDGE_CURVATURE    0x4
#define N4M_AUG_EDGE_ARTIFACTS_TRUNCATED_PEAKS   0x8
typedef struct n4m_aug_edge_artifacts_handle_t
    n4m_aug_edge_artifacts_handle_t;
N4M_API n4m_status_t n4m_augmentation_edge_artifacts_create(
    n4m_aug_edge_artifacts_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    int32_t enabled_flags,
    double  overall_strength,
    int32_t detector_model);
N4M_API n4m_status_t n4m_augmentation_edge_artifacts_apply(
    const n4m_aug_edge_artifacts_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t wavelengths,
    n4m_matrix_view_t out);
N4M_API void n4m_augmentation_edge_artifacts_destroy(
    n4m_aug_edge_artifacts_handle_t* handle);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* N4M_AUGMENTATION_INSTRUMENT_H */
