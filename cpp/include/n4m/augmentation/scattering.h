/* SPDX-License-Identifier: CECILL-2.1 */
/* cpp/include/n4m/augmentation/scattering.h — augmentation.scattering methods (ABI 2.0). */
#ifndef N4M_AUGMENTATION_SCATTERING_H
#define N4M_AUGMENTATION_SCATTERING_H
#include "n4m/n4m.h"   /* shared infra: status, matrix view, context, N4M_API */

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- ScatterSimulationMSC ---------- */
typedef struct n4m_aug_scatter_sim_handle_t n4m_aug_scatter_sim_handle_t;
N4M_API n4m_status_t n4m_augmentation_scatter_sim_msc_create(n4m_aug_scatter_sim_handle_t** out,
                                                 n4m_rng_pcg64_state_t* rng,
                                                 double a_low, double a_high,
                                                 double b_low, double b_high);
N4M_API n4m_status_t n4m_augmentation_scatter_sim_msc_apply(const n4m_aug_scatter_sim_handle_t* handle,
                                                n4m_matrix_view_t X,
                                                n4m_matrix_view_t out);
N4M_API void         n4m_augmentation_scatter_sim_msc_destroy(n4m_aug_scatter_sim_handle_t* handle);

/* ---------- ParticleSizeAugmenter ---------- */
typedef struct n4m_aug_particle_size_handle_t n4m_aug_particle_size_handle_t;
/* `use_size_range`: 0 = sample sizes from N(mean, variation), clipped to
 *   [5, 500]; 1 = sample uniform[size_range_low, size_range_high].
 * `include_path_length`: 0 = skip the multiplicative path-length step;
 *   1 = apply  factor = clip(1 + path_length_sensitivity * log(size_ratio),
 *                            0.7, 1.5).
 */
N4M_API n4m_status_t n4m_augmentation_particle_size_create(
    n4m_aug_particle_size_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double mean_size_um, double size_variation_um,
    int    use_size_range, double size_range_low_um, double size_range_high_um,
    double reference_size_um, double wavelength_exponent,
    double size_effect_strength,
    int    include_path_length, double path_length_sensitivity,
    const double* wavelengths, int64_t n_wavelengths);
N4M_API n4m_status_t n4m_augmentation_particle_size_apply(
    const n4m_aug_particle_size_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void         n4m_augmentation_particle_size_destroy(
    n4m_aug_particle_size_handle_t* handle);

/* ---------- EMSCDistortionAugmenter ---------- */
typedef struct n4m_aug_emsc_distort_handle_t n4m_aug_emsc_distort_handle_t;
N4M_API n4m_status_t n4m_augmentation_emsc_distort_create(
    n4m_aug_emsc_distort_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double mult_low, double mult_high,
    double add_low,  double add_high,
    int32_t polynomial_order,
    double  polynomial_strength,
    double  correlation,
    const double* wavelengths, int64_t n_wavelengths);
N4M_API n4m_status_t n4m_augmentation_emsc_distort_apply(
    const n4m_aug_emsc_distort_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void         n4m_augmentation_emsc_distort_destroy(
    n4m_aug_emsc_distort_handle_t* handle);

/* ---------- BatchEffectAugmenter ---------- */
typedef struct n4m_aug_batch_effect_handle_t n4m_aug_batch_effect_handle_t;
/* `variation_scope`: 0 = per-sample, 1 = batch.
 * `wavelengths` may be NULL → x axis derived from the integer index. */
N4M_API n4m_status_t n4m_augmentation_batch_effect_create(
    n4m_aug_batch_effect_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double offset_std, double slope_std, double gain_std,
    int32_t variation_scope,
    const double* wavelengths, int64_t n_wavelengths);
N4M_API n4m_status_t n4m_augmentation_batch_effect_apply(
    const n4m_aug_batch_effect_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void         n4m_augmentation_batch_effect_destroy(
    n4m_aug_batch_effect_handle_t* handle);

/* ---------- InstrumentalBroadeningAugmenter ---------- */
typedef struct n4m_aug_instrument_broaden_handle_t
                n4m_aug_instrument_broaden_handle_t;
/* `use_fwhm_range`: 0 = fixed `fwhm` for all rows (no RNG draws);
 *   1 = sample FWHM uniformly in [fwhm_low, fwhm_high]. */
N4M_API n4m_status_t n4m_augmentation_instrument_broaden_create(
    n4m_aug_instrument_broaden_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double fwhm,
    int    use_fwhm_range, double fwhm_low, double fwhm_high,
    int32_t variation_scope,
    const double* wavelengths, int64_t n_wavelengths);
N4M_API n4m_status_t n4m_augmentation_instrument_broaden_apply(
    const n4m_aug_instrument_broaden_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void         n4m_augmentation_instrument_broaden_destroy(
    n4m_aug_instrument_broaden_handle_t* handle);

/* ---------- DeadBandAugmenter ---------- */
typedef struct n4m_aug_dead_band_handle_t n4m_aug_dead_band_handle_t;
N4M_API n4m_status_t n4m_augmentation_dead_band_create(
    n4m_aug_dead_band_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    int32_t n_bands,
    int32_t width_low, int32_t width_high,
    double  noise_std, double probability,
    int32_t variation_scope);
N4M_API n4m_status_t n4m_augmentation_dead_band_apply(
    const n4m_aug_dead_band_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void         n4m_augmentation_dead_band_destroy(
    n4m_aug_dead_band_handle_t* handle);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* N4M_AUGMENTATION_SCATTERING_H */
