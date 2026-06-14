/* SPDX-License-Identifier: CECILL-2.1 */
/* cpp/include/n4m/transform/resampling.h — transform.resampling methods (ABI 2.0). */
#ifndef N4M_TRANSFORM_RESAMPLING_H
#define N4M_TRANSFORM_RESAMPLING_H
#include "n4m/n4m.h"   /* shared infra: status, matrix view, context, N4M_API */

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- Resampler (stateful) -------------------------------------- */
typedef struct n4m_pp_resampler_handle_t n4m_pp_resampler_handle_t;
N4M_API n4m_status_t n4m_transform_resampler_create(n4m_pp_resampler_handle_t** out,
                                              const double* target_wl,
                                              int64_t n_target,
                                              int32_t method,
                                              double crop_min, double crop_max,
                                              int use_crop, double fill_value,
                                              int bounds_error,
                                              int extrapolate);
N4M_API void         n4m_transform_resampler_destroy(n4m_pp_resampler_handle_t* h);
N4M_API n4m_status_t n4m_transform_resampler_fit(n4m_pp_resampler_handle_t* h,
                                           const double* source_wl,
                                           int64_t n_source);
N4M_API n4m_status_t n4m_transform_resampler_is_fitted(
    const n4m_pp_resampler_handle_t* h, int* out_fitted);
N4M_API int64_t      n4m_transform_resampler_output_cols(
    const n4m_pp_resampler_handle_t* h);
N4M_API n4m_status_t n4m_transform_resampler_transform(
    const n4m_pp_resampler_handle_t* h,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

/* ---------- CropTransformer (stateless) ------------------------------- */
typedef struct n4m_pp_crop_handle_t n4m_pp_crop_handle_t;
N4M_API n4m_status_t n4m_transform_crop_create(n4m_pp_crop_handle_t** out,
                                         int64_t start, int64_t end);
N4M_API void         n4m_transform_crop_destroy(n4m_pp_crop_handle_t* h);
N4M_API int64_t      n4m_transform_crop_output_cols(const n4m_pp_crop_handle_t* h,
                                              int64_t input_cols);
N4M_API n4m_status_t n4m_transform_crop_transform(const n4m_pp_crop_handle_t* h,
                                            n4m_matrix_view_t X,
                                            n4m_matrix_view_t out);

/* ---------- ResampleTransformer (stateless) --------------------------- */
typedef struct n4m_pp_resample_handle_t n4m_pp_resample_handle_t;
N4M_API n4m_status_t n4m_transform_resample_transformer_create(n4m_pp_resample_handle_t** out,
                                             int64_t num_samples);
N4M_API void         n4m_transform_resample_transformer_destroy(n4m_pp_resample_handle_t* h);
N4M_API int64_t      n4m_transform_resample_transformer_output_cols(
    const n4m_pp_resample_handle_t* h, int64_t input_cols);
N4M_API n4m_status_t n4m_transform_resample_transformer_transform(
    const n4m_pp_resample_handle_t* h,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

/* ---------- IntegerKBinsDiscretizer (stateful, int32 output) ---------- */
typedef struct n4m_pp_kbins_disc_handle_t n4m_pp_kbins_disc_handle_t;
N4M_API n4m_status_t n4m_transform_kbins_discretizer_create(n4m_pp_kbins_disc_handle_t** out,
                                               int32_t n_bins,
                                               int32_t strategy);
N4M_API void         n4m_transform_kbins_discretizer_destroy(n4m_pp_kbins_disc_handle_t* h);
N4M_API n4m_status_t n4m_transform_kbins_discretizer_fit(n4m_pp_kbins_disc_handle_t* h,
                                            n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_transform_kbins_discretizer_is_fitted(
    const n4m_pp_kbins_disc_handle_t* h, int* out_fitted);
N4M_API n4m_status_t n4m_transform_kbins_discretizer_transform(
    const n4m_pp_kbins_disc_handle_t* h,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

/* ---------- RangeDiscretizer (stateless, int32 output) ---------------- */
typedef struct n4m_pp_range_disc_handle_t n4m_pp_range_disc_handle_t;
N4M_API n4m_status_t n4m_transform_range_discretizer_create(n4m_pp_range_disc_handle_t** out,
                                               const double* bins,
                                               int64_t n_edges);
N4M_API void         n4m_transform_range_discretizer_destroy(n4m_pp_range_disc_handle_t* h);
N4M_API n4m_status_t n4m_transform_range_discretizer_transform(
    const n4m_pp_range_disc_handle_t* h,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* N4M_TRANSFORM_RESAMPLING_H */
