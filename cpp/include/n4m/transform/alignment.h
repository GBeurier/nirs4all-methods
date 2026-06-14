/* SPDX-License-Identifier: CECILL-2.1 */
/* cpp/include/n4m/transform/alignment.h — transform.alignment methods (ABI 2.0). */
#ifndef N4M_TRANSFORM_ALIGNMENT_H
#define N4M_TRANSFORM_ALIGNMENT_H
#include "n4m/n4m.h"   /* shared infra: status, matrix view, context, N4M_API */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_pp_xcorr_align_handle_t n4m_pp_xcorr_align_handle_t;
typedef struct n4m_pp_icoshift_align_handle_t n4m_pp_icoshift_align_handle_t;
typedef struct n4m_pp_dtw_align_handle_t n4m_pp_dtw_align_handle_t;
typedef struct n4m_pp_cow_align_handle_t n4m_pp_cow_align_handle_t;
N4M_API n4m_status_t n4m_transform_xcorr_align_create(
    n4m_pp_xcorr_align_handle_t** out, const double* reference,
    int64_t n_reference, int32_t interval_size, int32_t max_shift);
N4M_API void n4m_transform_xcorr_align_destroy(n4m_pp_xcorr_align_handle_t* handle);
N4M_API n4m_status_t n4m_transform_xcorr_align_fit(
    n4m_pp_xcorr_align_handle_t* handle, n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_transform_xcorr_align_transform(
    const n4m_pp_xcorr_align_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_transform_xcorr_align_is_fitted(
    const n4m_pp_xcorr_align_handle_t* handle, int* out_fitted);
N4M_API n4m_status_t n4m_transform_icoshift_align_create(
    n4m_pp_icoshift_align_handle_t** out, const double* reference,
    int64_t n_reference, int32_t interval_size, int32_t max_shift);
N4M_API void n4m_transform_icoshift_align_destroy(
    n4m_pp_icoshift_align_handle_t* handle);
N4M_API n4m_status_t n4m_transform_icoshift_align_fit(
    n4m_pp_icoshift_align_handle_t* handle, n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_transform_icoshift_align_transform(
    const n4m_pp_icoshift_align_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_transform_icoshift_align_is_fitted(
    const n4m_pp_icoshift_align_handle_t* handle, int* out_fitted);
N4M_API n4m_status_t n4m_transform_dtw_align_create(
    n4m_pp_dtw_align_handle_t** out, const double* reference,
    int64_t n_reference, int32_t interval_size, int32_t max_shift);
N4M_API void n4m_transform_dtw_align_destroy(n4m_pp_dtw_align_handle_t* handle);
N4M_API n4m_status_t n4m_transform_dtw_align_fit(
    n4m_pp_dtw_align_handle_t* handle, n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_transform_dtw_align_transform(
    const n4m_pp_dtw_align_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_transform_dtw_align_is_fitted(
    const n4m_pp_dtw_align_handle_t* handle, int* out_fitted);
N4M_API n4m_status_t n4m_transform_cow_align_create(
    n4m_pp_cow_align_handle_t** out, const double* reference,
    int64_t n_reference, int32_t interval_size, int32_t max_shift);
N4M_API void n4m_transform_cow_align_destroy(n4m_pp_cow_align_handle_t* handle);
N4M_API n4m_status_t n4m_transform_cow_align_fit(
    n4m_pp_cow_align_handle_t* handle, n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_transform_cow_align_transform(
    const n4m_pp_cow_align_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_transform_cow_align_is_fitted(
    const n4m_pp_cow_align_handle_t* handle, int* out_fitted);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* N4M_TRANSFORM_ALIGNMENT_H */
