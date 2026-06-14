/* SPDX-License-Identifier: CECILL-2.1 */
/* cpp/include/n4m/transform/signal_conversion.h — transform.signal_conversion methods (ABI 2.0). */
#ifndef N4M_TRANSFORM_SIGNAL_CONVERSION_H
#define N4M_TRANSFORM_SIGNAL_CONVERSION_H
#include "n4m/n4m.h"   /* shared infra: status, matrix view, context, N4M_API */

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- ToAbsorbance ----------------------------------------------- */
typedef struct n4m_pp_to_absorbance_handle_t n4m_pp_to_absorbance_handle_t;
N4M_API n4m_status_t n4m_transform_to_absorbance_create(
    n4m_pp_to_absorbance_handle_t** out,
    int is_percent, double epsilon, int clip_negative);
N4M_API void         n4m_transform_to_absorbance_destroy(
    n4m_pp_to_absorbance_handle_t* handle);
N4M_API n4m_status_t n4m_transform_to_absorbance_transform(
    const n4m_pp_to_absorbance_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

/* ---------- FromAbsorbance --------------------------------------------- */
typedef struct n4m_pp_from_absorbance_handle_t n4m_pp_from_absorbance_handle_t;
N4M_API n4m_status_t n4m_transform_from_absorbance_create(
    n4m_pp_from_absorbance_handle_t** out, int is_percent);
N4M_API void         n4m_transform_from_absorbance_destroy(
    n4m_pp_from_absorbance_handle_t* handle);
N4M_API n4m_status_t n4m_transform_from_absorbance_transform(
    const n4m_pp_from_absorbance_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

/* ---------- PercentToFraction ----------------------------------------- */
typedef struct n4m_pp_pct_to_frac_handle_t n4m_pp_pct_to_frac_handle_t;
N4M_API n4m_status_t n4m_transform_percent_to_fraction_create(
    n4m_pp_pct_to_frac_handle_t** out);
N4M_API void         n4m_transform_percent_to_fraction_destroy(
    n4m_pp_pct_to_frac_handle_t* handle);
N4M_API n4m_status_t n4m_transform_percent_to_fraction_transform(
    const n4m_pp_pct_to_frac_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

/* ---------- FractionToPercent ----------------------------------------- */
typedef struct n4m_pp_frac_to_pct_handle_t n4m_pp_frac_to_pct_handle_t;
N4M_API n4m_status_t n4m_transform_fraction_to_percent_create(
    n4m_pp_frac_to_pct_handle_t** out);
N4M_API void         n4m_transform_fraction_to_percent_destroy(
    n4m_pp_frac_to_pct_handle_t* handle);
N4M_API n4m_status_t n4m_transform_fraction_to_percent_transform(
    const n4m_pp_frac_to_pct_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

/* ---------- KubelkaMunk ----------------------------------------------- */
typedef struct n4m_pp_kubelka_munk_handle_t n4m_pp_kubelka_munk_handle_t;
N4M_API n4m_status_t n4m_transform_kubelka_munk_create(
    n4m_pp_kubelka_munk_handle_t** out, int is_percent, double epsilon);
N4M_API void         n4m_transform_kubelka_munk_destroy(
    n4m_pp_kubelka_munk_handle_t* handle);
N4M_API n4m_status_t n4m_transform_kubelka_munk_transform(
    const n4m_pp_kubelka_munk_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

/* ---------- 20.1 SignalTypeDetector ----------------------------------- */
typedef enum n4m_signal_type_t {
    N4M_SIGNAL_UNKNOWN               = 0,
    N4M_SIGNAL_ABSORBANCE            = 1,
    N4M_SIGNAL_REFLECTANCE           = 2,
    N4M_SIGNAL_REFLECTANCE_PERCENT   = 3,
    N4M_SIGNAL_TRANSMITTANCE         = 4,
    N4M_SIGNAL_TRANSMITTANCE_PERCENT = 5,
    N4M_SIGNAL_KUBELKA_MUNK          = 6,
    N4M_SIGNAL_LOG_1_R               = 7,
    N4M_SIGNAL_LOG_1_T               = 8
} n4m_signal_type_t;

N4M_API n4m_status_t n4m_transform_signal_type_detector(n4m_matrix_view_t X,
                                        const double* wavelengths_optional,
                                        int64_t wl_length,
                                        double confidence_threshold,
                                        n4m_signal_type_t* type_out,
                                        double* confidence_out,
                                        char reason_buf[256]);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* N4M_TRANSFORM_SIGNAL_CONVERSION_H */
