/* SPDX-License-Identifier: CECILL-2.1 */
#ifndef N4M_FILTER_ROLE_H
#define N4M_FILTER_ROLE_H
#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Closed, train/apply filter roles. They reuse the seven existing kernels;
 * neither role serializes fitted state. Views must be contiguous F64. An
 * optional Y must be one column with the same row count as X. */
typedef enum n4m_sample_filter_kind_t {
    N4M_SAMPLE_FILTER_Y_OUTLIER = 0,
    N4M_SAMPLE_FILTER_X_OUTLIER = 1,
    N4M_SAMPLE_FILTER_HIGH_LEVERAGE = 2,
    N4M_SAMPLE_FILTER_SPECTRAL_QUALITY = 3,
    N4M_SAMPLE_FILTER_COMPOSITE = 4
} n4m_sample_filter_kind_t;

typedef enum n4m_feature_filter_kind_t {
    N4M_FEATURE_FILTER_VARIANCE = 0,
    N4M_FEATURE_FILTER_CORRELATION = 1
} n4m_feature_filter_kind_t;

typedef struct n4m_sample_filter_t n4m_sample_filter_t;
typedef struct n4m_feature_filter_t n4m_feature_filter_t;

/* Positional constructor schema (exact lengths, no omitted entries):
 * Y: i=[method], d=[threshold,lower_pct,upper_pct]
 * X: i=[method,use_threshold,n_components,n_estimators,max_samples],
 *    d=[threshold,contamination], seed used
 * leverage: i=[method,use_absolute,n_components,center],
 *    d=[threshold_multiplier,absolute_threshold]
 * quality: i=[use_max,use_min,check_inf],
 *    d=[max_nan_ratio,max_zero_ratio,min_variance,max_value,min_value]
 * composite: i=[mode], d=[]; only leverage/quality children are accepted.
 * Boolean fields must be 0 or 1. No defaults are supplied by this ABI. */
N4M_API n4m_status_t n4m_sample_filter_create(
    int32_t kind, const int64_t* i, int32_t ni, const double* d, int32_t nd,
    uint64_t seed, n4m_sample_filter_t** out);
N4M_API n4m_status_t n4m_sample_filter_add_child(
    n4m_sample_filter_t* parent, int32_t kind, const int64_t* i, int32_t ni,
    const double* d, int32_t nd, uint64_t seed);
/* Child ownership transfers to parent only on success; destruction always
 * destroys the native composite before its owned children. */
N4M_API n4m_status_t n4m_sample_filter_fit(
    n4m_sample_filter_t* filter, const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y);
N4M_API n4m_status_t n4m_sample_filter_apply(
    const n4m_sample_filter_t* filter, const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y, uint8_t* mask, n4m_filter_stats_t* stats);
N4M_API n4m_status_t n4m_sample_filter_is_fitted(
    const n4m_sample_filter_t* filter, int* out);
N4M_API void n4m_sample_filter_destroy(n4m_sample_filter_t* filter);

/* threshold/top_k semantics are identical to the direct native selectors.
 * Correlation requires univariate Y at fit; variance requires Y == NULL.
 * selected_indices are zero-based input column indices in transform order. */
N4M_API n4m_status_t n4m_feature_filter_create(
    int32_t kind, double threshold, int32_t top_k,
    n4m_feature_filter_t** out);
N4M_API n4m_status_t n4m_feature_filter_fit(
    n4m_feature_filter_t* filter, const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y);
N4M_API n4m_status_t n4m_feature_filter_output_cols(
    const n4m_feature_filter_t* filter, int64_t* out);
/* out=NULL,capacity=0 queries the count. On insufficient capacity, count
 * receives the required count and out is unchanged. */
N4M_API n4m_status_t n4m_feature_filter_selected_indices(
    const n4m_feature_filter_t* filter, int64_t* out, int64_t capacity,
    int64_t* count);
N4M_API n4m_status_t n4m_feature_filter_transform(
    const n4m_feature_filter_t* filter, const n4m_matrix_view_t* X,
    n4m_matrix_view_t* out);
N4M_API n4m_status_t n4m_feature_filter_is_fitted(
    const n4m_feature_filter_t* filter, int* out);
N4M_API void n4m_feature_filter_destroy(n4m_feature_filter_t* filter);

#ifdef __cplusplus
}
#endif
#endif
