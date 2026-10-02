/* SPDX-License-Identifier: CECILL-2.1 */
#ifndef N4M_MULTIMODAL_H
#define N4M_MULTIMODAL_H

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Dense, complete-source early fusion. No folds, scoring or row alignment
 * live here. The caller supplies already aligned rows for this fit scope.
 * All pointers are borrowed for a call; fitted state owns no input rows.
 * Numeric tensors use explicit element strides and sample axis zero. */
typedef enum n4m_multimodal_encoder_t {
    N4M_MULTIMODAL_STANDARD_SCALER = 1,
    N4M_MULTIMODAL_TENSOR_PCA = 2,
    N4M_MULTIMODAL_COLUMN_TRANSFORMER = 3
} n4m_multimodal_encoder_t;

typedef struct n4m_multimodal_source_spec_v1_t {
    uint32_t struct_size;
    const char* name;
    const char* representation_id;
    const char* dtype;
    const void* identity_utf8; /* independently supplied canonical IO descriptor */
    size_t identity_bytes;
    int32_t ndim;             /* excludes the sample axis; 1..7 */
    const int64_t* shape;
    uint32_t encoder;
    double weight;
    int64_t n_components;     /* PCA: explicit positive count */
    int64_t random_state;     /* recorded; full SVD does not draw random numbers */
    int32_t with_mean;
    int32_t with_std;
    int32_t whiten;           /* first profile requires zero */
    int32_t ignore_unknown;   /* mixed table requires one */
    int64_t numeric_column;   /* mixed table: declared numeric position */
    int64_t categorical_column; /* mixed table: declared UTF-8 position */
} n4m_multimodal_source_spec_v1_t;

typedef struct n4m_multimodal_recipe_v1_t {
    uint32_t struct_size;
    int32_t n_sources;
    const n4m_multimodal_source_spec_v1_t* sources; /* explicit fusion order */
    double alpha;
    int32_t center_x;
    int32_t center_y;
    int32_t scale_x;          /* first profile requires zero */
} n4m_multimodal_recipe_v1_t;

typedef struct n4m_multimodal_source_view_v1_t {
    uint32_t struct_size;
    const char* name;
    const char* representation_id;
    const char* dtype;
    const void* identity_utf8;
    size_t identity_bytes;
    int32_t rank;             /* includes the sample axis; 2..8 */
    const int64_t* shape;
    const int64_t* strides;   /* element strides; numeric tensors only */
    const void* numeric_data;
    n4m_dtype_t numeric_dtype;
    /* A mixed table has shape (rows,2): numeric_data is a (rows,1)
     * numeric column with strides[0], and categorical cells are exact
     * length-delimited UTF-8, one per row. No learned host category codes.
     * offsets has rows+1 entries, starts at zero and ends at utf8_bytes. */
    const void* categorical_utf8;
    size_t utf8_bytes;
    const uint64_t* categorical_offsets;
} n4m_multimodal_source_view_v1_t;

typedef struct n4m_multimodal_pipeline_s n4m_multimodal_pipeline_t;

N4M_API n4m_status_t n4m_multimodal_pipeline_create(
    n4m_context_t* ctx, const n4m_multimodal_recipe_v1_t* recipe,
    n4m_multimodal_pipeline_t** out);
N4M_API void n4m_multimodal_pipeline_destroy(n4m_multimodal_pipeline_t* pipeline);
/* Transactional fit: failure preserves the previous fitted state. */
N4M_API n4m_status_t n4m_multimodal_pipeline_fit(
    n4m_context_t* ctx, n4m_multimodal_pipeline_t* pipeline, int32_t n_sources,
    const n4m_multimodal_source_view_v1_t* sources, const n4m_matrix_view_t* Y);
N4M_API n4m_status_t n4m_multimodal_pipeline_predict(
    n4m_context_t* ctx, const n4m_multimodal_pipeline_t* pipeline, int32_t n_sources,
    const n4m_multimodal_source_view_v1_t* sources, n4m_matrix_view_t* out);
/* The actual weighted encoded features, useful for independent parity.
 * No fit is performed. Output width is queried separately. */
N4M_API n4m_status_t n4m_multimodal_pipeline_transform_cols(
    const n4m_multimodal_pipeline_t* pipeline, int64_t* out);
N4M_API n4m_status_t n4m_multimodal_pipeline_transform(
    n4m_context_t* ctx, const n4m_multimodal_pipeline_t* pipeline, int32_t n_sources,
    const n4m_multimodal_source_view_v1_t* sources, n4m_matrix_view_t* out);

/* N4MF, LE format 1: complete resolved recipe, source identities, scaler /
 * PCA N4ME states, learned UTF-8 vocabulary and Ridge N4ME; final FNV1a64
 * integrity checksum. No training rows. Limit min(context limit,64MiB).
 * Import requires an independently supplied expected recipe/schema and
 * validates the entire payload before publishing a fitted handle. */
N4M_API n4m_status_t n4m_multimodal_pipeline_export_size(
    n4m_context_t* ctx, const n4m_multimodal_pipeline_t* pipeline, size_t* out);
N4M_API n4m_status_t n4m_multimodal_pipeline_export_to_buffer(
    n4m_context_t* ctx, const n4m_multimodal_pipeline_t* pipeline,
    void* buffer, size_t buffer_size, size_t* out_written);
N4M_API n4m_status_t n4m_multimodal_pipeline_import_from_buffer(
    n4m_context_t* ctx, const n4m_multimodal_recipe_v1_t* expected_recipe,
    const void* buffer, size_t buffer_size, n4m_multimodal_pipeline_t** out);

#ifdef __cplusplus
}
#endif
#endif
