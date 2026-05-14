// SPDX-License-Identifier: CeCILL-2.1
//
// Phase 0 stubs for the model lifecycle, predict/transform, get_array, and
// binary serialization. Every status-returning function returns
// P4A_ERR_NOT_IMPLEMENTED; void destroys are NULL-safe no-ops. Phase 1
// replaces this TU with the real NIPALS / SIMPLS implementations.

#include <stddef.h>
#include <stdint.h>

#include "pls4all/p4a.h"

// Empty body — Phase 0 has no model state. The fit function never allocates
// a model_s, so deletes are always safe.
struct p4a_model_s {};
struct p4a_array_s {};

extern "C" {

P4A_API p4a_status_t p4a_model_fit(
    p4a_context_t* /*ctx*/, const p4a_config_t* /*cfg*/,
    const p4a_matrix_view_t* /*X*/, const p4a_matrix_view_t* /*Y*/,
    p4a_model_t** out_model) {
    if (out_model != nullptr) *out_model = nullptr;
    return P4A_ERR_NOT_IMPLEMENTED;
}

P4A_API void p4a_model_destroy(p4a_model_t* model) {
    try { delete model; } catch (...) {}
}

P4A_API p4a_status_t p4a_model_predict(
    p4a_context_t* /*ctx*/, const p4a_model_t* /*model*/,
    const p4a_matrix_view_t* /*X*/, p4a_matrix_view_t* /*out*/) {
    return P4A_ERR_NOT_IMPLEMENTED;
}

P4A_API p4a_status_t p4a_model_transform(
    p4a_context_t* /*ctx*/, const p4a_model_t* /*model*/,
    const p4a_matrix_view_t* /*X*/, p4a_matrix_view_t* /*out_scores*/) {
    return P4A_ERR_NOT_IMPLEMENTED;
}

P4A_API p4a_status_t p4a_model_predict_alloc(
    p4a_context_t* /*ctx*/, const p4a_model_t* /*model*/,
    const p4a_matrix_view_t* /*X*/, p4a_array_t** out) {
    if (out != nullptr) *out = nullptr;
    return P4A_ERR_NOT_IMPLEMENTED;
}

P4A_API p4a_status_t p4a_model_transform_alloc(
    p4a_context_t* /*ctx*/, const p4a_model_t* /*model*/,
    const p4a_matrix_view_t* /*X*/, p4a_array_t** out_scores) {
    if (out_scores != nullptr) *out_scores = nullptr;
    return P4A_ERR_NOT_IMPLEMENTED;
}

P4A_API p4a_status_t p4a_model_get_array(
    p4a_context_t* /*ctx*/, const p4a_model_t* /*model*/,
    p4a_model_array_t /*which*/, p4a_array_t** out) {
    if (out != nullptr) *out = nullptr;
    return P4A_ERR_NOT_IMPLEMENTED;
}

P4A_API p4a_status_t p4a_model_get_n_components(const p4a_model_t* /*m*/,
                                                  int32_t* /*out*/) {
    return P4A_ERR_NOT_IMPLEMENTED;
}
P4A_API p4a_status_t p4a_model_get_n_features(const p4a_model_t* /*m*/,
                                                int32_t* /*out*/) {
    return P4A_ERR_NOT_IMPLEMENTED;
}
P4A_API p4a_status_t p4a_model_get_n_targets(const p4a_model_t* /*m*/,
                                               int32_t* /*out*/) {
    return P4A_ERR_NOT_IMPLEMENTED;
}

/* --- Serialization (Phase 0 stubs) --- */

P4A_API p4a_status_t p4a_model_export_size(const p4a_model_t* /*model*/,
                                             size_t* out_size) {
    if (out_size != nullptr) *out_size = 0;
    return P4A_ERR_NOT_IMPLEMENTED;
}

P4A_API p4a_status_t p4a_model_export_to_buffer(
    const p4a_model_t* /*model*/, void* /*buffer*/, size_t /*buffer_size*/,
    size_t* out_written) {
    if (out_written != nullptr) *out_written = 0;
    return P4A_ERR_NOT_IMPLEMENTED;
}

P4A_API p4a_status_t p4a_model_import_from_buffer(
    p4a_context_t* /*ctx*/, const void* /*buffer*/, size_t /*buffer_size*/,
    p4a_model_t** out_model) {
    if (out_model != nullptr) *out_model = nullptr;
    return P4A_ERR_NOT_IMPLEMENTED;
}

P4A_API p4a_status_t p4a_serialization_inspect(
    const void* /*buffer*/, size_t /*buffer_size*/,
    uint32_t* out_format_version,
    uint32_t* out_writer_abi_major,
    uint32_t* out_writer_abi_minor,
    uint32_t* out_writer_abi_patch) {
    if (out_format_version)    *out_format_version    = 0;
    if (out_writer_abi_major)  *out_writer_abi_major  = 0;
    if (out_writer_abi_minor)  *out_writer_abi_minor  = 0;
    if (out_writer_abi_patch)  *out_writer_abi_patch  = 0;
    return P4A_ERR_NOT_IMPLEMENTED;
}

/* --- Output array accessor (Phase 0 stubs — no array can ever exist) --- */

P4A_API p4a_status_t p4a_array_dtype(const p4a_array_t* /*arr*/,
                                      p4a_dtype_t* out) {
    if (out != nullptr) *out = P4A_DTYPE_UNKNOWN;
    return P4A_ERR_NOT_IMPLEMENTED;
}

P4A_API p4a_status_t p4a_array_shape(const p4a_array_t* /*arr*/,
                                      int64_t* rows, int64_t* cols) {
    if (rows != nullptr) *rows = 0;
    if (cols != nullptr) *cols = 0;
    return P4A_ERR_NOT_IMPLEMENTED;
}

P4A_API p4a_status_t p4a_array_view(const p4a_array_t* /*arr*/,
                                     p4a_matrix_view_t* /*out*/) {
    return P4A_ERR_NOT_IMPLEMENTED;
}

P4A_API void p4a_array_free(p4a_array_t* arr) {
    try { delete arr; } catch (...) {}
}

}  // extern "C"
