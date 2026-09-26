/* SPDX-License-Identifier: CECILL-2.1 */
/* cpp/include/n4m/estimator.h — generic estimator roles (ABI 2.13).
 *
 * Every catalog method with a reusable fitted state is exposed through one
 * life cycle: create (method id + named parameters) -> fit -> transform /
 * predict on new rows -> export / import (N4ME bytes). Bindings and
 * controllers consume this surface instead of per-method functions.
 * Design: docs/abi/estimator_roles_design.md.
 *
 * Conventions:
 *   - Descriptor outputs (`*_info_v1_t`) carry `struct_size`, set by the
 *     caller. The library writes min(struct_size, sizeof(current)) bytes,
 *     zeroes that prefix on error, and rejects a struct_size smaller than the
 *     v1 layout with N4M_ERR_INVALID_ARGUMENT.
 *   - Input structs (`n4m_fit_inputs_v1_t`) carry `struct_size`; fields past
 *     the caller's struct_size are treated as absent.
 *   - Strings returned in descriptors are library-owned static storage.
 *   - Matrix views may be strided.
 */
#ifndef N4M_ESTIMATOR_H
#define N4M_ESTIMATOR_H
#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Roles, kinds, capabilities ------------------------------------- */

typedef enum n4m_method_kind_t {
    N4M_METHOD_ESTIMATOR = 1,
    N4M_METHOD_PROCEDURE = 2
} n4m_method_kind_t;

#define N4M_ROLE_TRANSFORMER   (1u << 0)
#define N4M_ROLE_REGRESSOR     (1u << 1)
#define N4M_ROLE_CLASSIFIER    (1u << 2)
#define N4M_ROLE_SELECTOR      (1u << 3)
#define N4M_ROLE_SAMPLE_FILTER (1u << 4)

#define N4M_CAP_TRANSFORM             (UINT64_C(1) << 0)
#define N4M_CAP_PREDICT               (UINT64_C(1) << 1)
#define N4M_CAP_PREDICT_PROBA         (UINT64_C(1) << 2)
#define N4M_CAP_DECISION_FUNCTION     (UINT64_C(1) << 3)
#define N4M_CAP_PREDICT_LABELS        (UINT64_C(1) << 4)
#define N4M_CAP_SELECTED_INDICES      (UINT64_C(1) << 5)
#define N4M_CAP_APPLY_MASK            (UINT64_C(1) << 6)
#define N4M_CAP_SERIALIZABLE          (UINT64_C(1) << 7)
#define N4M_CAP_AFFINE                (UINT64_C(1) << 8)
#define N4M_CAP_RETAINS_TRAINING_ROWS (UINT64_C(1) << 9)

/* Input requirement levels, one per n4m_fit_inputs_v1_t input. */
typedef enum n4m_input_requirement_t {
    N4M_INPUT_NONE = 0,
    N4M_INPUT_OPTIONAL = 1,
    N4M_INPUT_REQUIRED = 2
} n4m_input_requirement_t;

typedef enum n4m_fit_input_t {
    N4M_FIT_INPUT_Y = 0,
    N4M_FIT_INPUT_LABELS = 1,
    N4M_FIT_INPUT_SAMPLE_WEIGHT = 2,
    N4M_FIT_INPUT_GROUPS = 3,
    N4M_FIT_INPUT_FEATURE_GROUPS = 4,
    N4M_FIT_INPUT_BLOCKS = 5,
    N4M_FIT_INPUT_AXIS = 6,
    N4M_FIT_INPUT_TARGET_DOMAIN = 7,
    N4M_FIT_INPUT_FOLD_IDS = 8,
    N4M_FIT_INPUT_COUNT = 9
} n4m_fit_input_t;

typedef enum n4m_method_param_type_t {
    N4M_METHOD_PARAM_INT = 1,
    N4M_METHOD_PARAM_DOUBLE = 2,
    N4M_METHOD_PARAM_BOOL = 3,
    N4M_METHOD_PARAM_ENUM = 4,
    N4M_METHOD_PARAM_INT_ARRAY = 5,
    N4M_METHOD_PARAM_DOUBLE_ARRAY = 6
} n4m_method_param_type_t;

/* ---- Introspection (the native manifest) ----------------------------- */

typedef struct n4m_method_info_v1_t {
    uint32_t struct_size;
    int32_t kind;                 /* n4m_method_kind_t */
    const char* method_id;        /* "models.pls.cppls" */
    const char* fq_name;          /* "n4m.estimators.pls.cppls" */
    uint32_t roles;               /* N4M_ROLE_* mask */
    int32_t n_params;
    uint64_t capabilities;        /* N4M_CAP_* of a fitted estimator */
    const char* state_format;     /* N4ME state block format, "" if none */
    int32_t inputs[N4M_FIT_INPUT_COUNT]; /* n4m_input_requirement_t */
} n4m_method_info_v1_t;

typedef struct n4m_param_info_v1_t {
    uint32_t struct_size;
    int32_t type;                 /* n4m_method_param_type_t */
    const char* name;
    int32_t has_default;          /* 0: the parameter is required */
    int32_t n_choices;            /* N4M_METHOD_PARAM_ENUM only */
    int64_t default_length;       /* 1 for scalars, n for arrays */
    double min_value;             /* NaN when unbounded */
    double max_value;             /* NaN when unbounded */
    const char* const* choices;   /* N4M_METHOD_PARAM_ENUM labels */
} n4m_param_info_v1_t;

N4M_API n4m_status_t n4m_method_count(int32_t* out_count);
/* N4M_ERR_INVALID_ARGUMENT when the id is unknown; *out_index is then -1. */
N4M_API n4m_status_t n4m_method_find(const char* method_id, int32_t* out_index);
N4M_API n4m_status_t n4m_method_info_v1(int32_t index, n4m_method_info_v1_t* out);
N4M_API n4m_status_t n4m_method_param_info_v1(int32_t index, int32_t param,
                                              n4m_param_info_v1_t* out);
/* Default values of a parameter. INT/BOOL/ENUM/INT_ARRAY defaults are read
 * with the _int variant (ENUM as a choice index), DOUBLE/DOUBLE_ARRAY with
 * the _double variant. out=NULL,capacity=0 queries *out_count. */
N4M_API n4m_status_t n4m_method_param_default_int(int32_t index, int32_t param,
                                                  int64_t* out, int64_t capacity,
                                                  int64_t* out_count);
N4M_API n4m_status_t n4m_method_param_default_double(int32_t index, int32_t param,
                                                     double* out, int64_t capacity,
                                                     int64_t* out_count);

/* ---- Named, typed parameters ---------------------------------------- */

typedef struct n4m_params_s n4m_params_t;

N4M_API n4m_status_t n4m_params_create(n4m_context_t* ctx, int32_t method_index,
                                       n4m_params_t** out);
N4M_API void n4m_params_destroy(n4m_params_t* params);
/* Setters validate name, type and bounds; they return
 * N4M_ERR_INVALID_ARGUMENT without a message. n4m_params_validate reports
 * the first invalid or missing required parameter by name through ctx. */
N4M_API n4m_status_t n4m_params_set_int(n4m_params_t*, const char* name, int64_t value);
N4M_API n4m_status_t n4m_params_set_double(n4m_params_t*, const char* name, double value);
N4M_API n4m_status_t n4m_params_set_bool(n4m_params_t*, const char* name, int32_t value);
N4M_API n4m_status_t n4m_params_set_enum(n4m_params_t*, const char* name, const char* choice);
N4M_API n4m_status_t n4m_params_set_int_array(n4m_params_t*, const char* name,
                                              const int64_t* values, int64_t n);
N4M_API n4m_status_t n4m_params_set_double_array(n4m_params_t*, const char* name,
                                                 const double* values, int64_t n);
N4M_API n4m_status_t n4m_params_validate(n4m_context_t* ctx, const n4m_params_t* params);
/* Resolved value (explicit or default). ENUM and BOOL read through _int. */
N4M_API n4m_status_t n4m_params_get_int(const n4m_params_t*, const char* name,
                                        int64_t* out, int64_t capacity, int64_t* out_count);
N4M_API n4m_status_t n4m_params_get_double(const n4m_params_t*, const char* name,
                                           double* out, int64_t capacity, int64_t* out_count);

/* ---- Fit inputs ------------------------------------------------------ */

typedef struct n4m_fit_inputs_v1_t {
    uint32_t struct_size;
    const n4m_matrix_view_t* X;              /* required */
    const n4m_matrix_view_t* Y;              /* continuous targets */
    const int64_t* labels;                   /* class IDs, one per row */
    int64_t n_labels;
    const double* sample_weight;
    int64_t n_sample_weight;
    const int64_t* groups;                   /* sample groups, one per row */
    int64_t n_groups;
    const int64_t* feature_groups;           /* group ID per column */
    int64_t n_feature_groups;
    const int64_t* block_sizes;              /* multiblock column partition */
    int64_t n_blocks;
    const double* axis;                      /* spectral axis, one per column */
    int64_t n_axis;
    const n4m_matrix_view_t* X_target;       /* target-domain / slave spectra */
    const int64_t* fold_ids;                 /* internal-CV test fold per row */
    int64_t n_fold_ids;
} n4m_fit_inputs_v1_t;

/* ---- Estimator life cycle ------------------------------------------- */

typedef struct n4m_estimator_s n4m_estimator_t;

/* params may be NULL (all defaults). The estimator copies params. */
N4M_API n4m_status_t n4m_estimator_create(n4m_context_t* ctx, const char* method_id,
                                          const n4m_params_t* params,
                                          n4m_estimator_t** out);
N4M_API void n4m_estimator_destroy(n4m_estimator_t* est);
/* Refitting replaces the previous state. On failure the estimator is
 * unfitted. Missing required inputs are named in the context message. */
N4M_API n4m_status_t n4m_estimator_fit(n4m_context_t* ctx, n4m_estimator_t* est,
                                       const n4m_fit_inputs_v1_t* inputs);
N4M_API n4m_status_t n4m_estimator_is_fitted(const n4m_estimator_t* est, int32_t* out);
/* Method index, and the capabilities of the fitted state (0 when unfitted). */
N4M_API n4m_status_t n4m_estimator_info(const n4m_estimator_t* est, int32_t* out_method_index,
                                        uint64_t* out_capabilities);
N4M_API n4m_status_t n4m_estimator_get_params(n4m_context_t* ctx, const n4m_estimator_t* est,
                                              n4m_params_t** out_copy);
N4M_API n4m_status_t n4m_estimator_n_features_in(const n4m_estimator_t* est, int64_t* out);
/* Output width of transform or predict for a fitted estimator. */
N4M_API n4m_status_t n4m_estimator_transform_cols(const n4m_estimator_t* est, int64_t* out);
N4M_API n4m_status_t n4m_estimator_n_outputs(const n4m_estimator_t* est, int64_t* out);

/* Operations return N4M_ERR_UNSUPPORTED when the capability is absent and
 * N4M_ERR_NOT_FITTED before a successful fit. `out` must have X->rows rows. */
N4M_API n4m_status_t n4m_estimator_transform(n4m_context_t* ctx, const n4m_estimator_t* est,
                                             const n4m_matrix_view_t* X, n4m_matrix_view_t* out);
N4M_API n4m_status_t n4m_estimator_predict(n4m_context_t* ctx, const n4m_estimator_t* est,
                                           const n4m_matrix_view_t* X, n4m_matrix_view_t* out);
N4M_API n4m_status_t n4m_estimator_decision_function(n4m_context_t* ctx,
                                                     const n4m_estimator_t* est,
                                                     const n4m_matrix_view_t* X,
                                                     n4m_matrix_view_t* out);
N4M_API n4m_status_t n4m_estimator_predict_proba(n4m_context_t* ctx, const n4m_estimator_t* est,
                                                 const n4m_matrix_view_t* X,
                                                 n4m_matrix_view_t* out);
N4M_API n4m_status_t n4m_estimator_predict_labels(n4m_context_t* ctx,
                                                  const n4m_estimator_t* est,
                                                  const n4m_matrix_view_t* X,
                                                  int64_t* out, int64_t n);
N4M_API n4m_status_t n4m_estimator_classes(const n4m_estimator_t* est, int64_t* out,
                                           int64_t capacity, int64_t* out_count);
N4M_API n4m_status_t n4m_estimator_selected_indices(const n4m_estimator_t* est, int64_t* out,
                                                    int64_t capacity, int64_t* out_count);
N4M_API n4m_status_t n4m_estimator_apply_mask(n4m_context_t* ctx, const n4m_estimator_t* est,
                                              const n4m_matrix_view_t* X,
                                              const n4m_matrix_view_t* Y, uint8_t* mask,
                                              int64_t n);
/* Borrowed fit diagnostics; valid until the next fit, import or destroy of
 * this estimator. N4M_ERR_UNSUPPORTED when the method keeps none (for
 * example after import). */
N4M_API n4m_status_t n4m_estimator_fit_result(const n4m_estimator_t* est,
                                              const n4m_method_result_t** out_borrowed);

/* ---- N4ME fitted-state serialization -------------------------------- */

#define N4M_ESTIMATOR_SERIALIZATION_FORMAT_VERSION 1u
/* Flags for export. States that retain training rows (capability
 * N4M_CAP_RETAINS_TRAINING_ROWS) export only with this flag. */
#define N4M_EXPORT_ALLOW_TRAINING_ROWS (1u << 0)

/* Import refuses payloads larger than this per-context limit (default
 * 256 MiB). */
N4M_API n4m_status_t n4m_context_set_max_state_bytes(n4m_context_t* ctx, uint64_t max_bytes);

N4M_API n4m_status_t n4m_estimator_export_size(n4m_context_t* ctx, const n4m_estimator_t* est,
                                               uint32_t flags, size_t* out_size);
N4M_API n4m_status_t n4m_estimator_export_to_buffer(n4m_context_t* ctx,
                                                    const n4m_estimator_t* est,
                                                    uint32_t flags, void* buffer,
                                                    size_t buffer_size, size_t* out_written);
N4M_API n4m_status_t n4m_estimator_import_from_buffer(n4m_context_t* ctx, const void* buffer,
                                                      size_t buffer_size,
                                                      n4m_estimator_t** out);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* N4M_ESTIMATOR_H */
