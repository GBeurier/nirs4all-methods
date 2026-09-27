/* SPDX-License-Identifier: CECILL-2.1 */
/* cpp/include/n4m/estimator.h — generic estimator roles (ABI 2.13).
 *
 * Every catalog method with a reusable fitted state is exposed through one
 * life cycle: create (method id + named parameters) -> fit -> transform /
 * predict on new rows -> export / import (N4ME bytes). Methods without
 * reusable state (splitters, augmenters, diagnostics) are procedures, run
 * once through n4m_procedure_run with the same parameters and inputs.
 * Bindings and controllers consume this surface instead of per-method
 * functions.
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
/* Procedure roles: a procedure declares exactly one. */
#define N4M_ROLE_SPLITTER      (1u << 5)
#define N4M_ROLE_AUGMENTER     (1u << 6)
#define N4M_ROLE_GENERIC       (1u << 7)

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
    uint64_t capabilities;        /* N4M_CAP_* of a fitted estimator, 0 for procedures */
    const char* state_format;     /* N4ME state block format, "" if none */
    int32_t inputs[N4M_FIT_INPUT_COUNT]; /* n4m_input_requirement_t */
} n4m_method_info_v1_t;

typedef struct n4m_param_info_v1_t {
    uint32_t struct_size;
    int32_t type;                 /* n4m_method_param_type_t */
    const char* name;
    int32_t has_default;          /* 0: the parameter is required */
    int32_t n_choices;            /* N4M_METHOD_PARAM_ENUM only */
    int64_t default_length;       /* 1 for scalars, n for arrays; 0 for an
                                     optional scalar (has_default = 1): no
                                     published default, the caller may leave
                                     it unset (seeds: unset means 0) */
    double min_value;             /* NaN when unbounded */
    double max_value;             /* NaN when unbounded */
    const char* const* choices;   /* N4M_METHOD_PARAM_ENUM labels */
    /* ABI 2.14; callers with the shorter 2.13 layout get the prefix. */
    int32_t recorded;             /* 1: the fitted state records the value (or
                                     a dimension it fixes) and N4ME import
                                     refuses a contradicting value */
} n4m_param_info_v1_t;

N4M_API n4m_status_t n4m_method_count(int32_t* out_count);
/* The whole manifest as JSON (UTF-8, no NUL): per method its id, kind,
 * roles, DAG-ML node kinds, capabilities, fit inputs and typed parameters
 * with defaults (NaN written as null). With out == NULL only *out_size is
 * set; otherwise capacity must be at least that size. */
N4M_API n4m_status_t n4m_method_manifest_json(char* out, size_t capacity, size_t* out_size);
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
 * the first invalid or missing required parameter by name through ctx.
 * Doubles must be finite, except that a parameter whose default is NaN
 * (an optional value, unused when NaN) also accepts NaN. */
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
    const double* axis;                      /* spectral axis, one per column;
                                                wavelengths in nm, finite and
                                                strictly increasing, for the
                                                augmenters with nm constants */
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
/* Refitting replaces the previous state only when the fit succeeds: a failed
 * fit or refit leaves the estimator as it was (still fitted with its previous
 * state, or unfitted). Every per-row input (Y rows, labels, sample_weight,
 * groups, fold_ids) must have exactly X->rows entries and per-column inputs
 * (feature_groups, axis) X->cols; nothing is broadcast. Missing required and
 * mismatched inputs are named in the context message. */
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
 * N4M_ERR_NOT_FITTED before a successful fit. `out` must be a valid F64 view
 * of exactly X->rows rows and the operation's width (transform_cols for
 * transform, n_outputs otherwise); any other view is refused with
 * N4M_ERR_SHAPE_MISMATCH before anything is written (checked since 2.14). */
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

/* ---- Procedures ------------------------------------------------------ */

/* Runs a procedure (kind N4M_METHOD_PROCEDURE) once. `params` may be NULL
 * (all defaults); inputs are checked against the manifest as in
 * n4m_estimator_fit. The caller owns *out (n4m_method_result_destroy). The
 * result depends on the procedure role:
 *   N4M_ROLE_SPLITTER   every fold, read with n4m_method_result_get_n_folds
 *                       and n4m_method_result_get_fold;
 *   N4M_ROLE_AUGMENTER  double matrix "X": the augmented rows, same shape and
 *                       row order as inputs->X (train-only, no fitted state);
 *                       augmenters that mix rows (y required) also return
 *                       double matrix "Y": the targets mixed with the same
 *                       draw, same shape as inputs->Y, row i paired with
 *                       row i of "X";
 *   N4M_ROLE_GENERIC    the named outputs of the method's C function.
 * N4M_ERR_INVALID_ARGUMENT for an estimator. */
N4M_API n4m_status_t n4m_procedure_run(n4m_context_t* ctx, int32_t method_index,
                                       const n4m_params_t* params,
                                       const n4m_fit_inputs_v1_t* inputs,
                                       n4m_method_result_t** out);
/* Folds of a splitter result: zero-based row indices of inputs->X in the
 * splitter's order. The arrays are borrowed from the result.
 * N4M_ERR_INVALID_ARGUMENT when the result holds no folds or the fold is
 * out of range. */
N4M_API n4m_status_t n4m_method_result_get_n_folds(const n4m_method_result_t* result,
                                                   int32_t* out_n_folds);
N4M_API n4m_status_t n4m_method_result_get_fold(const n4m_method_result_t* result, int32_t fold,
                                                const int64_t** out_train, int64_t* out_n_train,
                                                const int64_t** out_test, int64_t* out_n_test);

/* ---- N4ME fitted-state serialization -------------------------------- */

#define N4M_ESTIMATOR_SERIALIZATION_FORMAT_VERSION 1u
/* Flags for export. States that retain training rows (capability
 * N4M_CAP_RETAINS_TRAINING_ROWS) export only with this flag, which every
 * binding facade exposes as an explicit opt-in defaulting to off. */
#define N4M_EXPORT_ALLOW_TRAINING_ROWS (1u << 0)

/* ABI 2.14. *out = 1 when the fitted (or imported) state embeds training
 * rows, so that exporting it shares training data and needs
 * N4M_EXPORT_ALLOW_TRAINING_ROWS; 0 otherwise. N4M_ERR_NOT_FITTED before a
 * fit. The method-level capability "retains_training_rows" of the manifest
 * says which methods can produce such states. */
N4M_API n4m_status_t n4m_estimator_contains_training_rows(const n4m_estimator_t* est,
                                                         int32_t* out);

/* Import refuses payloads larger than this per-context limit (default
 * 256 MiB), and payloads whose parameters contradict their state: a
 * parameter the state records (its value, or a dimension it fixes, such as
 * n_components against the stored loadings) must equal what the state
 * implies. */
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

/* ---- Role pipelines (ABI 2.14) -------------------------------------- */

/* A trained linear recipe of estimators, owned natively so every binding
 * shares one implementation of the portable trained pipeline
 * (docs/abi/estimator_roles_design.md, D9). Steps, in order: zero or more
 * sample filters (fitted and applied to the training rows only; they keep
 * no state), zero or more transformers / selectors, then exactly one
 * regressor or classifier. A method with several roles plays the one its
 * position gives it (PLS is a transformer inside, a regressor last).
 *
 * Fit routing (n4m_fit_inputs_v1_t of the whole pipeline):
 *   - Y reaches every step whose manifest requires y, with all its columns.
 *     Without Y, the non-terminal steps that require y receive the class
 *     labels as one double column (classification recipes).
 *   - labels, sample_weight, groups and fold_ids reach the steps that
 *     declare them. Sample filters subset every row-aligned input.
 *   - feature_groups, blocks, axis and X_target describe the pipeline input
 *     columns: they reach the steps that declare them only while no
 *     transformer or selector has changed the columns.
 *   - An input no step uses is refused by name.
 * Every error names the step: "step <i> (<method id>): <message>".
 *
 * Feature identity: names set with n4m_role_pipeline_set_feature_names are
 * part of the fitted pipeline; n4m_role_pipeline_check_features refuses a
 * width mismatch and, when names are stored and given, any missing, renamed
 * or reordered column. Arrays without names are positional.
 *
 * Operations follow n4m_estimator_*: the terminal step answers predict,
 * predict_labels, decision_function and predict_proba (N4M_ERR_UNSUPPORTED
 * when its role lacks the operation); transform runs the transformers and
 * selectors only. Output views must have X->rows rows and the operation's
 * width (n4m_role_pipeline_transform_cols / _n_outputs), else
 * N4M_ERR_SHAPE_MISMATCH. N4M_ERR_NOT_FITTED before a successful fit or
 * import. */
typedef struct n4m_role_pipeline_s n4m_role_pipeline_t;

typedef struct n4m_role_pipeline_step_info_v1_t {
    uint32_t struct_size;
    int32_t method_index;
    const char* method_id;          /* library-owned static string */
    uint32_t role;                  /* the one N4M_ROLE_* the step plays */
    int32_t state_index;            /* index among the stateful steps; -1 for
                                       a sample filter (train-only, no state) */
    int32_t contains_training_rows; /* 1 when the fitted state embeds training
                                       rows (export needs
                                       N4M_EXPORT_ALLOW_TRAINING_ROWS) */
    int64_t n_features_in;          /* fitted widths; 0 while unfitted */
    int64_t n_features_out;
} n4m_role_pipeline_step_info_v1_t;

/* method_ids: n_steps catalog ids. params: NULL (all defaults) or n_steps
 * entries, each NULL (defaults) or created for that step's method; the
 * pipeline copies them. Refuses an empty recipe, unknown methods,
 * procedures, foreign params, missing required parameters and any other
 * role order. */
N4M_API n4m_status_t n4m_role_pipeline_create(n4m_context_t* ctx, int32_t n_steps,
                                              const char* const* method_ids,
                                              const n4m_params_t* const* params,
                                              n4m_role_pipeline_t** out);
N4M_API void n4m_role_pipeline_destroy(n4m_role_pipeline_t* pipeline);
/* UTF-8, unique names of the input columns, set before fit or import
 * (N4M_ERR_INVALID_ARGUMENT once fitted); n = 0 clears them. Fit and import
 * refuse a count different from the input width. */
N4M_API n4m_status_t n4m_role_pipeline_set_feature_names(n4m_context_t* ctx,
                                                         n4m_role_pipeline_t* pipeline,
                                                         const char* const* names, int64_t n);
/* Fits every step in order. On failure the pipeline is unfitted. */
N4M_API n4m_status_t n4m_role_pipeline_fit(n4m_context_t* ctx, n4m_role_pipeline_t* pipeline,
                                           const n4m_fit_inputs_v1_t* inputs);
/* Rebuilds a fitted pipeline from one N4ME state per stateful step, in step
 * order. Refuses a state count that differs from the recipe, a state of
 * another method, a state whose parameters differ from the recipe's
 * (typed comparison after default resolution), a state without the
 * operation its role needs, and widths that do not chain. On failure the
 * pipeline is unfitted. */
N4M_API n4m_status_t n4m_role_pipeline_import_states(n4m_context_t* ctx,
                                                     n4m_role_pipeline_t* pipeline,
                                                     int32_t n_states,
                                                     const void* const* states,
                                                     const size_t* state_sizes);
N4M_API n4m_status_t n4m_role_pipeline_is_fitted(const n4m_role_pipeline_t* pipeline,
                                                 int32_t* out);
N4M_API n4m_status_t n4m_role_pipeline_n_steps(const n4m_role_pipeline_t* pipeline,
                                               int32_t* out);
/* Number of stateful steps (all but the sample filters). */
N4M_API n4m_status_t n4m_role_pipeline_n_states(const n4m_role_pipeline_t* pipeline,
                                                int32_t* out);
N4M_API n4m_status_t n4m_role_pipeline_step_info_v1(const n4m_role_pipeline_t* pipeline,
                                                    int32_t step,
                                                    n4m_role_pipeline_step_info_v1_t* out);
/* Input width of a fitted pipeline. */
N4M_API n4m_status_t n4m_role_pipeline_n_features_in(const n4m_role_pipeline_t* pipeline,
                                                     int64_t* out);
/* Stored feature names: count (0 when positional), then each name,
 * borrowed until the next set, fit, import or destroy. */
N4M_API n4m_status_t n4m_role_pipeline_n_feature_names(const n4m_role_pipeline_t* pipeline,
                                                       int64_t* out);
N4M_API n4m_status_t n4m_role_pipeline_feature_name(const n4m_role_pipeline_t* pipeline,
                                                    int64_t index, const char** out_borrowed);
/* Checks new input columns against the fitted pipeline: N4M_ERR_SHAPE_MISMATCH
 * when n_columns differs from the input width; N4M_ERR_INVALID_ARGUMENT when
 * names (n_columns entries, or NULL for positional data) and the stored
 * names differ at any position. */
N4M_API n4m_status_t n4m_role_pipeline_check_features(n4m_context_t* ctx,
                                                      const n4m_role_pipeline_t* pipeline,
                                                      int64_t n_columns,
                                                      const char* const* names);
/* Output width of transform (the input of the terminal step) and of the
 * terminal step's predict / decision_function / predict_proba. */
N4M_API n4m_status_t n4m_role_pipeline_transform_cols(const n4m_role_pipeline_t* pipeline,
                                                      int64_t* out);
N4M_API n4m_status_t n4m_role_pipeline_n_outputs(const n4m_role_pipeline_t* pipeline,
                                                 int64_t* out);
N4M_API n4m_status_t n4m_role_pipeline_transform(n4m_context_t* ctx,
                                                 const n4m_role_pipeline_t* pipeline,
                                                 const n4m_matrix_view_t* X,
                                                 n4m_matrix_view_t* out);
N4M_API n4m_status_t n4m_role_pipeline_predict(n4m_context_t* ctx,
                                               const n4m_role_pipeline_t* pipeline,
                                               const n4m_matrix_view_t* X,
                                               n4m_matrix_view_t* out);
N4M_API n4m_status_t n4m_role_pipeline_decision_function(n4m_context_t* ctx,
                                                         const n4m_role_pipeline_t* pipeline,
                                                         const n4m_matrix_view_t* X,
                                                         n4m_matrix_view_t* out);
N4M_API n4m_status_t n4m_role_pipeline_predict_proba(n4m_context_t* ctx,
                                                     const n4m_role_pipeline_t* pipeline,
                                                     const n4m_matrix_view_t* X,
                                                     n4m_matrix_view_t* out);
N4M_API n4m_status_t n4m_role_pipeline_predict_labels(n4m_context_t* ctx,
                                                      const n4m_role_pipeline_t* pipeline,
                                                      const n4m_matrix_view_t* X, int64_t* out,
                                                      int64_t n);
/* Class ids of a classifier pipeline, ascending. */
N4M_API n4m_status_t n4m_role_pipeline_classes(const n4m_role_pipeline_t* pipeline,
                                               int64_t* out, int64_t capacity,
                                               int64_t* out_count);
/* N4ME state of stateful step `state` (see state_index). States that embed
 * training rows export only with N4M_EXPORT_ALLOW_TRAINING_ROWS. */
N4M_API n4m_status_t n4m_role_pipeline_export_state_size(n4m_context_t* ctx,
                                                         const n4m_role_pipeline_t* pipeline,
                                                         int32_t state, uint32_t flags,
                                                         size_t* out_size);
N4M_API n4m_status_t n4m_role_pipeline_export_state_to_buffer(
    n4m_context_t* ctx, const n4m_role_pipeline_t* pipeline, int32_t state, uint32_t flags,
    void* buffer, size_t buffer_size, size_t* out_written);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* N4M_ESTIMATOR_H */
