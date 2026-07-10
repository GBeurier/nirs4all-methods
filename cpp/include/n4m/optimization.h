/* SPDX-License-Identifier: CECILL-2.1 */
/* cpp/include/n4m/optimization.h — native hyperparameter-optimization role
 * header (ABI 2.1). A handle-based ask/tell optimizer + typed search space,
 * with all samplers/pruners behind reserved enum values so later phases are
 * additive (no ABI change). See docs/FINETUNING_F0_PR.md. */
#ifndef N4M_OPTIMIZATION_H
#define N4M_OPTIMIZATION_H
#include "n4m/n4m.h"   /* status, matrix view, context, config, algorithm,
                          array, validation_plan, method_result, N4M_API */

#ifdef __cplusplus
extern "C" {
#endif

/* ==== enums (all 4-byte, static-asserted in n4m.h) ======================== */

typedef enum n4m_param_kind_t {
    N4M_PARAM_INT = 0, N4M_PARAM_FLOAT = 1, N4M_PARAM_LOG_INT = 2,
    N4M_PARAM_LOG_FLOAT = 3, N4M_PARAM_CATEGORICAL = 4, N4M_PARAM_ORDINAL = 5,
    N4M_PARAM_SORTED_TUPLE = 6
} n4m_param_kind_t;

/* Typed categorical payload (D7): the `values` pointer in
 * n4m_search_space_add_categorical is reinterpreted per this tag. */
typedef enum n4m_cat_type_t {
    N4M_CAT_STR = 0,   /* const char* const*  */
    N4M_CAT_INT = 1,   /* const int64_t*      */
    N4M_CAT_FLOAT = 2, /* const double*       */
    N4M_CAT_BOOL = 3   /* const int32_t* (0/1)*/
} n4m_cat_type_t;

/* Generic declarative constraints (D8) — covers the dag-ml generation
 * vocabulary (mutex/requires/exclude + conditional activation). */
typedef enum n4m_constraint_kind_t {
    N4M_CONSTRAINT_MUTEX_GROUP = 0,   /* the whole (param,label) group may not co-occur */
    N4M_CONSTRAINT_REQUIRES = 1,      /* ref[0] present ⇒ ref[1] present            */
    N4M_CONSTRAINT_EXCLUDE = 2,       /* ref[0] and ref[1] may not both be present  */
    N4M_CONSTRAINT_CONDITION_IN = 3,  /* child active iff parent label ∈ {refs[1..]}*/
    N4M_CONSTRAINT_CONDITION_NOT_IN = 4
} n4m_constraint_kind_t;

/* Full numeric range reserved now; values not yet implemented cause
 * n4m_optimizer_create to return N4M_ERR_NOT_IMPLEMENTED. */
typedef enum n4m_sampler_kind_t {
    N4M_SAMPLER_RANDOM = 0, N4M_SAMPLER_SOBOL = 1, N4M_SAMPLER_LHS = 2,
    N4M_SAMPLER_TERNARY = 3, N4M_SAMPLER_GA = 4, N4M_SAMPLER_PSO = 5,
    N4M_SAMPLER_CMAES = 6, N4M_SAMPLER_TPE = 7, N4M_SAMPLER_GP_EI = 8
} n4m_sampler_kind_t;

typedef enum n4m_pruner_kind_t {
    N4M_PRUNER_NONE = 0, N4M_PRUNER_MEDIAN = 1, N4M_PRUNER_ASHA = 2,
    N4M_PRUNER_HYPERBAND = 3, N4M_PRUNER_RACING = 4
} n4m_pruner_kind_t;

typedef enum n4m_opt_direction_t {
    N4M_OPT_AUTO = 0,      /* derive from metric */
    N4M_OPT_MINIMIZE = 1, N4M_OPT_MAXIMIZE = 2
} n4m_opt_direction_t;

typedef enum n4m_eval_mode_t {   /* fold-score aggregation (nirs4all parity) */
    N4M_EVAL_BEST = 0, N4M_EVAL_MEAN = 1, N4M_EVAL_ROBUST_BEST = 2
} n4m_eval_mode_t;

typedef enum n4m_metric_t {      /* regression + classification (D6) */
    N4M_METRIC_RMSE = 0, N4M_METRIC_MSE = 1, N4M_METRIC_MAE = 2, N4M_METRIC_R2 = 3,
    N4M_METRIC_ACCURACY = 16, N4M_METRIC_BALANCED_ACCURACY = 17,
    N4M_METRIC_F1 = 18, N4M_METRIC_LOGLOSS = 19
} n4m_metric_t;

typedef enum n4m_liar_kind_t {   /* constant-liar for batch/parallel ask (D5) */
    N4M_LIAR_NONE = 0, N4M_LIAR_MIN = 1, N4M_LIAR_MEAN = 2, N4M_LIAR_MAX = 3
} n4m_liar_kind_t;

typedef enum n4m_trial_status_t {   /* D4 */
    N4M_TRIAL_RUNNING = 0, N4M_TRIAL_COMPLETED = 1,
    N4M_TRIAL_PRUNED = 2, N4M_TRIAL_FAILED = 3
} n4m_trial_status_t;

/* ==== forward-compatible options struct (D9) ============================= */

typedef struct n4m_optimizer_options_t {
    uint64_t            struct_size;      /* MUST equal sizeof(n4m_optimizer_options_t) */
    n4m_sampler_kind_t  sampler;
    n4m_pruner_kind_t   pruner;
    n4m_opt_direction_t direction;
    n4m_eval_mode_t     eval_mode;
    n4m_metric_t        metric;
    n4m_liar_kind_t     liar;
    int32_t             n_startup_trials; /* random seeding before adaptive kicks in */
    uint64_t            seed;
    double              timeout_seconds;  /* 0 = none; on expiry ask returns N4M_ERR_CANCELLED */
    uint8_t             reserved[64];     /* forward-compat; must be zeroed */
} n4m_optimizer_options_t;

/* Fill struct_size + defaults (sampler=RANDOM, pruner=NONE, direction=MINIMIZE,
 * eval_mode=MEAN, metric=RMSE, liar=NONE, n_startup_trials=10, seed=0). */
N4M_API void n4m_optimizer_options_init(n4m_optimizer_options_t* opts);

/* ==== search space ======================================================= */

typedef struct n4m_search_space_s n4m_search_space_t;

N4M_API n4m_status_t n4m_search_space_create(n4m_search_space_t** out);
N4M_API void         n4m_search_space_destroy(n4m_search_space_t* space);
N4M_API n4m_status_t n4m_search_space_add_int(n4m_search_space_t* space,
                        const char* name, int64_t low, int64_t high,
                        int64_t step, int32_t log);
N4M_API n4m_status_t n4m_search_space_add_float(n4m_search_space_t* space,
                        const char* name, double low, double high,
                        double step, int32_t log);   /* step<=0 ⇒ continuous */
N4M_API n4m_status_t n4m_search_space_add_categorical(n4m_search_space_t* space,
                        const char* name, n4m_cat_type_t type,
                        const void* values, int32_t n_values);
N4M_API n4m_status_t n4m_search_space_add_ordinal(n4m_search_space_t* space,
                        const char* name, const double* values, int32_t n_values);
N4M_API n4m_status_t n4m_search_space_add_sorted_tuple(n4m_search_space_t* space,
                        const char* name, int32_t length, double low, double high,
                        int32_t element_is_int);
/* refs are parallel arrays of param names + labels (label may be NULL where a
 * bare param reference suffices, e.g. mutex over presence). */
N4M_API n4m_status_t n4m_search_space_add_constraint(n4m_search_space_t* space,
                        n4m_constraint_kind_t kind, const char* const* param_refs,
                        const char* const* label_refs, int32_t n_refs);
N4M_API n4m_status_t n4m_search_space_num_params(const n4m_search_space_t* space,
                        int32_t* out_n);

/* ==== optimizer handle + ask/tell ======================================== */

typedef struct n4m_optimizer_s n4m_optimizer_t;
typedef struct n4m_trial_s     n4m_trial_t;   /* BORROWED — core-owned, valid until optimizer destroy */

N4M_API n4m_status_t n4m_optimizer_create(n4m_context_t* ctx,
                        const n4m_search_space_t* space,
                        const n4m_optimizer_options_t* opts,
                        n4m_optimizer_t** out);
N4M_API void         n4m_optimizer_destroy(n4m_optimizer_t* opt);
/* warm-start: force the next ask to return these numeric params (a categorical
 * value is passed as its choice index cast to double). */
N4M_API n4m_status_t n4m_optimizer_enqueue(n4m_optimizer_t* opt,
                        const char* const* names, const double* values, int32_t n);
N4M_API n4m_status_t n4m_optimizer_ask(n4m_optimizer_t* opt, n4m_trial_t** out_trial);
N4M_API n4m_status_t n4m_optimizer_ask_batch(n4m_optimizer_t* opt, int32_t n,
                        n4m_trial_t** out_trials, int32_t* out_count);
N4M_API n4m_status_t n4m_optimizer_tell(n4m_optimizer_t* opt, int64_t trial_id, double score);
N4M_API n4m_status_t n4m_optimizer_tell_result(n4m_optimizer_t* opt, int64_t trial_id,
                        n4m_trial_status_t status, double score, const char* error);
N4M_API n4m_status_t n4m_optimizer_tell_intermediate(n4m_optimizer_t* opt, int64_t trial_id,
                        int32_t step, double score, int32_t* out_should_prune);
N4M_API n4m_status_t n4m_optimizer_best(const n4m_optimizer_t* opt,
                        n4m_trial_t** out_best, double* out_score);
N4M_API n4m_status_t n4m_optimizer_get_trials(const n4m_optimizer_t* opt,
                        int64_t since_id, n4m_method_result_t** out);

/* Reserved persistence (D10) — return N4M_ERR_NOT_IMPLEMENTED until F7. */
N4M_API n4m_status_t n4m_optimizer_save(const n4m_optimizer_t* opt, n4m_array_t** out_blob);
N4M_API n4m_status_t n4m_optimizer_load(n4m_context_t* ctx, const uint8_t* blob,
                        uint64_t n, n4m_optimizer_t** out);

/* ==== trial accessors ==================================================== */

N4M_API n4m_status_t n4m_trial_get_id(const n4m_trial_t* trial, int64_t* out);
N4M_API n4m_status_t n4m_trial_get_int(const n4m_trial_t* trial, const char* name, int64_t* out);
N4M_API n4m_status_t n4m_trial_get_float(const n4m_trial_t* trial, const char* name, double* out);
N4M_API n4m_status_t n4m_trial_get_category(const n4m_trial_t* trial, const char* name,
                        int32_t* out_index, const char** out_label); /* label core-owned, UTF-8 */
N4M_API n4m_status_t n4m_trial_is_active(const n4m_trial_t* trial, const char* name, int32_t* out);
N4M_API n4m_status_t n4m_trial_get_rung(const n4m_trial_t* trial, int32_t* out);
N4M_API n4m_status_t n4m_trial_get_status(const n4m_trial_t* trial, n4m_trial_status_t* out);
N4M_API n4m_status_t n4m_trial_get_duration(const n4m_trial_t* trial, double* out_seconds);

/* ==== pure-native single-level driver (D11) ==============================
 * Thin driver over the same ask/tell primitives: objective = internal CV of a
 * native estimator over ONE validation_plan (no nested CV / selection / leakage
 * — those are dag-ml's). The search space is expected to declare the estimator
 * hyperparameters (F0: `n_components`). Result carries best_params + the trial
 * trace + best_score. */
N4M_API n4m_status_t n4m_finetune_estimator(n4m_context_t* ctx, n4m_algorithm_t estimator,
                        const n4m_matrix_view_t* X, const n4m_matrix_view_t* Y,
                        const n4m_validation_plan_t* plan, const n4m_search_space_t* space,
                        const n4m_optimizer_options_t* opts, int32_t n_trials,
                        n4m_method_result_t** out_result);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* N4M_OPTIMIZATION_H */
