/* SPDX-License-Identifier: CECILL-2.1 */
/* cpp/include/n4m/optimization.h — native hyperparameter-optimization role
 * header (ABI 2.2). A handle-based ask/tell optimizer + typed search space,
 * with nine samplers and five pruners behind one stable enum-based ABI. See
 * docs/FINETUNING_F0_PR.md for the original surface contract. */
#ifndef N4M_OPTIMIZATION_H
#define N4M_OPTIMIZATION_H
#include "n4m/n4m.h"   /* status, matrix view, context, config, algorithm,
                          array, validation_plan, method_result, N4M_API */

#ifdef __cplusplus
extern "C" {
#endif

/* ==== enums (all 4-byte, static-asserted below) =========================== */

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
 * vocabulary (mutex/requires/exclude + conditional activation). A "(param,label)
 * ref is PRESENT in a trial" iff the param is active and — when the ref carries a
 * label — the param's chosen categorical label equals it. */
typedef enum n4m_constraint_kind_t {
    /* Only the all-present combination is forbidden (matches the nirs4all `_mutex_`
     * issubset rule); every proper subset is allowed. For pairwise mutual
     * exclusion (at most one) use N4M_CONSTRAINT_EXCLUDE. */
    N4M_CONSTRAINT_MUTEX_GROUP = 0,
    N4M_CONSTRAINT_REQUIRES = 1,      /* ref[0] present ⇒ ref[1] present           */
    N4M_CONSTRAINT_EXCLUDE = 2,       /* ref[0] and ref[1] may not both be present */
    /* param_refs = {child, parent} (exactly 2); child is active iff the parent's
     * chosen label == label_refs[1] AND the parent is itself active. Repeat the
     * constraint (same child + parent) to activate on a SET of parent labels.
     * Conditions NEST: a child may in turn be another param's parent, and a
     * grandchild is active only when its whole ancestor chain is — so operator
     * attributes vanish when an outer pipeline slot does not select that operator.
     * Other shapes (n_refs != 2, or a second constraint with a different parent
     * for the same child) are rejected with N4M_ERR_UNSUPPORTED. */
    N4M_CONSTRAINT_CONDITION_IN = 3,
    N4M_CONSTRAINT_CONDITION_NOT_IN = 4
} n4m_constraint_kind_t;

/* Unknown/out-of-range values cause n4m_optimizer_create to return
 * N4M_ERR_NOT_IMPLEMENTED. */
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
    N4M_TRIAL_PRUNED = 2, N4M_TRIAL_FAILED = 3,
    N4M_TRIAL_CANCELLED = 4
} n4m_trial_status_t;

/* Version discriminators for the in-memory rich trace and structured-error
 * text wire. These are source constants, not durable save/load formats. */
#define N4M_TRIAL_TRACE_FORMAT_VERSION 1
#define N4M_TRIAL_ERROR_WIRE_PREFIX "n4m.error.v1|"

/* ABI guard rails — every HPO enum is 4-byte. Kept in this header (not n4m.h)
 * so it stays independently includable. */
N4M_STATIC_ASSERT(sizeof(n4m_param_kind_t)      == 4, "n4m_param_kind_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_cat_type_t)        == 4, "n4m_cat_type_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_constraint_kind_t) == 4, "n4m_constraint_kind_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_sampler_kind_t)    == 4, "n4m_sampler_kind_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_pruner_kind_t)     == 4, "n4m_pruner_kind_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_opt_direction_t)   == 4, "n4m_opt_direction_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_eval_mode_t)       == 4, "n4m_eval_mode_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_metric_t)          == 4, "n4m_metric_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_liar_kind_t)       == 4, "n4m_liar_kind_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_trial_status_t)    == 4, "n4m_trial_status_t must be 4 bytes");

/* ==== forward-compatible options struct (D9) ============================= */

typedef struct n4m_optimizer_options_t {
    uint64_t            struct_size;      /* caller's struct size; current ABI requires >= sizeof */
    n4m_sampler_kind_t  sampler;
    n4m_pruner_kind_t   pruner;
    n4m_opt_direction_t direction;
    n4m_eval_mode_t     eval_mode;
    n4m_metric_t        metric;
    n4m_liar_kind_t     liar;
    int32_t             n_startup_trials; /* >=0; must be >0 for TPE, GP-EI, median */
    uint64_t            seed;
    double              timeout_seconds;  /* finite >=0; 0 = none; ask expiry => N4M_ERR_CANCELLED,
                                             without changing existing trial states */
    int32_t             max_resource;     /* hyperband: required >0; must be 0 for other pruners */
    int32_t             reduction_factor; /* asha/hyperband eta: 0 = default 3, otherwise >=2 */
    uint8_t             reserved[56];     /* forward-compat; must be zeroed */
} n4m_optimizer_options_t;

/* Fill struct_size + defaults (sampler=RANDOM, pruner=NONE, direction=AUTO,
 * eval_mode=MEAN, metric=RMSE, liar=NONE, n_startup_trials=10, seed=0). */
N4M_API void n4m_optimizer_options_init(n4m_optimizer_options_t* opts);

/* ==== search space ======================================================= */

typedef struct n4m_search_space_s n4m_search_space_t;

/* All text accepted by search-space builders must be valid UTF-8. Parameter
 * names are ordered identifiers. They must be non-empty and unique,
 * and must not contain '#', which is reserved for sorted-tuple components.
 * Builders copy their inputs. Final cross-axis validation (unique names,
 * constraints, condition labels/cycles and sampler compatibility) is performed
 * by n4m_optimizer_create / n4m_finetune_estimator. Integer domains,
 * N4M_CAT_INT choices and integer sorted-tuple bounds are limited to the
 * consecutive binary64-exact range [-2^53, 2^53]; values outside it return
 * N4M_ERR_INVALID_ARGUMENT. */
N4M_API n4m_status_t n4m_search_space_create(n4m_search_space_t** out);
N4M_API void         n4m_search_space_destroy(n4m_search_space_t* space);
N4M_API n4m_status_t n4m_search_space_add_int(n4m_search_space_t* space,
                        const char* name, int64_t low, int64_t high,
                        int64_t step, int32_t log); /* step>0; log requires step=1 */
N4M_API n4m_status_t n4m_search_space_add_float(n4m_search_space_t* space,
                        const char* name, double low, double high,
                        double step, int32_t log);   /* step=0 ⇒ continuous; log requires 0 */
N4M_API n4m_status_t n4m_search_space_add_categorical(n4m_search_space_t* space,
                        const char* name, n4m_cat_type_t type,
                        const void* values, int32_t n_values);
N4M_API n4m_status_t n4m_search_space_add_ordinal(n4m_search_space_t* space,
                        const char* name, const double* values, int32_t n_values);
N4M_API n4m_status_t n4m_search_space_add_sorted_tuple(n4m_search_space_t* space,
                        const char* name, int32_t length, double low, double high,
                        int32_t element_is_int);
/* refs are parallel arrays of param names + labels (label may be NULL where a
 * bare param reference suffices, e.g. mutex over presence). Labels must name an
 * existing categorical/ordinal choice. Conditions require exactly {child,
 * parent}; the child must already exist when the condition is added, while the
 * parent may be added later. The final activation graph must be acyclic.
 * Compatibility is uniform enforce-or-refuse. Hard mutex/requires/exclude
 * constraints are enforced by bounded rejection sampling for the random, LHS,
 * ternary and TPE samplers, and refused at create with N4M_ERR_UNSUPPORTED for
 * Sobol, GA, PSO, CMA-ES and GP-EI, whose generators cannot honour them without
 * a silent fitness penalty. Hard constraints also cannot reference sorted-tuple
 * axes for ANY sampler, because a tuple root has no trial-level presence
 * representation. Conditional activation (CONDITION_IN / _NOT_IN) is supported by
 * every sampler. */
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
/* Best-effort batch ask. `out_count` (required) receives the number of trials
 * committed and is set to 0 whenever possible. `n` must be >= 0; n < 0 returns
 * N4M_ERR_INVALID_ARGUMENT. For n == 0 nothing is dispensed (N4M_OK) and
 * `out_trials` may be NULL; for n > 0 `out_trials` must point at storage for n
 * pointers, all of which are set to NULL before the first ask. Every committed
 * slot [0, *out_count) is a borrowed RUNNING trial exactly as if produced by
 * n4m_optimizer_ask, in ask order; these are never rolled back and the caller
 * must eventually tell/cancel them. A benign stop after >= 1 commit returns
 * N4M_OK with 0 < *out_count < n: a population (GA/PSO/CMA-ES) generation
 * boundary, or a timeout. The same conditions hit at zero progress surface their
 * stable status (N4M_ERR_INVALID_ARGUMENT for a generation boundary at zero
 * capacity, N4M_ERR_CANCELLED for a timeout). Any other failure (unsatisfiable
 * constraints, an invalid queued warm-start, allocation/internal) returns its
 * exact non-OK status with [0, *out_count) still committed and the remaining
 * slots NULL. ask_batch(n) is exactly n sequential ask() calls with no
 * intervening tell(); the constant-liar policy stays NONE. For the same ordered
 * search space and seed, reproducible continuation requires the same ordered
 * ASK / INTERMEDIATE / TERMINAL event stream with the same payloads. This is not
 * a claim about arbitrary
 * wall-clock completion order in parallel hosts. */
N4M_API n4m_status_t n4m_optimizer_ask_batch(n4m_optimizer_t* opt, int32_t n,
                        n4m_trial_t** out_trials, int32_t* out_count);
N4M_API n4m_status_t n4m_optimizer_tell(n4m_optimizer_t* opt, int64_t trial_id, double score);
/* FAILED and CANCELLED persist an optimizer-owned structured error. A plain
 * UTF-8 `error` remains supported and is assigned a stable default code. Hosts
 * that need an explicit code/retry policy may pass this versioned wire form:
 *
 *   n4m.error.v1|UPPER_SNAKE_CODE|0-or-1|arbitrary UTF-8 message
 *
 * The first three separators are structural; `message` may contain `|`.
 * The `n4m.error.v` prefix namespace is reserved for versioned error records;
 * legacy plain messages must not begin with it.
 * Malformed versioned records and invalid UTF-8 fail closed with
 * N4M_ERR_INVALID_ARGUMENT.
 * COMPLETED/PRUNED reject a non-empty error. Terminal replay is idempotent only
 * when the status and its meaningful payload are identical: score for
 * COMPLETED, structured error for FAILED/CANCELLED; PRUNED has no payload. */
N4M_API n4m_status_t n4m_optimizer_tell_result(n4m_optimizer_t* opt, int64_t trial_id,
                        n4m_trial_status_t status, double score, const char* error);
/* Steps are non-negative and strictly increasing. An exact replay of the
 * latest (step,score) is idempotent and returns its original prune decision,
 * including when that decision already terminalized the trial as PRUNED. */
N4M_API n4m_status_t n4m_optimizer_tell_intermediate(n4m_optimizer_t* opt, int64_t trial_id,
                        int32_t step, double score, int32_t* out_should_prune);
N4M_API n4m_status_t n4m_optimizer_best(const n4m_optimizer_t* opt,
                        n4m_trial_t** out_best, double* out_score);
/* Return an owning, immutable snapshot for trials whose id is >= since_id;
 * since_id must be non-negative.
 * Existing ABI 2.1 fields remain unchanged:
 *   trial_ids/scores/status/rung/duration (1 x n double matrices), n_trials.
 * Rich trace v1 adds exact `trial_ids_i64`; row-major parameter values,
 * kinds/category types/integer metadata, category indices, activation bits and
 * UTF-8 name/label byte pools; flattened intermediate step/score/prune streams;
 * global ask/intermediate/terminal event sequences; and structured error
 * code/message/retryable streams.
 * Variable-length records use int64 offset vectors. All
 * strings are UTF-8 bytes stored as int32 vectors (0..255). See
 * docs/methods/optimization.md for the complete key/shape contract. The result
 * and all of its buffers remain valid after `opt` is destroyed and are released
 * with n4m_method_result_destroy. */
N4M_API n4m_status_t n4m_optimizer_get_trials(const n4m_optimizer_t* opt,
                        int64_t since_id, n4m_method_result_t** out);

/* Portable N4MOPT checkpoint persistence (format v1). Save returns an owning
 * 1 x n_words N4M_DTYPE_I64 array. Its backing bytes (obtained from
 * n4m_array_view) are the checkpoint and its byte length is n_words * 8; free
 * it with n4m_array_free. Load copies/owns every decoded object and is
 * transactional: `*out` remains NULL on every failure. The decoder rejects bad
 * magic/checksum/padding, truncation/trailing bytes, future format versions,
 * over-limit sizes/counts and inconsistent option/search-space/state payloads.
 * No host pointer, object representation, or steady-clock epoch is serialized. */
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
 * — those are dag-ml's). `estimator` selects a generic native regression route
 * from a closed internal registry: N4M_ALGO_PLS_REGRESSION, PLS_CANONICAL,
 * PLS_SVD, OPLS, SPARSE_PLS and PCR. Every other algorithm — the classification
 * chassis (PLS_DA / OPLS_DA), the non-generic MB_PLS / LW_PLS / AOM_PLS, and any
 * invalid enum — is rejected with N4M_ERR_UNSUPPORTED before a study is created.
 * The solver/deflation recipe is chosen by the core fit_model authority, not by
 * this driver.
 *
 * The search space declares the tuned axes. The five dense routes require a
 * single INT (non-log) `n_components` axis with low >= 1, high <= INT32_MAX and
 * step >= 1. SPARSE_PLS accepts any non-empty subset of {`n_components`,
 * `sparsity_lambda`}: `sparsity_lambda` is a FLOAT axis with 0 <= low <= high < 1
 * (LOG_FLOAT permitted only when low > 0 and step = 0); an omitted axis uses its
 * documented Config default (`n_components` = 2, `sparsity_lambda` = 0.0).
 * Unknown, duplicate, wrong-kind, out-of-domain or bad-step axes are stable
 * N4M_ERR_UNSUPPORTED preflight errors. Conditional axes are also refused: all
 * registered axes are numeric, while conditions require a categorical/ordinal
 * parent. Only pruner = N4M_PRUNER_NONE and the regression metrics
 * (RMSE / MSE / MAE / R2) are supported. X/Y and the complete validation plan
 * are checked before the first ASK, including finite values and
 * ABI-representable column counts. X and Y may independently use F32 or F64;
 * global input errors never become failed hyperparameter trials.
 *
 * This is selection-only convenience: it does not refit a final model on all
 * rows. A timeout before any completed trial returns N4M_ERR_CANCELLED; after at
 * least one completion it returns N4M_OK with the partial best/trace and
 * `timed_out` = 1. The result carries active `best.<param>` values (including
 * `best.sparsity_lambda` when sampled), `best_score`, `metric`, `estimator`,
 * `timed_out`, `requested_trials`, and the full owning trial trace. */
N4M_API n4m_status_t n4m_finetune_estimator(n4m_context_t* ctx, n4m_algorithm_t estimator,
                        const n4m_matrix_view_t* X, const n4m_matrix_view_t* Y,
                        const n4m_validation_plan_t* plan, const n4m_search_space_t* space,
                        const n4m_optimizer_options_t* opts, int32_t n_trials,
                        n4m_method_result_t** out_result);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif /* N4M_OPTIMIZATION_H */
