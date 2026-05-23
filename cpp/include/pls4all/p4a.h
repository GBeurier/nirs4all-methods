/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * pls4all — public C ABI v1.1.0.
 *
 * Stability: experimental until v1.0.0. Every breaking change before that
 * version bumps the ABI MAJOR (see n4m_version.h). After v1.0.0 the ABI
 * follows strict semver: breaking changes bump MAJOR, additive changes bump
 * MINOR, bugfixes bump PATCH.
 *
 * This is the ONLY header consumers of libn4m are expected to include.
 *
 * Universal rules of the surface:
 *   - Every exported function is `noexcept` — no C++ exception ever crosses
 *     this boundary. Status-returning wrappers translate exceptions via
 *     n4m_status_t; void destroy/free wrappers catch and swallow after
 *     best-effort cleanup.
 *   - No STL types, no Eigen types, no C++ classes appear in this surface.
 *   - All multi-byte serialized integers are little-endian.
 *   - Floating-point quantities are IEEE 754 binary64 (`double`).
 *   - String pointers returned by const-`char*` getters point into
 *     library-owned storage; callers must NOT free them.
 *   - Error messages for failed calls live in a per-context 4096-byte buffer
 *     (N4M_ERROR_BUFFER_BYTES) that is invalidated by the next call on the
 *     same context — bindings must copy if they retain the value.
 *   - Memory ownership is symmetric: callers free what they allocated, the
 *     core frees what it allocated. The only core-allocated outputs are
 *     `n4m_array_t*` and the opaque handles below; both have explicit
 *     `n4m_*_destroy` / `n4m_array_free` functions.
 *
 * Current implementation status (rev 1.1.0 of this header — May 2026):
 *   - Lifecycle / config / version / matrix-view are fully implemented.
 *   - Pipeline / operator-bank / gating-strategy / validation-plan
 *     lifecycle is implemented; AOM global and POP per-component
 *     selection are exposed by `n4m_aom_global_select` and
 *     `n4m_aom_per_component_select` (see §15).
 *     Pipeline `_fit` / `_transform` are live for N4M_OP_IDENTITY,
 *     N4M_OP_CENTER, N4M_OP_AUTOSCALE, N4M_OP_PARETO_SCALE, N4M_OP_SNV,
 *     N4M_OP_MSC, N4M_OP_EMSC, N4M_OP_DETREND_POLY, N4M_OP_SAVGOL_SMOOTH,
 *     N4M_OP_SAVGOL_DERIVATIVE, N4M_OP_NORRIS_WILLIAMS,
 *     N4M_OP_ASLS_BASELINE, N4M_OP_WAVELET_DENOISE, N4M_OP_OSC and
 *     N4M_OP_EPO.
 *     N4M_OP_FINITE_DIFFERENCE, N4M_OP_WHITTAKER and N4M_OP_FCK are
 *     currently implemented by the internal strict-linear AOM operator
 *     kernels.
 *   - n4m_model_fit implements dependency-free NIPALS, orthogonal-scores,
 *     SIMPLS, kernel, wide-kernel, SVD, power-iteration and randomized-SVD PLS
 *     regression (PLS1 / PLS2) for N4M_ALGO_PLS_REGRESSION +
 *     N4M_SOLVER_NIPALS, N4M_SOLVER_ORTHOGONAL_SCORES,
 *     N4M_SOLVER_SIMPLS, N4M_SOLVER_KERNEL_ALGORITHM,
 *     N4M_SOLVER_WIDE_KERNEL, N4M_SOLVER_SVD, N4M_SOLVER_POWER or
 *     N4M_SOLVER_RANDOMIZED_SVD + N4M_DEFLATION_REGRESSION; PLS canonical
 *     for N4M_ALGO_PLS_CANONICAL + N4M_SOLVER_NIPALS/N4M_SOLVER_SVD +
 *     N4M_DEFLATION_CANONICAL; direct cross-covariance PLSSVD scores for
 *     N4M_ALGO_PLS_SVD + N4M_SOLVER_SVD + N4M_DEFLATION_CANONICAL; PLS-DA
 *     dummy-response scores for N4M_ALGO_PLS_DA + the PLS regression solver set +
 *     N4M_DEFLATION_REGRESSION; OPLS / OPLS-DA common predictive scores
 *     for N4M_ALGO_OPLS or N4M_ALGO_OPLS_DA + N4M_SOLVER_NIPALS +
 *     N4M_DEFLATION_ORTHOGONAL; plus PCR for N4M_ALGO_PCR + N4M_SOLVER_SVD +
 *     N4M_DEFLATION_REGRESSION.
 *   - Predict / transform / model-array accessors and binary serialization
 *     are implemented for fitted core PLS models.
 *
 * Status-code contracts (apply uniformly unless overridden in a function's
 * own doc comment):
 *
 *   - `_create(out)` functions: return N4M_OK on success, N4M_ERR_NULL_POINTER
 *     when `out` is NULL, N4M_ERR_OUT_OF_MEMORY on allocation failure.
 *   - `_destroy(handle)` functions: void, no-op on NULL.
 *   - `_clone(src, out_dst)`: N4M_OK, N4M_ERR_NULL_POINTER, N4M_ERR_OUT_OF_MEMORY.
 *   - `_set_*(handle, value)` setters: N4M_OK, N4M_ERR_NULL_POINTER (handle
 *     is NULL), N4M_ERR_INVALID_ARGUMENT (value rejected). Failed setters
 *     leave the handle's state UNCHANGED — there is no partial mutation.
 *   - `_get_*(handle, out)` getters: N4M_OK, N4M_ERR_NULL_POINTER (handle or
 *     `out` is NULL).
 *   - Any function that touches `n4m_context_t*` may additionally return
 *     N4M_ERR_OUT_OF_MEMORY (from internal allocations) and N4M_ERR_INTERNAL
 *     (from an unexpected exception caught at the C boundary). Both record
 *     a message in `n4m_context_last_error`.
 *   - Functions that consume `n4m_matrix_view_t` may additionally return
 *     N4M_ERR_SHAPE_MISMATCH, N4M_ERR_DTYPE_MISMATCH, N4M_ERR_STRIDE_INVALID
 *     when the view fails n4m_matrix_view_validate.
 */
#ifndef PLS4ALL_P4A_H
#define PLS4ALL_P4A_H

#include <stddef.h>
#include <stdint.h>

#include "pls4all/p4a_export.h"
#include "pls4all/p4a_version.h"
#include "n4m/n4m.h"  /* Common-core: status/backend/dtype/matrix_view/context/version */

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 5. Opaque handles
 * ========================================================================== */

typedef struct n4m_config_s                     n4m_config_t;
typedef struct n4m_operator_bank_s              n4m_operator_bank_t;
typedef struct n4m_gating_strategy_s            n4m_gating_strategy_t;
typedef struct n4m_pipeline_s                   n4m_pipeline_t;
typedef struct n4m_model_s                      n4m_model_t;
typedef struct n4m_validation_plan_s            n4m_validation_plan_t;
typedef struct n4m_aom_global_result_s          n4m_aom_global_result_t;
typedef struct n4m_aom_per_component_result_s   n4m_aom_per_component_result_t;
typedef struct n4m_method_result_s              n4m_method_result_t;
typedef struct n4m_array_s                      n4m_array_t;

/* ============================================================================
 * 7. Algorithm / solver / deflation enums
 * ========================================================================== */

typedef enum n4m_algorithm_t {
    N4M_ALGO_PLS_REGRESSION = 0,
    N4M_ALGO_PLS_CANONICAL  = 1,
    N4M_ALGO_PLS_SVD        = 2,
    N4M_ALGO_PLS_DA         = 3,
    N4M_ALGO_OPLS           = 4,
    N4M_ALGO_OPLS_DA        = 5,
    N4M_ALGO_SPARSE_PLS     = 6,
    N4M_ALGO_MB_PLS         = 7,
    N4M_ALGO_LW_PLS         = 8,
    N4M_ALGO_AOM_PLS        = 9,
    N4M_ALGO_PCR            = 10
    /* Nonlinear kernel PLS (RBF / polynomial / sigmoid) is intentionally
     * deferred to Phase 4 along with the kernel-type enum + setters; we do
     * not pre-declare it here so we don't lock in an unfinished surface.
     * Orthogonal-scores, SIMPLS, kernel-algorithm, wide-kernel, power and
     * randomized-SVD are SOLVERS (see below), not algorithms. POP-PLS is a
     * GATING STRATEGY (see below). */
} n4m_algorithm_t;

typedef enum n4m_solver_t {
    N4M_SOLVER_NIPALS            = 0,
    N4M_SOLVER_SIMPLS            = 1,
    N4M_SOLVER_ORTHOGONAL_SCORES = 2,
    N4M_SOLVER_KERNEL_ALGORITHM  = 3,   /* linear kernel-algorithm PLS (Lindgren/Wold) */
    N4M_SOLVER_WIDE_KERNEL       = 4,
    N4M_SOLVER_SVD               = 5,
    N4M_SOLVER_POWER             = 6,
    N4M_SOLVER_RANDOMIZED_SVD    = 7
} n4m_solver_t;

typedef enum n4m_deflation_t {
    N4M_DEFLATION_REGRESSION  = 0,
    N4M_DEFLATION_CANONICAL   = 1,
    N4M_DEFLATION_X_ONLY      = 2,
    N4M_DEFLATION_XY          = 3,
    N4M_DEFLATION_ORTHOGONAL  = 4
} n4m_deflation_t;

/* ============================================================================
 * 8. Config lifecycle and setters
 * ==========================================================================
 *
 * A n4m_config_t is an opaque bag of knobs. Bindings build one with the
 * setters below, hand it to n4m_model_fit, and either destroy it or reuse it.
 *
 * Setter contract:
 *   - Every setter validates its inputs.
 *   - On rejection (N4M_ERR_INVALID_ARGUMENT or _NULL_POINTER), the config
 *     is left UNCHANGED — failed setters never partially mutate state.
 *   - Every setter has a corresponding getter that returns the most recent
 *     accepted value.
 */
N4M_API n4m_status_t n4m_config_create(n4m_config_t** out_cfg);
N4M_API void         n4m_config_destroy(n4m_config_t* cfg);
N4M_API n4m_status_t n4m_config_clone(const n4m_config_t* src,
                                       n4m_config_t** out_dst);

/* ---- Setters ---- */
N4M_API n4m_status_t n4m_config_set_algorithm        (n4m_config_t*, n4m_algorithm_t);
N4M_API n4m_status_t n4m_config_set_solver           (n4m_config_t*, n4m_solver_t);
N4M_API n4m_status_t n4m_config_set_deflation        (n4m_config_t*, n4m_deflation_t);
/* `n_components` must be >= 1. There is no upper cap at the ABI level; if
 * the value exceeds rank(X) at fit time, Phase 1+ returns
 * N4M_ERR_INVALID_ARGUMENT from n4m_model_fit with a descriptive message. */
N4M_API n4m_status_t n4m_config_set_n_components     (n4m_config_t*, int32_t n);
N4M_API n4m_status_t n4m_config_set_center_x         (n4m_config_t*, int32_t enabled);
N4M_API n4m_status_t n4m_config_set_scale_x          (n4m_config_t*, int32_t enabled);
N4M_API n4m_status_t n4m_config_set_center_y         (n4m_config_t*, int32_t enabled);
N4M_API n4m_status_t n4m_config_set_scale_y          (n4m_config_t*, int32_t enabled);
N4M_API n4m_status_t n4m_config_set_tol              (n4m_config_t*, double tol);
/* `max_iter` must be >= 1. There is no upper cap at the ABI level. */
N4M_API n4m_status_t n4m_config_set_max_iter         (n4m_config_t*, int32_t max_iter);
N4M_API n4m_status_t n4m_config_set_store_scores     (n4m_config_t*, int32_t enabled);
N4M_API n4m_status_t n4m_config_set_store_diagnostics(n4m_config_t*, int32_t enabled);
/* Switch `n4m_sparse_simpls_fit` between the default Chun & Keles 2010
 * spls algorithm (enabled=0; matches R `spls::spls`) and the legacy
 * per-component absolute soft-threshold of the SIMPLS direction
 * (enabled=1; behaviour of pls4all <= 0.97.3). */
N4M_API n4m_status_t n4m_config_set_sparse_simpls_legacy(n4m_config_t*,
                                                          int32_t enabled);
N4M_API n4m_status_t n4m_config_get_sparse_simpls_legacy(const n4m_config_t*,
                                                          int32_t* out_enabled);
/* Switch `n4m_robust_pls_fit` between Partial Robust M-regression
 * (enabled=0; default; matches R `chemometrics::prm` bit-for-bit) and the
 * legacy Huber-IRLS over weighted SIMPLS (enabled=1; behaviour of pls4all
 * <= 0.97.3). */
N4M_API n4m_status_t n4m_config_set_robust_pls_legacy(n4m_config_t*,
                                                       int32_t enabled);
N4M_API n4m_status_t n4m_config_get_robust_pls_legacy(const n4m_config_t*,
                                                       int32_t* out_enabled);
/* Switch `n4m_approximate_press_compute` between true leave-one-out PRESS
 * (enabled=0; default; matches R `pls::plsr(validation='LOO', method='simpls',
 * scale=FALSE)` bit-for-bit) and the legacy Eastment-Krzanowski leverage-
 * inflated training-residual approximation (enabled=1; behaviour of pls4all
 * <= 0.97.3). */
N4M_API n4m_status_t n4m_config_set_approximate_press_legacy(n4m_config_t*,
                                                              int32_t enabled);
N4M_API n4m_status_t n4m_config_get_approximate_press_legacy(const n4m_config_t*,
                                                              int32_t* out_enabled);
N4M_API n4m_status_t n4m_config_set_dtype            (n4m_config_t*, n4m_dtype_t);

/* Composability hooks — these are non-owning pointers; the lifetime of the
 * pipeline / bank / strategy is managed by the caller. Passing NULL clears
 * the previously-set hook. */
N4M_API n4m_status_t n4m_config_set_pipeline         (n4m_config_t*,
                                                       const n4m_pipeline_t* pipe);
N4M_API n4m_status_t n4m_config_set_operator_bank    (n4m_config_t*,
                                                       const n4m_operator_bank_t* bank);
N4M_API n4m_status_t n4m_config_set_gating_strategy  (n4m_config_t*,
                                                       const n4m_gating_strategy_t* gate);

/* ---- Getters ---- */
N4M_API n4m_status_t n4m_config_get_algorithm        (const n4m_config_t*, n4m_algorithm_t*);
N4M_API n4m_status_t n4m_config_get_solver           (const n4m_config_t*, n4m_solver_t*);
N4M_API n4m_status_t n4m_config_get_deflation        (const n4m_config_t*, n4m_deflation_t*);
N4M_API n4m_status_t n4m_config_get_n_components     (const n4m_config_t*, int32_t*);
N4M_API n4m_status_t n4m_config_get_center_x         (const n4m_config_t*, int32_t*);
N4M_API n4m_status_t n4m_config_get_scale_x          (const n4m_config_t*, int32_t*);
N4M_API n4m_status_t n4m_config_get_center_y         (const n4m_config_t*, int32_t*);
N4M_API n4m_status_t n4m_config_get_scale_y          (const n4m_config_t*, int32_t*);
N4M_API n4m_status_t n4m_config_get_tol              (const n4m_config_t*, double*);
N4M_API n4m_status_t n4m_config_get_max_iter         (const n4m_config_t*, int32_t*);
N4M_API n4m_status_t n4m_config_get_store_scores     (const n4m_config_t*, int32_t*);
N4M_API n4m_status_t n4m_config_get_store_diagnostics(const n4m_config_t*, int32_t*);
N4M_API n4m_status_t n4m_config_get_dtype            (const n4m_config_t*, n4m_dtype_t*);
N4M_API n4m_status_t n4m_config_get_pipeline         (const n4m_config_t*,
                                                       const n4m_pipeline_t** out);
N4M_API n4m_status_t n4m_config_get_operator_bank    (const n4m_config_t*,
                                                       const n4m_operator_bank_t** out);
N4M_API n4m_status_t n4m_config_get_gating_strategy  (const n4m_config_t*,
                                                       const n4m_gating_strategy_t** out);

/* ============================================================================
 * 9. Operator bank, gating strategy, preprocessing pipeline
 * ========================================================================== */

typedef enum n4m_operator_kind_t {
    N4M_OP_IDENTITY            = 0,
    N4M_OP_CENTER              = 1,
    N4M_OP_AUTOSCALE           = 2,
    N4M_OP_PARETO_SCALE        = 3,
    N4M_OP_SNV                 = 4,
    N4M_OP_MSC                 = 5,
    N4M_OP_EMSC                = 6,
    N4M_OP_DETREND_POLY        = 7,
    N4M_OP_SAVGOL_SMOOTH       = 8,
    N4M_OP_SAVGOL_DERIVATIVE   = 9,
    N4M_OP_NORRIS_WILLIAMS     = 10,
    N4M_OP_ASLS_BASELINE       = 11,
    N4M_OP_OSC                 = 12,
    N4M_OP_EPO                 = 13,
    N4M_OP_WAVELET_DENOISE     = 14,
    N4M_OP_FINITE_DIFFERENCE   = 15,
    N4M_OP_WHITTAKER           = 16,
    N4M_OP_FCK                 = 17
} n4m_operator_kind_t;

/* Operator-bank lifecycle. An operator bank is an unordered collection of
 * preprocessing operators used by AOM-PLS (Phase 6) — the gating strategy
 * picks (soft / hard / sparse) which one(s) to apply per component.
 *
 * Memory: `params` (when non-NULL) is COPIED into the bank by `_add`. The
 * caller's buffer may be released immediately after the call returns. */
N4M_API n4m_status_t n4m_operator_bank_create(n4m_operator_bank_t** out);
N4M_API void         n4m_operator_bank_destroy(n4m_operator_bank_t* bank);
N4M_API n4m_status_t n4m_operator_bank_add   (n4m_operator_bank_t* bank,
                                               n4m_operator_kind_t kind,
                                               const double* params,
                                               int32_t n_params);
N4M_API n4m_status_t n4m_operator_bank_size  (const n4m_operator_bank_t* bank,
                                               int32_t* out_size);

/* ---- Gating strategy ---- */
typedef enum n4m_gating_mode_t {
    N4M_GATING_HARD            = 0,
    N4M_GATING_SOFT            = 1,
    N4M_GATING_SPARSE          = 2,
    N4M_GATING_PER_COMPONENT   = 3,    /* POP-PLS */
    N4M_GATING_PER_BLOCK       = 4,
    N4M_GATING_PER_TARGET      = 5
} n4m_gating_mode_t;

N4M_API n4m_status_t n4m_gating_strategy_create(n4m_gating_strategy_t** out,
                                                 n4m_gating_mode_t mode);
N4M_API void         n4m_gating_strategy_destroy(n4m_gating_strategy_t* gs);

/* ---- Preprocessing pipeline ----
 * Pipeline = ordered list of operators that share fit/transform.
 * Phase 3a/3b/3c/3d/3e/3f/3g/3h/3i/3j implements identity, center, autoscale,
 * Pareto scaling, SNV, MSC, EMSC, polynomial detrending, Savitzky-Golay
 * smoothing/derivatives, Norris-Williams gap-segment derivatives, ASLS
 * baseline correction, Haar wavelet denoising and supervised one-component
 * OSC / EPO with leakage-safe training statistics.
 * Unsupported operators are accepted by add_operator so pipelines remain
 * serialisable later, but fit returns N4M_ERR_NOT_IMPLEMENTED until the
 * operator body lands.
 */
N4M_API n4m_status_t n4m_pipeline_create        (n4m_pipeline_t** out);
N4M_API void         n4m_pipeline_destroy       (n4m_pipeline_t* pipe);
N4M_API n4m_status_t n4m_pipeline_clone         (const n4m_pipeline_t* src,
                                                  n4m_pipeline_t** out_dst);
/* `params` (when non-NULL) is COPIED into the pipeline by `_add_operator`;
 * the caller may release the buffer immediately afterwards. */
N4M_API n4m_status_t n4m_pipeline_add_operator  (n4m_pipeline_t* pipe,
                                                  n4m_operator_kind_t kind,
                                                  const double* params,
                                                  int32_t n_params);
N4M_API n4m_status_t n4m_pipeline_size          (const n4m_pipeline_t* pipe,
                                                  int32_t* out_size);

/* `Y` is optional for unsupervised preprocessing operators and may be NULL.
 * Supervised operators such as OSC and EPO require a non-NULL Y at fit time. */
N4M_API n4m_status_t n4m_pipeline_fit            (n4m_context_t* ctx,
                                                   n4m_pipeline_t* pipe,
                                                   const n4m_matrix_view_t* X,
                                                   const n4m_matrix_view_t* Y);
N4M_API n4m_status_t n4m_pipeline_transform      (n4m_context_t* ctx,
                                                   const n4m_pipeline_t* pipe,
                                                   const n4m_matrix_view_t* X,
                                                   n4m_matrix_view_t* out);
N4M_API n4m_status_t n4m_pipeline_transform_alloc(n4m_context_t* ctx,
                                                   const n4m_pipeline_t* pipe,
                                                   const n4m_matrix_view_t* X,
                                                   n4m_array_t** out);

/* ============================================================================
 * 10. Model lifecycle
 * ==========================================================================
 *
 * The current core implements N4M_ALGO_PLS_REGRESSION with N4M_SOLVER_NIPALS,
 * N4M_SOLVER_ORTHOGONAL_SCORES, N4M_SOLVER_SIMPLS,
 * N4M_SOLVER_KERNEL_ALGORITHM, N4M_SOLVER_WIDE_KERNEL, N4M_SOLVER_SVD,
 * N4M_SOLVER_POWER or N4M_SOLVER_RANDOMIZED_SVD using
 * N4M_DEFLATION_REGRESSION; N4M_ALGO_PLS_CANONICAL with N4M_SOLVER_NIPALS or
 * N4M_SOLVER_SVD using N4M_DEFLATION_CANONICAL; N4M_ALGO_PLS_SVD with
 * N4M_SOLVER_SVD using N4M_DEFLATION_CANONICAL; N4M_ALGO_PLS_DA with the
 * PLS regression solver set using N4M_DEFLATION_REGRESSION; N4M_ALGO_OPLS
 * and N4M_ALGO_OPLS_DA with one or more Y targets, N4M_SOLVER_NIPALS and
 * N4M_DEFLATION_ORTHOGONAL; and N4M_ALGO_PCR with N4M_SOLVER_SVD using
 * N4M_DEFLATION_REGRESSION. Multi-response OPLS uses one shared predictive
 * score direction from the dominant singular pair of X.T @ Y. PLSSVD
 * transform returns direct X scores from SVD(X.T @ Y); predict uses the
 * deterministic latent projection X @ W @ V.T scaled back to Y.
 * Unsupported algorithms / solvers / deflation modes return
 * N4M_ERR_UNSUPPORTED. Phase 6 adds AOM-PLS through the operator_bank +
 * gating_strategy hooks set on the config.
 */
N4M_API n4m_status_t n4m_model_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    n4m_model_t** out_model);

N4M_API void         n4m_model_destroy(n4m_model_t* model);

/* ---- Predict / transform (caller-provided output buffer) ---- */
N4M_API n4m_status_t n4m_model_predict(
    n4m_context_t* ctx, const n4m_model_t* model,
    const n4m_matrix_view_t* X, n4m_matrix_view_t* out);

N4M_API n4m_status_t n4m_model_transform(
    n4m_context_t* ctx, const n4m_model_t* model,
    const n4m_matrix_view_t* X, n4m_matrix_view_t* out_scores);

/* ---- Predict / transform (core-allocated output) ---- */
N4M_API n4m_status_t n4m_model_predict_alloc(
    n4m_context_t* ctx, const n4m_model_t* model,
    const n4m_matrix_view_t* X, n4m_array_t** out);

N4M_API n4m_status_t n4m_model_transform_alloc(
    n4m_context_t* ctx, const n4m_model_t* model,
    const n4m_matrix_view_t* X, n4m_array_t** out_scores);

/* Tagged accessor for fitted-model arrays. */
typedef enum n4m_model_array_t {
    N4M_MODEL_COEFFICIENTS = 0,   /* (n_features, n_targets) */
    N4M_MODEL_INTERCEPT    = 1,   /* (n_targets,) */
    N4M_MODEL_X_MEAN       = 2,
    N4M_MODEL_X_SCALE      = 3,
    N4M_MODEL_Y_MEAN       = 4,
    N4M_MODEL_Y_SCALE      = 5,
    N4M_MODEL_WEIGHTS_W    = 6,   /* (n_features, n_components) */
    N4M_MODEL_LOADINGS_P   = 7,
    N4M_MODEL_Y_LOADINGS_Q = 8,
    N4M_MODEL_ROTATIONS_R  = 9,
    N4M_MODEL_SCORES_T     = 10,  /* optional — store_scores=1 */
    N4M_MODEL_Y_SCORES_U   = 11   /* optional — store_scores=1 */
} n4m_model_array_t;

N4M_API n4m_status_t n4m_model_get_array(
    n4m_context_t* ctx, const n4m_model_t* model,
    n4m_model_array_t which, n4m_array_t** out);

N4M_API n4m_status_t n4m_model_get_n_components(const n4m_model_t*, int32_t* out);
N4M_API n4m_status_t n4m_model_get_n_features  (const n4m_model_t*, int32_t* out);
N4M_API n4m_status_t n4m_model_get_n_targets   (const n4m_model_t*, int32_t* out);

/* ============================================================================
 * 11. Serialization
 * ==========================================================================
 *
 * The binary format starts with the four-byte magic "N4MM" (the first 4
 * bytes of N4M_SERIALIZATION_MAGIC — note that as a string literal that
 * macro is 5 bytes including the trailing NUL; consumers should compare or
 * memcpy exactly 4 bytes), followed by N4M_SERIALIZATION_FORMAT_VERSION
 * (4-byte LE) and the producing library's ABI version triple (3 × 4-byte
 * LE). The consumer rejects on N4M_SERIALIZATION_FORMAT_VERSION mismatch
 * with N4M_ERR_VERSION_INCOMPATIBLE and writes a warning to last_error if
 * the ABI version differs but the format is compatible. All multi-byte
 * integers are little-endian; all doubles are IEEE 754 binary64 LE.
 */

N4M_API n4m_status_t n4m_model_export_size(
    const n4m_model_t* model, size_t* out_size);

N4M_API n4m_status_t n4m_model_export_to_buffer(
    const n4m_model_t* model, void* buffer, size_t buffer_size,
    size_t* out_written);

N4M_API n4m_status_t n4m_model_import_from_buffer(
    n4m_context_t* ctx, const void* buffer, size_t buffer_size,
    n4m_model_t** out_model);

/* Inspect a buffer's header without fully importing it. Useful for ABI /
 * format compatibility checks before allocating model memory. */
N4M_API n4m_status_t n4m_serialization_inspect(
    const void* buffer, size_t buffer_size,
    uint32_t* out_format_version,
    uint32_t* out_writer_abi_major,
    uint32_t* out_writer_abi_minor,
    uint32_t* out_writer_abi_patch);

/* ============================================================================
 * 13. Output array helper — core-allocated arrays returned by *_alloc calls
 * ==========================================================================
 *
 * n4m_array_t is a non-opaque concept (you read its dtype/shape and obtain a
 * view), but the only ways to construct one are through library functions
 * such as `n4m_model_predict_alloc` and `n4m_pipeline_transform_alloc`.
 * Callers must release it with n4m_array_free. AOM/POP selection (§15)
 * exposes its arrays through result-handle accessors that point into
 * result-owned storage instead.
 */
N4M_API n4m_status_t n4m_array_dtype(const n4m_array_t* arr, n4m_dtype_t* out);
N4M_API n4m_status_t n4m_array_shape(const n4m_array_t* arr,
                                      int64_t* rows, int64_t* cols);
N4M_API n4m_status_t n4m_array_view (const n4m_array_t* arr,
                                      n4m_matrix_view_t* out);
N4M_API void         n4m_array_free (n4m_array_t* arr);

/* ============================================================================
 * 14. Validation plan — caller-built train/test fold list for selection
 * ==========================================================================
 *
 * A n4m_validation_plan_t carries the per-fold train and test sample indices
 * used by selection / cross-validation kernels. It is built by the caller via
 * `_set_n_samples` and one `_add_fold` per fold; index buffers are COPIED
 * into the plan (the caller may release their buffers immediately after each
 * call). Failed setters and failed `_add_fold` calls leave the plan
 * UNCHANGED — there is no partial mutation.
 *
 * Lifetime: the plan is heap-allocated by `_create` and must be released with
 * `_destroy`. A const plan pointer may be reused across multiple selection
 * calls.
 */
N4M_API n4m_status_t n4m_validation_plan_create(n4m_validation_plan_t** out);
N4M_API void         n4m_validation_plan_destroy(n4m_validation_plan_t* plan);

/* Sets the sample count the plan refers to. `n_samples` must be >= 0; a
 * non-positive value is allowed at construction (treated as "unset") so
 * bindings may set folds before learning the sample count, but selection
 * callers must set a value of at least 2 before invoking selection. Returns
 * N4M_ERR_NULL_POINTER if `plan` is NULL, N4M_ERR_INVALID_ARGUMENT if
 * `n_samples < 0`. */
N4M_API n4m_status_t n4m_validation_plan_set_n_samples(
    n4m_validation_plan_t* plan, int64_t n_samples);

/* Appends one fold. `n_train` and `n_test` must both be >= 1. Index buffers
 * must be non-NULL. Indices are not validated here against `n_samples`;
 * semantic checks (range, duplicates, train/test overlap) are run when a
 * selection kernel consumes the plan. The plan is left UNCHANGED on failure.
 * Returns N4M_ERR_NULL_POINTER for NULL plan or NULL index buffer, and
 * N4M_ERR_INVALID_ARGUMENT when n_train or n_test is zero or negative. */
N4M_API n4m_status_t n4m_validation_plan_add_fold(
    n4m_validation_plan_t* plan,
    const int64_t* train_indices, int64_t n_train,
    const int64_t* test_indices,  int64_t n_test);

N4M_API n4m_status_t n4m_validation_plan_get_n_samples(
    const n4m_validation_plan_t* plan, int64_t* out_n_samples);
N4M_API n4m_status_t n4m_validation_plan_get_n_folds(
    const n4m_validation_plan_t* plan, int32_t* out_n_folds);

/* ============================================================================
 * 15. AOM / POP selection -- public C ABI for the internal AOM-SIMPLS selector
 * ==========================================================================
 *
 * AOM-PLS picks one operator (global) or one operator per latent component
 * (POP) from a n4m_operator_bank_t, scored under a caller-provided
 * n4m_validation_plan_t. Phase 6 ships strict-linear AOM operators with the
 * SIMPLS solver in N4M_DEFLATION_REGRESSION; other algorithm/solver/
 * deflation combinations return N4M_ERR_UNSUPPORTED. Both kernels require
 * a non-empty bank of strict-linear operators (see N4M_OP_IDENTITY,
 * N4M_OP_DETREND_POLY, N4M_OP_SAVGOL_SMOOTH, N4M_OP_SAVGOL_DERIVATIVE,
 * N4M_OP_NORRIS_WILLIAMS, N4M_OP_FINITE_DIFFERENCE, N4M_OP_WHITTAKER,
 * N4M_OP_FCK). Non-strict operators (e.g. N4M_OP_SNV) return
 * N4M_ERR_UNSUPPORTED.
 *
 * On entry to *_select, *out_result is set to NULL. On success the out
 * pointer is a newly heap-allocated result handle that the caller MUST
 * release with the matching *_destroy. All getter buffer pointers point
 * into the result's owned storage and are invalidated when the result is
 * destroyed; bindings must copy if they need to retain the values past the
 * handle lifetime.
 *
 * Layout conventions (all row-major, contiguous, IEEE 754 binary64):
 *   - rmse_curves[op*max_components + k]: prefix-RMSE of operator `op`
 *     using the first (k+1) latent components.
 *   - component_scores[k*n_operators + op] (POP): per-component CV score
 *     of operator `op` at latent component k.
 *   - prefix_scores[k] (POP): cumulative best score using k+1 components.
 *   - operator_scores[op]: best prefix score for operator `op`.
 *   - operator_kinds[op]: bank-order operator kind of operator `op`.
 *   - selected_operator_indices[k] (POP, int32): bank index of the
 *     operator picked at component k for the selected prefix.
 *   - predictions: full-fit predictions, shape (X.rows, Y.cols).
 *
 * Errors: the universal status-code contract in the header prelude
 * applies. Additional cases:
 *   - Empty operator bank or empty plan -> N4M_ERR_INVALID_ARGUMENT.
 *   - Plan sample count not matching X.rows -> N4M_ERR_SHAPE_MISMATCH.
 *   - Duplicate, out-of-range, or train/test-overlapping fold indices
 *     -> N4M_ERR_INVALID_ARGUMENT.
 *   - max_components < 1, max_components > X.cols, or max_components >= X.rows
 *     -> N4M_ERR_INVALID_ARGUMENT.
 *   - Unsupported solver/algorithm/deflation -> N4M_ERR_UNSUPPORTED.
 *   - Non-strict operator in the bank -> N4M_ERR_UNSUPPORTED.
 *   - X.rows != Y.rows -> N4M_ERR_SHAPE_MISMATCH.
 */

N4M_API n4m_status_t n4m_aom_global_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_operator_bank_t* bank,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t max_components,
    n4m_aom_global_result_t** out_result);

N4M_API void n4m_aom_global_result_destroy(n4m_aom_global_result_t* result);

N4M_API n4m_status_t n4m_aom_global_result_get_n_operators(
    const n4m_aom_global_result_t* result, int32_t* out);
N4M_API n4m_status_t n4m_aom_global_result_get_max_components(
    const n4m_aom_global_result_t* result, int32_t* out);
N4M_API n4m_status_t n4m_aom_global_result_get_selected_operator_index(
    const n4m_aom_global_result_t* result, int32_t* out);
N4M_API n4m_status_t n4m_aom_global_result_get_selected_n_components(
    const n4m_aom_global_result_t* result, int32_t* out);
N4M_API n4m_status_t n4m_aom_global_result_get_best_score(
    const n4m_aom_global_result_t* result, double* out);

N4M_API n4m_status_t n4m_aom_global_result_get_operator_kinds(
    const n4m_aom_global_result_t* result,
    const n4m_operator_kind_t** out_data, int32_t* out_size);
N4M_API n4m_status_t n4m_aom_global_result_get_operator_scores(
    const n4m_aom_global_result_t* result,
    const double** out_data, int32_t* out_size);
N4M_API n4m_status_t n4m_aom_global_result_get_rmse_curves(
    const n4m_aom_global_result_t* result,
    const double** out_data, int32_t* out_rows, int32_t* out_cols);
N4M_API n4m_status_t n4m_aom_global_result_get_predictions(
    const n4m_aom_global_result_t* result,
    const double** out_data, int64_t* out_rows, int64_t* out_cols);

N4M_API n4m_status_t n4m_aom_per_component_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_operator_bank_t* bank,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t max_components,
    n4m_aom_per_component_result_t** out_result);

N4M_API void n4m_aom_per_component_result_destroy(
    n4m_aom_per_component_result_t* result);

N4M_API n4m_status_t n4m_aom_per_component_result_get_n_operators(
    const n4m_aom_per_component_result_t* result, int32_t* out);
N4M_API n4m_status_t n4m_aom_per_component_result_get_max_components(
    const n4m_aom_per_component_result_t* result, int32_t* out);
N4M_API n4m_status_t n4m_aom_per_component_result_get_selected_n_components(
    const n4m_aom_per_component_result_t* result, int32_t* out);
N4M_API n4m_status_t n4m_aom_per_component_result_get_best_score(
    const n4m_aom_per_component_result_t* result, double* out);

N4M_API n4m_status_t n4m_aom_per_component_result_get_operator_kinds(
    const n4m_aom_per_component_result_t* result,
    const n4m_operator_kind_t** out_data, int32_t* out_size);
N4M_API n4m_status_t n4m_aom_per_component_result_get_selected_operator_indices(
    const n4m_aom_per_component_result_t* result,
    const int32_t** out_data, int32_t* out_size);
N4M_API n4m_status_t n4m_aom_per_component_result_get_component_scores(
    const n4m_aom_per_component_result_t* result,
    const double** out_data, int32_t* out_rows, int32_t* out_cols);
N4M_API n4m_status_t n4m_aom_per_component_result_get_prefix_scores(
    const n4m_aom_per_component_result_t* result,
    const double** out_data, int32_t* out_size);
N4M_API n4m_status_t n4m_aom_per_component_result_get_predictions(
    const n4m_aom_per_component_result_t* result,
    const double** out_data, int64_t* out_rows, int64_t* out_cols);

/* ============================================================================
 * 16. Method-result handle — universal output container for the extra
 *     PLS / monitoring / diagnostics / ensemble fit functions
 * ==========================================================================
 *
 * n4m_method_result_t holds named double matrices, named int32 vectors and
 * named scalars. The fit functions below build one of these per call and
 * return it to the caller. Accessor functions look up outputs by name; the
 * caller must NOT free returned buffer pointers (they point into the
 * result-owned storage and are invalidated when the result is destroyed).
 */

N4M_API void n4m_method_result_destroy(n4m_method_result_t* result);

/* Read a named double matrix. Returns N4M_ERR_INVALID_ARGUMENT when the name is
 * absent. */
N4M_API n4m_status_t n4m_method_result_get_double_matrix(
    const n4m_method_result_t* result,
    const char* name,
    const double** out_data, int64_t* out_rows, int64_t* out_cols);

/* Read a named int32 vector. Returns N4M_ERR_INVALID_ARGUMENT when the name is
 * absent. */
N4M_API n4m_status_t n4m_method_result_get_int_vector(
    const n4m_method_result_t* result,
    const char* name,
    const int32_t** out_data, int32_t* out_size);

/* Read a named int64 vector. Returns N4M_ERR_INVALID_ARGUMENT when the name is
 * absent. Used by the §5 / §6 selectors to expose `selected_indices`,
 * `removed_indices`, etc. without losing precision for large feature counts. */
N4M_API n4m_status_t n4m_method_result_get_int64_vector(
    const n4m_method_result_t* result,
    const char* name,
    const int64_t** out_data, int64_t* out_size);

/* Read a named scalar. Returns N4M_ERR_INVALID_ARGUMENT when the name is absent. */
N4M_API n4m_status_t n4m_method_result_get_scalar(
    const n4m_method_result_t* result,
    const char* name,
    double* out_value);

/* ---- Fit entry points ---- */

/* Sparse SIMPLS (§7). Soft-thresholds the SIMPLS direction by
 * `sparsity_lambda` at each component. The result contains:
 *   "coefficients" (n_features x n_targets, row-major)
 *   "predictions"  (n_samples  x n_targets, row-major)
 *   "x_mean"       (1 x n_features)
 *   "y_mean"       (1 x n_targets)
 *   "weights_w"    (n_features x n_components)
 *   scalar "rmse"  in-sample RMSE
 */
N4M_API n4m_status_t n4m_sparse_simpls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    double sparsity_lambda,
    n4m_method_result_t** out_result);

/* Domain-invariant PLS (§13). Penalizes the SIMPLS direction's alignment
 * with the source-vs-target mean difference. The result contains:
 *   "coefficients"        (n_features x n_targets)
 *   "predictions"         (n_samples  x n_targets)
 *   "x_mean", "y_mean"
 *   scalar "rmse_source"  in-sample RMSE on source data
 */
N4M_API n4m_status_t n4m_di_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X_source,
    const n4m_matrix_view_t* Y_source,
    const n4m_matrix_view_t* X_target,
    double di_lambda,
    n4m_method_result_t** out_result);

/* Recursive (moving-window) PLS (§12). Predicts each sample at index >=
 * window_size from a SIMPLS model fitted on the previous window_size rows.
 * The result contains:
 *   "predictions" (n_samples x n_targets) — zeros for warmup samples
 *   int "in_window" (n_samples)           — 1 when predicted, 0 warmup
 *   scalar "rmse_predictable" on samples i >= window_size
 */
N4M_API n4m_status_t n4m_recursive_pls_run(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t window_size,
    n4m_method_result_t** out_result);

/* CPPLS — Canonical Powered PLS (§1). Each X column is rescaled by its
 * std^gamma before SIMPLS, then coefficients are unscaled. gamma in [0, 1]
 * interpolates between PLS (gamma=0) and a power-rescaled flavour
 * (gamma=1). The result contains:
 *   "coefficients" (n_features x n_targets, row-major)
 *   "predictions"  (n_samples  x n_targets)
 *   "x_mean", "y_mean"
 *   scalar "rmse", scalar "gamma"
 */
N4M_API n4m_status_t n4m_cppls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    double gamma,
    n4m_method_result_t** out_result);

/* Weighted PLS (§26). Pre-multiplies the centered (X, Y) rows by
 * sqrt(sample_weights[i]) before SIMPLS. `weights` must point to
 * `n_samples` strictly positive finite doubles. The result contains:
 *   "coefficients", "predictions", "x_mean", "y_mean"
 *   scalar "rmse"
 */
N4M_API n4m_status_t n4m_weighted_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const double* sample_weights,
    int64_t sample_weights_size,
    n4m_method_result_t** out_result);

/* Robust PLS via IRLS with Huber weights (§26). max_irls_iter <= 0 falls
 * back to 5. The result contains:
 *   "coefficients", "predictions", "x_mean", "y_mean"
 *   "final_weights" (1 x n_samples) — Huber weights at convergence
 *   scalar "rmse", scalar "huber_k"
 */
N4M_API n4m_status_t n4m_robust_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    double huber_k,
    int32_t max_irls_iter,
    n4m_method_result_t** out_result);

/* Ridge-augmented PLS (§26). Augments (X, Y) with sqrt(ridge_lambda)*I and
 * 0 rows, then runs SIMPLS. ridge_lambda must be >= 0. The result contains:
 *   "coefficients", "predictions", "x_mean", "y_mean"
 *   scalar "rmse", scalar "ridge_lambda"
 */
N4M_API n4m_status_t n4m_ridge_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    double ridge_lambda,
    n4m_method_result_t** out_result);

/* Continuum regression (§26). tau in [0, 1] interpolates between PLS
 * (tau=0) and a whitened-X / OLS-like flavour (tau=1). The result contains:
 *   "coefficients", "predictions", "x_mean", "y_mean"
 *   scalar "rmse", scalar "tau"
 */
N4M_API n4m_status_t n4m_continuum_regression_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    double tau,
    n4m_method_result_t** out_result);

/* N-PLS (§22). 3-way tensor regression via Bro's algorithm. X must be a
 * row-major (n_samples x (mode_j * mode_k)) flat view of the (n x J x K)
 * tensor. The result contains:
 *   "predictions"   (n_samples x n_targets)
 *   "coefficients"  ((mode_j*mode_k) x n_targets)
 *   "w_j"           (mode_j x n_components)
 *   "w_k"           (mode_k x n_components)
 *   "scores_t"      (n_samples x n_components)
 *   "x_mean", "y_mean"
 *   scalar "rmse"
 */
N4M_API n4m_status_t n4m_n_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X_flat,
    int32_t mode_j,
    int32_t mode_k,
    const n4m_matrix_view_t* Y,
    n4m_method_result_t** out_result);

/* Non-linear kernel PLS (§24). `kernel_type`:
 *   0 = linear, 1 = RBF, 2 = polynomial, 3 = sigmoid
 * `gamma`, `coef0`, `degree` are kernel parameters; ignored when not
 * applicable to the selected kernel. The result contains:
 *   "predictions"   (n_samples x n_targets)  in-sample predictions
 *   "alpha"         (n_samples x n_targets)  dual coefficients
 *   "y_mean"        (1 x n_targets)
 *   scalar "rmse", scalar "kernel_type" (as double)
 */
N4M_API n4m_status_t n4m_kernel_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    int32_t kernel_type,
    double gamma,
    double coef0,
    int32_t degree,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    n4m_method_result_t** out_result);

/* O2PLS (§16; Trygg & Wold 2003). Bi-directional OPLS with
 * `n_predictive` joint + `n_x_orthogonal` X-orthogonal +
 * `n_y_orthogonal` Y-orthogonal components. The result contains:
 *   "coefficients"    (n_features_x x n_features_y)
 *   "predictions"     (n_samples x n_features_y)
 *   "x_mean", "y_mean"
 *   "w_predictive"    (n_features_x x n_predictive)
 *   "c_predictive"    (n_features_y x n_predictive)
 *   "w_x_orthogonal"  (n_features_x x n_x_orthogonal)
 *   "c_y_orthogonal"  (n_features_y x n_y_orthogonal)
 *   "b_predictive"    (1 x n_predictive)
 *   scalar "rmse"
 */
N4M_API n4m_status_t n4m_o2pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t n_predictive,
    int32_t n_x_orthogonal,
    int32_t n_y_orthogonal,
    n4m_method_result_t** out_result);

/* Approximate-PRESS component selection (§29). For each component count
 * k in [1, max_components], fits SIMPLS, then approximates PRESS via
 * leverage-inflated in-sample residuals. The result contains:
 *   "press_per_component" (1 x max_components)
 *   "rmse_per_component"  (1 x max_components)
 *   int "selected_n_components" — argmin of press_per_component
 *   scalar "selected_n_components_d" — same as a double for convenience
 */
N4M_API n4m_status_t n4m_approximate_press_compute(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t max_components,
    n4m_method_result_t** out_result);

/* Sparse PLS-DA (§7). Dummy-encodes integer class labels then runs
 * sparse SIMPLS with the configured `sparsity_lambda` from `cfg`.
 * `y_labels` must be a length-n_samples buffer of non-negative class
 * IDs (max value implies n_classes). The result contains:
 *   "coefficients"  (n_features x n_classes)
 *   "predictions"   (n_samples x n_classes) — soft assignments
 *   "x_mean", "y_mean"
 *   "class_priors"  (1 x n_classes)
 *   scalar "rmse"
 */
N4M_API n4m_status_t n4m_sparse_pls_da_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const int32_t* y_labels,
    int64_t y_labels_size,
    n4m_method_result_t** out_result);

/* Group sparse PLS (§7). Soft-thresholds groups of features together.
 * `group_assignment` is a length-n_features buffer of 0-based group IDs.
 * Result contains coefficients, predictions, x_mean, y_mean, rmse, and
 * scalar `n_groups`.
 */
N4M_API n4m_status_t n4m_group_sparse_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const int32_t* group_assignment,
    int64_t group_assignment_size,
    double group_lambda,
    n4m_method_result_t** out_result);

/* Fused sparse PLS (§7). Combines L1 sparsity with fusion of
 * consecutive feature pairs. Result identical to group_sparse_pls_fit
 * plus scalars `l1_lambda` and `fusion_lambda`.
 */
N4M_API n4m_status_t n4m_fused_sparse_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    double l1_lambda,
    double fusion_lambda,
    n4m_method_result_t** out_result);

/* PLS process monitoring (§19). Computes Hotelling T² and Q-residual
 * thresholds from `X_reference` (phase-1) at confidence level
 * (1 - alpha), then evaluates `X_monitor` (phase-2) and flags rows
 * exceeding the thresholds. Result contains:
 *   "t2"            (1 x n_monitor)
 *   "q"             (1 x n_monitor)
 *   "t2_reference"  (1 x n_reference)
 *   "q_reference"   (1 x n_reference)
 *   int "t2_alarms"  (length n_monitor)
 *   int "q_alarms"   (length n_monitor)
 *   int "any_alarms" (length n_monitor)
 *   scalar "t2_threshold", scalar "q_threshold", scalar "alpha"
 */
N4M_API n4m_status_t n4m_pls_monitoring_run(
    n4m_context_t* ctx,
    const n4m_model_t* model,
    const n4m_matrix_view_t* X_reference,
    const n4m_matrix_view_t* X_monitor,
    double alpha,
    n4m_method_result_t** out_result);

/* One-SE rule for PLS component selection (§10). Given a (max_components
 * x n_folds) row-major matrix of fold RMSE values, returns the smallest
 * k whose mean fold RMSE is within one standard error of the best mean
 * fold RMSE. Result contains:
 *   "mean_rmse_per_component" (1 x max_components)
 *   int "best_n_components"       (length 1)
 *   int "one_se_n_components"     (length 1)
 *   scalar "one_se_standard_error", scalar "one_se_threshold"
 */
N4M_API n4m_status_t n4m_one_se_rule_compute(
    n4m_context_t* ctx,
    const double* fold_rmse_matrix,
    int32_t max_components,
    int32_t n_folds,
    n4m_method_result_t** out_result);

/* SO-PLS (§17). Sequential and Orthogonalized PLS for B X-blocks
 * predicting one Y. `X_blocks` is an array of `n_blocks`
 * n4m_matrix_view_t structs (all sharing X.rows = Y.rows).
 * `n_components_per_block` is a length-n_blocks int32 array.
 * The result contains:
 *   "predictions" (n_samples x n_targets)
 *   "y_mean"      (1 x n_targets)
 *   For each block b: "block_coefficients_<b>" of shape (p_b x n_targets)
 *   scalar "n_blocks"
 */
N4M_API n4m_status_t n4m_so_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X_blocks,
    int32_t n_blocks,
    const n4m_matrix_view_t* Y,
    const int32_t* n_components_per_block,
    int64_t n_components_per_block_size,
    n4m_method_result_t** out_result);

/* OnPLS (§18). Generalized orthogonal projection for multi-block PLS.
 * Removes joint and per-block unique components.
 * Result contains scalars `n_blocks` and `n_joint`, plus per-block
 * loading matrices "joint_loadings_<b>", "unique_loadings_<b>" and
 * "joint_scores_<b>".
 */
N4M_API n4m_status_t n4m_on_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X_blocks,
    int32_t n_blocks,
    int32_t n_joint,
    const int32_t* n_unique_per_block,
    int64_t n_unique_per_block_size,
    n4m_method_result_t** out_result);

/* ROSA (§19). Response-Oriented Sequential Alternation: at each
 * component, picks the block whose latent direction yields the highest
 * correlation with the current Y residual.
 * Result contains:
 *   "predictions"                  (n_samples x n_targets)
 *   "y_mean"
 *   "selected_block_per_component" int vector
 *   For each block b: "block_coefficients_<b>"
 *   scalar "n_components"
 */
N4M_API n4m_status_t n4m_rosa_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X_blocks,
    int32_t n_blocks,
    const n4m_matrix_view_t* Y,
    int32_t n_components,
    n4m_method_result_t** out_result);

/* Bagging PLS (§20). Bootstrap aggregation of `n_estimators` PLS
 * regressors with the configured seed. Returns the average regression
 * coefficient matrix:
 *   "coefficients"   (n_features x n_targets)
 *   "predictions"    (n_samples x n_targets)
 *   "x_mean", "y_mean"
 *   scalar "rmse", scalar "n_estimators"
 */
N4M_API n4m_status_t n4m_bagging_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t n_estimators,
    uint64_t seed,
    n4m_method_result_t** out_result);

/* Boosting PLS (§20). Gradient-boosting style stage-wise refit of
 * `n_estimators` PLS regressors with a per-stage `learning_rate`.
 * Output shape identical to bagging_pls_fit.
 */
N4M_API n4m_status_t n4m_boosting_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t n_estimators,
    double learning_rate,
    n4m_method_result_t** out_result);

/* Random-subspace PLS (§20). Each of `n_estimators` PLS regressors is
 * fit on a random subset of `features_per_subspace` columns. The
 * result averages predictions over the missing columns by zero-padding
 * coefficients; output shape identical to bagging_pls_fit plus a
 * scalar `features_per_subspace`.
 */
N4M_API n4m_status_t n4m_random_subspace_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t n_estimators,
    int32_t features_per_subspace,
    uint64_t seed,
    n4m_method_result_t** out_result);

/* GPR-on-PLS (§47): two-stage regression. First fits a SIMPLS PLS with
 * cfg.n_components latent components and rotation R (p x k). Then fits a
 * Gaussian Process with RBF kernel
 *     K(t_i, t_j) = exp(-||t_i - t_j||^2 / (2 * length_scale^2))
 * and diagonal noise sigma^2 (the `noise_level` parameter is interpreted
 * as **variance**, matching sklearn `WhiteKernel(noise_level=...)`, not
 * standard deviation). The GP solve uses Cholesky on (K + noise_level * I)
 * on the training scores T = (X - x_mean) @ R and centred y.
 * Y must be single-target (n x 1) in Phase 47.
 *
 * `seed` is reserved for ABI symmetry with the ensemble methods; the fit
 * is fully deterministic.
 *
 * Result keys:
 *   "rotation_r"           (n_features x n_components)
 *   "x_mean"               (1 x n_features)
 *   "alpha"                (n_samples x 1)            — GP dual weights
 *   "L_lower"              (n_samples x n_samples)    — Cholesky factor
 *   "training_scores"      (n_samples x n_components)
 *   "predictions"          (n_samples x 1)
 *   "predictive_variance"  (n_samples x 1)            — noise-free posterior
 *   scalar "length_scale", "noise_level", "n_components", "y_mean", "rmse", "seed"
 *
 * Returns N4M_ERR_INVALID_ARGUMENT for non-positive length_scale or
 * noise_level, or for n_components outside [1, min(n,p)].
 * Returns N4M_ERR_SHAPE_MISMATCH if Y has more than one column.
 * Returns N4M_ERR_NUMERICAL_FAILURE if the Cholesky of (K + noise*I)
 * fails (typically at very low noise with redundant score directions).
 */
N4M_API n4m_status_t n4m_gpr_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    double length_scale,
    double noise_level,
    uint64_t seed,
    n4m_method_result_t** out_result);

/* Simplified SIMPLS PLS regression — raw-pointer signature for
 * language bindings whose FFI layer struggles with the
 * `n4m_matrix_view_t*` parameter pattern (notably Emscripten 5.0.7
 * and Julia 1.12 ccall with `Ref{MatrixView}`).
 *
 * X is row-major (n × p), Y is row-major (n × q). The function fits
 * SIMPLS with `center_x = center_y = scale_x = scale_y = 1` and writes
 * into caller-provided buffers:
 *
 *   coefficients_out: (p × q) row-major regression coefficients
 *   x_mean_out:       (1 × p)
 *   y_mean_out:       (1 × q)
 *   predictions_out:  (n × q) row-major in-sample predictions; pass
 *                     NULL to skip the in-sample predict step.
 *
 * Returns N4M_OK on success or a N4M_ERR_* status. This is a stable
 * additive helper — implementations may grow new variants but the
 * shape of this one is fixed at ABI minor 1.13.
 */
N4M_API n4m_status_t n4m_pls_fit_simple(
    const double* x, const double* y,
    int32_t n, int32_t p, int32_t q, int32_t n_components,
    double* coefficients_out,
    double* x_mean_out,
    double* y_mean_out,
    double* predictions_out);

/* PLS-GLM (§5). PLS-reduced design feeding a softmax / Poisson IRLS.
 * `poisson` selects the Poisson-link path; otherwise a one-vs-rest
 * softmax-like fit on a continuous PLS regression on Y. The result
 * contains:
 *   "coefficients"  (n_features x n_classes)
 *   "intercept"     (1 x n_classes)
 *   "predictions"   (n_samples x n_classes)
 *   "x_mean"
 *   scalar "rmse", scalar "poisson" (0 or 1)
 */
N4M_API n4m_status_t n4m_pls_glm_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t poisson,
    n4m_method_result_t** out_result);

/* PLS-QDA (§5). Quadratic discriminant analysis on PLS scores.
 * `y_labels` is a length-n_samples buffer of non-negative class IDs.
 * The result contains:
 *   "class_means"        (n_classes x n_components)
 *   "class_covariances"  (n_classes x (n_components * n_components))
 *   "log_class_priors"   (1 x n_classes)
 *   "rotations_r"        (n_features x n_components)
 *   "x_mean"
 *   "predictions"        (n_samples x n_classes)  log-likelihood scores
 *   scalar "rmse"
 */
N4M_API n4m_status_t n4m_pls_qda_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const int32_t* y_labels,
    int64_t y_labels_size,
    n4m_method_result_t** out_result);

/* PLS-Cox (§5). PLS-reduced Cox proportional hazards with Breslow
 * baseline hazard. `survival_times` is the observed time, and
 * `event_indicators` is 1 (event) or 0 (censored). The result contains:
 *   "coefficients"     (n_features x 1)  linear predictor coefficients
 *   "baseline_hazard"  (1 x n_unique_event_times)
 *   "event_times"      (1 x n_unique_event_times)
 *   "x_mean"
 *   "predictions"      (n_samples x 1)  linear-predictor scores
 */
N4M_API n4m_status_t n4m_pls_cox_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const double* survival_times,
    int64_t survival_times_size,
    const int32_t* event_indicators,
    int64_t event_indicators_size,
    n4m_method_result_t** out_result);

/* PDS — Piecewise Direct Standardization (§13). Fits per-target-column
 * windowed least-squares maps from source instrument X_source to target
 * X_target. The result contains:
 *   "transformation"  (n_features_target x n_features_source)
 *   "predictions"     (n_samples x n_features_target) — X_source @ T^T
 *   scalar "rmse" — frobenius error vs X_target
 *   scalar "window_half_width"
 */
N4M_API n4m_status_t n4m_pds_fit(
    n4m_context_t* ctx,
    const n4m_matrix_view_t* X_source,
    const n4m_matrix_view_t* X_target,
    int32_t window_half_width,
    n4m_method_result_t** out_result);

/* DS — Direct Standardization (§13). Fits a full pt × ps least-squares
 * map from centered source X to centered target X plus a bias. The
 * result contains:
 *   "transformation" (n_features_source x n_features_target)
 *   "bias"           (1 x n_features_target)
 *   "predictions"    (n_samples x n_features_target)
 *   scalar "rmse"
 */
N4M_API n4m_status_t n4m_ds_fit(
    n4m_context_t* ctx,
    const n4m_matrix_view_t* X_source,
    const n4m_matrix_view_t* X_target,
    n4m_method_result_t** out_result);

/* MIR-PLS — Multiple Inverse Regression PLS (§13). Inverts the X→Y
 * relationship by running SIMPLS on (Y, X) and pseudoinverting the
 * resulting Y→X coefficients to obtain X→Y prediction coefficients.
 * The result contains:
 *   "coefficients"  (n_features x n_targets)
 *   "predictions"   (n_samples x n_targets)
 *   "x_mean", "y_mean"
 *   scalar "rmse"
 */
N4M_API n4m_status_t n4m_mir_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    n4m_method_result_t** out_result);

/* Missing-aware NIPALS (§13). Same shape as a regular PLS regression
 * model but tolerates NaN entries in X (replaced with the current
 * latent-space iterate during NIPALS). The result contains:
 *   "coefficients", "predictions", "x_mean", "y_mean"
 *   scalar "rmse"
 */
N4M_API n4m_status_t n4m_missing_aware_nipals_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    n4m_method_result_t** out_result);

/* PLS diagnostics (§9). Computes Hotelling T², Q residuals (SPE) and
 * DModX from a fitted model and a design matrix X. When `X_reference`
 * is NULL, score variances and sigma_train fall back to the model's
 * stored training scores — this requires `cfg.store_scores = 1` at fit
 * time. When `X_reference` is non-NULL, its rows define the reference
 * score distribution.
 *
 * The result contains:
 *   "t2"     (1 x n_samples) — Hotelling T² statistic per row
 *   "q"      (1 x n_samples) — squared reconstruction residual per row
 *   "dmodx"  (1 x n_samples) — distance-to-model X per row
 *   scalar "n_components", scalar "n_features"
 */
N4M_API n4m_status_t n4m_pls_diagnostics_compute(
    n4m_context_t* ctx,
    const n4m_model_t* model,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* X_reference,
    n4m_method_result_t** out_result);

/* ============================================================================
 * 17. ABI shims for §2 internal-only core algos (Phase 4r/4s/4p/4q)
 *     and §4 AOM preprocessing (Phase 6a)
 * ==========================================================================
 * These four core algorithms have full C++ implementations in cpp/src/core
 * but were not previously exposed via the public C ABI. Added 2026-05 to
 * close the parity-gate gap identified in docs/parity_audit_2026_05/.
 */

/* MB-PLS — block-weighted multi-block PLS (Phase 4r). Predicts on the
 * concatenated feature matrix X (n × Σ block_sizes). Result keys:
 *   "predictions"    (n × n_targets)
 *   "coefficients"   (Σ block_sizes × n_targets, original X scale)
 *   "x_mean"         (1 × Σ block_sizes)
 *   "x_scale"        (1 × Σ block_sizes)
 *   "intercept"      (1 × n_targets)
 *   "block_weights"  (1 × n_blocks)
 *   scalar "n_blocks", scalar "rmse"
 */
N4M_API n4m_status_t n4m_mb_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const int64_t* block_sizes,
    int64_t n_blocks,
    n4m_method_result_t** out_result);

/* LW-PLS — locally-weighted PLS with k-NN windows (Phase 4s). Predicts
 * each test row from a per-row PLS refit on its `n_neighbors` nearest
 * training rows. Currently the training set is X itself (in-sample). Result
 * keys:
 *   "predictions"            (n × n_targets) double matrix
 *   "neighbor_indices"       (n × n_neighbors) double matrix (cast from
 *                            int64 for unified matrix-shaped reads)
 *   int64 "neighbor_indices_i64" — same data as a row-major int64 vector
 *                            (preferred for index semantics)
 *   scalar "n_neighbors", scalar "n_components", scalar "rmse"
 */
N4M_API n4m_status_t n4m_lw_pls_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t n_neighbors,
    n4m_method_result_t** out_result);

/* PLS-LDA — Linear Discriminant Analysis on PLS scores (Phase 4p). Result
 * keys:
 *   "decision_scores"   (n × n_classes)
 *   int "predictions"   (n)
 *   scalar "n_classes", scalar "n_components"
 */
N4M_API n4m_status_t n4m_pls_lda_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const int32_t* y_labels,
    int64_t y_labels_size,
    int32_t n_classes,
    n4m_method_result_t** out_result);

/* PLS-Logistic — multinomial logistic regression on PLS scores (Phase 4q).
 * Result keys:
 *   "decision_scores"  (n × n_classes)
 *   "probabilities"    (n × n_classes)
 *   "intercepts"       (1 × (n_classes - 1))
 *   "coefficients"     ((n_classes - 1) × n_components)
 *   int "predictions"  (n)
 *   scalar "n_classes", scalar "n_components"
 */
N4M_API n4m_status_t n4m_pls_logistic_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const int32_t* y_labels,
    int64_t y_labels_size,
    int32_t n_classes,
    n4m_method_result_t** out_result);

/* AOM preprocessing fit/transform (Phase 6a). Applies an operator bank
 * through the gating strategy and returns both the per-operator outputs
 * and the gated/mixed transformed matrix. Y is optional (some operators
 * use it, e.g. EPO; pass NULL when not needed). Result keys:
 *   "transformed"        (n × n_features) — final gated transform
 *   "operator_outputs"   (n_operators × (n × n_features), operator-major,
 *                         stored as a (n_operators × (n*n_features)) matrix)
 *   "weights"            (1 × n_operators) — gating weights at fit time
 *   int64 "operator_kinds" (n_operators) — N4M_OP_* enum values
 *   scalar "n_operators", scalar "mode" (gating mode integer)
 */
N4M_API n4m_status_t n4m_aom_preprocess_fit(
    n4m_context_t* ctx,
    const n4m_operator_bank_t* bank,
    const n4m_gating_strategy_t* gate,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    n4m_method_result_t** out_result);

/* ============================================================================
 * 18. ABI shims for §6 Phase 5 variable-selection methods
 * ==========================================================================
 * Each selector returns a n4m_method_result_t with at minimum:
 *   int64 "selected_indices"     — top-k (or final) selected feature indices
 *   double "scores" or "rmse_*"  — algorithm-specific score/RMSE arrays
 *   scalar "best_rmse" (when applicable)
 *
 * All selectors that take a ValidationPlan use the same plan API as
 * `n4m_aom_global_select`. Selectors that take seeds expose them as
 * `uint64_t` parameters.
 */

/* VIP / coefficient-magnitude / selectivity-ratio rankers (Phase 5a). Method
 * enum: 0=VIP, 1=coefficient magnitude, 2=selectivity ratio. */
N4M_API n4m_status_t n4m_variable_select_rank(
    n4m_context_t* ctx,
    const n4m_model_t* model,
    const n4m_matrix_view_t* X,
    int32_t method,
    int32_t top_k,
    n4m_method_result_t** out_result);

/* Interval / iPLS / mwPLS scan (Phase 5b). */
N4M_API n4m_status_t n4m_interval_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t interval_width,
    int32_t step,
    n4m_method_result_t** out_result);

/* MCUVE / coefficient-stability selector (Phase 5c). */
N4M_API n4m_status_t n4m_stability_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t top_k,
    n4m_method_result_t** out_result);

/* UVE — artificial-noise-thresholded selector (Phase 5d). */
N4M_API n4m_status_t n4m_uve_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t noise_features,
    uint64_t noise_seed,
    n4m_method_result_t** out_result);

/* SPA — Successive Projections Algorithm (Phase 5e). */
N4M_API n4m_status_t n4m_spa_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t top_k,
    n4m_method_result_t** out_result);

/* CARS — Competitive Adaptive Reweighted Sampling (Phase 5f). */
N4M_API n4m_status_t n4m_cars_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t n_iterations,
    int32_t min_features,
    n4m_method_result_t** out_result);

/* Random Frog (Phase 5g). */
N4M_API n4m_status_t n4m_random_frog_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t n_iterations,
    int32_t initial_size,
    int32_t min_size,
    int32_t max_size,
    int32_t top_k,
    uint64_t seed,
    n4m_method_result_t** out_result);

/* SCARS — Stability + CARS (Phase 5h). */
N4M_API n4m_status_t n4m_scars_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t n_iterations,
    int32_t min_features,
    double sample_fraction,
    uint64_t seed,
    n4m_method_result_t** out_result);

/* GA-PLS (Phase 5i). */
N4M_API n4m_status_t n4m_ga_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t n_generations,
    int32_t population_size,
    int32_t min_features,
    int32_t max_features,
    double mutation_rate,
    uint64_t seed,
    n4m_method_result_t** out_result);

/* PSO-PLS (§48): Binary Particle Swarm Optimization variable selection
 * (Kennedy & Eberhart 1997). Each particle is a binary mask over the
 * p features; fitness is the CV-RMSE of a PLS regression on the
 * selected subset. Position update uses sigmoid stochastic threshold
 * on the continuous velocity.
 *
 * Defaults from Clerc & Kennedy (2002) convergence analysis:
 *   w = 0.729 (inertia), c1 = c2 = 1.494, v_max = 4.0.
 *
 * Result keys:
 *   "inclusion_frequencies"     (1 × n_features)
 *   "best_rmse_by_iteration"    (1 × n_iterations)
 *   "mean_rmse_by_iteration"    (1 × n_iterations)
 *   int64 "selected_indices"
 *   scalars: n_features, n_targets, n_components, n_swarm, n_iterations,
 *            w, c1, c2, v_max, seed, best_rmse
 */
N4M_API n4m_status_t n4m_pso_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t n_swarm,
    int32_t n_iterations,
    double w,
    double c1,
    double c2,
    double v_max,
    uint64_t seed,
    n4m_method_result_t** out_result);

/* VISSA-PLS (§49): Variable Iterative Space Shrinkage Approach.
 * Reference: Deng B., Yun Y., Liang Y. (2014) Anal. Chim. Acta 838:27-40.
 * Paper-only — no widely installable port.
 *
 * Weighted Binary Matrix Sampling iteratively shrinks the per-feature
 * inclusion probabilities by averaging the masks of the top-K best
 * submodels per iteration. floor_probability clamps w_j to
 * [floor, 1-floor] each iteration to preserve exploration.
 *
 * Defaults: n_iterations=20, n_submodels=100, ratio_kept=0.1,
 * threshold=0.5, floor_probability=0.01.
 *
 * Result keys:
 *   "final_probabilities"   (1 × p) — final w_j vector
 *   "inclusion_frequencies" (1 × p) — alias for final_probabilities
 *   "best_rmse_by_iteration"  (1 × n_iterations)
 *   "mean_rmse_by_iteration"  (1 × n_iterations)
 *   "top_k_per_iteration"   (n_iterations × p)
 *   int64 "selected_indices" — {j : w_j > threshold}
 *   scalars: n_features, n_targets, n_components, n_iterations,
 *            n_submodels, ratio_kept, threshold, floor_probability,
 *            seed, best_rmse
 */
N4M_API n4m_status_t n4m_vissa_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t n_iterations,
    int32_t n_submodels,
    double ratio_kept,
    double threshold,
    double floor_probability,
    uint64_t seed,
    n4m_method_result_t** out_result);

/* Shaving (Phase 5j). */
N4M_API n4m_status_t n4m_shaving_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t n_steps,
    int32_t min_features,
    double shave_fraction,
    n4m_method_result_t** out_result);

/* BVE-PLS (Phase 5k). */
N4M_API n4m_status_t n4m_bve_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t n_steps,
    int32_t min_features,
    n4m_method_result_t** out_result);

/* T2-PLS (Phase 5l). `alpha_thresholds` is an array of `n_alphas` α values
 * in (0, 1). */
N4M_API n4m_status_t n4m_t2_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    const double* alpha_thresholds,
    int64_t n_alphas,
    int32_t min_selected,
    n4m_method_result_t** out_result);

/* WVC-PLS (Phase 5m). */
N4M_API n4m_status_t n4m_wvc_select(
    n4m_context_t* ctx,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t n_components,
    int32_t top_k,
    int32_t normalize,
    n4m_method_result_t** out_result);

/* WVC-threshold (Phase 5r). */
N4M_API n4m_status_t n4m_wvc_threshold_select(
    n4m_context_t* ctx,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t n_components,
    int32_t normalize,
    double score_threshold,
    double threshold_factor,
    int32_t min_selected,
    n4m_method_result_t** out_result);

/* EMCUVE (Phase 5n). */
N4M_API n4m_status_t n4m_emcuve_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t noise_features,
    uint64_t noise_seed,
    int32_t n_ensembles,
    double vote_threshold,
    n4m_method_result_t** out_result);

/* Randomization-test selector (Phase 5o). */
N4M_API n4m_status_t n4m_randomization_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    int32_t n_permutations,
    uint64_t randomization_seed,
    double alpha,
    n4m_method_result_t** out_result);

/* biPLS — backward iPLS (Phase 5p). */
N4M_API n4m_status_t n4m_bipls_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t interval_width,
    int32_t min_intervals,
    n4m_method_result_t** out_result);

/* siPLS — synergistic interval PLS (Phase 5q). */
N4M_API n4m_status_t n4m_sipls_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t interval_width,
    int32_t combination_size,
    n4m_method_result_t** out_result);

/* REP-PLS (Phase 5s). */
N4M_API n4m_status_t n4m_rep_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t n_steps,
    int32_t min_features,
    int32_t remove_count,
    n4m_method_result_t** out_result);

/* IPW-PLS (Phase 5t). */
N4M_API n4m_status_t n4m_ipw_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t n_iterations,
    int32_t top_k,
    double damping,
    double weight_floor,
    n4m_method_result_t** out_result);

/* ST-PLS — score-threshold (Phase 5u). `thresholds` array of `n_thresholds`
 * positive doubles. */
N4M_API n4m_status_t n4m_st_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    const double* thresholds,
    int64_t n_thresholds,
    int32_t min_selected,
    n4m_method_result_t** out_result);

/* ============================================================================
 * 19. ABI shims for §6 Phase 50+ numerical methods (ECR / IRIV / IRF)
 * ==========================================================================*/

/* ECR — Elastic Component Regression (Phase 50). Liu 2009/2010.
 * `alpha` in [0, 1]: 0 = PCR-like, 1 = PLS-like. Values outside the
 * interval are silently clamped to [0, 1] (mirrors libPLS `ecr.m`).
 * cfg.n_components selects how many ECR components to extract. The
 * effective component count is clamped to min(n-1, p-1, cfg.n_components).
 * The result is shaped like the other "fit" methods:
 *   "coefficients"  (n_features x n_targets)
 *   "predictions"   (n_samples x n_targets)
 *   "x_mean", "y_mean", "x_scale", "y_scale"
 *   "weights_w"     (n_features x n_components)
 *   "loadings_p"    (n_features x n_components)
 *   "y_loadings"    (n_targets  x n_components)
 *   "wstar"         (n_features x n_components, so that X · wstar = T)
 *   "r2x"           (1 x n_components, % X variance per component)
 *   "r2y"           (1 x n_components, % Y variance per component)
 *   scalars: n_samples, n_features, n_targets, n_components, alpha, rmse
 */
N4M_API n4m_status_t n4m_ecr_fit(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    double alpha,
    n4m_method_result_t** out_result);

/* IRIV — Iteratively Retains Informative Variables (Phase 51). Yun 2014.
 * Iterative backward variable selection driven by a Mann-Whitney U test
 * on permuted-replacement CV-RMSE values.
 *   cfg.n_components — capped per round to surviving feature count.
 *   max_rounds       — hard cap on IRIV iterations (the algorithm stops
 *                      earlier if no variables are flagged uninformative).
 *   seed             — splitmix64 seed driving the binary-mask generator.
 * The result contains:
 *   "remaining_per_round" (1 x (n_rounds+1)) — features alive after each round
 *   "removed_per_round"   (1 x n_rounds)
 *   int64 "selected_indices"
 *   scalars: n_features, n_targets, n_components, n_rounds,
 *            binary_matrix_rows, seed
 */
N4M_API n4m_status_t n4m_iriv_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t max_rounds,
    uint64_t seed,
    n4m_method_result_t** out_result);

/* IRF — Interval Random Frog (Phase 52). Yun 2013. Random Frog applied to
 * fixed-width sliding spectral intervals (window of `window_size` features
 * each). Final selection is the union of features under the top-K
 * inclusion-frequency intervals.
 *   window_size       — interval width (1 <= w <= n_features)
 *   initial_intervals — Q in the libPLS paper (initial subset size)
 *   top_k             — number of best intervals to union into the
 *                       returned feature set
 *   seed              — splitmix64 seed
 * Result contains:
 *   "probability"        (1 x n_intervals)
 *   "rmse_by_iteration"  (1 x n_iterations)
 *   "subset_sizes"       (1 x n_iterations)
 *   int64 "top_k_intervals"
 *   int64 "selected_indices"  (union of top-K intervals)
 *   scalars: n_features, n_targets, n_components, n_iterations, window_size,
 *            initial_intervals, n_intervals, top_k, seed, best_rmse
 */
N4M_API n4m_status_t n4m_irf_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const n4m_validation_plan_t* plan,
    int32_t n_iterations,
    int32_t window_size,
    int32_t initial_intervals,
    int32_t top_k,
    uint64_t seed,
    n4m_method_result_t** out_result);

/* VIP_SPA — VIP-then-SPA hybrid variable selection (Phase 53).
 * Composition of Favilla 2013 VIP scoring (mask features by VIP > threshold)
 * and Araujo 2001 Successive Projections Algorithm (greedy collinearity-
 * minimising pick over the surviving subset). Matches auswahl.VIP_SPA's
 * LSX-UniWue algorithm; the SPA start is taken as argmax(VIP) within the
 * surviving subset (deterministic) rather than auswahl's seed enumeration,
 * and successive projections are computed on raw masked X to match
 * auswahl._spa._fit_spa (which L2-normalizes only the current direction).
 *   cfg.n_components — components used to fit PLS for VIP scoring
 *   vip_threshold    — drop any feature with VIP <= threshold (auswahl default 0.3)
 *   top_k            — upper bound on selected features; truncated to the
 *                      surviving-candidate count when that is smaller.
 * Result contains:
 *   "vip_scores"        (1 x n_features)
 *   "vip_mask"          (1 x n_features) as 0/1 doubles
 *   "selection_scores"  (1 x n_selected)
 *   int64 "selected_indices" (length n_selected)
 *   scalars: n_features, n_targets, n_components, top_k (requested),
 *            n_selected (actual count), n_candidates, vip_threshold
 */
N4M_API n4m_status_t n4m_vip_spa_select(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    double vip_threshold,
    int32_t top_k,
    n4m_method_result_t** out_result);

#ifdef __cplusplus
}  /* extern "C" */
#endif

/* ============================================================================
 * 16. ABI guard rails — fixed-size assertions on the C ABI shape
 * ==========================================================================
 *
 * Compilers may shrink enums under non-default flags (e.g. -fshort-enums on
 * gcc/clang). Because enums appear in function signatures and in the
 * n4m_matrix_view_t struct, that would silently break the ABI. We assert at
 * compile time that every public enum is sizeof(int) = 4 bytes, and that
 * n4m_matrix_view_t has its documented 48-byte LP64 layout.
 *
 * If any of these fail, the build fails fast — and the caller knows to use
 * a default-enum-size toolchain.
 */
#ifdef __cplusplus
#  define N4M_STATIC_ASSERT(cond, msg) static_assert((cond), msg)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#  define N4M_STATIC_ASSERT(cond, msg) _Static_assert((cond), msg)
#else
#  define N4M_STATIC_ASSERT(cond, msg) typedef char n4m_sa_##__LINE__[(cond) ? 1 : -1]
#endif

N4M_STATIC_ASSERT(sizeof(n4m_status_t)        == 4, "n4m_status_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_backend_t)       == 4, "n4m_backend_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_dtype_t)         == 4, "n4m_dtype_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_algorithm_t)     == 4, "n4m_algorithm_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_solver_t)        == 4, "n4m_solver_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_deflation_t)     == 4, "n4m_deflation_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_operator_kind_t) == 4, "n4m_operator_kind_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_gating_mode_t)   == 4, "n4m_gating_mode_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_model_array_t)   == 4, "n4m_model_array_t must be 4 bytes");

/* n4m_matrix_view_t must be 48 bytes on LP64 / LLP64. ILP32 is not
 * supported by Phase 0; on ILP32 platforms the layout would differ
 * (pointer is 4 bytes), which is why we exclude armeabi-v7a. */
#if defined(__LP64__) || defined(_WIN64)
N4M_STATIC_ASSERT(sizeof(n4m_matrix_view_t) == 48,
                  "n4m_matrix_view_t layout must be 48 bytes on LP64/LLP64");
#endif

#endif /* PLS4ALL_P4A_H */
