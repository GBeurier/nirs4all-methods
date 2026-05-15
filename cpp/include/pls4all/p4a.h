/* SPDX-License-Identifier: CeCILL-2.1 */
/*
 * pls4all — public C ABI v1.1.0.
 *
 * Stability: experimental until v1.0.0. Every breaking change before that
 * version bumps the ABI MAJOR (see p4a_version.h). After v1.0.0 the ABI
 * follows strict semver: breaking changes bump MAJOR, additive changes bump
 * MINOR, bugfixes bump PATCH.
 *
 * This is the ONLY header consumers of libp4a are expected to include.
 *
 * Universal rules of the surface:
 *   - Every exported function is `noexcept` — no C++ exception ever crosses
 *     this boundary. Status-returning wrappers translate exceptions via
 *     p4a_status_t; void destroy/free wrappers catch and swallow after
 *     best-effort cleanup.
 *   - No STL types, no Eigen types, no C++ classes appear in this surface.
 *   - All multi-byte serialized integers are little-endian.
 *   - Floating-point quantities are IEEE 754 binary64 (`double`).
 *   - String pointers returned by const-`char*` getters point into
 *     library-owned storage; callers must NOT free them.
 *   - Error messages for failed calls live in a per-context 4096-byte buffer
 *     (P4A_ERROR_BUFFER_BYTES) that is invalidated by the next call on the
 *     same context — bindings must copy if they retain the value.
 *   - Memory ownership is symmetric: callers free what they allocated, the
 *     core frees what it allocated. The only core-allocated outputs are
 *     `p4a_array_t*` and the opaque handles below; both have explicit
 *     `p4a_*_destroy` / `p4a_array_free` functions.
 *
 * Current implementation status (rev 1.1.0 of this header — May 2026):
 *   - Lifecycle / config / version / matrix-view are fully implemented.
 *   - Pipeline / operator-bank / gating-strategy / validation-plan
 *     lifecycle is implemented; AOM global and POP per-component
 *     selection are exposed by `p4a_aom_global_select` and
 *     `p4a_aom_per_component_select` (see §15).
 *     Pipeline `_fit` / `_transform` are live for P4A_OP_IDENTITY,
 *     P4A_OP_CENTER, P4A_OP_AUTOSCALE, P4A_OP_PARETO_SCALE, P4A_OP_SNV,
 *     P4A_OP_MSC, P4A_OP_EMSC, P4A_OP_DETREND_POLY, P4A_OP_SAVGOL_SMOOTH,
 *     P4A_OP_SAVGOL_DERIVATIVE, P4A_OP_NORRIS_WILLIAMS,
 *     P4A_OP_ASLS_BASELINE, P4A_OP_WAVELET_DENOISE, P4A_OP_OSC and
 *     P4A_OP_EPO.
 *     P4A_OP_FINITE_DIFFERENCE, P4A_OP_WHITTAKER and P4A_OP_FCK are
 *     currently implemented by the internal strict-linear AOM operator
 *     kernels.
 *   - p4a_model_fit implements dependency-free NIPALS, orthogonal-scores,
 *     SIMPLS, kernel, wide-kernel, SVD, power-iteration and randomized-SVD PLS
 *     regression (PLS1 / PLS2) for P4A_ALGO_PLS_REGRESSION +
 *     P4A_SOLVER_NIPALS, P4A_SOLVER_ORTHOGONAL_SCORES,
 *     P4A_SOLVER_SIMPLS, P4A_SOLVER_KERNEL_ALGORITHM,
 *     P4A_SOLVER_WIDE_KERNEL, P4A_SOLVER_SVD, P4A_SOLVER_POWER or
 *     P4A_SOLVER_RANDOMIZED_SVD + P4A_DEFLATION_REGRESSION; PLS canonical
 *     for P4A_ALGO_PLS_CANONICAL + P4A_SOLVER_NIPALS/P4A_SOLVER_SVD +
 *     P4A_DEFLATION_CANONICAL; direct cross-covariance PLSSVD scores for
 *     P4A_ALGO_PLS_SVD + P4A_SOLVER_SVD + P4A_DEFLATION_CANONICAL; PLS-DA
 *     dummy-response scores for P4A_ALGO_PLS_DA + the PLS regression solver set +
 *     P4A_DEFLATION_REGRESSION; OPLS / OPLS-DA common predictive scores
 *     for P4A_ALGO_OPLS or P4A_ALGO_OPLS_DA + P4A_SOLVER_NIPALS +
 *     P4A_DEFLATION_ORTHOGONAL; plus PCR for P4A_ALGO_PCR + P4A_SOLVER_SVD +
 *     P4A_DEFLATION_REGRESSION.
 *   - Predict / transform / model-array accessors and binary serialization
 *     are implemented for fitted core PLS models.
 *
 * Status-code contracts (apply uniformly unless overridden in a function's
 * own doc comment):
 *
 *   - `_create(out)` functions: return P4A_OK on success, P4A_ERR_NULL_POINTER
 *     when `out` is NULL, P4A_ERR_OUT_OF_MEMORY on allocation failure.
 *   - `_destroy(handle)` functions: void, no-op on NULL.
 *   - `_clone(src, out_dst)`: P4A_OK, P4A_ERR_NULL_POINTER, P4A_ERR_OUT_OF_MEMORY.
 *   - `_set_*(handle, value)` setters: P4A_OK, P4A_ERR_NULL_POINTER (handle
 *     is NULL), P4A_ERR_INVALID_ARGUMENT (value rejected). Failed setters
 *     leave the handle's state UNCHANGED — there is no partial mutation.
 *   - `_get_*(handle, out)` getters: P4A_OK, P4A_ERR_NULL_POINTER (handle or
 *     `out` is NULL).
 *   - Any function that touches `p4a_context_t*` may additionally return
 *     P4A_ERR_OUT_OF_MEMORY (from internal allocations) and P4A_ERR_INTERNAL
 *     (from an unexpected exception caught at the C boundary). Both record
 *     a message in `p4a_context_last_error`.
 *   - Functions that consume `p4a_matrix_view_t` may additionally return
 *     P4A_ERR_SHAPE_MISMATCH, P4A_ERR_DTYPE_MISMATCH, P4A_ERR_STRIDE_INVALID
 *     when the view fails p4a_matrix_view_validate.
 */
#ifndef PLS4ALL_P4A_H
#define PLS4ALL_P4A_H

#include <stddef.h>
#include <stdint.h>

#include "pls4all/p4a_export.h"
#include "pls4all/p4a_version.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 1. Status codes
 * ========================================================================== */

typedef enum p4a_status_t {
    P4A_OK                        =   0,
    P4A_ERR_INVALID_ARGUMENT      =   1,
    P4A_ERR_NULL_POINTER          =   2,
    P4A_ERR_SHAPE_MISMATCH        =   3,
    P4A_ERR_DTYPE_MISMATCH        =   4,
    P4A_ERR_STRIDE_INVALID        =   5,
    P4A_ERR_NOT_FITTED            =   6,
    P4A_ERR_NUMERICAL_FAILURE     =   7,
    P4A_ERR_CONVERGENCE_FAILED    =   8,
    P4A_ERR_OUT_OF_MEMORY         =   9,
    P4A_ERR_UNSUPPORTED           =  10,
    P4A_ERR_NOT_IMPLEMENTED       =  11,   /* deferred public surface */
    P4A_ERR_ABI_MISMATCH          =  12,
    P4A_ERR_IO                    =  13,
    P4A_ERR_CORRUPT_BUFFER        =  14,
    P4A_ERR_VERSION_INCOMPATIBLE  =  15,
    P4A_ERR_BACKEND_UNAVAILABLE   =  16,
    P4A_ERR_CANCELLED             =  17,
    P4A_ERR_INTERNAL              = 255
} p4a_status_t;

/* Returns a static NUL-terminated string describing the status code. Returns
 * "unknown status" for any integer outside the declared enum range; never
 * returns NULL. */
P4A_API const char* p4a_status_to_string(p4a_status_t s);

/* ============================================================================
 * 2. Backends
 * ==========================================================================
 *
 * Backends are runtime-selectable via p4a_context_set_backend. The reference
 * CPU backend is always available; everything else is built when the
 * corresponding PLS4ALL_WITH_* CMake option is enabled at compile time.
 */
typedef enum p4a_backend_t {
    P4A_BACKEND_AUTO            = 0,   /* library picks the best available */
    P4A_BACKEND_REFERENCE_CPU   = 1,   /* always present, portable */
    P4A_BACKEND_NATIVE_CPU      = 2,   /* SIMD-tuned scalar paths */
    P4A_BACKEND_BLAS            = 3,
    P4A_BACKEND_OPENMP          = 4,   /* combinable with another backend */
    P4A_BACKEND_CUDA            = 5,
    P4A_BACKEND_OPENCL          = 6,   /* reserved, not implemented */
    P4A_BACKEND_METAL           = 7    /* reserved, not implemented */
} p4a_backend_t;

/* Returns 1 if the backend was compiled into this build of libp4a, 0
 * otherwise. AUTO is always available (it resolves to REFERENCE_CPU at
 * minimum). */
P4A_API int          p4a_backend_is_available(p4a_backend_t backend);
P4A_API const char*  p4a_backend_to_string(p4a_backend_t backend);

/* ============================================================================
 * 3. Numerical dtypes
 * ========================================================================== */

typedef enum p4a_dtype_t {
    P4A_DTYPE_UNKNOWN = 0,
    P4A_DTYPE_F64     = 1,
    P4A_DTYPE_F32     = 2,
    P4A_DTYPE_I32     = 3,    /* reserved for label / index views */
    P4A_DTYPE_I64     = 4
} p4a_dtype_t;

/* Returns the size in bytes of one element of the given dtype.
 * Returns 0 for P4A_DTYPE_UNKNOWN or any out-of-range value. */
P4A_API size_t       p4a_dtype_size(p4a_dtype_t dtype);
P4A_API const char*  p4a_dtype_to_string(p4a_dtype_t dtype);

/* ============================================================================
 * 4. Matrix view — non-owning, stride-aware
 * ==========================================================================
 *
 * The view describes a 2-D array of `dtype` elements at `data`, with `rows`
 * and `cols` and explicit `row_stride` / `col_stride` measured in *elements*
 * (NOT bytes). This shape accepts:
 *   - NumPy row-major (col_stride=1, row_stride=cols)
 *   - R / MATLAB / Fortran column-major (row_stride=1, col_stride=rows)
 *   - Transposed sub-views (swap the two strides)
 *   - Slice views (arbitrary positive strides)
 *
 * Stride rules enforced by p4a_matrix_view_validate:
 *   - Negative strides are rejected.
 *   - For `rows >= 2`, `row_stride` must be >= 1.
 *   - For `cols >= 2`, `col_stride` must be >= 1.
 *   - For degenerate dimensions (`rows <= 1` or `cols <= 1`), the
 *     corresponding stride is unused and may be any non-negative value.
 *   - `rows == 0` or `cols == 0` is allowed (empty view); the buffer may be
 *     NULL in that case.
 *
 * Layout on LP64 (Linux, macOS) and LLP64 (Windows-64): 48 bytes total
 * (8+8+8+8+8+4+4). ILP32 is not supported; Android 32-bit (armeabi-v7a) is
 * explicitly excluded from Phase 0/2 — see roadmap.
 */
typedef struct p4a_matrix_view_t {
    void*        data;
    int64_t      rows;
    int64_t      cols;
    int64_t      row_stride;   /* elements between row i and i+1 */
    int64_t      col_stride;   /* elements between col j and j+1 */
    p4a_dtype_t  dtype;
    int32_t      reserved0;    /* keep struct size stable */
} p4a_matrix_view_t;

/* Initialise a row-major (C / NumPy default) view. row_stride = cols,
 * col_stride = 1. Returns P4A_ERR_NULL_POINTER if `out` is NULL or
 * `data` is NULL with rows*cols > 0; P4A_ERR_INVALID_ARGUMENT for negative
 * dimensions or P4A_DTYPE_UNKNOWN; P4A_OK on success. */
P4A_API p4a_status_t p4a_matrix_view_init_rowmajor(
    p4a_matrix_view_t* out, void* data,
    int64_t rows, int64_t cols, p4a_dtype_t dtype);

/* Initialise a column-major (Fortran / R / MATLAB default) view.
 * row_stride = 1, col_stride = rows. */
P4A_API p4a_status_t p4a_matrix_view_init_colmajor(
    p4a_matrix_view_t* out, void* data,
    int64_t rows, int64_t cols, p4a_dtype_t dtype);

/* Initialise a strided view with explicit positive strides. Useful for
 * slicing into a larger buffer or for transposed views. Strides must be
 * positive (>= 1) and ints; the caller is responsible for ensuring the
 * underlying buffer is large enough. */
P4A_API p4a_status_t p4a_matrix_view_init_strided(
    p4a_matrix_view_t* out, void* data,
    int64_t rows, int64_t cols,
    int64_t row_stride, int64_t col_stride,
    p4a_dtype_t dtype);

/* Validate that the view is well-formed. The rules are listed in §4 prelude
 * above. Returns:
 *   - P4A_OK if all rules hold.
 *   - P4A_ERR_NULL_POINTER  if `v` is NULL, or `v->data` is NULL with
 *                            rows*cols > 0.
 *   - P4A_ERR_STRIDE_INVALID if a stride is negative or zero where required.
 *   - P4A_ERR_INVALID_ARGUMENT for negative `rows` / `cols`, or dtype not
 *                              in {F64, F32, I32, I64}. */
P4A_API p4a_status_t p4a_matrix_view_validate(const p4a_matrix_view_t* v);

/* ============================================================================
 * 5. Opaque handles
 * ========================================================================== */

typedef struct p4a_context_s                    p4a_context_t;
typedef struct p4a_config_s                     p4a_config_t;
typedef struct p4a_operator_bank_s              p4a_operator_bank_t;
typedef struct p4a_gating_strategy_s            p4a_gating_strategy_t;
typedef struct p4a_pipeline_s                   p4a_pipeline_t;
typedef struct p4a_model_s                      p4a_model_t;
typedef struct p4a_array_s                      p4a_array_t;
typedef struct p4a_validation_plan_s            p4a_validation_plan_t;
typedef struct p4a_aom_global_result_s          p4a_aom_global_result_t;
typedef struct p4a_aom_per_component_result_s   p4a_aom_per_component_result_t;
typedef struct p4a_method_result_s              p4a_method_result_t;

/* ============================================================================
 * 6. Context lifecycle
 * ==========================================================================
 *
 * One p4a_context_t represents an isolated execution environment. It owns:
 *   - the per-context 4 KiB error buffer,
 *   - the current backend selection,
 *   - the RNG seed used by stochastic algorithms,
 *   - the thread-count hint for parallel backends,
 *   - an optional user-pointer for binding-side bookkeeping.
 *
 * Threading: a single p4a_context_t* must not be used concurrently from two
 * threads — use one context per thread. Across contexts the library is
 * thread-safe.
 *
 * Signal safety: no p4a_* function is async-signal-safe. Do not call any
 * p4a_* function from a POSIX signal handler.
 */
P4A_API p4a_status_t p4a_context_create(p4a_context_t** out_ctx);
P4A_API void         p4a_context_destroy(p4a_context_t* ctx);

P4A_API p4a_status_t p4a_context_set_seed(p4a_context_t* ctx, uint64_t seed);
P4A_API p4a_status_t p4a_context_get_seed(const p4a_context_t* ctx,
                                          uint64_t* out_seed);

P4A_API p4a_status_t p4a_context_set_backend(p4a_context_t* ctx,
                                             p4a_backend_t backend);
P4A_API p4a_status_t p4a_context_get_backend(const p4a_context_t* ctx,
                                             p4a_backend_t* out_backend);

/* `n_threads`: any int32_t is accepted. Values <= 0 are documented as
 * "library picks the default" (read by parallel backends later); positive
 * values pin the worker count. The setter does NOT validate the upper
 * bound — the caller is responsible for choosing a sensible cap. */
P4A_API p4a_status_t p4a_context_set_num_threads(p4a_context_t* ctx,
                                                 int32_t n_threads);
P4A_API p4a_status_t p4a_context_get_num_threads(const p4a_context_t* ctx,
                                                 int32_t* out_threads);

/* Returns a NUL-terminated UTF-8 string owned by the context. The pointer
 * remains valid until the next p4a_* call on this context — bindings must
 * copy if they need to retain the value. Never returns NULL: when no error
 * has occurred since context creation (or since the last clear_error),
 * returns the empty string "". If `ctx` is NULL, returns a static "" string
 * literal (still never NULL). The buffer capacity is fixed at
 * P4A_ERROR_BUFFER_BYTES (4096) — over-long messages are truncated at the
 * boundary, preserving the NUL terminator. */
P4A_API const char*  p4a_context_last_error(const p4a_context_t* ctx);
/* No-op when `ctx` is NULL. */
P4A_API void         p4a_context_clear_error(p4a_context_t* ctx);

/* Optional binding-side annotation. The library never reads or writes the
 * pointer's referent. `p4a_context_get_user_data(NULL)` returns NULL — this
 * is indistinguishable from a context that has had NULL stored explicitly;
 * if you need to distinguish, store a sentinel value. */
P4A_API p4a_status_t p4a_context_set_user_data(p4a_context_t* ctx, void* user);
P4A_API void*        p4a_context_get_user_data(const p4a_context_t* ctx);

/* ============================================================================
 * 7. Algorithm / solver / deflation enums
 * ========================================================================== */

typedef enum p4a_algorithm_t {
    P4A_ALGO_PLS_REGRESSION = 0,
    P4A_ALGO_PLS_CANONICAL  = 1,
    P4A_ALGO_PLS_SVD        = 2,
    P4A_ALGO_PLS_DA         = 3,
    P4A_ALGO_OPLS           = 4,
    P4A_ALGO_OPLS_DA        = 5,
    P4A_ALGO_SPARSE_PLS     = 6,
    P4A_ALGO_MB_PLS         = 7,
    P4A_ALGO_LW_PLS         = 8,
    P4A_ALGO_AOM_PLS        = 9,
    P4A_ALGO_PCR            = 10
    /* Nonlinear kernel PLS (RBF / polynomial / sigmoid) is intentionally
     * deferred to Phase 4 along with the kernel-type enum + setters; we do
     * not pre-declare it here so we don't lock in an unfinished surface.
     * Orthogonal-scores, SIMPLS, kernel-algorithm, wide-kernel, power and
     * randomized-SVD are SOLVERS (see below), not algorithms. POP-PLS is a
     * GATING STRATEGY (see below). */
} p4a_algorithm_t;

typedef enum p4a_solver_t {
    P4A_SOLVER_NIPALS            = 0,
    P4A_SOLVER_SIMPLS            = 1,
    P4A_SOLVER_ORTHOGONAL_SCORES = 2,
    P4A_SOLVER_KERNEL_ALGORITHM  = 3,   /* linear kernel-algorithm PLS (Lindgren/Wold) */
    P4A_SOLVER_WIDE_KERNEL       = 4,
    P4A_SOLVER_SVD               = 5,
    P4A_SOLVER_POWER             = 6,
    P4A_SOLVER_RANDOMIZED_SVD    = 7
} p4a_solver_t;

typedef enum p4a_deflation_t {
    P4A_DEFLATION_REGRESSION  = 0,
    P4A_DEFLATION_CANONICAL   = 1,
    P4A_DEFLATION_X_ONLY      = 2,
    P4A_DEFLATION_XY          = 3,
    P4A_DEFLATION_ORTHOGONAL  = 4
} p4a_deflation_t;

/* ============================================================================
 * 8. Config lifecycle and setters
 * ==========================================================================
 *
 * A p4a_config_t is an opaque bag of knobs. Bindings build one with the
 * setters below, hand it to p4a_model_fit, and either destroy it or reuse it.
 *
 * Setter contract:
 *   - Every setter validates its inputs.
 *   - On rejection (P4A_ERR_INVALID_ARGUMENT or _NULL_POINTER), the config
 *     is left UNCHANGED — failed setters never partially mutate state.
 *   - Every setter has a corresponding getter that returns the most recent
 *     accepted value.
 */
P4A_API p4a_status_t p4a_config_create(p4a_config_t** out_cfg);
P4A_API void         p4a_config_destroy(p4a_config_t* cfg);
P4A_API p4a_status_t p4a_config_clone(const p4a_config_t* src,
                                       p4a_config_t** out_dst);

/* ---- Setters ---- */
P4A_API p4a_status_t p4a_config_set_algorithm        (p4a_config_t*, p4a_algorithm_t);
P4A_API p4a_status_t p4a_config_set_solver           (p4a_config_t*, p4a_solver_t);
P4A_API p4a_status_t p4a_config_set_deflation        (p4a_config_t*, p4a_deflation_t);
/* `n_components` must be >= 1. There is no upper cap at the ABI level; if
 * the value exceeds rank(X) at fit time, Phase 1+ returns
 * P4A_ERR_INVALID_ARGUMENT from p4a_model_fit with a descriptive message. */
P4A_API p4a_status_t p4a_config_set_n_components     (p4a_config_t*, int32_t n);
P4A_API p4a_status_t p4a_config_set_center_x         (p4a_config_t*, int32_t enabled);
P4A_API p4a_status_t p4a_config_set_scale_x          (p4a_config_t*, int32_t enabled);
P4A_API p4a_status_t p4a_config_set_center_y         (p4a_config_t*, int32_t enabled);
P4A_API p4a_status_t p4a_config_set_scale_y          (p4a_config_t*, int32_t enabled);
P4A_API p4a_status_t p4a_config_set_tol              (p4a_config_t*, double tol);
/* `max_iter` must be >= 1. There is no upper cap at the ABI level. */
P4A_API p4a_status_t p4a_config_set_max_iter         (p4a_config_t*, int32_t max_iter);
P4A_API p4a_status_t p4a_config_set_store_scores     (p4a_config_t*, int32_t enabled);
P4A_API p4a_status_t p4a_config_set_store_diagnostics(p4a_config_t*, int32_t enabled);
P4A_API p4a_status_t p4a_config_set_dtype            (p4a_config_t*, p4a_dtype_t);

/* Composability hooks — these are non-owning pointers; the lifetime of the
 * pipeline / bank / strategy is managed by the caller. Passing NULL clears
 * the previously-set hook. */
P4A_API p4a_status_t p4a_config_set_pipeline         (p4a_config_t*,
                                                       const p4a_pipeline_t* pipe);
P4A_API p4a_status_t p4a_config_set_operator_bank    (p4a_config_t*,
                                                       const p4a_operator_bank_t* bank);
P4A_API p4a_status_t p4a_config_set_gating_strategy  (p4a_config_t*,
                                                       const p4a_gating_strategy_t* gate);

/* ---- Getters ---- */
P4A_API p4a_status_t p4a_config_get_algorithm        (const p4a_config_t*, p4a_algorithm_t*);
P4A_API p4a_status_t p4a_config_get_solver           (const p4a_config_t*, p4a_solver_t*);
P4A_API p4a_status_t p4a_config_get_deflation        (const p4a_config_t*, p4a_deflation_t*);
P4A_API p4a_status_t p4a_config_get_n_components     (const p4a_config_t*, int32_t*);
P4A_API p4a_status_t p4a_config_get_center_x         (const p4a_config_t*, int32_t*);
P4A_API p4a_status_t p4a_config_get_scale_x          (const p4a_config_t*, int32_t*);
P4A_API p4a_status_t p4a_config_get_center_y         (const p4a_config_t*, int32_t*);
P4A_API p4a_status_t p4a_config_get_scale_y          (const p4a_config_t*, int32_t*);
P4A_API p4a_status_t p4a_config_get_tol              (const p4a_config_t*, double*);
P4A_API p4a_status_t p4a_config_get_max_iter         (const p4a_config_t*, int32_t*);
P4A_API p4a_status_t p4a_config_get_store_scores     (const p4a_config_t*, int32_t*);
P4A_API p4a_status_t p4a_config_get_store_diagnostics(const p4a_config_t*, int32_t*);
P4A_API p4a_status_t p4a_config_get_dtype            (const p4a_config_t*, p4a_dtype_t*);
P4A_API p4a_status_t p4a_config_get_pipeline         (const p4a_config_t*,
                                                       const p4a_pipeline_t** out);
P4A_API p4a_status_t p4a_config_get_operator_bank    (const p4a_config_t*,
                                                       const p4a_operator_bank_t** out);
P4A_API p4a_status_t p4a_config_get_gating_strategy  (const p4a_config_t*,
                                                       const p4a_gating_strategy_t** out);

/* ============================================================================
 * 9. Operator bank, gating strategy, preprocessing pipeline
 * ========================================================================== */

typedef enum p4a_operator_kind_t {
    P4A_OP_IDENTITY            = 0,
    P4A_OP_CENTER              = 1,
    P4A_OP_AUTOSCALE           = 2,
    P4A_OP_PARETO_SCALE        = 3,
    P4A_OP_SNV                 = 4,
    P4A_OP_MSC                 = 5,
    P4A_OP_EMSC                = 6,
    P4A_OP_DETREND_POLY        = 7,
    P4A_OP_SAVGOL_SMOOTH       = 8,
    P4A_OP_SAVGOL_DERIVATIVE   = 9,
    P4A_OP_NORRIS_WILLIAMS     = 10,
    P4A_OP_ASLS_BASELINE       = 11,
    P4A_OP_OSC                 = 12,
    P4A_OP_EPO                 = 13,
    P4A_OP_WAVELET_DENOISE     = 14,
    P4A_OP_FINITE_DIFFERENCE   = 15,
    P4A_OP_WHITTAKER           = 16,
    P4A_OP_FCK                 = 17
} p4a_operator_kind_t;

/* Operator-bank lifecycle. An operator bank is an unordered collection of
 * preprocessing operators used by AOM-PLS (Phase 6) — the gating strategy
 * picks (soft / hard / sparse) which one(s) to apply per component.
 *
 * Memory: `params` (when non-NULL) is COPIED into the bank by `_add`. The
 * caller's buffer may be released immediately after the call returns. */
P4A_API p4a_status_t p4a_operator_bank_create(p4a_operator_bank_t** out);
P4A_API void         p4a_operator_bank_destroy(p4a_operator_bank_t* bank);
P4A_API p4a_status_t p4a_operator_bank_add   (p4a_operator_bank_t* bank,
                                               p4a_operator_kind_t kind,
                                               const double* params,
                                               int32_t n_params);
P4A_API p4a_status_t p4a_operator_bank_size  (const p4a_operator_bank_t* bank,
                                               int32_t* out_size);

/* ---- Gating strategy ---- */
typedef enum p4a_gating_mode_t {
    P4A_GATING_HARD            = 0,
    P4A_GATING_SOFT            = 1,
    P4A_GATING_SPARSE          = 2,
    P4A_GATING_PER_COMPONENT   = 3,    /* POP-PLS */
    P4A_GATING_PER_BLOCK       = 4,
    P4A_GATING_PER_TARGET      = 5
} p4a_gating_mode_t;

P4A_API p4a_status_t p4a_gating_strategy_create(p4a_gating_strategy_t** out,
                                                 p4a_gating_mode_t mode);
P4A_API void         p4a_gating_strategy_destroy(p4a_gating_strategy_t* gs);

/* ---- Preprocessing pipeline ----
 * Pipeline = ordered list of operators that share fit/transform.
 * Phase 3a/3b/3c/3d/3e/3f/3g/3h/3i/3j implements identity, center, autoscale,
 * Pareto scaling, SNV, MSC, EMSC, polynomial detrending, Savitzky-Golay
 * smoothing/derivatives, Norris-Williams gap-segment derivatives, ASLS
 * baseline correction, Haar wavelet denoising and supervised one-component
 * OSC / EPO with leakage-safe training statistics.
 * Unsupported operators are accepted by add_operator so pipelines remain
 * serialisable later, but fit returns P4A_ERR_NOT_IMPLEMENTED until the
 * operator body lands.
 */
P4A_API p4a_status_t p4a_pipeline_create        (p4a_pipeline_t** out);
P4A_API void         p4a_pipeline_destroy       (p4a_pipeline_t* pipe);
P4A_API p4a_status_t p4a_pipeline_clone         (const p4a_pipeline_t* src,
                                                  p4a_pipeline_t** out_dst);
/* `params` (when non-NULL) is COPIED into the pipeline by `_add_operator`;
 * the caller may release the buffer immediately afterwards. */
P4A_API p4a_status_t p4a_pipeline_add_operator  (p4a_pipeline_t* pipe,
                                                  p4a_operator_kind_t kind,
                                                  const double* params,
                                                  int32_t n_params);
P4A_API p4a_status_t p4a_pipeline_size          (const p4a_pipeline_t* pipe,
                                                  int32_t* out_size);

/* `Y` is optional for unsupervised preprocessing operators and may be NULL.
 * Supervised operators such as OSC and EPO require a non-NULL Y at fit time. */
P4A_API p4a_status_t p4a_pipeline_fit            (p4a_context_t* ctx,
                                                   p4a_pipeline_t* pipe,
                                                   const p4a_matrix_view_t* X,
                                                   const p4a_matrix_view_t* Y);
P4A_API p4a_status_t p4a_pipeline_transform      (p4a_context_t* ctx,
                                                   const p4a_pipeline_t* pipe,
                                                   const p4a_matrix_view_t* X,
                                                   p4a_matrix_view_t* out);
P4A_API p4a_status_t p4a_pipeline_transform_alloc(p4a_context_t* ctx,
                                                   const p4a_pipeline_t* pipe,
                                                   const p4a_matrix_view_t* X,
                                                   p4a_array_t** out);

/* ============================================================================
 * 10. Model lifecycle
 * ==========================================================================
 *
 * The current core implements P4A_ALGO_PLS_REGRESSION with P4A_SOLVER_NIPALS,
 * P4A_SOLVER_ORTHOGONAL_SCORES, P4A_SOLVER_SIMPLS,
 * P4A_SOLVER_KERNEL_ALGORITHM, P4A_SOLVER_WIDE_KERNEL, P4A_SOLVER_SVD,
 * P4A_SOLVER_POWER or P4A_SOLVER_RANDOMIZED_SVD using
 * P4A_DEFLATION_REGRESSION; P4A_ALGO_PLS_CANONICAL with P4A_SOLVER_NIPALS or
 * P4A_SOLVER_SVD using P4A_DEFLATION_CANONICAL; P4A_ALGO_PLS_SVD with
 * P4A_SOLVER_SVD using P4A_DEFLATION_CANONICAL; P4A_ALGO_PLS_DA with the
 * PLS regression solver set using P4A_DEFLATION_REGRESSION; P4A_ALGO_OPLS
 * and P4A_ALGO_OPLS_DA with one or more Y targets, P4A_SOLVER_NIPALS and
 * P4A_DEFLATION_ORTHOGONAL; and P4A_ALGO_PCR with P4A_SOLVER_SVD using
 * P4A_DEFLATION_REGRESSION. Multi-response OPLS uses one shared predictive
 * score direction from the dominant singular pair of X.T @ Y. PLSSVD
 * transform returns direct X scores from SVD(X.T @ Y); predict uses the
 * deterministic latent projection X @ W @ V.T scaled back to Y.
 * Unsupported algorithms / solvers / deflation modes return
 * P4A_ERR_UNSUPPORTED. Phase 6 adds AOM-PLS through the operator_bank +
 * gating_strategy hooks set on the config.
 */
P4A_API p4a_status_t p4a_model_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    p4a_model_t** out_model);

P4A_API void         p4a_model_destroy(p4a_model_t* model);

/* ---- Predict / transform (caller-provided output buffer) ---- */
P4A_API p4a_status_t p4a_model_predict(
    p4a_context_t* ctx, const p4a_model_t* model,
    const p4a_matrix_view_t* X, p4a_matrix_view_t* out);

P4A_API p4a_status_t p4a_model_transform(
    p4a_context_t* ctx, const p4a_model_t* model,
    const p4a_matrix_view_t* X, p4a_matrix_view_t* out_scores);

/* ---- Predict / transform (core-allocated output) ---- */
P4A_API p4a_status_t p4a_model_predict_alloc(
    p4a_context_t* ctx, const p4a_model_t* model,
    const p4a_matrix_view_t* X, p4a_array_t** out);

P4A_API p4a_status_t p4a_model_transform_alloc(
    p4a_context_t* ctx, const p4a_model_t* model,
    const p4a_matrix_view_t* X, p4a_array_t** out_scores);

/* Tagged accessor for fitted-model arrays. */
typedef enum p4a_model_array_t {
    P4A_MODEL_COEFFICIENTS = 0,   /* (n_features, n_targets) */
    P4A_MODEL_INTERCEPT    = 1,   /* (n_targets,) */
    P4A_MODEL_X_MEAN       = 2,
    P4A_MODEL_X_SCALE      = 3,
    P4A_MODEL_Y_MEAN       = 4,
    P4A_MODEL_Y_SCALE      = 5,
    P4A_MODEL_WEIGHTS_W    = 6,   /* (n_features, n_components) */
    P4A_MODEL_LOADINGS_P   = 7,
    P4A_MODEL_Y_LOADINGS_Q = 8,
    P4A_MODEL_ROTATIONS_R  = 9,
    P4A_MODEL_SCORES_T     = 10,  /* optional — store_scores=1 */
    P4A_MODEL_Y_SCORES_U   = 11   /* optional — store_scores=1 */
} p4a_model_array_t;

P4A_API p4a_status_t p4a_model_get_array(
    p4a_context_t* ctx, const p4a_model_t* model,
    p4a_model_array_t which, p4a_array_t** out);

P4A_API p4a_status_t p4a_model_get_n_components(const p4a_model_t*, int32_t* out);
P4A_API p4a_status_t p4a_model_get_n_features  (const p4a_model_t*, int32_t* out);
P4A_API p4a_status_t p4a_model_get_n_targets   (const p4a_model_t*, int32_t* out);

/* ============================================================================
 * 11. Serialization
 * ==========================================================================
 *
 * The binary format starts with the four-byte magic "P4AM" (the first 4
 * bytes of P4A_SERIALIZATION_MAGIC — note that as a string literal that
 * macro is 5 bytes including the trailing NUL; consumers should compare or
 * memcpy exactly 4 bytes), followed by P4A_SERIALIZATION_FORMAT_VERSION
 * (4-byte LE) and the producing library's ABI version triple (3 × 4-byte
 * LE). The consumer rejects on P4A_SERIALIZATION_FORMAT_VERSION mismatch
 * with P4A_ERR_VERSION_INCOMPATIBLE and writes a warning to last_error if
 * the ABI version differs but the format is compatible. All multi-byte
 * integers are little-endian; all doubles are IEEE 754 binary64 LE.
 */

P4A_API p4a_status_t p4a_model_export_size(
    const p4a_model_t* model, size_t* out_size);

P4A_API p4a_status_t p4a_model_export_to_buffer(
    const p4a_model_t* model, void* buffer, size_t buffer_size,
    size_t* out_written);

P4A_API p4a_status_t p4a_model_import_from_buffer(
    p4a_context_t* ctx, const void* buffer, size_t buffer_size,
    p4a_model_t** out_model);

/* Inspect a buffer's header without fully importing it. Useful for ABI /
 * format compatibility checks before allocating model memory. */
P4A_API p4a_status_t p4a_serialization_inspect(
    const void* buffer, size_t buffer_size,
    uint32_t* out_format_version,
    uint32_t* out_writer_abi_major,
    uint32_t* out_writer_abi_minor,
    uint32_t* out_writer_abi_patch);

/* ============================================================================
 * 12. ABI version + build info
 * ========================================================================== */

P4A_API uint32_t     p4a_get_abi_version_major(void);
P4A_API uint32_t     p4a_get_abi_version_minor(void);
P4A_API uint32_t     p4a_get_abi_version_patch(void);
P4A_API uint32_t     p4a_get_abi_version_int(void);   /* MAJOR*10000 + MINOR*100 + PATCH */
P4A_API const char*  p4a_get_version_string(void);    /* e.g. "0.72.0+abi.1.3.0" */
P4A_API const char*  p4a_get_build_info(void);        /* compiler / flags / backends */
P4A_API const char*  p4a_get_git_revision(void);      /* git rev at build time, or "" */

/* Returns P4A_OK if the header's ABI MAJOR == library MAJOR and the header's
 * MINOR <= library MINOR (forward compat). Returns P4A_ERR_ABI_MISMATCH on
 * MAJOR mismatch, P4A_ERR_VERSION_INCOMPATIBLE on MINOR-too-high. */
P4A_API p4a_status_t p4a_check_abi_compatibility(
    uint32_t header_abi_major,
    uint32_t header_abi_minor);

/* ============================================================================
 * 13. Output array helper — core-allocated arrays returned by *_alloc calls
 * ==========================================================================
 *
 * p4a_array_t is a non-opaque concept (you read its dtype/shape and obtain a
 * view), but the only ways to construct one are through library functions
 * such as `p4a_model_predict_alloc` and `p4a_pipeline_transform_alloc`.
 * Callers must release it with p4a_array_free. AOM/POP selection (§15)
 * exposes its arrays through result-handle accessors that point into
 * result-owned storage instead.
 */
P4A_API p4a_status_t p4a_array_dtype(const p4a_array_t* arr, p4a_dtype_t* out);
P4A_API p4a_status_t p4a_array_shape(const p4a_array_t* arr,
                                      int64_t* rows, int64_t* cols);
P4A_API p4a_status_t p4a_array_view (const p4a_array_t* arr,
                                      p4a_matrix_view_t* out);
P4A_API void         p4a_array_free (p4a_array_t* arr);

/* ============================================================================
 * 14. Validation plan — caller-built train/test fold list for selection
 * ==========================================================================
 *
 * A p4a_validation_plan_t carries the per-fold train and test sample indices
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
P4A_API p4a_status_t p4a_validation_plan_create(p4a_validation_plan_t** out);
P4A_API void         p4a_validation_plan_destroy(p4a_validation_plan_t* plan);

/* Sets the sample count the plan refers to. `n_samples` must be >= 0; a
 * non-positive value is allowed at construction (treated as "unset") so
 * bindings may set folds before learning the sample count, but selection
 * callers must set a value of at least 2 before invoking selection. Returns
 * P4A_ERR_NULL_POINTER if `plan` is NULL, P4A_ERR_INVALID_ARGUMENT if
 * `n_samples < 0`. */
P4A_API p4a_status_t p4a_validation_plan_set_n_samples(
    p4a_validation_plan_t* plan, int64_t n_samples);

/* Appends one fold. `n_train` and `n_test` must both be >= 1. Index buffers
 * must be non-NULL. Indices are not validated here against `n_samples`;
 * semantic checks (range, duplicates, train/test overlap) are run when a
 * selection kernel consumes the plan. The plan is left UNCHANGED on failure.
 * Returns P4A_ERR_NULL_POINTER for NULL plan or NULL index buffer, and
 * P4A_ERR_INVALID_ARGUMENT when n_train or n_test is zero or negative. */
P4A_API p4a_status_t p4a_validation_plan_add_fold(
    p4a_validation_plan_t* plan,
    const int64_t* train_indices, int64_t n_train,
    const int64_t* test_indices,  int64_t n_test);

P4A_API p4a_status_t p4a_validation_plan_get_n_samples(
    const p4a_validation_plan_t* plan, int64_t* out_n_samples);
P4A_API p4a_status_t p4a_validation_plan_get_n_folds(
    const p4a_validation_plan_t* plan, int32_t* out_n_folds);

/* ============================================================================
 * 15. AOM / POP selection -- public C ABI for the internal AOM-SIMPLS selector
 * ==========================================================================
 *
 * AOM-PLS picks one operator (global) or one operator per latent component
 * (POP) from a p4a_operator_bank_t, scored under a caller-provided
 * p4a_validation_plan_t. Phase 6 ships strict-linear AOM operators with the
 * SIMPLS solver in P4A_DEFLATION_REGRESSION; other algorithm/solver/
 * deflation combinations return P4A_ERR_UNSUPPORTED. Both kernels require
 * a non-empty bank of strict-linear operators (see P4A_OP_IDENTITY,
 * P4A_OP_DETREND_POLY, P4A_OP_SAVGOL_SMOOTH, P4A_OP_SAVGOL_DERIVATIVE,
 * P4A_OP_NORRIS_WILLIAMS, P4A_OP_FINITE_DIFFERENCE, P4A_OP_WHITTAKER,
 * P4A_OP_FCK). Non-strict operators (e.g. P4A_OP_SNV) return
 * P4A_ERR_UNSUPPORTED.
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
 *   - Empty operator bank or empty plan -> P4A_ERR_INVALID_ARGUMENT.
 *   - Plan sample count not matching X.rows -> P4A_ERR_SHAPE_MISMATCH.
 *   - Duplicate, out-of-range, or train/test-overlapping fold indices
 *     -> P4A_ERR_INVALID_ARGUMENT.
 *   - max_components < 1, max_components > X.cols, or max_components >= X.rows
 *     -> P4A_ERR_INVALID_ARGUMENT.
 *   - Unsupported solver/algorithm/deflation -> P4A_ERR_UNSUPPORTED.
 *   - Non-strict operator in the bank -> P4A_ERR_UNSUPPORTED.
 *   - X.rows != Y.rows -> P4A_ERR_SHAPE_MISMATCH.
 */

P4A_API p4a_status_t p4a_aom_global_select(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_operator_bank_t* bank,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    const p4a_validation_plan_t* plan,
    int32_t max_components,
    p4a_aom_global_result_t** out_result);

P4A_API void p4a_aom_global_result_destroy(p4a_aom_global_result_t* result);

P4A_API p4a_status_t p4a_aom_global_result_get_n_operators(
    const p4a_aom_global_result_t* result, int32_t* out);
P4A_API p4a_status_t p4a_aom_global_result_get_max_components(
    const p4a_aom_global_result_t* result, int32_t* out);
P4A_API p4a_status_t p4a_aom_global_result_get_selected_operator_index(
    const p4a_aom_global_result_t* result, int32_t* out);
P4A_API p4a_status_t p4a_aom_global_result_get_selected_n_components(
    const p4a_aom_global_result_t* result, int32_t* out);
P4A_API p4a_status_t p4a_aom_global_result_get_best_score(
    const p4a_aom_global_result_t* result, double* out);

P4A_API p4a_status_t p4a_aom_global_result_get_operator_kinds(
    const p4a_aom_global_result_t* result,
    const p4a_operator_kind_t** out_data, int32_t* out_size);
P4A_API p4a_status_t p4a_aom_global_result_get_operator_scores(
    const p4a_aom_global_result_t* result,
    const double** out_data, int32_t* out_size);
P4A_API p4a_status_t p4a_aom_global_result_get_rmse_curves(
    const p4a_aom_global_result_t* result,
    const double** out_data, int32_t* out_rows, int32_t* out_cols);
P4A_API p4a_status_t p4a_aom_global_result_get_predictions(
    const p4a_aom_global_result_t* result,
    const double** out_data, int64_t* out_rows, int64_t* out_cols);

P4A_API p4a_status_t p4a_aom_per_component_select(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_operator_bank_t* bank,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    const p4a_validation_plan_t* plan,
    int32_t max_components,
    p4a_aom_per_component_result_t** out_result);

P4A_API void p4a_aom_per_component_result_destroy(
    p4a_aom_per_component_result_t* result);

P4A_API p4a_status_t p4a_aom_per_component_result_get_n_operators(
    const p4a_aom_per_component_result_t* result, int32_t* out);
P4A_API p4a_status_t p4a_aom_per_component_result_get_max_components(
    const p4a_aom_per_component_result_t* result, int32_t* out);
P4A_API p4a_status_t p4a_aom_per_component_result_get_selected_n_components(
    const p4a_aom_per_component_result_t* result, int32_t* out);
P4A_API p4a_status_t p4a_aom_per_component_result_get_best_score(
    const p4a_aom_per_component_result_t* result, double* out);

P4A_API p4a_status_t p4a_aom_per_component_result_get_operator_kinds(
    const p4a_aom_per_component_result_t* result,
    const p4a_operator_kind_t** out_data, int32_t* out_size);
P4A_API p4a_status_t p4a_aom_per_component_result_get_selected_operator_indices(
    const p4a_aom_per_component_result_t* result,
    const int32_t** out_data, int32_t* out_size);
P4A_API p4a_status_t p4a_aom_per_component_result_get_component_scores(
    const p4a_aom_per_component_result_t* result,
    const double** out_data, int32_t* out_rows, int32_t* out_cols);
P4A_API p4a_status_t p4a_aom_per_component_result_get_prefix_scores(
    const p4a_aom_per_component_result_t* result,
    const double** out_data, int32_t* out_size);
P4A_API p4a_status_t p4a_aom_per_component_result_get_predictions(
    const p4a_aom_per_component_result_t* result,
    const double** out_data, int64_t* out_rows, int64_t* out_cols);

/* ============================================================================
 * 16. Method-result handle — universal output container for the extra
 *     PLS / monitoring / diagnostics / ensemble fit functions
 * ==========================================================================
 *
 * p4a_method_result_t holds named double matrices, named int32 vectors and
 * named scalars. The fit functions below build one of these per call and
 * return it to the caller. Accessor functions look up outputs by name; the
 * caller must NOT free returned buffer pointers (they point into the
 * result-owned storage and are invalidated when the result is destroyed).
 */

P4A_API void p4a_method_result_destroy(p4a_method_result_t* result);

/* Read a named double matrix. Returns P4A_ERR_INVALID_ARGUMENT when the name is
 * absent. */
P4A_API p4a_status_t p4a_method_result_get_double_matrix(
    const p4a_method_result_t* result,
    const char* name,
    const double** out_data, int64_t* out_rows, int64_t* out_cols);

/* Read a named int32 vector. Returns P4A_ERR_INVALID_ARGUMENT when the name is
 * absent. */
P4A_API p4a_status_t p4a_method_result_get_int_vector(
    const p4a_method_result_t* result,
    const char* name,
    const int32_t** out_data, int32_t* out_size);

/* Read a named scalar. Returns P4A_ERR_INVALID_ARGUMENT when the name is absent. */
P4A_API p4a_status_t p4a_method_result_get_scalar(
    const p4a_method_result_t* result,
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
P4A_API p4a_status_t p4a_sparse_simpls_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    double sparsity_lambda,
    p4a_method_result_t** out_result);

/* Domain-invariant PLS (§13). Penalizes the SIMPLS direction's alignment
 * with the source-vs-target mean difference. The result contains:
 *   "coefficients"        (n_features x n_targets)
 *   "predictions"         (n_samples  x n_targets)
 *   "x_mean", "y_mean"
 *   scalar "rmse_source"  in-sample RMSE on source data
 */
P4A_API p4a_status_t p4a_di_pls_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X_source,
    const p4a_matrix_view_t* Y_source,
    const p4a_matrix_view_t* X_target,
    double di_lambda,
    p4a_method_result_t** out_result);

/* Recursive (moving-window) PLS (§12). Predicts each sample at index >=
 * window_size from a SIMPLS model fitted on the previous window_size rows.
 * The result contains:
 *   "predictions" (n_samples x n_targets) — zeros for warmup samples
 *   int "in_window" (n_samples)           — 1 when predicted, 0 warmup
 *   scalar "rmse_predictable" on samples i >= window_size
 */
P4A_API p4a_status_t p4a_recursive_pls_run(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    int32_t window_size,
    p4a_method_result_t** out_result);

/* CPPLS — Canonical Powered PLS (§1). Each X column is rescaled by its
 * std^gamma before SIMPLS, then coefficients are unscaled. gamma in [0, 1]
 * interpolates between PLS (gamma=0) and a power-rescaled flavour
 * (gamma=1). The result contains:
 *   "coefficients" (n_features x n_targets, row-major)
 *   "predictions"  (n_samples  x n_targets)
 *   "x_mean", "y_mean"
 *   scalar "rmse", scalar "gamma"
 */
P4A_API p4a_status_t p4a_cppls_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    double gamma,
    p4a_method_result_t** out_result);

/* Weighted PLS (§26). Pre-multiplies the centered (X, Y) rows by
 * sqrt(sample_weights[i]) before SIMPLS. `weights` must point to
 * `n_samples` strictly positive finite doubles. The result contains:
 *   "coefficients", "predictions", "x_mean", "y_mean"
 *   scalar "rmse"
 */
P4A_API p4a_status_t p4a_weighted_pls_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    const double* sample_weights,
    int64_t sample_weights_size,
    p4a_method_result_t** out_result);

/* Robust PLS via IRLS with Huber weights (§26). max_irls_iter <= 0 falls
 * back to 5. The result contains:
 *   "coefficients", "predictions", "x_mean", "y_mean"
 *   "final_weights" (1 x n_samples) — Huber weights at convergence
 *   scalar "rmse", scalar "huber_k"
 */
P4A_API p4a_status_t p4a_robust_pls_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    double huber_k,
    int32_t max_irls_iter,
    p4a_method_result_t** out_result);

/* Ridge-augmented PLS (§26). Augments (X, Y) with sqrt(ridge_lambda)*I and
 * 0 rows, then runs SIMPLS. ridge_lambda must be >= 0. The result contains:
 *   "coefficients", "predictions", "x_mean", "y_mean"
 *   scalar "rmse", scalar "ridge_lambda"
 */
P4A_API p4a_status_t p4a_ridge_pls_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    double ridge_lambda,
    p4a_method_result_t** out_result);

/* Continuum regression (§26). tau in [0, 1] interpolates between PLS
 * (tau=0) and a whitened-X / OLS-like flavour (tau=1). The result contains:
 *   "coefficients", "predictions", "x_mean", "y_mean"
 *   scalar "rmse", scalar "tau"
 */
P4A_API p4a_status_t p4a_continuum_regression_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    double tau,
    p4a_method_result_t** out_result);

#ifdef __cplusplus
}  /* extern "C" */
#endif

/* ============================================================================
 * 16. ABI guard rails — fixed-size assertions on the C ABI shape
 * ==========================================================================
 *
 * Compilers may shrink enums under non-default flags (e.g. -fshort-enums on
 * gcc/clang). Because enums appear in function signatures and in the
 * p4a_matrix_view_t struct, that would silently break the ABI. We assert at
 * compile time that every public enum is sizeof(int) = 4 bytes, and that
 * p4a_matrix_view_t has its documented 48-byte LP64 layout.
 *
 * If any of these fail, the build fails fast — and the caller knows to use
 * a default-enum-size toolchain.
 */
#ifdef __cplusplus
#  define P4A_STATIC_ASSERT(cond, msg) static_assert((cond), msg)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#  define P4A_STATIC_ASSERT(cond, msg) _Static_assert((cond), msg)
#else
#  define P4A_STATIC_ASSERT(cond, msg) typedef char p4a_sa_##__LINE__[(cond) ? 1 : -1]
#endif

P4A_STATIC_ASSERT(sizeof(p4a_status_t)        == 4, "p4a_status_t must be 4 bytes");
P4A_STATIC_ASSERT(sizeof(p4a_backend_t)       == 4, "p4a_backend_t must be 4 bytes");
P4A_STATIC_ASSERT(sizeof(p4a_dtype_t)         == 4, "p4a_dtype_t must be 4 bytes");
P4A_STATIC_ASSERT(sizeof(p4a_algorithm_t)     == 4, "p4a_algorithm_t must be 4 bytes");
P4A_STATIC_ASSERT(sizeof(p4a_solver_t)        == 4, "p4a_solver_t must be 4 bytes");
P4A_STATIC_ASSERT(sizeof(p4a_deflation_t)     == 4, "p4a_deflation_t must be 4 bytes");
P4A_STATIC_ASSERT(sizeof(p4a_operator_kind_t) == 4, "p4a_operator_kind_t must be 4 bytes");
P4A_STATIC_ASSERT(sizeof(p4a_gating_mode_t)   == 4, "p4a_gating_mode_t must be 4 bytes");
P4A_STATIC_ASSERT(sizeof(p4a_model_array_t)   == 4, "p4a_model_array_t must be 4 bytes");

/* p4a_matrix_view_t must be 48 bytes on LP64 / LLP64. ILP32 is not
 * supported by Phase 0; on ILP32 platforms the layout would differ
 * (pointer is 4 bytes), which is why we exclude armeabi-v7a. */
#if defined(__LP64__) || defined(_WIN64)
P4A_STATIC_ASSERT(sizeof(p4a_matrix_view_t) == 48,
                  "p4a_matrix_view_t layout must be 48 bytes on LP64/LLP64");
#endif

#endif /* PLS4ALL_P4A_H */
