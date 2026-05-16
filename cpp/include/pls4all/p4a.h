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
P4A_API const char*  p4a_get_version_string(void);    /* e.g. "0.97.0+abi.1.14.0" */
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

/* Read a named int64 vector. Returns P4A_ERR_INVALID_ARGUMENT when the name is
 * absent. Used by the §5 / §6 selectors to expose `selected_indices`,
 * `removed_indices`, etc. without losing precision for large feature counts. */
P4A_API p4a_status_t p4a_method_result_get_int64_vector(
    const p4a_method_result_t* result,
    const char* name,
    const int64_t** out_data, int64_t* out_size);

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
P4A_API p4a_status_t p4a_n_pls_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X_flat,
    int32_t mode_j,
    int32_t mode_k,
    const p4a_matrix_view_t* Y,
    p4a_method_result_t** out_result);

/* Non-linear kernel PLS (§24). `kernel_type`:
 *   0 = linear, 1 = RBF, 2 = polynomial, 3 = sigmoid
 * `gamma`, `coef0`, `degree` are kernel parameters; ignored when not
 * applicable to the selected kernel. The result contains:
 *   "predictions"   (n_samples x n_targets)  in-sample predictions
 *   "alpha"         (n_samples x n_targets)  dual coefficients
 *   "y_mean"        (1 x n_targets)
 *   scalar "rmse", scalar "kernel_type" (as double)
 */
P4A_API p4a_status_t p4a_kernel_pls_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    int32_t kernel_type,
    double gamma,
    double coef0,
    int32_t degree,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    p4a_method_result_t** out_result);

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
P4A_API p4a_status_t p4a_o2pls_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    int32_t n_predictive,
    int32_t n_x_orthogonal,
    int32_t n_y_orthogonal,
    p4a_method_result_t** out_result);

/* Approximate-PRESS component selection (§29). For each component count
 * k in [1, max_components], fits SIMPLS, then approximates PRESS via
 * leverage-inflated in-sample residuals. The result contains:
 *   "press_per_component" (1 x max_components)
 *   "rmse_per_component"  (1 x max_components)
 *   int "selected_n_components" — argmin of press_per_component
 *   scalar "selected_n_components_d" — same as a double for convenience
 */
P4A_API p4a_status_t p4a_approximate_press_compute(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    int32_t max_components,
    p4a_method_result_t** out_result);

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
P4A_API p4a_status_t p4a_sparse_pls_da_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const int32_t* y_labels,
    int64_t y_labels_size,
    p4a_method_result_t** out_result);

/* Group sparse PLS (§7). Soft-thresholds groups of features together.
 * `group_assignment` is a length-n_features buffer of 0-based group IDs.
 * Result contains coefficients, predictions, x_mean, y_mean, rmse, and
 * scalar `n_groups`.
 */
P4A_API p4a_status_t p4a_group_sparse_pls_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    const int32_t* group_assignment,
    int64_t group_assignment_size,
    double group_lambda,
    p4a_method_result_t** out_result);

/* Fused sparse PLS (§7). Combines L1 sparsity with fusion of
 * consecutive feature pairs. Result identical to group_sparse_pls_fit
 * plus scalars `l1_lambda` and `fusion_lambda`.
 */
P4A_API p4a_status_t p4a_fused_sparse_pls_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    double l1_lambda,
    double fusion_lambda,
    p4a_method_result_t** out_result);

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
P4A_API p4a_status_t p4a_pls_monitoring_run(
    p4a_context_t* ctx,
    const p4a_model_t* model,
    const p4a_matrix_view_t* X_reference,
    const p4a_matrix_view_t* X_monitor,
    double alpha,
    p4a_method_result_t** out_result);

/* One-SE rule for PLS component selection (§10). Given a (max_components
 * x n_folds) row-major matrix of fold RMSE values, returns the smallest
 * k whose mean fold RMSE is within one standard error of the best mean
 * fold RMSE. Result contains:
 *   "mean_rmse_per_component" (1 x max_components)
 *   int "best_n_components"       (length 1)
 *   int "one_se_n_components"     (length 1)
 *   scalar "one_se_standard_error", scalar "one_se_threshold"
 */
P4A_API p4a_status_t p4a_one_se_rule_compute(
    p4a_context_t* ctx,
    const double* fold_rmse_matrix,
    int32_t max_components,
    int32_t n_folds,
    p4a_method_result_t** out_result);

/* SO-PLS (§17). Sequential and Orthogonalized PLS for B X-blocks
 * predicting one Y. `X_blocks` is an array of `n_blocks`
 * p4a_matrix_view_t structs (all sharing X.rows = Y.rows).
 * `n_components_per_block` is a length-n_blocks int32 array.
 * The result contains:
 *   "predictions" (n_samples x n_targets)
 *   "y_mean"      (1 x n_targets)
 *   For each block b: "block_coefficients_<b>" of shape (p_b x n_targets)
 *   scalar "n_blocks"
 */
P4A_API p4a_status_t p4a_so_pls_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X_blocks,
    int32_t n_blocks,
    const p4a_matrix_view_t* Y,
    const int32_t* n_components_per_block,
    int64_t n_components_per_block_size,
    p4a_method_result_t** out_result);

/* OnPLS (§18). Generalized orthogonal projection for multi-block PLS.
 * Removes joint and per-block unique components.
 * Result contains scalars `n_blocks` and `n_joint`, plus per-block
 * loading matrices "joint_loadings_<b>", "unique_loadings_<b>" and
 * "joint_scores_<b>".
 */
P4A_API p4a_status_t p4a_on_pls_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X_blocks,
    int32_t n_blocks,
    int32_t n_joint,
    const int32_t* n_unique_per_block,
    int64_t n_unique_per_block_size,
    p4a_method_result_t** out_result);

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
P4A_API p4a_status_t p4a_rosa_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X_blocks,
    int32_t n_blocks,
    const p4a_matrix_view_t* Y,
    int32_t n_components,
    p4a_method_result_t** out_result);

/* Bagging PLS (§20). Bootstrap aggregation of `n_estimators` PLS
 * regressors with the configured seed. Returns the average regression
 * coefficient matrix:
 *   "coefficients"   (n_features x n_targets)
 *   "predictions"    (n_samples x n_targets)
 *   "x_mean", "y_mean"
 *   scalar "rmse", scalar "n_estimators"
 */
P4A_API p4a_status_t p4a_bagging_pls_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    int32_t n_estimators,
    uint64_t seed,
    p4a_method_result_t** out_result);

/* Boosting PLS (§20). Gradient-boosting style stage-wise refit of
 * `n_estimators` PLS regressors with a per-stage `learning_rate`.
 * Output shape identical to bagging_pls_fit.
 */
P4A_API p4a_status_t p4a_boosting_pls_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    int32_t n_estimators,
    double learning_rate,
    p4a_method_result_t** out_result);

/* Random-subspace PLS (§20). Each of `n_estimators` PLS regressors is
 * fit on a random subset of `features_per_subspace` columns. The
 * result averages predictions over the missing columns by zero-padding
 * coefficients; output shape identical to bagging_pls_fit plus a
 * scalar `features_per_subspace`.
 */
P4A_API p4a_status_t p4a_random_subspace_pls_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    int32_t n_estimators,
    int32_t features_per_subspace,
    uint64_t seed,
    p4a_method_result_t** out_result);

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
 * Returns P4A_ERR_INVALID_ARGUMENT for non-positive length_scale or
 * noise_level, or for n_components outside [1, min(n,p)].
 * Returns P4A_ERR_SHAPE_MISMATCH if Y has more than one column.
 * Returns P4A_ERR_NUMERICAL_FAILURE if the Cholesky of (K + noise*I)
 * fails (typically at very low noise with redundant score directions).
 */
P4A_API p4a_status_t p4a_gpr_pls_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    double length_scale,
    double noise_level,
    uint64_t seed,
    p4a_method_result_t** out_result);

/* Simplified SIMPLS PLS regression — raw-pointer signature for
 * language bindings whose FFI layer struggles with the
 * `p4a_matrix_view_t*` parameter pattern (notably Emscripten 5.0.7
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
 * Returns P4A_OK on success or a P4A_ERR_* status. This is a stable
 * additive helper — implementations may grow new variants but the
 * shape of this one is fixed at ABI minor 1.13.
 */
P4A_API p4a_status_t p4a_pls_fit_simple(
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
P4A_API p4a_status_t p4a_pls_glm_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    int32_t poisson,
    p4a_method_result_t** out_result);

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
P4A_API p4a_status_t p4a_pls_qda_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const int32_t* y_labels,
    int64_t y_labels_size,
    p4a_method_result_t** out_result);

/* PLS-Cox (§5). PLS-reduced Cox proportional hazards with Breslow
 * baseline hazard. `survival_times` is the observed time, and
 * `event_indicators` is 1 (event) or 0 (censored). The result contains:
 *   "coefficients"     (n_features x 1)  linear predictor coefficients
 *   "baseline_hazard"  (1 x n_unique_event_times)
 *   "event_times"      (1 x n_unique_event_times)
 *   "x_mean"
 *   "predictions"      (n_samples x 1)  linear-predictor scores
 */
P4A_API p4a_status_t p4a_pls_cox_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const double* survival_times,
    int64_t survival_times_size,
    const int32_t* event_indicators,
    int64_t event_indicators_size,
    p4a_method_result_t** out_result);

/* PDS — Piecewise Direct Standardization (§13). Fits per-target-column
 * windowed least-squares maps from source instrument X_source to target
 * X_target. The result contains:
 *   "transformation"  (n_features_target x n_features_source)
 *   "predictions"     (n_samples x n_features_target) — X_source @ T^T
 *   scalar "rmse" — frobenius error vs X_target
 *   scalar "window_half_width"
 */
P4A_API p4a_status_t p4a_pds_fit(
    p4a_context_t* ctx,
    const p4a_matrix_view_t* X_source,
    const p4a_matrix_view_t* X_target,
    int32_t window_half_width,
    p4a_method_result_t** out_result);

/* DS — Direct Standardization (§13). Fits a full pt × ps least-squares
 * map from centered source X to centered target X plus a bias. The
 * result contains:
 *   "transformation" (n_features_source x n_features_target)
 *   "bias"           (1 x n_features_target)
 *   "predictions"    (n_samples x n_features_target)
 *   scalar "rmse"
 */
P4A_API p4a_status_t p4a_ds_fit(
    p4a_context_t* ctx,
    const p4a_matrix_view_t* X_source,
    const p4a_matrix_view_t* X_target,
    p4a_method_result_t** out_result);

/* MIR-PLS — Multiple Inverse Regression PLS (§13). Inverts the X→Y
 * relationship by running SIMPLS on (Y, X) and pseudoinverting the
 * resulting Y→X coefficients to obtain X→Y prediction coefficients.
 * The result contains:
 *   "coefficients"  (n_features x n_targets)
 *   "predictions"   (n_samples x n_targets)
 *   "x_mean", "y_mean"
 *   scalar "rmse"
 */
P4A_API p4a_status_t p4a_mir_pls_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    p4a_method_result_t** out_result);

/* Missing-aware NIPALS (§13). Same shape as a regular PLS regression
 * model but tolerates NaN entries in X (replaced with the current
 * latent-space iterate during NIPALS). The result contains:
 *   "coefficients", "predictions", "x_mean", "y_mean"
 *   scalar "rmse"
 */
P4A_API p4a_status_t p4a_missing_aware_nipals_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    p4a_method_result_t** out_result);

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
P4A_API p4a_status_t p4a_pls_diagnostics_compute(
    p4a_context_t* ctx,
    const p4a_model_t* model,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* X_reference,
    p4a_method_result_t** out_result);

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
P4A_API p4a_status_t p4a_mb_pls_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    const int64_t* block_sizes,
    int64_t n_blocks,
    p4a_method_result_t** out_result);

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
P4A_API p4a_status_t p4a_lw_pls_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    int32_t n_neighbors,
    p4a_method_result_t** out_result);

/* PLS-LDA — Linear Discriminant Analysis on PLS scores (Phase 4p). Result
 * keys:
 *   "decision_scores"   (n × n_classes)
 *   int "predictions"   (n)
 *   scalar "n_classes", scalar "n_components"
 */
P4A_API p4a_status_t p4a_pls_lda_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const int32_t* y_labels,
    int64_t y_labels_size,
    int32_t n_classes,
    p4a_method_result_t** out_result);

/* PLS-Logistic — multinomial logistic regression on PLS scores (Phase 4q).
 * Result keys:
 *   "decision_scores"  (n × n_classes)
 *   "probabilities"    (n × n_classes)
 *   "intercepts"       (1 × (n_classes - 1))
 *   "coefficients"     ((n_classes - 1) × n_components)
 *   int "predictions"  (n)
 *   scalar "n_classes", scalar "n_components"
 */
P4A_API p4a_status_t p4a_pls_logistic_fit(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const int32_t* y_labels,
    int64_t y_labels_size,
    int32_t n_classes,
    p4a_method_result_t** out_result);

/* AOM preprocessing fit/transform (Phase 6a). Applies an operator bank
 * through the gating strategy and returns both the per-operator outputs
 * and the gated/mixed transformed matrix. Y is optional (some operators
 * use it, e.g. EPO; pass NULL when not needed). Result keys:
 *   "transformed"        (n × n_features) — final gated transform
 *   "operator_outputs"   (n_operators × (n × n_features), operator-major,
 *                         stored as a (n_operators × (n*n_features)) matrix)
 *   "weights"            (1 × n_operators) — gating weights at fit time
 *   int64 "operator_kinds" (n_operators) — P4A_OP_* enum values
 *   scalar "n_operators", scalar "mode" (gating mode integer)
 */
P4A_API p4a_status_t p4a_aom_preprocess_fit(
    p4a_context_t* ctx,
    const p4a_operator_bank_t* bank,
    const p4a_gating_strategy_t* gate,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    p4a_method_result_t** out_result);

/* ============================================================================
 * 18. ABI shims for §6 Phase 5 variable-selection methods
 * ==========================================================================
 * Each selector returns a p4a_method_result_t with at minimum:
 *   int64 "selected_indices"     — top-k (or final) selected feature indices
 *   double "scores" or "rmse_*"  — algorithm-specific score/RMSE arrays
 *   scalar "best_rmse" (when applicable)
 *
 * All selectors that take a ValidationPlan use the same plan API as
 * `p4a_aom_global_select`. Selectors that take seeds expose them as
 * `uint64_t` parameters.
 */

/* VIP / coefficient-magnitude / selectivity-ratio rankers (Phase 5a). Method
 * enum: 0=VIP, 1=coefficient magnitude, 2=selectivity ratio. */
P4A_API p4a_status_t p4a_variable_select_rank(
    p4a_context_t* ctx,
    const p4a_model_t* model,
    const p4a_matrix_view_t* X,
    int32_t method,
    int32_t top_k,
    p4a_method_result_t** out_result);

/* Interval / iPLS / mwPLS scan (Phase 5b). */
P4A_API p4a_status_t p4a_interval_select(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    const p4a_validation_plan_t* plan,
    int32_t interval_width,
    int32_t step,
    p4a_method_result_t** out_result);

/* MCUVE / coefficient-stability selector (Phase 5c). */
P4A_API p4a_status_t p4a_stability_select(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    const p4a_validation_plan_t* plan,
    int32_t top_k,
    p4a_method_result_t** out_result);

/* UVE — artificial-noise-thresholded selector (Phase 5d). */
P4A_API p4a_status_t p4a_uve_select(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    const p4a_validation_plan_t* plan,
    int32_t noise_features,
    uint64_t noise_seed,
    p4a_method_result_t** out_result);

/* SPA — Successive Projections Algorithm (Phase 5e). */
P4A_API p4a_status_t p4a_spa_select(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    int32_t top_k,
    p4a_method_result_t** out_result);

/* CARS — Competitive Adaptive Reweighted Sampling (Phase 5f). */
P4A_API p4a_status_t p4a_cars_select(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    const p4a_validation_plan_t* plan,
    int32_t n_iterations,
    int32_t min_features,
    p4a_method_result_t** out_result);

/* Random Frog (Phase 5g). */
P4A_API p4a_status_t p4a_random_frog_select(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    const p4a_validation_plan_t* plan,
    int32_t n_iterations,
    int32_t initial_size,
    int32_t min_size,
    int32_t max_size,
    int32_t top_k,
    uint64_t seed,
    p4a_method_result_t** out_result);

/* SCARS — Stability + CARS (Phase 5h). */
P4A_API p4a_status_t p4a_scars_select(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    const p4a_validation_plan_t* plan,
    int32_t n_iterations,
    int32_t min_features,
    double sample_fraction,
    uint64_t seed,
    p4a_method_result_t** out_result);

/* GA-PLS (Phase 5i). */
P4A_API p4a_status_t p4a_ga_select(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    const p4a_validation_plan_t* plan,
    int32_t n_generations,
    int32_t population_size,
    int32_t min_features,
    int32_t max_features,
    double mutation_rate,
    uint64_t seed,
    p4a_method_result_t** out_result);

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
P4A_API p4a_status_t p4a_pso_select(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    const p4a_validation_plan_t* plan,
    int32_t n_swarm,
    int32_t n_iterations,
    double w,
    double c1,
    double c2,
    double v_max,
    uint64_t seed,
    p4a_method_result_t** out_result);

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
P4A_API p4a_status_t p4a_vissa_select(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    const p4a_validation_plan_t* plan,
    int32_t n_iterations,
    int32_t n_submodels,
    double ratio_kept,
    double threshold,
    double floor_probability,
    uint64_t seed,
    p4a_method_result_t** out_result);

/* Shaving (Phase 5j). */
P4A_API p4a_status_t p4a_shaving_select(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    const p4a_validation_plan_t* plan,
    int32_t n_steps,
    int32_t min_features,
    double shave_fraction,
    p4a_method_result_t** out_result);

/* BVE-PLS (Phase 5k). */
P4A_API p4a_status_t p4a_bve_select(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    const p4a_validation_plan_t* plan,
    int32_t n_steps,
    int32_t min_features,
    p4a_method_result_t** out_result);

/* T2-PLS (Phase 5l). `alpha_thresholds` is an array of `n_alphas` α values
 * in (0, 1). */
P4A_API p4a_status_t p4a_t2_select(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    const p4a_validation_plan_t* plan,
    const double* alpha_thresholds,
    int64_t n_alphas,
    int32_t min_selected,
    p4a_method_result_t** out_result);

/* WVC-PLS (Phase 5m). */
P4A_API p4a_status_t p4a_wvc_select(
    p4a_context_t* ctx,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    int32_t n_components,
    int32_t top_k,
    int32_t normalize,
    p4a_method_result_t** out_result);

/* WVC-threshold (Phase 5r). */
P4A_API p4a_status_t p4a_wvc_threshold_select(
    p4a_context_t* ctx,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    int32_t n_components,
    int32_t normalize,
    double score_threshold,
    double threshold_factor,
    int32_t min_selected,
    p4a_method_result_t** out_result);

/* EMCUVE (Phase 5n). */
P4A_API p4a_status_t p4a_emcuve_select(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    const p4a_validation_plan_t* plan,
    int32_t noise_features,
    uint64_t noise_seed,
    int32_t n_ensembles,
    double vote_threshold,
    p4a_method_result_t** out_result);

/* Randomization-test selector (Phase 5o). */
P4A_API p4a_status_t p4a_randomization_select(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    int32_t n_permutations,
    uint64_t randomization_seed,
    double alpha,
    p4a_method_result_t** out_result);

/* biPLS — backward iPLS (Phase 5p). */
P4A_API p4a_status_t p4a_bipls_select(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    const p4a_validation_plan_t* plan,
    int32_t interval_width,
    int32_t min_intervals,
    p4a_method_result_t** out_result);

/* siPLS — synergistic interval PLS (Phase 5q). */
P4A_API p4a_status_t p4a_sipls_select(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    const p4a_validation_plan_t* plan,
    int32_t interval_width,
    int32_t combination_size,
    p4a_method_result_t** out_result);

/* REP-PLS (Phase 5s). */
P4A_API p4a_status_t p4a_rep_select(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    const p4a_validation_plan_t* plan,
    int32_t n_steps,
    int32_t min_features,
    int32_t remove_count,
    p4a_method_result_t** out_result);

/* IPW-PLS (Phase 5t). */
P4A_API p4a_status_t p4a_ipw_select(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    const p4a_validation_plan_t* plan,
    int32_t n_iterations,
    int32_t top_k,
    double damping,
    double weight_floor,
    p4a_method_result_t** out_result);

/* ST-PLS — score-threshold (Phase 5u). `thresholds` array of `n_thresholds`
 * positive doubles. */
P4A_API p4a_status_t p4a_st_select(
    p4a_context_t* ctx,
    const p4a_config_t* cfg,
    const p4a_matrix_view_t* X,
    const p4a_matrix_view_t* Y,
    const p4a_validation_plan_t* plan,
    const double* thresholds,
    int64_t n_thresholds,
    int32_t min_selected,
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
