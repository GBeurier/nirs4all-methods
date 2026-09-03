/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * nirs4all-methods — public C ABI (version: see N4M_ABI_VERSION_* in n4m_version.h).
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
 *     core frees what it allocated. The only core-allocated outputs are the
 *     opaque operator handles; each has an explicit `n4m_*_destroy` function.
 *
 * Phase 3 trim (May 2026): prior revisions of this header inherited a large
 * set of PLS-domain declarations from the legacy pre-rename surface
 * (config, pipeline, model,
 * operator bank, gating strategy, AOM, validation plan, method result,
 * n4m_array_t). None of those had implementations in the nirs4all-methods
 * core; they were placeholders for an algorithm surface that has been
 * deferred to later phases (or removed entirely). The header has therefore
 * been pruned down to the canonical nirs4all-methods surface: status codes,
 * matrix views, context lifecycle, RNG (Phase 1) and preprocessing operators
 * (Phases 2-3). Future phases append new sections in dedicated banners and
 * bump N4M_ABI_VERSION_MINOR.
 *
 * Status-code contracts (apply uniformly unless overridden in a function's
 * own doc comment):
 *
 *   - `_create(out, ...)` functions: return N4M_OK on success, N4M_ERR_NULL_POINTER
 *     when `out` is NULL, N4M_ERR_INVALID_ARGUMENT when constructor parameters
 *     are rejected, N4M_ERR_OUT_OF_MEMORY on allocation failure. `*out` is
 *     set to NULL on every failure.
 *   - `_destroy(handle)` functions: void, no-op on NULL.
 *   - For stateful operators (Phase 3): `_transform` returns N4M_ERR_NOT_FITTED
 *     when called before `_fit`. Calling `_fit` again replaces the prior
 *     state (sklearn-style refit semantics).
 *   - Functions that consume `n4m_matrix_view_t` may additionally return
 *     N4M_ERR_SHAPE_MISMATCH, N4M_ERR_DTYPE_MISMATCH, N4M_ERR_STRIDE_INVALID
 *     when the view fails validation.
 */
#ifndef N4M_N4M_H
#define N4M_N4M_H

#include <stddef.h>
#include <stdint.h>

#include "n4m/n4m_export.h"
#include "n4m/n4m_version.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 1. Status codes
 * ========================================================================== */

typedef enum n4m_status_t {
    N4M_OK                        =   0,
    N4M_ERR_INVALID_ARGUMENT      =   1,
    N4M_ERR_NULL_POINTER          =   2,
    N4M_ERR_SHAPE_MISMATCH        =   3,
    N4M_ERR_DTYPE_MISMATCH        =   4,
    N4M_ERR_STRIDE_INVALID        =   5,
    N4M_ERR_NOT_FITTED            =   6,
    N4M_ERR_NUMERICAL_FAILURE     =   7,
    N4M_ERR_CONVERGENCE_FAILED    =   8,
    N4M_ERR_OUT_OF_MEMORY         =   9,
    N4M_ERR_UNSUPPORTED           =  10,
    N4M_ERR_NOT_IMPLEMENTED       =  11,   /* deferred public surface */
    N4M_ERR_ABI_MISMATCH          =  12,
    N4M_ERR_IO                    =  13,
    N4M_ERR_CORRUPT_BUFFER        =  14,
    N4M_ERR_VERSION_INCOMPATIBLE  =  15,
    N4M_ERR_BACKEND_UNAVAILABLE   =  16,
    N4M_ERR_CANCELLED             =  17,
    N4M_ERR_INTERNAL              = 255
} n4m_status_t;

/* Returns a static NUL-terminated string describing the status code. Returns
 * "unknown status" for any integer outside the declared enum range; never
 * returns NULL. */
N4M_API const char* n4m_status_to_string(n4m_status_t s);

/* ============================================================================
 * 2. Backends and numerical dtypes
 * ==========================================================================
 *
 * Each build of libn4m has ONE active numerical path, fixed at COMPILE time:
 * the accelerator enabled via the matching N4M_WITH_* CMake option (BLAS,
 * OpenMP, or CUDA), or the portable reference CPU path when none is enabled.
 * linalg dispatch is compile-time, so n4m_context_set_backend validates and
 * records the requested backend but does NOT re-route compute at runtime.
 * A build compiled with an accelerator routes all GEMM/GEMV through it; the
 * reference scalar path is compiled in only when no accelerator is enabled.
 * In particular a CUDA build is GPU-TARGETED: it loads and answers metadata
 * queries without a GPU, but any method that computes a GEMM raises at runtime
 * on a GPU-less machine — it does not fall back to the CPU reference.
 */
typedef enum n4m_backend_t {
    N4M_BACKEND_AUTO            = 0,   /* library picks the best available */
    N4M_BACKEND_REFERENCE_CPU   = 1,   /* always present, portable */
    N4M_BACKEND_NATIVE_CPU      = 2,   /* SIMD-tuned scalar paths */
    N4M_BACKEND_BLAS            = 3,
    N4M_BACKEND_OPENMP          = 4,   /* combinable with another backend */
    N4M_BACKEND_CUDA            = 5,
    N4M_BACKEND_OPENCL          = 6,   /* reserved, not implemented */
    N4M_BACKEND_METAL           = 7    /* reserved, not implemented */
} n4m_backend_t;

/* Returns 1 if the backend is usable in this build of libn4m, 0 otherwise.
 * "Usable" means compiled in (via the matching N4M_WITH_* option) AND, for
 * CUDA, a CUDA-capable GPU present at runtime. AUTO and REFERENCE_CPU are
 * always available. Because dispatch is compile-time (see above), this reports
 * what the build CAN do, not a per-context runtime switch. */
N4M_API int          n4m_backend_is_available(n4m_backend_t backend);
N4M_API const char*  n4m_backend_to_string(n4m_backend_t backend);

typedef enum n4m_dtype_t {
    N4M_DTYPE_UNKNOWN = 0,
    N4M_DTYPE_F64     = 1,
    N4M_DTYPE_F32     = 2,
    N4M_DTYPE_I32     = 3,    /* reserved for label / index views */
    N4M_DTYPE_I64     = 4
} n4m_dtype_t;

/* Returns the size in bytes of one element of the given dtype.
 * Returns 0 for N4M_DTYPE_UNKNOWN or any out-of-range value. */
N4M_API size_t       n4m_dtype_size(n4m_dtype_t dtype);
N4M_API const char*  n4m_dtype_to_string(n4m_dtype_t dtype);

/* ============================================================================
 * 3. Matrix view — non-owning, stride-aware
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
 * Stride rules enforced by n4m_matrix_view_validate:
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
 * explicitly excluded.
 */
typedef struct n4m_matrix_view_t {
    void*        data;
    int64_t      rows;
    int64_t      cols;
    int64_t      row_stride;   /* elements between row i and i+1 */
    int64_t      col_stride;   /* elements between col j and j+1 */
    n4m_dtype_t  dtype;
    int32_t      reserved0;    /* keep struct size stable */
} n4m_matrix_view_t;

/* Initialise a row-major (C / NumPy default) view. row_stride = cols,
 * col_stride = 1. Returns N4M_ERR_NULL_POINTER if `out` is NULL or
 * `data` is NULL with rows*cols > 0; N4M_ERR_INVALID_ARGUMENT for negative
 * dimensions or N4M_DTYPE_UNKNOWN; N4M_OK on success. */
N4M_API n4m_status_t n4m_matrix_view_init_rowmajor(
    n4m_matrix_view_t* out, void* data,
    int64_t rows, int64_t cols, n4m_dtype_t dtype);

/* Initialise a column-major (Fortran / R / MATLAB default) view.
 * row_stride = 1, col_stride = rows. */
N4M_API n4m_status_t n4m_matrix_view_init_colmajor(
    n4m_matrix_view_t* out, void* data,
    int64_t rows, int64_t cols, n4m_dtype_t dtype);

/* Initialise a strided view with explicit positive strides. Useful for
 * slicing into a larger buffer or for transposed views. Strides must be
 * positive (>= 1) and ints; the caller is responsible for ensuring the
 * underlying buffer is large enough. */
N4M_API n4m_status_t n4m_matrix_view_init_strided(
    n4m_matrix_view_t* out, void* data,
    int64_t rows, int64_t cols,
    int64_t row_stride, int64_t col_stride,
    n4m_dtype_t dtype);

/* Validate that the view is well-formed. The rules are listed in §3 prelude
 * above. Returns:
 *   - N4M_OK if all rules hold.
 *   - N4M_ERR_NULL_POINTER  if `v` is NULL, or `v->data` is NULL with
 *                            rows*cols > 0.
 *   - N4M_ERR_STRIDE_INVALID if a stride is negative or zero where required.
 *   - N4M_ERR_INVALID_ARGUMENT for negative `rows` / `cols`, or dtype not
 *                              in {F64, F32, I32, I64}. */
N4M_API n4m_status_t n4m_matrix_view_validate(const n4m_matrix_view_t* v);

/* ============================================================================
 * 4. Context lifecycle
 * ==========================================================================
 *
 * One n4m_context_t represents an isolated execution environment. It owns:
 *   - the per-context 4 KiB error buffer,
 *   - the current backend selection,
 *   - the RNG seed used by stochastic algorithms,
 *   - the thread-count hint for parallel backends,
 *   - an optional user-pointer for binding-side bookkeeping.
 *
 * Threading: a single n4m_context_t* must not be used concurrently from two
 * threads — use one context per thread. Across contexts the (default CPU)
 * library is thread-safe.
 * EXCEPTION: a cuda-on build routes all GEMM/GEMV/GER through a single
 * process-wide cuBLAS handle, which is NOT safe for concurrent use from
 * multiple host threads. Until the deferred per-stream handle pool lands
 * (see DEFERRALS.md "GPU — batched execution path"), treat a cuda-on build
 * as single-threaded: do not fit/predict concurrently across contexts.
 *
 * Signal safety: no n4m_* function is async-signal-safe. Do not call any
 * n4m_* function from a POSIX signal handler.
 */
typedef struct n4m_context_s n4m_context_t;

N4M_API n4m_status_t n4m_context_create(n4m_context_t** out_ctx);
N4M_API void         n4m_context_destroy(n4m_context_t* ctx);

N4M_API n4m_status_t n4m_context_set_seed(n4m_context_t* ctx, uint64_t seed);
N4M_API n4m_status_t n4m_context_get_seed(const n4m_context_t* ctx,
                                          uint64_t* out_seed);

N4M_API n4m_status_t n4m_context_set_backend(n4m_context_t* ctx,
                                             n4m_backend_t backend);
N4M_API n4m_status_t n4m_context_get_backend(const n4m_context_t* ctx,
                                             n4m_backend_t* out_backend);

/* `n_threads`: any int32_t is accepted. Values <= 0 are documented as
 * "library picks the default" (read by parallel backends later); positive
 * values pin the worker count. The setter does NOT validate the upper
 * bound — the caller is responsible for choosing a sensible cap. */
N4M_API n4m_status_t n4m_context_set_num_threads(n4m_context_t* ctx,
                                                 int32_t n_threads);
N4M_API n4m_status_t n4m_context_get_num_threads(const n4m_context_t* ctx,
                                                 int32_t* out_threads);

/* Returns a NUL-terminated UTF-8 string owned by the context. The pointer
 * remains valid until the next n4m_* call on this context — bindings must
 * copy if they need to retain the value. Never returns NULL: when no error
 * has occurred since context creation (or since the last clear_error),
 * returns the empty string "". If `ctx` is NULL, returns a static "" string
 * literal (still never NULL). The buffer capacity is fixed at
 * N4M_ERROR_BUFFER_BYTES (4096) — over-long messages are truncated at the
 * boundary, preserving the NUL terminator. */
N4M_API const char*  n4m_context_last_error(const n4m_context_t* ctx);
/* No-op when `ctx` is NULL. */
N4M_API void         n4m_context_clear_error(n4m_context_t* ctx);

/* Optional binding-side annotation. The library never reads or writes the
 * pointer's referent. `n4m_context_get_user_data(NULL)` returns NULL — this
 * is indistinguishable from a context that has had NULL stored explicitly;
 * if you need to distinguish, store a sentinel value. */
N4M_API n4m_status_t n4m_context_set_user_data(n4m_context_t* ctx, void* user);
N4M_API void*        n4m_context_get_user_data(const n4m_context_t* ctx);

/* ============================================================================
 * 5. ABI naming convention and operator contracts
 * ==========================================================================
 *
 * Reserved prefixes for the n4m_* namespace:
 *
 *   n4m_pp_*       preprocessing operators (Phases 2-7, 10)
 *   n4m_split_*    sample splitters       (Phase 11)
 *   n4m_filter_*   sample filters         (Phases 12-14)
 *   n4m_aug_*      data augmentations     (Phases 15-18)
 *   n4m_util_*     generic utilities      (cross-phase)
 *   n4m_rng_*      random number generators (Phase 1)
 *   n4m_metric_*   metrics and diagnostics (Phases 19-20)
 *
 * Operator lifecycle contracts:
 *
 *   * Stateless operators expose `create / transform / destroy`:
 *       - `_create(handle**, ...)` allocates an opaque handle parameterised
 *         by the constructor arguments. No fitting needed; the transform is
 *         a pure function of the inputs.
 *       - `_transform(handle, X_view, out_view)` applies the operator.
 *       - `_destroy(handle)` releases the handle. NULL-safe.
 *
 *   * Stateful operators expose `create / fit / transform / destroy` (with
 *     optional `inverse_transform` and `is_fitted`):
 *       - `_create(handle**, ...)` allocates an opaque handle parameterised
 *         by the constructor arguments. The handle starts in the *unfitted*
 *         state.
 *       - `_fit(handle, X_view)` learns the operator's parameters from X.
 *         Calling `_fit` again replaces the prior state (sklearn-style
 *         refit semantics).
 *       - `_transform(handle, X_view, out_view)` applies the fitted
 *         operator. Calling `_transform` (or `_inverse_transform`) before
 *         `_fit` returns N4M_ERR_NOT_FITTED.
 *       - `_inverse_transform(handle, X_view, out_view)` reverses the
 *         transform when the operator is invertible. Operators with no
 *         meaningful inverse omit this entry point.
 *       - `_is_fitted(handle, int* out)` reports the fitted state (1 if
 *         fitted, 0 otherwise).
 *       - `_destroy(handle)` releases the handle. NULL-safe.
 *
 * Matrix view inputs must be row-major contiguous F64 for the current
 * implementation; non-contiguous and non-F64 views return
 * N4M_ERR_STRIDE_INVALID or N4M_ERR_DTYPE_MISMATCH respectively.
 */

/* ============================================================================
 * 6. ABI version + build info
 * ========================================================================== */

N4M_API uint32_t     n4m_get_abi_version_major(void);
N4M_API uint32_t     n4m_get_abi_version_minor(void);
N4M_API uint32_t     n4m_get_abi_version_patch(void);
N4M_API uint32_t     n4m_get_abi_version_int(void);   /* MAJOR*10000 + MINOR*100 + PATCH */
N4M_API const char*  n4m_get_version_string(void);    /* e.g. "0.1.0+abi.1.3.0" */
N4M_API const char*  n4m_get_build_info(void);        /* compiler / flags / backends */
N4M_API const char*  n4m_get_git_revision(void);      /* git rev at build time, or "" */

/* Returns N4M_OK if the header's ABI MAJOR == library MAJOR and the header's
 * MINOR <= library MINOR (forward compat). Returns N4M_ERR_ABI_MISMATCH on
 * MAJOR mismatch, N4M_ERR_VERSION_INCOMPATIBLE on MINOR-too-high. */
N4M_API n4m_status_t n4m_check_abi_compatibility(
    uint32_t header_abi_major,
    uint32_t header_abi_minor);

/* ============================================================================
 * 7. Phase 1 — PCG64 RNG
 * ============================================================================
 *
 * Pseudo-random number generator providing bit-exact parity against
 * NumPy 1.26.4's `numpy.random.PCG64`. All subsequent stochastic operators
 * in nirs4all-methods (augmenters, IsolationForest, KMeans, CARS, MCUVE)
 * consume a `n4m_rng_pcg64_state_t*` so that pipelines run on this library
 * yield identical results to their reference NumPy implementations.
 *
 * Algorithm: PCG-XSL-RR 128/64 with a 128-bit state and 128-bit increment.
 * Seeding from a single uint64 reproduces `np.random.default_rng(seed)`
 * via NumPy's SeedSequence (pool_size=4) mixer. Normal samples use the
 * 256-region Ziggurat algorithm with NumPy's vendored constants.
 *
 * Parity guarantees:
 *   - n4m_rng_pcg64_uint64_fill          ≡ np.random.default_rng(s)
 *                                            .bit_generator.random_raw(n)
 *     byte-for-byte exact.
 *   - n4m_rng_pcg64_standard_normal_fill ≡ np.random.default_rng(s)
 *                                            .standard_normal(n)
 *     bit-exact double-for-double.
 *
 * Lifecycle: `n4m_rng_pcg64_create` allocates + seeds the handle;
 * `n4m_rng_pcg64_destroy` releases it (NULL-safe, no-op on NULL).
 * Status semantics follow the universal rules at the top of this header.
 */
typedef struct n4m_rng_pcg64_state_t n4m_rng_pcg64_state_t;

N4M_API n4m_status_t n4m_rng_pcg64_create(uint64_t seed,
                                           n4m_rng_pcg64_state_t** out);

N4M_API void n4m_rng_pcg64_destroy(n4m_rng_pcg64_state_t* rng);

N4M_API n4m_status_t n4m_rng_pcg64_set_seed(n4m_rng_pcg64_state_t* rng,
                                             uint64_t seed);

N4M_API n4m_status_t n4m_rng_pcg64_uint64(n4m_rng_pcg64_state_t* rng,
                                           uint64_t* out);

N4M_API n4m_status_t n4m_rng_pcg64_uint64_fill(n4m_rng_pcg64_state_t* rng,
                                                uint64_t* out, size_t n);

N4M_API n4m_status_t n4m_rng_pcg64_standard_normal_fill(
    n4m_rng_pcg64_state_t* rng, double* out, size_t n);

N4M_API n4m_status_t n4m_rng_pcg64_advance(n4m_rng_pcg64_state_t* rng,
                                            uint64_t delta);


/* ============================================================================
 * 8. PLS-domain shared infrastructure (config, operator bank, gating,
 *    pipeline, model lifecycle, serialization, output array, validation
 *    plan, method-result container) — formerly n4m/pls.h
 * ============================================================================ */

/* ---------- Shared result type ----------------------------------------- */
typedef struct n4m_split_result_t {
    int64_t* train_idx;
    int64_t  n_train;
    int64_t* test_idx;
    int64_t  n_test;
    void*    _owner;
} n4m_split_result_t;

N4M_API void n4m_split_result_destroy(n4m_split_result_t* r);

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
    N4M_ALGO_PCR            = 10,
    /* A self-described, PREDICT-only affine model imported from an external
     * fitted predictor.  It is never produced by n4m_model_fit and has no
     * latent-score/transform semantics. */
    N4M_ALGO_IMPORTED_LINEAR_PREDICTOR = 11
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

/* Random-number generator used by stochastic methods (variable selectors,
 * ensemble PLS, randomized SVD). The default SPLITMIX64 reproduces n4m's
 * historical streams bit-for-bit, so leaving it unset changes nothing. The
 * other kinds make a method draw from an external library's exact RNG so its
 * stochastic output can match that reference:
 *   PCG64    ≡ numpy.random.default_rng
 *   MT_R     ≡ base R set.seed / runif / rnorm (Inversion)
 *   NUMPY_MT ≡ numpy.random.RandomState (legacy MT19937; sklearn / auswahl) */
typedef enum n4m_rng_kind_t {
    N4M_RNG_SPLITMIX64 = 0,
    N4M_RNG_PCG64      = 1,
    N4M_RNG_MT_R       = 2,
    N4M_RNG_NUMPY_MT   = 3
} n4m_rng_kind_t;

/* AOM sweep route policy. AUTO uses exact operator-moment routes when the
 * current guards consider them valid and falls back to materialized chains
 * otherwise. MATERIALIZED forces the legacy materialized-chain screen, useful
 * for timing/score comparisons and for avoiding moment-route regressions.
 * FORCE_MOMENTS rejects any candidate screen that would need a materialized
 * fallback; the selected chain may still be materialized once after scoring
 * to populate public predictions and input-space coefficients. */
typedef enum n4m_aom_moment_policy_t {
    N4M_AOM_MOMENT_AUTO          = 0,
    N4M_AOM_MOMENT_MATERIALIZED  = 1,
    N4M_AOM_MOMENT_FORCE_MOMENTS = 2
} n4m_aom_moment_policy_t;

typedef enum n4m_aom_pls_score_mode_t {
    N4M_AOM_PLS_SCORE_CV        = 0,
    N4M_AOM_PLS_SCORE_GCV_PROXY = 1
} n4m_aom_pls_score_mode_t;

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
N4M_API n4m_status_t n4m_config_set_rng_kind         (n4m_config_t*, n4m_rng_kind_t);
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
/* Switch `n4m_estimators_sparse_simpls_fit` between the default Chun & Keles 2010
 * spls algorithm (enabled=0; matches R `spls::spls`) and the legacy
 * per-component absolute soft-threshold of the SIMPLS direction
 * (enabled=1; behaviour of nirs4all-methods <= 0.97.3). */
N4M_API n4m_status_t n4m_config_set_sparse_simpls_legacy(n4m_config_t*,
                                                          int32_t enabled);
N4M_API n4m_status_t n4m_config_get_sparse_simpls_legacy(const n4m_config_t*,
                                                          int32_t* out_enabled);
/* Switch `n4m_estimators_robust_pls_fit` between Partial Robust M-regression
 * (enabled=0; default; matches R `chemometrics::prm` bit-for-bit) and the
 * legacy Huber-IRLS over weighted SIMPLS (enabled=1; behaviour of
 * nirs4all-methods <= 0.97.3). */
N4M_API n4m_status_t n4m_config_set_robust_pls_legacy(n4m_config_t*,
                                                       int32_t enabled);
N4M_API n4m_status_t n4m_config_get_robust_pls_legacy(const n4m_config_t*,
                                                       int32_t* out_enabled);
/* Switch `n4m_metrics_approximate_press_compute` between true leave-one-out PRESS
 * (enabled=0; default; matches R `pls::plsr(validation='LOO', method='simpls',
 * scale=FALSE)` bit-for-bit) and the legacy Eastment-Krzanowski leverage-
 * inflated training-residual approximation (enabled=1; behaviour of
 * nirs4all-methods <= 0.97.3). */
N4M_API n4m_status_t n4m_config_set_approximate_press_legacy(n4m_config_t*,
                                                              int32_t enabled);
N4M_API n4m_status_t n4m_config_get_approximate_press_legacy(const n4m_config_t*,
                                                              int32_t* out_enabled);
N4M_API n4m_status_t n4m_config_set_aom_moment_policy(n4m_config_t*,
                                                       n4m_aom_moment_policy_t);
N4M_API n4m_status_t n4m_config_get_aom_moment_policy(const n4m_config_t*,
                                                       n4m_aom_moment_policy_t*);
N4M_API n4m_status_t n4m_config_set_aom_pls_score_mode(n4m_config_t*,
                                                       n4m_aom_pls_score_mode_t);
N4M_API n4m_status_t n4m_config_get_aom_pls_score_mode(
    const n4m_config_t*,
    n4m_aom_pls_score_mode_t*);
/* When enabled on AOM sweep methods, return the candidate table and selected
 * identifiers without the final selected-chain refit/materialization outputs.
 * This is intended for very large preprocessing screens where downstream code
 * only needs ranks/scores first. Default is disabled. */
N4M_API n4m_status_t n4m_config_set_aom_score_only(n4m_config_t*, int32_t enabled);
N4M_API n4m_status_t n4m_config_get_aom_score_only(const n4m_config_t*, int32_t*);
/* Optional CUDA scheduling knob for exact PLS1 moment CV. When enabled on a
 * CUDA build, independent fold/moment jobs may run in bounded stream-parallel
 * batches on the single selected GPU. Scores are unchanged; this only affects
 * scheduling and timings. Default is disabled. */
N4M_API n4m_status_t n4m_config_set_cuda_pls_parallel_folds(n4m_config_t*,
                                                             int32_t enabled);
N4M_API n4m_status_t n4m_config_get_cuda_pls_parallel_folds(const n4m_config_t*,
                                                             int32_t*);
/* Optional CUDA PLS1 moment routing threshold. The default is 1024 features,
 * matching the conservative built-in guard. Set a lower positive value only
 * when benchmarking whether medium-width exact moment PLS screens benefit
 * from the device route on the selected single GPU. */
N4M_API n4m_status_t n4m_config_set_cuda_pls_min_device_features(
    n4m_config_t*, int32_t n_features);
N4M_API n4m_status_t n4m_config_get_cuda_pls_min_device_features(
    const n4m_config_t*, int32_t*);
/* Optional CUDA PLS1 moment many-design batching knob. When enabled on a CUDA
 * build with more than one independent fold/moment job, the tiled
 * strided-batched device path is used instead of the reusable sequential
 * workspace. Scores are unchanged; this only affects scheduling and timings.
 * Default is disabled; the N4M_CUDA_PLS_MANY_BATCHED env var is an equivalent
 * fallback opt-in. */
N4M_API n4m_status_t n4m_config_set_cuda_pls_many_batched(n4m_config_t*,
                                                          int32_t enabled);
N4M_API n4m_status_t n4m_config_get_cuda_pls_many_batched(const n4m_config_t*,
                                                          int32_t*);
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
N4M_API n4m_status_t n4m_config_get_rng_kind         (const n4m_config_t*, n4m_rng_kind_t*);
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
    N4M_OP_FCK                 = 17,
    N4M_OP_GAUSSIAN            = 18
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

/* Exact affine predictor import for migration of an already-fitted external
 * model.  The caller supplies row-major coefficients of shape
 * (n_features, n_targets), and an intercept vector of length n_targets:
 *
 *   y_hat = intercept + X @ coefficients
 *
 * All supplied values must be finite.  The library copies the inputs, so
 * callers retain ownership and may release their buffers after the call.
 * `source_training_samples` is provenance only and may be zero if the source
 * format does not disclose it.  The resulting model is explicitly marked
 * N4M_ALGO_IMPORTED_LINEAR_PREDICTOR, is N4MM-exportable/importable and
 * supports PREDICT only.  n4m_model_transform returns N4M_ERR_UNSUPPORTED:
 * an affine projection does not reconstruct a PLS latent decomposition.
 */
typedef struct n4m_linear_predictor_spec_t {
    int64_t source_training_samples;
    int32_t n_features;
    int32_t n_targets;
    const double* coefficients;
    const double* intercept;
} n4m_linear_predictor_spec_t;

N4M_API n4m_status_t n4m_model_import_linear_predictor(
    n4m_context_t* ctx,
    const n4m_linear_predictor_spec_t* spec,
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

/* Versioned, allocation-free inspection of a complete N4MM fitted-model
 * payload. Unlike n4m_serialization_inspect (which intentionally reads only
 * the 20-byte header), this entry point validates the magic, supported wire
 * version, checksum, model recipe, dimensions and every length-delimited
 * section, plus the bounded format-2 pipeline block when present, before
 * publishing any metadata. The `_v1` suffix identifies this descriptor
 * schema; the function accepts both N4MM wire versions 1 and 2.
 *
 * `out_info` is byte-zeroed before any validation. It therefore remains all
 * zero on N4M_ERR_NULL_POINTER, N4M_ERR_CORRUPT_BUFFER,
 * N4M_ERR_VERSION_INCOMPATIBLE or N4M_ERR_UNSUPPORTED. Unknown algorithm,
 * solver or deflation values are unsupported rather than trusted as
 * capabilities. The capability mask is derived solely from the validated
 * N4MM bytes; callers must not augment it from an external manifest.
 */
#define N4M_SERIALIZED_MODEL_INFO_SCHEMA_V1 1u

#define N4M_SERIALIZED_MODEL_CAPABILITY_PREDICT   (UINT64_C(1) << 0)
#define N4M_SERIALIZED_MODEL_CAPABILITY_TRANSFORM (UINT64_C(1) << 1)
#define N4M_SERIALIZED_MODEL_CAPABILITY_AFFINE    (UINT64_C(1) << 2)
#define N4M_SERIALIZED_MODEL_CAPABILITY_PIPELINE  (UINT64_C(1) << 3)

typedef struct n4m_serialized_model_info_v1_t {
    uint32_t schema_version;
    uint32_t format_version;
    uint32_t writer_abi_major;
    uint32_t writer_abi_minor;
    uint32_t writer_abi_patch;
    n4m_algorithm_t algorithm;
    n4m_solver_t solver;
    n4m_deflation_t deflation;
    int64_t training_samples;
    int32_t n_features;
    int32_t n_targets;
    int32_t n_components;
    uint32_t reserved0;
    uint64_t capabilities;
} n4m_serialized_model_info_v1_t;

N4M_API n4m_status_t n4m_serialization_inspect_model_v1(
    const void* buffer, size_t buffer_size,
    n4m_serialized_model_info_v1_t* out_info);

/* Versioned inspection of the optional fitted-model preprocessing pipeline.
 * The caller passes its output allocation size explicitly; current callers
 * must pass sizeof(n4m_serialized_pipeline_info_v1_t). The function validates
 * the complete N4MM payload through the same authoritative decoder as
 * n4m_serialization_inspect_model_v1 before publishing any field.
 *
 * N4MM format 1 returns N4M_OK with present=0 and all pipeline-specific fields
 * zero. Format 2 returns the exact validated SNV -> Savitzky-Golay plan. The
 * two feature widths are separate ABI fields even though this bounded plan is
 * dimension-preserving. The semantic profile fixes row-wise SNV with mean/std
 * and ddof=0 followed by Savitzky-Golay interp mode with cval=0. fingerprint is
 * FNV-1a-64 over the canonical 84-byte little-endian pipeline block (operator
 * count/tags/parameters plus the semantic-profile fields), and is therefore
 * stable across machines and writer ABI versions.
 *
 * On every error, the writable prefix min(out_info_size, sizeof(*out_info)) is
 * byte-zeroed. A short output allocation returns N4M_ERR_INVALID_ARGUMENT.
 */
#define N4M_SERIALIZED_PIPELINE_INFO_SCHEMA_V1 1u
#define N4M_SERIALIZED_PIPELINE_MAX_OPERATORS  2u
#define N4M_PIPELINE_SEMANTIC_PROFILE_NONE 0u
#define N4M_PIPELINE_SEMANTIC_PROFILE_NIRS4ALL_SNV_SAVGOL_V1 1u
#define N4M_PIPELINE_FINGERPRINT_NONE          0u
#define N4M_PIPELINE_FINGERPRINT_FNV1A64_V1    1u

typedef struct n4m_serialized_pipeline_info_v1_t {
    uint32_t schema_version;
    uint32_t struct_size;
    uint32_t present;
    uint32_t operator_count;
    n4m_operator_kind_t operators[N4M_SERIALIZED_PIPELINE_MAX_OPERATORS];
    int32_t savgol_window;
    int32_t savgol_poly_degree;
    int32_t savgol_derivative;
    uint32_t semantic_profile;
    double savgol_delta;
    int32_t raw_n_features;
    int32_t model_n_features;
    uint32_t fingerprint_algorithm;
    int32_t snv_axis;
    uint64_t fingerprint;
    uint32_t snv_with_mean;
    uint32_t snv_with_std;
    int32_t snv_ddof;
    int32_t savgol_mode; /* Numeric n4m_pp_savgol_mode_t value. */
    double savgol_cval;
} n4m_serialized_pipeline_info_v1_t;

N4M_API n4m_status_t n4m_serialization_inspect_pipeline_v1(
    const void* buffer, size_t buffer_size,
    n4m_serialized_pipeline_info_v1_t* out_info, size_t out_info_size);

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

/* PLS-only cross-validation surface.
 *
 * ABI 1.22 provides a stable reference endpoint for exact PLS CV. The current
 * implementation delegates to n4m_model_selection_sweep_run with heads_mask=PLS, so scores and
 * route counters match the PLS branch of the sweep path. It is not yet the
 * grouped/fused multi-chain IKPLS grinder; that executor can replace the
 * internals later without changing this signature.
 *
 * `fold_ids` follows n4m_model_selection_sweep_run semantics: when provided,
 * `n_fold_ids == n_samples`; `n_folds <= 0` means infer max(fold_ids)+1.
 * `component_grid` is a non-empty array of positive component counts.
 */
N4M_API n4m_status_t n4m_pls_cross_validate(
    n4m_context_t* ctx,
    const n4m_config_t* cfg,
    const n4m_matrix_view_t* X,
    const n4m_matrix_view_t* Y,
    const int32_t* fold_ids,
    int64_t n_fold_ids,
    int32_t n_folds,
    const int32_t* component_grid,
    int64_t n_component_grid,
    n4m_method_result_t** out_result);

#ifdef __cplusplus
}  /* extern "C" */
#endif

/* ============================================================================
 * 9. Role headers — every method declaration lives in a role header or
 *    its second-level subheader; included here so consumers need only
 *    #include "n4m/n4m.h".
 * ============================================================================ */

/* N4M_STATIC_ASSERT is defined here — before the role headers — so each role
 * header can static-assert its own enum sizes and remain independently
 * includable. */
#ifdef __cplusplus
#  define N4M_STATIC_ASSERT(cond, msg) static_assert((cond), msg)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#  define N4M_STATIC_ASSERT(cond, msg) _Static_assert((cond), msg)
#else
#  define N4M_STATIC_ASSERT(cond, msg) typedef char n4m_sa_##__LINE__[(cond) ? 1 : -1]
#endif

#include "n4m/transform.h"
#include "n4m/augmentation.h"
#include "n4m/estimators.h"
#include "n4m/feature_selection.h"
#include "n4m/model_selection.h"
#include "n4m/domain_adaptation.h"
#include "n4m/outlier_detection.h"
#include "n4m/ensemble.h"
#include "n4m/compose.h"
#include "n4m/metrics.h"
#include "n4m/decomposition.h"
#include "n4m/lowlevel.h"
#include "n4m/optimization.h"

/* ============================================================================
 * 10. ABI guard rails — fixed-size assertions on the C ABI shape
 * ============================================================================ */

N4M_STATIC_ASSERT(sizeof(n4m_status_t)         == 4, "n4m_status_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_backend_t)        == 4, "n4m_backend_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_dtype_t)          == 4, "n4m_dtype_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_pp_lsnv_pad_mode_t) == 4,
                  "n4m_pp_lsnv_pad_mode_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_pp_area_method_t) == 4,
                  "n4m_pp_area_method_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_pp_savgol_mode_t) == 4,
                  "n4m_pp_savgol_mode_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_pp_gaussian_mode_t) == 4,
                  "n4m_pp_gaussian_mode_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_y_outlier_method_t) == 4,
                  "n4m_y_outlier_method_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_composite_mode_t) == 4,
                  "n4m_composite_mode_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_signal_type_t) == 4,
                  "n4m_signal_type_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_split_kbins_strategy_t) == 4,
                  "n4m_split_kbins_strategy_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_split_y_metric_t) == 4,
                  "n4m_split_y_metric_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_split_aggregation_t) == 4,
                  "n4m_split_aggregation_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_pp_wavelet_family_t) == 4,
                  "n4m_pp_wavelet_family_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_pp_wavelet_boundary_t) == 4,
                  "n4m_pp_wavelet_boundary_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_pp_wavelet_threshold_t) == 4,
                  "n4m_pp_wavelet_threshold_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_pp_wavelet_noise_t) == 4,
                  "n4m_pp_wavelet_noise_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_filter_x_outlier_method_t) == 4,
                  "n4m_filter_x_outlier_method_t must be 4 bytes");

/* n4m_matrix_view_t must be 48 bytes on LP64 / LLP64. ILP32 is not
 * supported; on ILP32 platforms the layout would differ (pointer is 4 bytes),
 * which is why we exclude armeabi-v7a. */
#if defined(__LP64__) || defined(_WIN64)
N4M_STATIC_ASSERT(sizeof(n4m_matrix_view_t) == 48,
                  "n4m_matrix_view_t layout must be 48 bytes on LP64/LLP64");
#endif

N4M_STATIC_ASSERT(sizeof(n4m_algorithm_t)     == 4, "n4m_algorithm_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_solver_t)        == 4, "n4m_solver_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_deflation_t)     == 4, "n4m_deflation_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_operator_kind_t) == 4, "n4m_operator_kind_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_gating_mode_t)   == 4, "n4m_gating_mode_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_model_array_t)   == 4, "n4m_model_array_t must be 4 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_serialized_model_info_v1_t) == 64,
                  "n4m_serialized_model_info_v1_t must be 64 bytes");
N4M_STATIC_ASSERT(sizeof(n4m_serialized_pipeline_info_v1_t) == 96,
                  "n4m_serialized_pipeline_info_v1_t must be 96 bytes");
N4M_STATIC_ASSERT(offsetof(n4m_serialized_pipeline_info_v1_t, semantic_profile) == 36,
                  "serialized pipeline semantic_profile offset must be 36");
N4M_STATIC_ASSERT(offsetof(n4m_serialized_pipeline_info_v1_t, snv_axis) == 60,
                  "serialized pipeline snv_axis offset must be 60");
N4M_STATIC_ASSERT(offsetof(n4m_serialized_pipeline_info_v1_t, fingerprint) == 64,
                  "serialized pipeline fingerprint offset must be 64");
N4M_STATIC_ASSERT(offsetof(n4m_serialized_pipeline_info_v1_t, savgol_cval) == 88,
                  "serialized pipeline savgol_cval offset must be 88");
/* HPO enum guard rails live in n4m/optimization.h so that header is
 * independently includable. */

#endif /* N4M_N4M_H */
