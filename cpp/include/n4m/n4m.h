/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * nirs4all-methods — public C ABI v1.8.0.
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
 * set of PLS-domain declarations from pls4all (config, pipeline, model,
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
 * threads — use one context per thread. Across contexts the library is
 * thread-safe.
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
 * 8. Phase 2 — Stateless scatter / scaling preprocessings
 * ============================================================================
 *
 * Seven stateless row- or column-wise transforms with bit-exact NumPy
 * parity (where applicable) against nirs4all 0.8.x. Each operator exposes a
 * `create / transform / destroy` triplet:
 *
 *   - `_create(out, ...)` allocates an opaque handle initialised with the
 *     constructor parameters; writes NULL to *out on every failure.
 *   - `_transform(handle, X_view, out_view)` applies the operator on the
 *     input matrix view and writes the result into the output matrix view.
 *     Both views must be row-major contiguous, F64, same shape; the X and
 *     out buffers may overlap (in-place is supported).
 *   - `_destroy(handle)` releases the handle. NULL-safe.
 *
 * Status semantics follow the universal rules at the top of this header.
 * Matrix-view validation may additionally return N4M_ERR_DTYPE_MISMATCH
 * (non-F64), N4M_ERR_STRIDE_INVALID (non-contiguous row-major), or
 * N4M_ERR_SHAPE_MISMATCH (X and out shapes disagree).
 */

/* ---------- SNV (Standard Normal Variate) -------------------------------- */
typedef struct n4m_pp_snv_handle_t n4m_pp_snv_handle_t;
N4M_API n4m_status_t n4m_pp_snv_create(n4m_pp_snv_handle_t** out,
                                        int with_mean, int with_std, int ddof);
N4M_API void         n4m_pp_snv_destroy(n4m_pp_snv_handle_t* handle);
N4M_API n4m_status_t n4m_pp_snv_transform(const n4m_pp_snv_handle_t* handle,
                                           n4m_matrix_view_t X,
                                           n4m_matrix_view_t out);

/* ---------- LocalSNV (sliding-window SNV) -------------------------------- */
typedef struct n4m_pp_lsnv_handle_t n4m_pp_lsnv_handle_t;
typedef enum n4m_pp_lsnv_pad_mode_t {
    N4M_PP_LSNV_PAD_REFLECT  = 0,
    N4M_PP_LSNV_PAD_EDGE     = 1,
    N4M_PP_LSNV_PAD_CONSTANT = 2
} n4m_pp_lsnv_pad_mode_t;
/* `window` must be an odd integer >= 3. */
N4M_API n4m_status_t n4m_pp_lsnv_create(n4m_pp_lsnv_handle_t** out,
                                         int32_t window, int32_t pad_mode,
                                         double constant_value);
N4M_API void         n4m_pp_lsnv_destroy(n4m_pp_lsnv_handle_t* handle);
N4M_API n4m_status_t n4m_pp_lsnv_transform(const n4m_pp_lsnv_handle_t* handle,
                                            n4m_matrix_view_t X,
                                            n4m_matrix_view_t out);

/* ---------- RobustSNV (median + k * MAD) --------------------------------- */
typedef struct n4m_pp_rnv_handle_t n4m_pp_rnv_handle_t;
N4M_API n4m_status_t n4m_pp_rnv_create(n4m_pp_rnv_handle_t** out,
                                        int with_center, int with_scale,
                                        double k);
N4M_API void         n4m_pp_rnv_destroy(n4m_pp_rnv_handle_t* handle);
N4M_API n4m_status_t n4m_pp_rnv_transform(const n4m_pp_rnv_handle_t* handle,
                                           n4m_matrix_view_t X,
                                           n4m_matrix_view_t out);

/* ---------- AreaNormalization -------------------------------------------- */
typedef struct n4m_pp_area_handle_t n4m_pp_area_handle_t;
typedef enum n4m_pp_area_method_t {
    N4M_PP_AREA_SUM     = 0,
    N4M_PP_AREA_ABS_SUM = 1,
    N4M_PP_AREA_TRAPZ   = 2
} n4m_pp_area_method_t;
N4M_API n4m_status_t n4m_pp_area_create(n4m_pp_area_handle_t** out,
                                         int32_t method);
N4M_API void         n4m_pp_area_destroy(n4m_pp_area_handle_t* handle);
N4M_API n4m_status_t n4m_pp_area_transform(const n4m_pp_area_handle_t* handle,
                                            n4m_matrix_view_t X,
                                            n4m_matrix_view_t out);

/* ---------- Normalize (column-wise) -------------------------------------- */
typedef struct n4m_pp_normalize_handle_t n4m_pp_normalize_handle_t;
/* `(feature_min, feature_max) == (-1, 1)` selects linalg-norm mode; any
 * other pair selects user-defined-range mode. */
N4M_API n4m_status_t n4m_pp_normalize_create(n4m_pp_normalize_handle_t** out,
                                              double feature_min,
                                              double feature_max);
N4M_API void         n4m_pp_normalize_destroy(n4m_pp_normalize_handle_t* handle);
N4M_API n4m_status_t n4m_pp_normalize_transform(
    const n4m_pp_normalize_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out);

/* ---------- SimpleScale (column-wise min-max → [0, 1]) ------------------- */
typedef struct n4m_pp_simple_scale_handle_t n4m_pp_simple_scale_handle_t;
N4M_API n4m_status_t n4m_pp_simple_scale_create(
    n4m_pp_simple_scale_handle_t** out);
N4M_API void         n4m_pp_simple_scale_destroy(
    n4m_pp_simple_scale_handle_t* handle);
N4M_API n4m_status_t n4m_pp_simple_scale_transform(
    const n4m_pp_simple_scale_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out);

/* ---------- LogTransform ------------------------------------------------- */
typedef struct n4m_pp_log_handle_t n4m_pp_log_handle_t;
/* `base == 0.0` selects the natural logarithm. `base` must be > 0 and != 1
 * otherwise. `min_value` must be > 0 (positive clamp target).
 *
 * Lifecycle (Phase 5b split):
 *
 *   - When ``auto_offset == 0`` the operator is STATELESS: the user-supplied
 *     ``offset`` is applied verbatim. ``_transform`` may be called directly
 *     without calling ``_fit`` first.
 *
 *   - When ``auto_offset != 0`` the offset is captured at FIT TIME: the
 *     caller must invoke ``n4m_pp_log_fit`` once on the training matrix.
 *     The fitted offset is cached on the handle and re-used by subsequent
 *     ``_transform`` calls. Calling ``_transform`` before ``_fit`` returns
 *     N4M_ERR_NOT_FITTED. Calling ``_fit`` again replaces the prior fit
 *     (sklearn-style refit semantics).
 *
 *   This preserves the pre-Phase-5b behaviour of ``auto_offset == 0`` while
 *   giving ``auto_offset != 0`` proper sklearn-style fit-once/transform-many
 *   semantics.
 */
N4M_API n4m_status_t n4m_pp_log_create(n4m_pp_log_handle_t** out,
                                        double base, double offset,
                                        int auto_offset, double min_value);
N4M_API void         n4m_pp_log_destroy(n4m_pp_log_handle_t* handle);
N4M_API n4m_status_t n4m_pp_log_fit(n4m_pp_log_handle_t* handle,
                                     n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_pp_log_is_fitted(const n4m_pp_log_handle_t* handle,
                                           int* out_fitted);
N4M_API n4m_status_t n4m_pp_log_transform(const n4m_pp_log_handle_t* handle,
                                           n4m_matrix_view_t X,
                                           n4m_matrix_view_t out);

/* ============================================================================
 * 9. Phase 3 — Stateful scatter preprocessings
 * ============================================================================
 *
 * Four stateful operators implementing the `_create / _fit / _transform /
 * _destroy` ABI contract from §5. Each operator stores parameters learned
 * from a training matrix and replays them on subsequent calls. Calling
 * `_transform` (or `_inverse_transform`) before `_fit` returns
 * N4M_ERR_NOT_FITTED.
 *
 * Reference: nirs4all 0.8.x operators in `operators/transforms/{nirs,
 * signal,scalers}.py`.
 */

/* ---------- MSC (Multiplicative Scatter Correction) ---------------------- */
/*
 * Conventional row-wise scatter correction calibrated against the mean
 * spectrum of the training matrix:
 *   reference[j] = mean(X[:, j])              # length n_features
 *   (offset[i], slope[i]) = ols(X[i, :] ~ 1 + reference)
 *   X'[i, j] = (X[i, j] - offset[i]) / slope[i]
 * Inverse uses the row coefficients stored by the last `_transform` call on
 * the same handle:
 *   X[i, j] = X'[i, j] * slope[i] + offset[i]
 *
 * Matches `prospectr::msc` / `pls::msc` for the default reference-spectrum
 * contract. The historical nirs4all column-regression variant is intentionally
 * not the validation target for this ABI.
 *
 * `_fit` requires at least one row and at least two columns.
 */
typedef struct n4m_pp_msc_handle_t n4m_pp_msc_handle_t;
N4M_API n4m_status_t n4m_pp_msc_create(n4m_pp_msc_handle_t** out);
N4M_API void         n4m_pp_msc_destroy(n4m_pp_msc_handle_t* handle);
N4M_API n4m_status_t n4m_pp_msc_fit(n4m_pp_msc_handle_t* handle,
                                     n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_pp_msc_transform(n4m_pp_msc_handle_t* handle,
                                           n4m_matrix_view_t X,
                                           n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_msc_inverse_transform(
    const n4m_pp_msc_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_msc_is_fitted(const n4m_pp_msc_handle_t* handle,
                                           int* out_fitted);

/* ---------- EMSC (Extended Multiplicative Scatter Correction) ------------ */
/*
 * Per-sample scatter correction with polynomial wavelength terms. For each
 * row x_i, solve the least-squares system
 *     x_i ≈ c[0] * ref + sum_{d=1..degree} c[d] * wavelengths^d
 * then subtract the polynomial portion and divide by the reference
 * coefficient:
 *     x'_i = (x_i - sum_{d=1..degree} c[d] * wavelengths^d) / c[0]
 *
 * `ref` is the per-column mean of the training matrix (length n_features).
 * `wavelengths` is the integer index 0, 1, …, n_features - 1 (matching the
 * nirs4all convention; rescaling the axis to [-1, 1] would change the
 * polynomial coefficients but not the result after subtraction-and-divide).
 *
 * No inverse transform: the polynomial subtraction is per-row data-driven
 * and discards the row-specific coefficients, so inversion would require
 * re-storing them.
 *
 * `_create` requires `degree >= 1`.
 * `_fit`    requires `X.rows >= 1` and `X.cols >= degree + 2` (basis
 *           dimensionality + reference).
 *
 * Matches `nirs4all.operators.transforms.nirs.ExtendedMultiplicativeScatterCorrection`
 * with `scale=False`.
 */
typedef struct n4m_pp_emsc_handle_t n4m_pp_emsc_handle_t;
N4M_API n4m_status_t n4m_pp_emsc_create(n4m_pp_emsc_handle_t** out,
                                         int32_t degree);
N4M_API void         n4m_pp_emsc_destroy(n4m_pp_emsc_handle_t* handle);
N4M_API n4m_status_t n4m_pp_emsc_fit(n4m_pp_emsc_handle_t* handle,
                                      n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_pp_emsc_transform(const n4m_pp_emsc_handle_t* handle,
                                            n4m_matrix_view_t X,
                                            n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_emsc_is_fitted(const n4m_pp_emsc_handle_t* handle,
                                            int* out_fitted);

/* ---------- Baseline (column-mean centering) ----------------------------- */
/*
 * Per-column mean centering. `_fit` learns `mean[j] = mean(X[:, j])`.
 * `_transform` writes `out[i, j] = X[i, j] - mean[j]`.
 * `_inverse_transform` writes `out[i, j] = X[i, j] + mean[j]`.
 *
 * Matches `nirs4all.operators.transforms.signal.Baseline` — equivalent to
 * `sklearn.preprocessing.StandardScaler(with_std=False)`. Note the class
 * name is historical; the operator does NOT perform baseline correction in
 * the spectroscopic sense (a polynomial baseline removal lives in Phase 5).
 */
typedef struct n4m_pp_baseline_handle_t n4m_pp_baseline_handle_t;
N4M_API n4m_status_t n4m_pp_baseline_create(n4m_pp_baseline_handle_t** out);
N4M_API void         n4m_pp_baseline_destroy(n4m_pp_baseline_handle_t* handle);
N4M_API n4m_status_t n4m_pp_baseline_fit(n4m_pp_baseline_handle_t* handle,
                                          n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_pp_baseline_transform(
    const n4m_pp_baseline_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_baseline_inverse_transform(
    const n4m_pp_baseline_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_baseline_is_fitted(
    const n4m_pp_baseline_handle_t* handle,
    int* out_fitted);

/* ---------- Derivate (finite-difference derivative along axis=1) --------- */
/*
 * Finite-difference derivative of order `order` (>= 1) along the wavelength
 * axis (axis=1). The output has shape (rows, cols - order); each derivative
 * step shortens the column count by one. Division by `delta**order` yields
 * the physical-units derivative when `delta` is the wavelength sample
 * spacing.
 *
 *   out = np.diff(X, n=order, axis=1) / delta ** order
 *
 * `_fit` memoises the input column count for shape validation at transform
 * time — it carries no statistical learned state, but it does record `cols`
 * so that `_transform` can reject inputs whose width disagrees with what
 * was seen at fit time. Calling `_transform` before `_fit` returns
 * N4M_ERR_NOT_FITTED; calling it with a column count different from the
 * fitted value returns N4M_ERR_SHAPE_MISMATCH. This lets bindings treat
 * Derivate uniformly with the other stateful preprocessings without a
 * special case.
 *
 * Use `n4m_pp_derivate_output_cols(order, input_cols)` to compute the
 * output column count expected by `_transform`. The helper returns 0 when
 * `order >= input_cols`.
 */
typedef struct n4m_pp_derivate_handle_t n4m_pp_derivate_handle_t;
N4M_API n4m_status_t n4m_pp_derivate_create(n4m_pp_derivate_handle_t** out,
                                             int32_t order, double delta);
N4M_API void         n4m_pp_derivate_destroy(n4m_pp_derivate_handle_t* handle);
N4M_API n4m_status_t n4m_pp_derivate_fit(n4m_pp_derivate_handle_t* handle,
                                          n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_pp_derivate_transform(
    const n4m_pp_derivate_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out);
N4M_API int64_t      n4m_pp_derivate_output_cols(int32_t order,
                                                  int64_t input_cols);

/* ============================================================================
 * 10. Phase 4 — Stateless derivatives & smoothing preprocessings
 * ============================================================================
 *
 * Five stateless smoothing / derivative operators that complete the spectral
 * preprocessing toolkit alongside the Phase 2/3 scatter and scaling stages.
 * All implement the `_create / _transform / _destroy` ABI contract from §5
 * (no `_fit`).
 *
 *   - SavitzkyGolay     : polynomial least-squares smoothing & differentiation
 *                         along axis=1 with 5 boundary modes.  Matches
 *                         `scipy.signal.savgol_filter`.
 *   - FirstDerivative   : `np.gradient(X, delta, axis=1, edge_order=...)`.
 *                         Shape-preserving central differences.
 *   - SecondDerivative  : two passes of `np.gradient` (shape-preserving).
 *   - NorrisWilliams    : segment-smooth + gap-difference, applied
 *                         `derivative_order` times.  Matches
 *                         `nirs4all.operators.transforms.norris_williams`.
 *   - Gaussian          : `scipy.ndimage.gaussian_filter1d` with the same
 *                         5 boundary modes as the SciPy reference.
 *
 * Status semantics follow the universal rules at the top of this header.
 * Matrix-view validation may additionally return N4M_ERR_DTYPE_MISMATCH
 * (non-F64), N4M_ERR_STRIDE_INVALID (non-contiguous row-major), or
 * N4M_ERR_SHAPE_MISMATCH (X and out shapes disagree).
 */

/* ---------- SavitzkyGolay -------------------------------------------------
 *
 * Parameters:
 *   window_length : odd integer >= 1
 *   polyorder     : 0 <= polyorder < window_length
 *   deriv         : derivative order >= 0
 *   delta         : sample spacing (non-zero)
 *   mode          : boundary handling, see n4m_pp_savgol_mode_t below
 *   cval          : fill value used when mode == N4M_PP_SAVGOL_CONSTANT
 *
 * The five modes mirror `scipy.signal.savgol_filter`:
 *   - MIRROR   : reflection without repeating the edge (default).
 *   - CONSTANT : pad with `cval`.
 *   - NEAREST  : replicate the edge sample.
 *   - WRAP     : cyclic.
 *   - INTERP   : fit `polyorder` polynomials to the first and last
 *                window_length samples, evaluate at the half-window
 *                positions on each side.
 */
typedef struct n4m_pp_savgol_handle_t n4m_pp_savgol_handle_t;
typedef enum n4m_pp_savgol_mode_t {
    N4M_PP_SAVGOL_MIRROR   = 0,
    N4M_PP_SAVGOL_CONSTANT = 1,
    N4M_PP_SAVGOL_NEAREST  = 2,
    N4M_PP_SAVGOL_WRAP     = 3,
    N4M_PP_SAVGOL_INTERP   = 4
} n4m_pp_savgol_mode_t;
N4M_API n4m_status_t n4m_pp_savgol_create(n4m_pp_savgol_handle_t** out,
                                           int32_t window_length,
                                           int32_t polyorder,
                                           int32_t deriv,
                                           double delta,
                                           n4m_pp_savgol_mode_t mode,
                                           double cval);
N4M_API void         n4m_pp_savgol_destroy(n4m_pp_savgol_handle_t* handle);
N4M_API n4m_status_t n4m_pp_savgol_transform(
    const n4m_pp_savgol_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out);

/* ---------- FirstDerivative -----------------------------------------------
 *
 * `np.gradient(X, delta, axis=1, edge_order)` with edge_order in {1, 2}.
 * Output shape matches input shape.  `delta` must be non-zero.
 */
typedef struct n4m_pp_first_derivative_handle_t
    n4m_pp_first_derivative_handle_t;
N4M_API n4m_status_t n4m_pp_first_derivative_create(
    n4m_pp_first_derivative_handle_t** out, double delta, int32_t edge_order);
N4M_API void         n4m_pp_first_derivative_destroy(
    n4m_pp_first_derivative_handle_t* handle);
N4M_API n4m_status_t n4m_pp_first_derivative_transform(
    const n4m_pp_first_derivative_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out);

/* ---------- SecondDerivative ----------------------------------------------
 *
 * Two passes of `np.gradient(X, delta, axis=1, edge_order)`.  Shape-preserving.
 */
typedef struct n4m_pp_second_derivative_handle_t
    n4m_pp_second_derivative_handle_t;
N4M_API n4m_status_t n4m_pp_second_derivative_create(
    n4m_pp_second_derivative_handle_t** out, double delta, int32_t edge_order);
N4M_API void         n4m_pp_second_derivative_destroy(
    n4m_pp_second_derivative_handle_t* handle);
N4M_API n4m_status_t n4m_pp_second_derivative_transform(
    const n4m_pp_second_derivative_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out);

/* ---------- NorrisWilliams -------------------------------------------------
 *
 * `segment` must be odd and >= 1 (1 disables smoothing).  `gap` must be >= 1.
 * `derivative_order` must be 1 or 2.  `delta` must be non-zero.  Shape-preserving.
 */
typedef struct n4m_pp_norris_williams_handle_t
    n4m_pp_norris_williams_handle_t;
N4M_API n4m_status_t n4m_pp_norris_williams_create(
    n4m_pp_norris_williams_handle_t** out,
    int32_t segment, int32_t gap, int32_t derivative_order, double delta);
N4M_API void         n4m_pp_norris_williams_destroy(
    n4m_pp_norris_williams_handle_t* handle);
N4M_API n4m_status_t n4m_pp_norris_williams_transform(
    const n4m_pp_norris_williams_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out);

/* ---------- Gaussian -------------------------------------------------------
 *
 * `scipy.ndimage.gaussian_filter1d(X, sigma, order, axis=1, mode, cval,
 * truncate)`.  Kernel radius is `int(truncate * sigma + 0.5)`.  Sigma must be
 * positive, order >= 0, truncate >= 0.
 *
 * Boundary modes mirror SciPy.ndimage:
 *   - REFLECT  : edge-repeating reflection (default).
 *   - CONSTANT : pad with `cval`.
 *   - NEAREST  : replicate the edge sample.
 *   - MIRROR   : non-edge-repeating reflection.
 *   - WRAP     : cyclic.
 */
typedef struct n4m_pp_gaussian_handle_t n4m_pp_gaussian_handle_t;
typedef enum n4m_pp_gaussian_mode_t {
    N4M_PP_GAUSSIAN_REFLECT  = 0,
    N4M_PP_GAUSSIAN_CONSTANT = 1,
    N4M_PP_GAUSSIAN_NEAREST  = 2,
    N4M_PP_GAUSSIAN_MIRROR   = 3,
    N4M_PP_GAUSSIAN_WRAP     = 4
} n4m_pp_gaussian_mode_t;
N4M_API n4m_status_t n4m_pp_gaussian_create(
    n4m_pp_gaussian_handle_t** out,
    double sigma, int32_t order,
    n4m_pp_gaussian_mode_t mode,
    double cval, double truncate);
N4M_API void         n4m_pp_gaussian_destroy(n4m_pp_gaussian_handle_t* handle);
N4M_API n4m_status_t n4m_pp_gaussian_transform(
    const n4m_pp_gaussian_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out);

/* ============================================================================
 * 11. Phase 5a — Baseline correction (core)
 * ============================================================================
 *
 * Four stateless baseline correction operators, all implementing the
 * `_create / _transform / _destroy` ABI contract from §5 (no `_fit`).
 *
 *   - Detrend   : polynomial baseline subtraction (np.polyfit / np.polyval).
 *   - AsLS      : Asymmetric Least Squares (Eilers & Boelens 2005).
 *   - AirPLS    : Adaptive iteratively reweighted PLS (Zhang 2010).
 *   - ArPLS     : Asymmetrically reweighted PLS (Baek 2015).
 *
 * The three iterative operators (AsLS, AirPLS, ArPLS) all use the same
 * underlying pentadiagonal LDLT solver against a 2nd-order penalty
 * `lam * D2^T D2`. They iterate the reweighting loop up to `max_iter`
 * times; on max_iter exhaustion they silently return the last iterate
 * (matching pybaselines 1.1.4 semantics — convergence does NOT raise).
 *
 * Output convention: `out[i, j] = X[i, j] - baseline[i, j]`. For all four
 * operators the output has the same shape as the input.
 *
 * Parity reference: the frozen NumPy reference under
 *   parity/python_generator/src/n4m_parity_pybaselines_ref/
 * validated once against pybaselines==1.1.4 (see parity/README.md).
 */

/* ---------- Detrend (polynomial baseline subtraction) ------------------- */
typedef struct n4m_pp_detrend_handle_t n4m_pp_detrend_handle_t;
/* `polyorder` >= 0. Default in pybaselines.polynomial: 1 (linear detrend). */
N4M_API n4m_status_t n4m_pp_detrend_create(n4m_pp_detrend_handle_t** out,
                                            int32_t polyorder);
N4M_API void         n4m_pp_detrend_destroy(n4m_pp_detrend_handle_t* handle);
N4M_API n4m_status_t n4m_pp_detrend_transform(
    const n4m_pp_detrend_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out);

/* ---------- AsLS (Eilers & Boelens 2005) -------------------------------- */
typedef struct n4m_pp_asls_handle_t n4m_pp_asls_handle_t;
/* `lam` > 0  (smoothing penalty; default 1e6).
 * `p`        (asymmetry, 0 < p < 1; default 1e-2).
 * `max_iter` >= 0 (default 50).
 * `tol`      >= 0 (default 1e-3, relative L2 weight change). */
N4M_API n4m_status_t n4m_pp_asls_create(n4m_pp_asls_handle_t** out,
                                         double lam, double p,
                                         int32_t max_iter, double tol);
N4M_API void         n4m_pp_asls_destroy(n4m_pp_asls_handle_t* handle);
N4M_API n4m_status_t n4m_pp_asls_transform(const n4m_pp_asls_handle_t* handle,
                                            n4m_matrix_view_t X,
                                            n4m_matrix_view_t out);

/* ---------- AirPLS (Zhang 2010) ----------------------------------------- */
typedef struct n4m_pp_airpls_handle_t n4m_pp_airpls_handle_t;
/* `lam`      > 0 (default 1e6).
 * `max_iter` >= 0 (default 50).
 * `tol`      >= 0 (default 1e-3, |sum(neg residuals)| / |y|_1). */
N4M_API n4m_status_t n4m_pp_airpls_create(n4m_pp_airpls_handle_t** out,
                                           double lam,
                                           int32_t max_iter, double tol);
N4M_API void         n4m_pp_airpls_destroy(n4m_pp_airpls_handle_t* handle);
N4M_API n4m_status_t n4m_pp_airpls_transform(
    const n4m_pp_airpls_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out);

/* ---------- ArPLS (Baek 2015) ------------------------------------------- */
typedef struct n4m_pp_arpls_handle_t n4m_pp_arpls_handle_t;
/* `lam`      > 0 (default 1e5).
 * `max_iter` >= 0 (default 50).
 * `tol`      >= 0 (default 1e-3, relative L2 weight change). */
N4M_API n4m_status_t n4m_pp_arpls_create(n4m_pp_arpls_handle_t** out,
                                          double lam,
                                          int32_t max_iter, double tol);
N4M_API void         n4m_pp_arpls_destroy(n4m_pp_arpls_handle_t* handle);
N4M_API n4m_status_t n4m_pp_arpls_transform(const n4m_pp_arpls_handle_t* handle,
                                             n4m_matrix_view_t X,
                                             n4m_matrix_view_t out);

/* ============================================================================
 * 12. Phase 5b — Baseline correction (rest)
 * ============================================================================
 *
 * Six additional stateless baseline correction operators completing the
 * family started in Phase 5a:
 *
 *   - ModPoly       : Iterative polynomial baseline with peak-clipping
 *                     (Lieber & Mahadevan-Jansen 2003).
 *   - IModPoly      : Improved ModPoly with σ-based stopping (Gan 2006).
 *   - SNIP          : Statistics-sensitive Non-linear Iterative Peak-clipping
 *                     (Ryan 1988, Morháč 1997).
 *   - RollingBall   : Morphological rolling-ball baseline (Kneen & Annegarn
 *                     1996) — min-then-max filter with window `half_window`.
 *   - IAsLS         : Improved AsLS (He 2014) — polynomial prefit followed by
 *                     AsLS-style banded reweighting.
 *   - BEADS         : Baseline Estimation And Denoising with Sparsity
 *                     (Ning & Selesnick 2014), matching pybaselines' full
 *                     banded BEADS contract for the exposed parameters.
 *
 * All six operators implement the `_create / _transform / _destroy` ABI
 * contract from §5 (stateless — no `_fit`). Each operator subtracts the
 * estimated baseline and writes `out = X - baseline`. Convergence-failure
 * semantics match Phase 5a: silently returns the last iterate at max_iter
 * exhaustion.
 *
 * Parity reference: pybaselines snapshots stored under
 *   benchmarks/reference_snapshots/cross_binding/
 * and generated with reference-library metadata for reproducibility.
 */

/* ---------- ModPoly (Lieber & Mahadevan-Jansen 2003) -------------------- */
typedef struct n4m_pp_modpoly_handle_t n4m_pp_modpoly_handle_t;
/* `polyorder` >= 0 (default 2).
 * `max_iter`  >= 0 (default 250).
 * `tol`       >= 0 (default 1e-3, relative L2 of the baseline change). */
N4M_API n4m_status_t n4m_pp_modpoly_create(n4m_pp_modpoly_handle_t** out,
                                            int32_t polyorder,
                                            int32_t max_iter, double tol);
N4M_API void         n4m_pp_modpoly_destroy(n4m_pp_modpoly_handle_t* handle);
N4M_API n4m_status_t n4m_pp_modpoly_transform(
    const n4m_pp_modpoly_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out);

/* ---------- IModPoly (Gan, Ruan, Mo 2006) ------------------------------ */
typedef struct n4m_pp_imodpoly_handle_t n4m_pp_imodpoly_handle_t;
/* `polyorder` >= 0 (default 2).
 * `max_iter`  >= 0 (default 250).
 * `tol`       >= 0 (default 1e-3, relative change in residual stdev). */
N4M_API n4m_status_t n4m_pp_imodpoly_create(n4m_pp_imodpoly_handle_t** out,
                                             int32_t polyorder,
                                             int32_t max_iter, double tol);
N4M_API void         n4m_pp_imodpoly_destroy(n4m_pp_imodpoly_handle_t* handle);
N4M_API n4m_status_t n4m_pp_imodpoly_transform(
    const n4m_pp_imodpoly_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out);

/* ---------- SNIP (Ryan 1988, Morháč 1997) ------------------------------ */
typedef struct n4m_pp_snip_handle_t n4m_pp_snip_handle_t;
/* `max_half_window` >= 1 (default 20). The algorithm matches
 * pybaselines.smooth.snip's raw-data filter_order=2 contract: linear edge
 * extrapolation, then simultaneous local-min clipping for each window width
 * w in [1, max_half_window]. */
N4M_API n4m_status_t n4m_pp_snip_create(n4m_pp_snip_handle_t** out,
                                         int32_t max_half_window);
N4M_API void         n4m_pp_snip_destroy(n4m_pp_snip_handle_t* handle);
N4M_API n4m_status_t n4m_pp_snip_transform(const n4m_pp_snip_handle_t* handle,
                                            n4m_matrix_view_t X,
                                            n4m_matrix_view_t out);

/* ---------- RollingBall (Kneen & Annegarn 1996) ------------------------ */
typedef struct n4m_pp_rolling_ball_handle_t n4m_pp_rolling_ball_handle_t;
/* `half_window`        >= 1 (default 20). Window radius for the min-then-max
 *                            morphological filter.
 * `smooth_half_window` >= 0 (default 0). Optional moving-average smoothing of
 *                            the final baseline. 0 disables smoothing. */
N4M_API n4m_status_t n4m_pp_rolling_ball_create(
    n4m_pp_rolling_ball_handle_t** out,
    int32_t half_window, int32_t smooth_half_window);
N4M_API void         n4m_pp_rolling_ball_destroy(
    n4m_pp_rolling_ball_handle_t* handle);
N4M_API n4m_status_t n4m_pp_rolling_ball_transform(
    const n4m_pp_rolling_ball_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out);

/* ---------- IAsLS (He 2014) -------------------------------------------- */
typedef struct n4m_pp_iasls_handle_t n4m_pp_iasls_handle_t;
/* `lam`        > 0 (default 1e6).
 * `p`          (asymmetry, 0 < p < 1; default 1e-2).
 * `lam_1`      > 0 (default 1e-4, first-derivative residual penalty).
 * `polyorder`  >= 0 (default 2, prefit polynomial degree; pybaselines uses 2).
 * `diff_order` == 2 in this ABI revision.
 * `max_iter`   >= 0 (default 50).
 * `tol`        >= 0 (default 1e-3, relative L2 weight change). */
N4M_API n4m_status_t n4m_pp_iasls_create(n4m_pp_iasls_handle_t** out,
                                          double lam, double p,
                                          int32_t polyorder,
                                          int32_t max_iter, double tol);
N4M_API n4m_status_t n4m_pp_iasls_create_ex(n4m_pp_iasls_handle_t** out,
                                             double lam, double p,
                                             double lam_1,
                                             int32_t polyorder,
                                             int32_t diff_order,
                                             int32_t max_iter, double tol);
N4M_API void         n4m_pp_iasls_destroy(n4m_pp_iasls_handle_t* handle);
N4M_API n4m_status_t n4m_pp_iasls_transform(const n4m_pp_iasls_handle_t* handle,
                                             n4m_matrix_view_t X,
                                             n4m_matrix_view_t out);

/* ---------- BEADS — Ning & Selesnick 2014 ------------------------------ */
typedef struct n4m_pp_beads_handle_t n4m_pp_beads_handle_t;
/* `lam_0`    > 0 (default 1e2, sparsity weight on baseline residual).
 * `lam_1`    > 0 (default 0.5, banded 1st-difference smoothness weight).
 * `lam_2`    > 0 (default 0.5, banded 2nd-difference smoothness weight).
 * `max_iter` >= 0 (default 50).
 * `tol`      >= 0 (default 1e-3, relative L2 baseline change).
 *
 * Fixed BEADS options match pybaselines defaults for the public n4m surface:
 * freq_cutoff=0.005, asymmetry=6, filter_type=1, cost_function=2,
 * eps_0=eps_1=1e-6, fit_parabola=true. */
N4M_API n4m_status_t n4m_pp_beads_create(n4m_pp_beads_handle_t** out,
                                          double lam_0, double lam_1,
                                          double lam_2,
                                          int32_t max_iter, double tol);
N4M_API void         n4m_pp_beads_destroy(n4m_pp_beads_handle_t* handle);
N4M_API n4m_status_t n4m_pp_beads_transform(const n4m_pp_beads_handle_t* handle,
                                             n4m_matrix_view_t X,
                                             n4m_matrix_view_t out);

/* ============================================================================
 * 13. Phase 7 — Signal conversion preprocessings
 * ============================================================================
 *
 * Five closed-form spectroscopy unit conversions:
 *
 *   - ToAbsorbance      : A = -log10(max(R, eps)).
 *   - FromAbsorbance    : R = 10^(-A). Optional %-scaling.
 *   - PercentToFraction : R_frac = R_pct / 100.
 *   - FractionToPercent : R_pct  = R_frac * 100.
 *   - KubelkaMunk       : KM = (1 - R)^2 / (2 R), with R guarded by epsilon.
 *
 * All five operators are stateless (`_create / _transform / _destroy`).
 * Output buffers may alias the input (in-place is supported).
 */

/* ---------- ToAbsorbance ----------------------------------------------- */
typedef struct n4m_pp_to_absorbance_handle_t n4m_pp_to_absorbance_handle_t;
N4M_API n4m_status_t n4m_pp_to_absorbance_create(
    n4m_pp_to_absorbance_handle_t** out,
    int is_percent, double epsilon, int clip_negative);
N4M_API void         n4m_pp_to_absorbance_destroy(
    n4m_pp_to_absorbance_handle_t* handle);
N4M_API n4m_status_t n4m_pp_to_absorbance_transform(
    const n4m_pp_to_absorbance_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

/* ---------- FromAbsorbance --------------------------------------------- */
typedef struct n4m_pp_from_absorbance_handle_t n4m_pp_from_absorbance_handle_t;
N4M_API n4m_status_t n4m_pp_from_absorbance_create(
    n4m_pp_from_absorbance_handle_t** out, int is_percent);
N4M_API void         n4m_pp_from_absorbance_destroy(
    n4m_pp_from_absorbance_handle_t* handle);
N4M_API n4m_status_t n4m_pp_from_absorbance_transform(
    const n4m_pp_from_absorbance_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

/* ---------- PercentToFraction ----------------------------------------- */
typedef struct n4m_pp_pct_to_frac_handle_t n4m_pp_pct_to_frac_handle_t;
N4M_API n4m_status_t n4m_pp_pct_to_frac_create(
    n4m_pp_pct_to_frac_handle_t** out);
N4M_API void         n4m_pp_pct_to_frac_destroy(
    n4m_pp_pct_to_frac_handle_t* handle);
N4M_API n4m_status_t n4m_pp_pct_to_frac_transform(
    const n4m_pp_pct_to_frac_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

/* ---------- FractionToPercent ----------------------------------------- */
typedef struct n4m_pp_frac_to_pct_handle_t n4m_pp_frac_to_pct_handle_t;
N4M_API n4m_status_t n4m_pp_frac_to_pct_create(
    n4m_pp_frac_to_pct_handle_t** out);
N4M_API void         n4m_pp_frac_to_pct_destroy(
    n4m_pp_frac_to_pct_handle_t* handle);
N4M_API n4m_status_t n4m_pp_frac_to_pct_transform(
    const n4m_pp_frac_to_pct_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

/* ---------- KubelkaMunk ----------------------------------------------- */
typedef struct n4m_pp_kubelka_munk_handle_t n4m_pp_kubelka_munk_handle_t;
N4M_API n4m_status_t n4m_pp_kubelka_munk_create(
    n4m_pp_kubelka_munk_handle_t** out, int is_percent, double epsilon);
N4M_API void         n4m_pp_kubelka_munk_destroy(
    n4m_pp_kubelka_munk_handle_t* handle);
N4M_API n4m_status_t n4m_pp_kubelka_munk_transform(
    const n4m_pp_kubelka_munk_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

/* ============================================================================
 * 14. Phase 8 — Orthogonalization (OSC, EPO)
 * ============================================================================
 *
 * Both operators expose the stateful `_create / _fit / _transform / _destroy
 * / _is_fitted` ABI contract. OSC additionally exposes `_inverse_transform`
 * which always returns N4M_ERR_UNSUPPORTED (the orthogonal-component
 * deflation is many-to-one). EPO has its own `_inverse_transform`.
 */

/* ---------- Orthogonal Signal Correction (Wold 1998) ------------------- */
typedef struct n4m_pp_osc_handle_t n4m_pp_osc_handle_t;
N4M_API n4m_status_t n4m_pp_osc_create(n4m_pp_osc_handle_t** out,
                                        int32_t n_components, int scale);
N4M_API void         n4m_pp_osc_destroy(n4m_pp_osc_handle_t* handle);
N4M_API n4m_status_t n4m_pp_osc_fit(n4m_pp_osc_handle_t* handle,
                                     n4m_matrix_view_t X,
                                     const double* y, int64_t y_len);
N4M_API n4m_status_t n4m_pp_osc_transform(const n4m_pp_osc_handle_t* handle,
                                           n4m_matrix_view_t X,
                                           n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_osc_inverse_transform(
    const n4m_pp_osc_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_osc_is_fitted(const n4m_pp_osc_handle_t* handle,
                                           int* out_fitted);

/* ---------- External Parameter Orthogonalisation (Roger 2003) ---------- */
typedef struct n4m_pp_epo_handle_t n4m_pp_epo_handle_t;
N4M_API n4m_status_t n4m_pp_epo_create(n4m_pp_epo_handle_t** out, int scale);
N4M_API void         n4m_pp_epo_destroy(n4m_pp_epo_handle_t* handle);
N4M_API n4m_status_t n4m_pp_epo_fit(n4m_pp_epo_handle_t* handle,
                                     n4m_matrix_view_t X,
                                     const double* d, int64_t d_len);
N4M_API n4m_status_t n4m_pp_epo_transform(const n4m_pp_epo_handle_t* handle,
                                           n4m_matrix_view_t X,
                                           n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_epo_transform_with_d(
    const n4m_pp_epo_handle_t* handle,
    n4m_matrix_view_t X,
    const double* d, int64_t d_len,
    n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_epo_inverse_transform(
    const n4m_pp_epo_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_epo_is_fitted(const n4m_pp_epo_handle_t* handle,
                                           int* out_fitted);

/* ============================================================================
 * 15. Phase 9 — Feature selection (FlexiblePCA, FlexibleSVD)
 * ============================================================================
 *
 * Stateful dimensionality reduction. `n_components` may be a positive
 * integer (number of components) or a value in (0, 1] interpreted as the
 * target cumulative explained-variance ratio (FlexiblePCA only). Use
 * `_output_cols` to query the post-fit output column count.
 */

/* ---------- FlexiblePCA ------------------------------------------------ */
typedef struct n4m_pp_flex_pca_handle_t n4m_pp_flex_pca_handle_t;
N4M_API n4m_status_t n4m_pp_flex_pca_create(n4m_pp_flex_pca_handle_t** out,
                                             double n_components);
N4M_API void         n4m_pp_flex_pca_destroy(n4m_pp_flex_pca_handle_t* handle);
N4M_API n4m_status_t n4m_pp_flex_pca_fit(n4m_pp_flex_pca_handle_t* handle,
                                          n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_pp_flex_pca_transform(
    const n4m_pp_flex_pca_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_flex_pca_is_fitted(
    const n4m_pp_flex_pca_handle_t* handle, int* out_fitted);
N4M_API n4m_status_t n4m_pp_flex_pca_output_cols(
    const n4m_pp_flex_pca_handle_t* handle, int64_t* out_cols);

/* ---------- FlexibleSVD ------------------------------------------------ */
typedef struct n4m_pp_flex_svd_handle_t n4m_pp_flex_svd_handle_t;
N4M_API n4m_status_t n4m_pp_flex_svd_create(n4m_pp_flex_svd_handle_t** out,
                                             double n_components);
N4M_API void         n4m_pp_flex_svd_destroy(n4m_pp_flex_svd_handle_t* handle);
N4M_API n4m_status_t n4m_pp_flex_svd_fit(n4m_pp_flex_svd_handle_t* handle,
                                          n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_pp_flex_svd_transform(
    const n4m_pp_flex_svd_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_flex_svd_is_fitted(
    const n4m_pp_flex_svd_handle_t* handle, int* out_fitted);
N4M_API n4m_status_t n4m_pp_flex_svd_output_cols(
    const n4m_pp_flex_svd_handle_t* handle, int64_t* out_cols);

/* ============================================================================
 * 16. Phase 10 — Resampling, cropping, target discretization
 * ============================================================================
 *
 * Five operators covering wavelength resampling, cropping and target-value
 * discretization.
 *
 *   - Resampler        : interpolate spectra onto a target wavelength grid
 *                        (linear / cubic / nearest).
 *   - CropTransformer  : slice columns [start, end).
 *   - ResampleTransformer : trim/replicate columns to a target length.
 *   - IntegerKBinsDiscretizer : per-column equal-width / quantile binning.
 *   - RangeDiscretizer : monotonic edge-based binning (integer output).
 *
 * The two discretizers produce int32 outputs and use the I32 dtype on the
 * output view.
 */

/* ---------- Resampler (stateful) -------------------------------------- */
typedef struct n4m_pp_resampler_handle_t n4m_pp_resampler_handle_t;
N4M_API n4m_status_t n4m_pp_resampler_create(n4m_pp_resampler_handle_t** out,
                                              const double* target_wl,
                                              int64_t n_target,
                                              int32_t method,
                                              double crop_min, double crop_max,
                                              int use_crop, double fill_value,
                                              int bounds_error,
                                              int extrapolate);
N4M_API void         n4m_pp_resampler_destroy(n4m_pp_resampler_handle_t* h);
N4M_API n4m_status_t n4m_pp_resampler_fit(n4m_pp_resampler_handle_t* h,
                                           const double* source_wl,
                                           int64_t n_source);
N4M_API n4m_status_t n4m_pp_resampler_is_fitted(
    const n4m_pp_resampler_handle_t* h, int* out_fitted);
N4M_API int64_t      n4m_pp_resampler_output_cols(
    const n4m_pp_resampler_handle_t* h);
N4M_API n4m_status_t n4m_pp_resampler_transform(
    const n4m_pp_resampler_handle_t* h,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

/* ---------- CropTransformer (stateless) ------------------------------- */
typedef struct n4m_pp_crop_handle_t n4m_pp_crop_handle_t;
N4M_API n4m_status_t n4m_pp_crop_create(n4m_pp_crop_handle_t** out,
                                         int64_t start, int64_t end);
N4M_API void         n4m_pp_crop_destroy(n4m_pp_crop_handle_t* h);
N4M_API int64_t      n4m_pp_crop_output_cols(const n4m_pp_crop_handle_t* h,
                                              int64_t input_cols);
N4M_API n4m_status_t n4m_pp_crop_transform(const n4m_pp_crop_handle_t* h,
                                            n4m_matrix_view_t X,
                                            n4m_matrix_view_t out);

/* ---------- ResampleTransformer (stateless) --------------------------- */
typedef struct n4m_pp_resample_handle_t n4m_pp_resample_handle_t;
N4M_API n4m_status_t n4m_pp_resample_create(n4m_pp_resample_handle_t** out,
                                             int64_t num_samples);
N4M_API void         n4m_pp_resample_destroy(n4m_pp_resample_handle_t* h);
N4M_API int64_t      n4m_pp_resample_output_cols(
    const n4m_pp_resample_handle_t* h, int64_t input_cols);
N4M_API n4m_status_t n4m_pp_resample_transform(
    const n4m_pp_resample_handle_t* h,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

/* ---------- IntegerKBinsDiscretizer (stateful, int32 output) ---------- */
typedef struct n4m_pp_kbins_disc_handle_t n4m_pp_kbins_disc_handle_t;
N4M_API n4m_status_t n4m_pp_kbins_disc_create(n4m_pp_kbins_disc_handle_t** out,
                                               int32_t n_bins,
                                               int32_t strategy);
N4M_API void         n4m_pp_kbins_disc_destroy(n4m_pp_kbins_disc_handle_t* h);
N4M_API n4m_status_t n4m_pp_kbins_disc_fit(n4m_pp_kbins_disc_handle_t* h,
                                            n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_pp_kbins_disc_is_fitted(
    const n4m_pp_kbins_disc_handle_t* h, int* out_fitted);
N4M_API n4m_status_t n4m_pp_kbins_disc_transform(
    const n4m_pp_kbins_disc_handle_t* h,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

/* ---------- RangeDiscretizer (stateless, int32 output) ---------------- */
typedef struct n4m_pp_range_disc_handle_t n4m_pp_range_disc_handle_t;
N4M_API n4m_status_t n4m_pp_range_disc_create(n4m_pp_range_disc_handle_t** out,
                                               const double* bins,
                                               int64_t n_edges);
N4M_API void         n4m_pp_range_disc_destroy(n4m_pp_range_disc_handle_t* h);
N4M_API n4m_status_t n4m_pp_range_disc_transform(
    const n4m_pp_range_disc_handle_t* h,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

/* ============================================================================
 * 17. Phase 11 — Splitters (n4m_split_*)
 * ============================================================================
 *
 * Nine train/test splitting strategies. Each splitter returns a
 * `n4m_split_result_t` containing two heap-allocated index arrays that the
 * caller releases via `n4m_split_result_destroy`.
 *
 *   - KennardStone, SPXY, SPXYFold, SPXYGFold (group-aware), KMeans,
 *     KBinsStratified, BinnedStratifiedGroupKFold, SystematicCircular,
 *     SPlit (data twinning).
 */

/* ---------- Shared result type ----------------------------------------- */
typedef struct n4m_split_result_t {
    int64_t* train_idx;
    int64_t  n_train;
    int64_t* test_idx;
    int64_t  n_test;
    void*    _owner;
} n4m_split_result_t;

N4M_API void n4m_split_result_destroy(n4m_split_result_t* r);

/* ---------- Splitter enums (passed as int32_t through the create API) -- */
/* Bin-edge strategy for KBinsStratified + BinnedStratifiedGroupKFold. */
typedef enum n4m_split_kbins_strategy_t {
    N4M_SPLIT_KBINS_UNIFORM  = 0,  /* equal-width bins between y.min/y.max  */
    N4M_SPLIT_KBINS_QUANTILE = 1   /* equal-frequency bins (quantile edges) */
} n4m_split_kbins_strategy_t;

/* Y-metric mode for SPXYFold / SPXYGFold. Selects how Y participates in
 * the alternating max-min distance computation:
 *   NONE       : ignore Y entirely (pure Kennard-Stone on X)
 *   EUCLIDEAN  : standard SPXY (X and Y both Euclidean, summed)
 *   HAMMING    : SPXY with Hamming on Y (classification / multilabel) */
typedef enum n4m_split_y_metric_t {
    N4M_SPLIT_Y_METRIC_NONE      = 0,
    N4M_SPLIT_Y_METRIC_EUCLIDEAN = 1,
    N4M_SPLIT_Y_METRIC_HAMMING   = 2
} n4m_split_y_metric_t;

/* Group-aggregation mode for SPXYGFold (per-group representative). */
typedef enum n4m_split_aggregation_t {
    N4M_SPLIT_AGGREGATION_MEAN   = 0,
    N4M_SPLIT_AGGREGATION_MEDIAN = 1
} n4m_split_aggregation_t;

/* ---------- KennardStone ----------------------------------------------- */
typedef struct n4m_split_kennard_stone_handle_t n4m_split_kennard_stone_handle_t;
N4M_API n4m_status_t n4m_split_kennard_stone_create(
    n4m_split_kennard_stone_handle_t** out, double test_size);
N4M_API void n4m_split_kennard_stone_destroy(n4m_split_kennard_stone_handle_t* h);
N4M_API n4m_status_t n4m_split_kennard_stone_split(
    const n4m_split_kennard_stone_handle_t* h,
    n4m_matrix_view_t X, n4m_split_result_t* out);

/* ---------- SPXY (single train/test, joint X-Y) ----------------------- */
typedef struct n4m_split_spxy_handle_t n4m_split_spxy_handle_t;
N4M_API n4m_status_t n4m_split_spxy_create(n4m_split_spxy_handle_t** out,
                                            double test_size);
N4M_API void n4m_split_spxy_destroy(n4m_split_spxy_handle_t* h);
N4M_API n4m_status_t n4m_split_spxy_split(const n4m_split_spxy_handle_t* h,
                                           n4m_matrix_view_t X,
                                           n4m_matrix_view_t Y,
                                           n4m_split_result_t* out);

/* ---------- SPXYFold (k-fold) ---------------------------------------- */
typedef struct n4m_split_spxy_fold_handle_t n4m_split_spxy_fold_handle_t;
N4M_API n4m_status_t n4m_split_spxy_fold_create(n4m_split_spxy_fold_handle_t** out,
                                                 int32_t n_splits,
                                                 int32_t y_metric);
N4M_API void n4m_split_spxy_fold_destroy(n4m_split_spxy_fold_handle_t* h);
N4M_API n4m_status_t n4m_split_spxy_fold_n_splits(
    const n4m_split_spxy_fold_handle_t* h, int32_t* out);
N4M_API n4m_status_t n4m_split_spxy_fold_split_fold(
    const n4m_split_spxy_fold_handle_t* h,
    n4m_matrix_view_t X, n4m_matrix_view_t Y,
    int32_t fold_idx, n4m_split_result_t* out);

/* ---------- SPXYGFold (group-aware k-fold) --------------------------- */
typedef struct n4m_split_spxy_g_fold_handle_t n4m_split_spxy_g_fold_handle_t;
N4M_API n4m_status_t n4m_split_spxy_g_fold_create(n4m_split_spxy_g_fold_handle_t** out,
                                                    int32_t n_splits,
                                                    int32_t y_metric,
                                                    int32_t aggregation);
N4M_API void n4m_split_spxy_g_fold_destroy(n4m_split_spxy_g_fold_handle_t* h);
N4M_API n4m_status_t n4m_split_spxy_g_fold_n_splits(
    const n4m_split_spxy_g_fold_handle_t* h, int32_t* out);
N4M_API n4m_status_t n4m_split_spxy_g_fold_split_fold(
    const n4m_split_spxy_g_fold_handle_t* h,
    n4m_matrix_view_t X, n4m_matrix_view_t Y,
    const int64_t* groups, int64_t groups_len,
    int32_t fold_idx, n4m_split_result_t* out);

/* ---------- KMeans (k-means++) ---------------------------------------- */
typedef struct n4m_split_kmeans_handle_t n4m_split_kmeans_handle_t;
N4M_API n4m_status_t n4m_split_kmeans_create(n4m_split_kmeans_handle_t** out,
                                               double test_size, uint64_t seed,
                                               int32_t max_iter);
N4M_API void n4m_split_kmeans_destroy(n4m_split_kmeans_handle_t* h);
N4M_API n4m_status_t n4m_split_kmeans_split(const n4m_split_kmeans_handle_t* h,
                                              n4m_matrix_view_t X,
                                              n4m_split_result_t* out);

/* ---------- KBinsStratified ------------------------------------------ */
typedef struct n4m_split_kbins_stratified_handle_t n4m_split_kbins_stratified_handle_t;
N4M_API n4m_status_t n4m_split_kbins_stratified_create(
    n4m_split_kbins_stratified_handle_t** out,
    double test_size, uint64_t seed,
    int32_t n_bins, int32_t strategy);
N4M_API void n4m_split_kbins_stratified_destroy(n4m_split_kbins_stratified_handle_t* h);
N4M_API n4m_status_t n4m_split_kbins_stratified_split(
    const n4m_split_kbins_stratified_handle_t* h,
    n4m_matrix_view_t Y, n4m_split_result_t* out);

/* ---------- BinnedStratifiedGroupKFold ------------------------------- */
typedef struct n4m_split_binned_strat_group_kfold_handle_t
    n4m_split_binned_strat_group_kfold_handle_t;
N4M_API n4m_status_t n4m_split_binned_strat_group_kfold_create(
    n4m_split_binned_strat_group_kfold_handle_t** out,
    int32_t n_splits, int32_t n_bins, int32_t strategy,
    int32_t shuffle, uint64_t seed);
N4M_API void n4m_split_binned_strat_group_kfold_destroy(
    n4m_split_binned_strat_group_kfold_handle_t* h);
N4M_API n4m_status_t n4m_split_binned_strat_group_kfold_n_splits(
    const n4m_split_binned_strat_group_kfold_handle_t* h, int32_t* out);
N4M_API n4m_status_t n4m_split_binned_strat_group_kfold_split_fold(
    const n4m_split_binned_strat_group_kfold_handle_t* h,
    n4m_matrix_view_t Y,
    const int64_t* groups, int64_t groups_len,
    int32_t fold_idx, n4m_split_result_t* out);

/* ---------- SystematicCircular --------------------------------------- */
typedef struct n4m_split_systematic_circular_handle_t n4m_split_systematic_circular_handle_t;
N4M_API n4m_status_t n4m_split_systematic_circular_create(
    n4m_split_systematic_circular_handle_t** out, double test_size, uint64_t seed);
N4M_API void n4m_split_systematic_circular_destroy(
    n4m_split_systematic_circular_handle_t* h);
N4M_API n4m_status_t n4m_split_systematic_circular_split(
    const n4m_split_systematic_circular_handle_t* h,
    n4m_matrix_view_t Y, n4m_split_result_t* out);

/* ---------- SPlit (data twinning) ------------------------------------ */
typedef struct n4m_split_split_splitter_handle_t n4m_split_split_splitter_handle_t;
N4M_API n4m_status_t n4m_split_split_splitter_create(
    n4m_split_split_splitter_handle_t** out, double test_size, uint64_t seed);
N4M_API void n4m_split_split_splitter_destroy(n4m_split_split_splitter_handle_t* h);
N4M_API n4m_status_t n4m_split_split_splitter_split(
    const n4m_split_split_splitter_handle_t* h,
    n4m_matrix_view_t X, n4m_split_result_t* out);

/* ============================================================================
 * 18. Phase 12 — Sample filters: Y-based outliers (n4m_filter_*)
 * ============================================================================
 *
 * Sample-level filters return a per-row keep-mask (`uint8_t`, 1 = keep,
 * 0 = exclude) plus a `n4m_filter_stats_t` summary. Mask + stats buffers
 * are caller-allocated.
 *
 * Phase 12 introduces the `n4m_filter_stats_t` struct shared with later
 * filter phases (P14, future P13).
 */

/* ---------- Shared filter result type --------------------------------- */
typedef struct n4m_filter_stats_t {
    int64_t n_samples;        /* total samples seen by the filter         */
    int64_t n_kept;           /* count of mask[i] == 1                    */
    int64_t n_excluded;       /* n_samples - n_kept                       */
    double  exclusion_rate;   /* n_excluded / n_samples; 0.0 when n == 0  */
} n4m_filter_stats_t;

/* ---------- Y-outlier method enum ------------------------------------- */
typedef enum n4m_y_outlier_method_t {
    N4M_Y_OUTLIER_IQR        = 0,
    N4M_Y_OUTLIER_ZSCORE     = 1,
    N4M_Y_OUTLIER_PERCENTILE = 2,
    N4M_Y_OUTLIER_MAD        = 3
} n4m_y_outlier_method_t;

/* ---------- YOutlierFilter -------------------------------------------- *
 *
 * The filter follows the classical sklearn fit/apply split:
 *   - `_create` constructs an unfitted handle from the method + thresholds.
 *   - `_fit(handle, y, n)` learns the per-method bounds from the training
 *     y vector (one shot; idempotent — recomputes on every call).
 *   - `_apply(handle, y, n, mask, stats)` applies the previously learned
 *     bounds to a (possibly different) y vector, returning the keep-mask
 *     and statistics. The handle must be fitted; returns
 *     `N4M_ERR_NOT_FITTED` when called on an unfitted handle.
 *   - `_is_fitted(handle, out)` writes 1 when `_fit` has been called at
 *     least once on this handle, 0 otherwise.
 */
typedef struct n4m_filter_y_outlier_handle_t n4m_filter_y_outlier_handle_t;
N4M_API n4m_status_t n4m_filter_y_outlier_create(
    n4m_filter_y_outlier_handle_t** out,
    int32_t method,
    double threshold, double lower_pct, double upper_pct);
N4M_API void         n4m_filter_y_outlier_destroy(
    n4m_filter_y_outlier_handle_t* handle);
N4M_API n4m_status_t n4m_filter_y_outlier_fit(
    n4m_filter_y_outlier_handle_t* handle,
    const double* y, int64_t n);
N4M_API n4m_status_t n4m_filter_y_outlier_apply(
    const n4m_filter_y_outlier_handle_t* handle,
    const double* y, int64_t n,
    uint8_t* mask_out, n4m_filter_stats_t* stats_out);
N4M_API n4m_status_t n4m_filter_y_outlier_is_fitted(
    const n4m_filter_y_outlier_handle_t* handle, int* out);

/* ============================================================================
 * 19. Phase 14 — Leverage / Quality / Composite filters (n4m_filter_*)
 * ============================================================================
 *
 * Three meta-filter operators that reuse the §18 `n4m_filter_stats_t`.
 *
 *   - HighLeverageFilter   : hat-matrix or PCA score-space leverage.
 *   - SpectralQualityFilter : six all-or-nothing thresholds on each row.
 *   - CompositeFilter      : combines sub-filter keep-masks via ANY or ALL.
 *
 * Composite filter ownership:
 *   Sub-filter handles passed to `n4m_filter_composite_add_*` are stored by
 *   reference (the composite does not take ownership and does not copy the
 *   handle). The caller MUST keep every sub-filter alive at least as long
 *   as the composite that references it. `n4m_filter_composite_destroy`
 *   does NOT free sub-filters — the caller must call the matching
 *   `n4m_filter_<kind>_destroy` for each one after the composite is gone.
 */

/* ---------- HighLeverageFilter (stateful) ----------------------------- */
typedef struct n4m_filter_leverage_handle_t n4m_filter_leverage_handle_t;
N4M_API n4m_status_t n4m_filter_leverage_create(
    n4m_filter_leverage_handle_t** out,
    int32_t method,
    double  threshold_multiplier,
    int     use_absolute_threshold,
    double  absolute_threshold,
    int32_t n_components,
    int     center);
N4M_API void         n4m_filter_leverage_destroy(
    n4m_filter_leverage_handle_t* handle);
N4M_API n4m_status_t n4m_filter_leverage_fit(
    n4m_filter_leverage_handle_t* handle, n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_filter_leverage_is_fitted(
    const n4m_filter_leverage_handle_t* handle, int* out_fitted);
N4M_API n4m_status_t n4m_filter_leverage_apply(
    const n4m_filter_leverage_handle_t* handle,
    n4m_matrix_view_t X,
    uint8_t* mask_out, n4m_filter_stats_t* stats_out);
N4M_API double       n4m_filter_leverage_threshold(
    const n4m_filter_leverage_handle_t* handle);

/* ---------- SpectralQualityFilter (stateless) ------------------------- */
typedef struct n4m_filter_quality_handle_t n4m_filter_quality_handle_t;
N4M_API n4m_status_t n4m_filter_quality_create(
    n4m_filter_quality_handle_t** out,
    double max_nan_ratio, double max_zero_ratio,
    double min_variance,
    int use_max, double max_value,
    int use_min, double min_value,
    int check_inf);
N4M_API void         n4m_filter_quality_destroy(
    n4m_filter_quality_handle_t* handle);
N4M_API n4m_status_t n4m_filter_quality_apply(
    const n4m_filter_quality_handle_t* handle,
    n4m_matrix_view_t X,
    uint8_t* mask_out, n4m_filter_stats_t* stats_out);

/* ---------- CompositeFilter ------------------------------------------- */
typedef enum n4m_composite_mode_t {
    N4M_COMPOSITE_ANY = 0,    /* exclude if ANY sub-filter excludes */
    N4M_COMPOSITE_ALL = 1     /* exclude only if ALL sub-filters exclude */
} n4m_composite_mode_t;

typedef struct n4m_filter_composite_handle_t n4m_filter_composite_handle_t;
N4M_API n4m_status_t n4m_filter_composite_create(
    n4m_filter_composite_handle_t** out, int32_t mode);
N4M_API void         n4m_filter_composite_destroy(
    n4m_filter_composite_handle_t* handle);
N4M_API n4m_status_t n4m_filter_composite_add_leverage(
    n4m_filter_composite_handle_t* handle,
    n4m_filter_leverage_handle_t* sub);
N4M_API n4m_status_t n4m_filter_composite_add_quality(
    n4m_filter_composite_handle_t* handle,
    n4m_filter_quality_handle_t* sub);
N4M_API n4m_status_t n4m_filter_composite_apply(
    const n4m_filter_composite_handle_t* handle,
    n4m_matrix_view_t X,
    uint8_t* mask_out, n4m_filter_stats_t* stats_out);

/* ============================================================================
 * 20. Phase 19 — Signal type detection, NIRS metrics, T² / Q residuals
 * ============================================================================
 *
 * Three groups of free-function utilities:
 *
 *   - n4m_signal_* : auto-detect spectrum type (absorbance / reflectance /
 *                    transmittance / Kubelka-Munk / log(1/R) / log(1/T))
 *                    plus the percent-scaled variants.
 *   - n4m_metric_* : eight closed-form regression metrics (RMSE, MAE,
 *                    bias, SEP, RPD, RPIQ, R², NRMSE).
 *   - n4m_util_*   : multivariate outlier statistics (Hotelling T²,
 *                    Q-residuals) with their analytical upper control limits.
 */

/* ---------- 20.1 SignalTypeDetector ----------------------------------- */
typedef enum n4m_signal_type_t {
    N4M_SIGNAL_UNKNOWN               = 0,
    N4M_SIGNAL_ABSORBANCE            = 1,
    N4M_SIGNAL_REFLECTANCE           = 2,
    N4M_SIGNAL_REFLECTANCE_PERCENT   = 3,
    N4M_SIGNAL_TRANSMITTANCE         = 4,
    N4M_SIGNAL_TRANSMITTANCE_PERCENT = 5,
    N4M_SIGNAL_KUBELKA_MUNK          = 6,
    N4M_SIGNAL_LOG_1_R               = 7,
    N4M_SIGNAL_LOG_1_T               = 8
} n4m_signal_type_t;

N4M_API n4m_status_t n4m_signal_detect(n4m_matrix_view_t X,
                                        const double* wavelengths_optional,
                                        int64_t wl_length,
                                        double confidence_threshold,
                                        n4m_signal_type_t* type_out,
                                        double* confidence_out,
                                        char reason_buf[256]);

/* ---------- 20.2 NIRS regression metrics ------------------------------ */
N4M_API n4m_status_t n4m_metric_rmse (const double* y_true,
                                       const double* y_pred,
                                       int64_t n, double* out);
N4M_API n4m_status_t n4m_metric_mae  (const double* y_true,
                                       const double* y_pred,
                                       int64_t n, double* out);
N4M_API n4m_status_t n4m_metric_bias (const double* y_true,
                                       const double* y_pred,
                                       int64_t n, double* out);
N4M_API n4m_status_t n4m_metric_sep  (const double* y_true,
                                       const double* y_pred,
                                       int64_t n, double* out);
N4M_API n4m_status_t n4m_metric_rpd  (const double* y_true,
                                       const double* y_pred,
                                       int64_t n, double* out);
N4M_API n4m_status_t n4m_metric_rpiq (const double* y_true,
                                       const double* y_pred,
                                       int64_t n, double* out);
N4M_API n4m_status_t n4m_metric_r2   (const double* y_true,
                                       const double* y_pred,
                                       int64_t n, double* out);
N4M_API n4m_status_t n4m_metric_nrmse(const double* y_true,
                                       const double* y_pred,
                                       int64_t n, double* out);

/* ---------- 20.3 Multivariate outlier statistics ---------------------- */
N4M_API n4m_status_t n4m_util_hotelling_t2(n4m_matrix_view_t X,
                                            int32_t n_components,
                                            double alpha,
                                            double* t2_per_sample,
                                            int64_t n_samples,
                                            double* ucl_out);
N4M_API n4m_status_t n4m_util_q_residuals(n4m_matrix_view_t X,
                                           int32_t n_components,
                                           double alpha,
                                           double* q_per_sample,
                                           int64_t n_samples,
                                           double* ucl_out);

/* ============================================================================
 * 21. Phase 20 — Transfer metrics utility (n4m_transfer_*)
 * ============================================================================
 *
 * `n4m_transfer_metrics_compute` computes nine alignment metrics between two
 * datasets (source and target), as defined in
 * nirs4all/analysis/transfer_metrics.py. The struct is plain-data — every
 * field is a `double`. NaN encodes "metric not applicable" (e.g. Grassmann
 * when the two datasets do not share a feature count).
 */
typedef struct n4m_transfer_metrics_t {
    double centroid_distance;
    double cka_similarity;
    double grassmann_distance;
    double rv_coefficient;
    double procrustes_disparity;
    double trustworthiness;
    double spread_distance;
    double evr_source;
    double evr_target;
} n4m_transfer_metrics_t;

N4M_API n4m_status_t n4m_transfer_metrics_compute(
    n4m_matrix_view_t X_source,
    n4m_matrix_view_t X_target,
    int32_t n_components,
    int32_t k_neighbors,
    uint64_t seed,
    n4m_transfer_metrics_t* out);

/* ============================================================================
 * 22. Phase 21 — FCK static transformer (n4m_pp_fck_static_*)
 * ============================================================================
 *
 * FCKStaticTransformer applies a precomputed bank of L = n_orders * n_scales
 * fractional convolutional kernels to each spectrum row. For an input of
 * shape (n, p) the output has shape (n, L * p) with the L convolved signals
 * horizontally concatenated in (sample, kernel, feature) order.
 *
 * Each kernel of length `kernel_size` is the standard Gaussian × fractional
 * spatial-derivative recipe (see docs/algorithms/fck_static.md); the bank is
 * the Cartesian product of `filter_orders` × `filter_scales` with `alpha`
 * varying slowest.
 */
typedef struct n4m_pp_fck_static_handle_t n4m_pp_fck_static_handle_t;
N4M_API n4m_status_t n4m_pp_fck_static_create(
    n4m_pp_fck_static_handle_t** out,
    int32_t kernel_size,
    const double* filter_orders, int32_t n_orders,
    const double* filter_scales, int32_t n_scales);
N4M_API void n4m_pp_fck_static_destroy(n4m_pp_fck_static_handle_t* handle);
N4M_API n4m_status_t n4m_pp_fck_static_transform(
    const n4m_pp_fck_static_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_fck_static_output_cols(int32_t n_kernels,
                                                    int32_t n_features,
                                                    int32_t* out);

/* ============================================================================
 * 23. Phase 6 — Wavelets & denoising (n4m_pp_wavelet_*, n4m_pp_haar_*,
 *               n4m_pp_wavelet_denoise_*, n4m_pp_wavelet_features_*,
 *               n4m_pp_wavelet_pca_*, n4m_pp_wavelet_svd_*)
 * ============================================================================
 *
 * Six wavelet-domain operators backed by the shared filter banks in
 * `core/common/wavelet_kernels`. Supported families: haar, db4, sym4, coif1.
 * Supported boundary modes: periodization (default), symmetric, zero.
 *
 *   - Wavelet         : stateless single-level DWT. Output is the cA||cD
 *                       concatenation of length `output_cols(input_cols)`.
 *   - Haar            : convenience alias for Wavelet(haar, periodization).
 *   - WaveletDenoise  : multi-level DWT with VisuShrink thresholding; output
 *                       has the same width as the input.
 *   - WaveletFeatures : multi-level DWT statistical features (mean, std,
 *                       energy, entropy per decomposition level).
 *   - WaveletPCA      : stateful — fit_then_transform with embedded PCA over
 *                       the flattened DWT coefficient matrix.
 *   - WaveletSVD      : stateful — same as WaveletPCA but with SVD scoring.
 *
 * The threshold and noise-estimator selections for WaveletDenoise are
 * enumerated by `n4m_pp_wavelet_threshold_t` and `n4m_pp_wavelet_noise_t`.
 */

typedef enum n4m_pp_wavelet_family_t {
    N4M_PP_WAVELET_HAAR  = 0,
    N4M_PP_WAVELET_DB4   = 1,
    N4M_PP_WAVELET_SYM4  = 2,
    N4M_PP_WAVELET_COIF1 = 3
} n4m_pp_wavelet_family_t;

typedef enum n4m_pp_wavelet_boundary_t {
    N4M_PP_WAVELET_BOUNDARY_PERIODIZATION = 0,
    N4M_PP_WAVELET_BOUNDARY_SYMMETRIC     = 1,
    N4M_PP_WAVELET_BOUNDARY_ZERO          = 2
} n4m_pp_wavelet_boundary_t;

typedef enum n4m_pp_wavelet_threshold_t {
    N4M_PP_WAVELET_THRESHOLD_SOFT = 0,
    N4M_PP_WAVELET_THRESHOLD_HARD = 1
} n4m_pp_wavelet_threshold_t;

typedef enum n4m_pp_wavelet_noise_t {
    N4M_PP_WAVELET_NOISE_MEDIAN = 0,
    N4M_PP_WAVELET_NOISE_STD    = 1
} n4m_pp_wavelet_noise_t;

typedef enum n4m_pp_wavelet_features_entropy_t {
    N4M_PP_WAVELET_FEATURES_ENTROPY_ENERGY    = 0,
    N4M_PP_WAVELET_FEATURES_ENTROPY_HISTOGRAM = 1
} n4m_pp_wavelet_features_entropy_t;

typedef struct n4m_pp_wavelet_handle_t          n4m_pp_wavelet_handle_t;
typedef struct n4m_pp_haar_handle_t             n4m_pp_haar_handle_t;
typedef struct n4m_pp_wavelet_denoise_handle_t  n4m_pp_wavelet_denoise_handle_t;
typedef struct n4m_pp_wavelet_features_handle_t n4m_pp_wavelet_features_handle_t;
typedef struct n4m_pp_wavelet_pca_handle_t      n4m_pp_wavelet_pca_handle_t;
typedef struct n4m_pp_wavelet_svd_handle_t      n4m_pp_wavelet_svd_handle_t;

N4M_API n4m_status_t n4m_pp_wavelet_create(n4m_pp_wavelet_handle_t** out,
                                            n4m_pp_wavelet_family_t   family,
                                            n4m_pp_wavelet_boundary_t mode);
N4M_API void         n4m_pp_wavelet_destroy(n4m_pp_wavelet_handle_t* handle);
N4M_API n4m_status_t n4m_pp_wavelet_output_cols(
    const n4m_pp_wavelet_handle_t* handle,
    int64_t input_cols, int64_t* out_cols);
N4M_API n4m_status_t n4m_pp_wavelet_transform(
    const n4m_pp_wavelet_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

N4M_API n4m_status_t n4m_pp_haar_create(n4m_pp_haar_handle_t** out);
N4M_API void         n4m_pp_haar_destroy(n4m_pp_haar_handle_t* handle);
N4M_API n4m_status_t n4m_pp_haar_output_cols(int64_t input_cols,
                                              int64_t* out_cols);
N4M_API n4m_status_t n4m_pp_haar_transform(const n4m_pp_haar_handle_t* handle,
                                            n4m_matrix_view_t X,
                                            n4m_matrix_view_t out);

N4M_API n4m_status_t n4m_pp_wavelet_denoise_create(
    n4m_pp_wavelet_denoise_handle_t** out,
    n4m_pp_wavelet_family_t    family,
    n4m_pp_wavelet_boundary_t  mode,
    int32_t                    level,
    n4m_pp_wavelet_threshold_t threshold_mode,
    n4m_pp_wavelet_noise_t     noise_estimator);
N4M_API void         n4m_pp_wavelet_denoise_destroy(
    n4m_pp_wavelet_denoise_handle_t* handle);
N4M_API n4m_status_t n4m_pp_wavelet_denoise_transform(
    const n4m_pp_wavelet_denoise_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

N4M_API n4m_status_t n4m_pp_wavelet_features_create(
    n4m_pp_wavelet_features_handle_t** out,
    n4m_pp_wavelet_family_t   family,
    n4m_pp_wavelet_boundary_t mode,
    int32_t                   max_level);
N4M_API n4m_status_t n4m_pp_wavelet_features_create_ex(
    n4m_pp_wavelet_features_handle_t** out,
    n4m_pp_wavelet_family_t            family,
    n4m_pp_wavelet_boundary_t          mode,
    int32_t                            max_level,
    n4m_pp_wavelet_features_entropy_t  entropy_mode);
N4M_API void         n4m_pp_wavelet_features_destroy(
    n4m_pp_wavelet_features_handle_t* handle);
N4M_API n4m_status_t n4m_pp_wavelet_features_output_cols(
    const n4m_pp_wavelet_features_handle_t* handle,
    int64_t input_cols, int64_t* out_cols);
N4M_API n4m_status_t n4m_pp_wavelet_features_transform(
    const n4m_pp_wavelet_features_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

N4M_API n4m_status_t n4m_pp_wavelet_pca_create(
    n4m_pp_wavelet_pca_handle_t** out,
    n4m_pp_wavelet_family_t   family,
    n4m_pp_wavelet_boundary_t mode,
    int32_t                   max_level,
    double                    n_components);
N4M_API void         n4m_pp_wavelet_pca_destroy(
    n4m_pp_wavelet_pca_handle_t* handle);
N4M_API n4m_status_t n4m_pp_wavelet_pca_fit(
    n4m_pp_wavelet_pca_handle_t* handle, n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_pp_wavelet_pca_transform(
    const n4m_pp_wavelet_pca_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_wavelet_pca_is_fitted(
    const n4m_pp_wavelet_pca_handle_t* handle, int* out_fitted);
N4M_API n4m_status_t n4m_pp_wavelet_pca_output_cols(
    const n4m_pp_wavelet_pca_handle_t* handle, int64_t* out_cols);

N4M_API n4m_status_t n4m_pp_wavelet_svd_create(
    n4m_pp_wavelet_svd_handle_t** out,
    n4m_pp_wavelet_family_t   family,
    n4m_pp_wavelet_boundary_t mode,
    int32_t                   max_level,
    double                    n_components);
N4M_API void         n4m_pp_wavelet_svd_destroy(
    n4m_pp_wavelet_svd_handle_t* handle);
N4M_API n4m_status_t n4m_pp_wavelet_svd_fit(
    n4m_pp_wavelet_svd_handle_t* handle, n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_pp_wavelet_svd_transform(
    const n4m_pp_wavelet_svd_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_wavelet_svd_is_fitted(
    const n4m_pp_wavelet_svd_handle_t* handle, int* out_fitted);
N4M_API n4m_status_t n4m_pp_wavelet_svd_output_cols(
    const n4m_pp_wavelet_svd_handle_t* handle, int64_t* out_cols);

/* ============================================================================
 * 24. Phase 13 — XOutlierFilter (n4m_filter_x_outlier_*)
 * ============================================================================
 *
 * Single operator with six selectable methods over the same opaque handle.
 * Methods are selected via `n4m_filter_x_outlier_method_t` at construction
 * time. The filter follows the same `_create / _fit / _is_fitted / _apply /
 * _destroy` pattern as the Phase 12 Y-outlier filter, and reuses the public
 * `n4m_filter_stats_t` (§18) for its `_apply` statistics output.
 *
 * Methods:
 *   0  N4M_X_OUTLIER_MAHALANOBIS         Empirical-cov Mahalanobis distance.
 *   1  N4M_X_OUTLIER_ROBUST_MAHALANOBIS  Mahalanobis with MinCovDet shrinkage.
 *   2  N4M_X_OUTLIER_PCA_RESIDUAL        Q-statistic on PCA reconstruction.
 *   3  N4M_X_OUTLIER_PCA_LEVERAGE        Hotelling T² in PCA score space.
 *   4  N4M_X_OUTLIER_ISOLATION_FOREST    Sklearn-style ensemble.
 *   5  N4M_X_OUTLIER_LOF                 Local Outlier Factor (k=min(20,n-1)).
 *
 * `use_threshold` = 0 selects contamination-quantile mode (`contamination`
 * controls the discard fraction); `use_threshold` = 1 selects fixed-threshold
 * mode (`threshold` is the score cutoff). For Isolation Forest /
 * LOF the `n_estimators` and `max_samples` knobs steer the ensemble; for the
 * PCA-backed methods `n_components` selects the PCA rank.
 */

typedef enum n4m_filter_x_outlier_method_t {
    N4M_X_OUTLIER_MAHALANOBIS         = 0,
    N4M_X_OUTLIER_ROBUST_MAHALANOBIS  = 1,
    N4M_X_OUTLIER_PCA_RESIDUAL        = 2,
    N4M_X_OUTLIER_PCA_LEVERAGE        = 3,
    N4M_X_OUTLIER_ISOLATION_FOREST    = 4,
    N4M_X_OUTLIER_LOF                 = 5
} n4m_filter_x_outlier_method_t;

typedef struct n4m_filter_x_outlier_handle_t n4m_filter_x_outlier_handle_t;
N4M_API n4m_status_t n4m_filter_x_outlier_create(
    n4m_filter_x_outlier_handle_t** out,
    int32_t  method,
    int      use_threshold,
    double   threshold,
    int32_t  n_components,
    double   contamination,
    uint64_t seed,
    int32_t  n_estimators,
    int64_t  max_samples);
N4M_API void n4m_filter_x_outlier_destroy(n4m_filter_x_outlier_handle_t* handle);
N4M_API n4m_status_t n4m_filter_x_outlier_fit(
    n4m_filter_x_outlier_handle_t* handle, n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_filter_x_outlier_is_fitted(
    const n4m_filter_x_outlier_handle_t* handle, int* out_fitted);
N4M_API n4m_status_t n4m_filter_x_outlier_apply(
    const n4m_filter_x_outlier_handle_t* handle,
    n4m_matrix_view_t X,
    uint8_t* mask_out,
    n4m_filter_stats_t* stats_out);

/* ============================================================================
 * 25. Phase 15 — Augmenters: noise + drift (n4m_aug_*)
 * ============================================================================
 *
 * Seven stochastic augmenters that consume a `n4m_rng_pcg64_state_t*` handle
 * supplied by the caller. The augmenter does NOT own the RNG — the caller
 * must keep it alive for the augmenter's lifetime. RNG state advances on
 * every `_apply` call so consecutive calls produce independent draws.
 *
 * Universal triple:
 *   * `_create(out, rng, ...)`   — allocate the opaque handle.
 *   * `_apply(handle, X, out)`   — write the augmented matrix into out.
 *   * `_destroy(handle)`         — release. NULL-safe.
 *
 * Parity is bit-exact (1e-15) against the frozen NumPy Generator(PCG64) refs
 * in `n4m_parity_augmenters_ref`. The reference operates on row-major F64
 * matrices; the augmenter validates dtype/stride/shape on each `_apply`.
 */

typedef struct n4m_aug_gaussian_noise_handle_t       n4m_aug_gaussian_noise_handle_t;
typedef struct n4m_aug_multiplicative_noise_handle_t n4m_aug_multiplicative_noise_handle_t;
typedef struct n4m_aug_spike_noise_handle_t          n4m_aug_spike_noise_handle_t;
typedef struct n4m_aug_hetero_noise_handle_t         n4m_aug_hetero_noise_handle_t;
typedef struct n4m_aug_linear_drift_handle_t         n4m_aug_linear_drift_handle_t;
typedef struct n4m_aug_poly_drift_handle_t           n4m_aug_poly_drift_handle_t;
typedef struct n4m_aug_path_length_handle_t          n4m_aug_path_length_handle_t;

N4M_API n4m_status_t n4m_aug_gaussian_noise_create(
    n4m_aug_gaussian_noise_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double sigma);
N4M_API void n4m_aug_gaussian_noise_destroy(n4m_aug_gaussian_noise_handle_t* handle);
N4M_API n4m_status_t n4m_aug_gaussian_noise_apply(
    const n4m_aug_gaussian_noise_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

N4M_API n4m_status_t n4m_aug_multiplicative_noise_create(
    n4m_aug_multiplicative_noise_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double sigma_gain);
N4M_API void n4m_aug_multiplicative_noise_destroy(
    n4m_aug_multiplicative_noise_handle_t* handle);
N4M_API n4m_status_t n4m_aug_multiplicative_noise_apply(
    const n4m_aug_multiplicative_noise_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

N4M_API n4m_status_t n4m_aug_spike_noise_create(
    n4m_aug_spike_noise_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    int32_t n_spikes_min, int32_t n_spikes_max,
    double amplitude_min, double amplitude_max);
N4M_API void n4m_aug_spike_noise_destroy(n4m_aug_spike_noise_handle_t* handle);
N4M_API n4m_status_t n4m_aug_spike_noise_apply(
    const n4m_aug_spike_noise_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

N4M_API n4m_status_t n4m_aug_hetero_noise_create(
    n4m_aug_hetero_noise_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double noise_base, double noise_signal_dep);
N4M_API void n4m_aug_hetero_noise_destroy(n4m_aug_hetero_noise_handle_t* handle);
N4M_API n4m_status_t n4m_aug_hetero_noise_apply(
    const n4m_aug_hetero_noise_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

N4M_API n4m_status_t n4m_aug_linear_drift_create(
    n4m_aug_linear_drift_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double offset_min, double offset_max,
    double slope_min,  double slope_max);
N4M_API void n4m_aug_linear_drift_destroy(n4m_aug_linear_drift_handle_t* handle);
N4M_API n4m_status_t n4m_aug_linear_drift_apply(
    const n4m_aug_linear_drift_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

N4M_API n4m_status_t n4m_aug_poly_drift_create(
    n4m_aug_poly_drift_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    int32_t degree,
    const double* coeff_min, const double* coeff_max);
N4M_API void n4m_aug_poly_drift_destroy(n4m_aug_poly_drift_handle_t* handle);
N4M_API n4m_status_t n4m_aug_poly_drift_apply(
    const n4m_aug_poly_drift_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

N4M_API n4m_status_t n4m_aug_path_length_create(
    n4m_aug_path_length_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double path_length_std, double min_path_length);
N4M_API void n4m_aug_path_length_destroy(n4m_aug_path_length_handle_t* handle);
N4M_API n4m_status_t n4m_aug_path_length_apply(
    const n4m_aug_path_length_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

/* ============================================================================
 * 26. Phase 16 — Augmenters: wavelength + spectral (n4m_aug_*)
 * ============================================================================
 *
 * Ten stochastic augmenters that operate on wavelength-aware or
 * spectral-domain features. Same universal triple as §25; consumers must
 * supply a non-NULL `n4m_rng_pcg64_state_t*` handle.
 *
 * For augmenters that take a `wavelengths` array, the handle copies the
 * wavelength axis at create time; the caller may free the pointer
 * immediately after `_create` returns. `_apply` enforces n_features == the
 * `n_wavelengths` captured at create time.
 *
 * BandMasking / ChannelDropout `mode`: 0 = zero-fill, 1 = linear interp.
 */

typedef struct n4m_aug_wavelength_shift_handle_t   n4m_aug_wavelength_shift_handle_t;
typedef struct n4m_aug_wavelength_stretch_handle_t n4m_aug_wavelength_stretch_handle_t;
typedef struct n4m_aug_local_warp_handle_t         n4m_aug_local_warp_handle_t;
typedef struct n4m_aug_band_perturb_handle_t       n4m_aug_band_perturb_handle_t;
typedef struct n4m_aug_band_mask_handle_t          n4m_aug_band_mask_handle_t;
typedef struct n4m_aug_channel_dropout_handle_t    n4m_aug_channel_dropout_handle_t;
typedef struct n4m_aug_gauss_jitter_handle_t       n4m_aug_gauss_jitter_handle_t;
typedef struct n4m_aug_unsharp_mask_handle_t       n4m_aug_unsharp_mask_handle_t;
typedef struct n4m_aug_magnitude_warp_handle_t     n4m_aug_magnitude_warp_handle_t;
typedef struct n4m_aug_local_clip_handle_t         n4m_aug_local_clip_handle_t;

N4M_API n4m_status_t n4m_aug_wavelength_shift_create(
    n4m_aug_wavelength_shift_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double shift_lo, double shift_hi,
    const double* wavelengths, int64_t n_wavelengths);
N4M_API n4m_status_t n4m_aug_wavelength_shift_apply(
    const n4m_aug_wavelength_shift_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void n4m_aug_wavelength_shift_destroy(n4m_aug_wavelength_shift_handle_t* handle);

N4M_API n4m_status_t n4m_aug_wavelength_stretch_create(
    n4m_aug_wavelength_stretch_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double stretch_lo, double stretch_hi,
    const double* wavelengths, int64_t n_wavelengths);
N4M_API n4m_status_t n4m_aug_wavelength_stretch_apply(
    const n4m_aug_wavelength_stretch_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void n4m_aug_wavelength_stretch_destroy(n4m_aug_wavelength_stretch_handle_t* handle);

N4M_API n4m_status_t n4m_aug_local_warp_create(
    n4m_aug_local_warp_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    int32_t n_control_points,
    double  max_shift,
    const double* wavelengths, int64_t n_wavelengths);
N4M_API n4m_status_t n4m_aug_local_warp_apply(
    const n4m_aug_local_warp_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void n4m_aug_local_warp_destroy(n4m_aug_local_warp_handle_t* handle);

N4M_API n4m_status_t n4m_aug_band_perturb_create(
    n4m_aug_band_perturb_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    int32_t n_bands,
    int32_t bw_lo, int32_t bw_hi,
    double  gain_lo, double  gain_hi,
    double  offset_lo, double offset_hi);
N4M_API n4m_status_t n4m_aug_band_perturb_apply(
    const n4m_aug_band_perturb_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void n4m_aug_band_perturb_destroy(n4m_aug_band_perturb_handle_t* handle);

N4M_API n4m_status_t n4m_aug_band_mask_create(
    n4m_aug_band_mask_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    int32_t n_bands_lo, int32_t n_bands_hi,
    int32_t bw_lo, int32_t bw_hi,
    int32_t mode);
N4M_API n4m_status_t n4m_aug_band_mask_apply(
    const n4m_aug_band_mask_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void n4m_aug_band_mask_destroy(n4m_aug_band_mask_handle_t* handle);

N4M_API n4m_status_t n4m_aug_channel_dropout_create(
    n4m_aug_channel_dropout_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double  dropout_prob,
    int32_t mode);
N4M_API n4m_status_t n4m_aug_channel_dropout_apply(
    const n4m_aug_channel_dropout_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void n4m_aug_channel_dropout_destroy(n4m_aug_channel_dropout_handle_t* handle);

N4M_API n4m_status_t n4m_aug_gauss_jitter_create(
    n4m_aug_gauss_jitter_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double  sigma_lo, double sigma_hi,
    int32_t kernel_width);
N4M_API n4m_status_t n4m_aug_gauss_jitter_apply(
    const n4m_aug_gauss_jitter_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void n4m_aug_gauss_jitter_destroy(n4m_aug_gauss_jitter_handle_t* handle);

N4M_API n4m_status_t n4m_aug_unsharp_mask_create(
    n4m_aug_unsharp_mask_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double  amount_lo, double amount_hi,
    double  sigma, int32_t kernel_width);
N4M_API n4m_status_t n4m_aug_unsharp_mask_apply(
    const n4m_aug_unsharp_mask_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void n4m_aug_unsharp_mask_destroy(n4m_aug_unsharp_mask_handle_t* handle);

N4M_API n4m_status_t n4m_aug_magnitude_warp_create(
    n4m_aug_magnitude_warp_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    int32_t n_control_points,
    double  gain_lo, double gain_hi,
    const double* wavelengths, int64_t n_wavelengths);
N4M_API n4m_status_t n4m_aug_magnitude_warp_apply(
    const n4m_aug_magnitude_warp_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void n4m_aug_magnitude_warp_destroy(n4m_aug_magnitude_warp_handle_t* handle);

N4M_API n4m_status_t n4m_aug_local_clip_create(
    n4m_aug_local_clip_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    int32_t n_regions,
    int32_t width_lo, int32_t width_hi);
N4M_API n4m_status_t n4m_aug_local_clip_apply(
    const n4m_aug_local_clip_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void n4m_aug_local_clip_destroy(n4m_aug_local_clip_handle_t* handle);

/* ============================================================================
 * 27. Phase 17 — Augmenters: mixup + physical + environmental (n4m_aug_*)
 * ============================================================================
 *
 * Ten stochastic augmenters. Same universal triple as §25.
 *
 * Wavelength-aware augmenters take a `wavelengths` array at create time; the
 * handle copies the wavelengths internally. `_apply` enforces n_features ==
 * n_wavelengths captured at create time.
 *
 * MixupAugmenter / LocalMixupAugmenter combine TWO samples within a single
 * batch. The output rows are pairwise convex combinations of two input rows.
 */

/* ---------- MixupAugmenter ---------- */
typedef struct n4m_aug_mixup_handle_t n4m_aug_mixup_handle_t;
N4M_API n4m_status_t n4m_aug_mixup_create(n4m_aug_mixup_handle_t** out,
                                           n4m_rng_pcg64_state_t* rng,
                                           double alpha);
N4M_API n4m_status_t n4m_aug_mixup_apply(const n4m_aug_mixup_handle_t* handle,
                                          n4m_matrix_view_t X,
                                          n4m_matrix_view_t out);
N4M_API void         n4m_aug_mixup_destroy(n4m_aug_mixup_handle_t* handle);

/* ---------- LocalMixupAugmenter ---------- */
typedef struct n4m_aug_local_mixup_handle_t n4m_aug_local_mixup_handle_t;
N4M_API n4m_status_t n4m_aug_local_mixup_create(n4m_aug_local_mixup_handle_t** out,
                                                 n4m_rng_pcg64_state_t* rng,
                                                 double alpha,
                                                 int32_t k_neighbors);
N4M_API n4m_status_t n4m_aug_local_mixup_apply(const n4m_aug_local_mixup_handle_t* handle,
                                                n4m_matrix_view_t X,
                                                n4m_matrix_view_t out);
N4M_API void         n4m_aug_local_mixup_destroy(n4m_aug_local_mixup_handle_t* handle);

/* ---------- ScatterSimulationMSC ---------- */
typedef struct n4m_aug_scatter_sim_handle_t n4m_aug_scatter_sim_handle_t;
N4M_API n4m_status_t n4m_aug_scatter_sim_create(n4m_aug_scatter_sim_handle_t** out,
                                                 n4m_rng_pcg64_state_t* rng,
                                                 double a_low, double a_high,
                                                 double b_low, double b_high);
N4M_API n4m_status_t n4m_aug_scatter_sim_apply(const n4m_aug_scatter_sim_handle_t* handle,
                                                n4m_matrix_view_t X,
                                                n4m_matrix_view_t out);
N4M_API void         n4m_aug_scatter_sim_destroy(n4m_aug_scatter_sim_handle_t* handle);

/* ---------- ParticleSizeAugmenter ---------- */
typedef struct n4m_aug_particle_size_handle_t n4m_aug_particle_size_handle_t;
/* `use_size_range`: 0 = sample sizes from N(mean, variation), clipped to
 *   [5, 500]; 1 = sample uniform[size_range_low, size_range_high].
 * `include_path_length`: 0 = skip the multiplicative path-length step;
 *   1 = apply  factor = clip(1 + path_length_sensitivity * log(size_ratio),
 *                            0.7, 1.5).
 */
N4M_API n4m_status_t n4m_aug_particle_size_create(
    n4m_aug_particle_size_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double mean_size_um, double size_variation_um,
    int    use_size_range, double size_range_low_um, double size_range_high_um,
    double reference_size_um, double wavelength_exponent,
    double size_effect_strength,
    int    include_path_length, double path_length_sensitivity,
    const double* wavelengths, int64_t n_wavelengths);
N4M_API n4m_status_t n4m_aug_particle_size_apply(
    const n4m_aug_particle_size_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void         n4m_aug_particle_size_destroy(
    n4m_aug_particle_size_handle_t* handle);

/* ---------- EMSCDistortionAugmenter ---------- */
typedef struct n4m_aug_emsc_distort_handle_t n4m_aug_emsc_distort_handle_t;
N4M_API n4m_status_t n4m_aug_emsc_distort_create(
    n4m_aug_emsc_distort_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double mult_low, double mult_high,
    double add_low,  double add_high,
    int32_t polynomial_order,
    double  polynomial_strength,
    double  correlation,
    const double* wavelengths, int64_t n_wavelengths);
N4M_API n4m_status_t n4m_aug_emsc_distort_apply(
    const n4m_aug_emsc_distort_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void         n4m_aug_emsc_distort_destroy(
    n4m_aug_emsc_distort_handle_t* handle);

/* ---------- BatchEffectAugmenter ---------- */
typedef struct n4m_aug_batch_effect_handle_t n4m_aug_batch_effect_handle_t;
/* `variation_scope`: 0 = per-sample, 1 = batch.
 * `wavelengths` may be NULL → x axis derived from the integer index. */
N4M_API n4m_status_t n4m_aug_batch_effect_create(
    n4m_aug_batch_effect_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double offset_std, double slope_std, double gain_std,
    int32_t variation_scope,
    const double* wavelengths, int64_t n_wavelengths);
N4M_API n4m_status_t n4m_aug_batch_effect_apply(
    const n4m_aug_batch_effect_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void         n4m_aug_batch_effect_destroy(
    n4m_aug_batch_effect_handle_t* handle);

/* ---------- InstrumentalBroadeningAugmenter ---------- */
typedef struct n4m_aug_instrument_broaden_handle_t
                n4m_aug_instrument_broaden_handle_t;
/* `use_fwhm_range`: 0 = fixed `fwhm` for all rows (no RNG draws);
 *   1 = sample FWHM uniformly in [fwhm_low, fwhm_high]. */
N4M_API n4m_status_t n4m_aug_instrument_broaden_create(
    n4m_aug_instrument_broaden_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double fwhm,
    int    use_fwhm_range, double fwhm_low, double fwhm_high,
    int32_t variation_scope,
    const double* wavelengths, int64_t n_wavelengths);
N4M_API n4m_status_t n4m_aug_instrument_broaden_apply(
    const n4m_aug_instrument_broaden_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void         n4m_aug_instrument_broaden_destroy(
    n4m_aug_instrument_broaden_handle_t* handle);

/* ---------- DeadBandAugmenter ---------- */
typedef struct n4m_aug_dead_band_handle_t n4m_aug_dead_band_handle_t;
N4M_API n4m_status_t n4m_aug_dead_band_create(
    n4m_aug_dead_band_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    int32_t n_bands,
    int32_t width_low, int32_t width_high,
    double  noise_std, double probability,
    int32_t variation_scope);
N4M_API n4m_status_t n4m_aug_dead_band_apply(
    const n4m_aug_dead_band_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void         n4m_aug_dead_band_destroy(
    n4m_aug_dead_band_handle_t* handle);

/* ---------- TemperatureAugmenter ---------- */
typedef struct n4m_aug_temperature_handle_t n4m_aug_temperature_handle_t;
N4M_API n4m_status_t n4m_aug_temperature_create(
    n4m_aug_temperature_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double temperature_delta,
    int    use_temp_range, double temp_low, double temp_high,
    int    enable_shift, int enable_intensity, int enable_broadening,
    int    region_specific,
    const double* wavelengths, int64_t n_wavelengths);
N4M_API n4m_status_t n4m_aug_temperature_apply(
    const n4m_aug_temperature_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void         n4m_aug_temperature_destroy(
    n4m_aug_temperature_handle_t* handle);

/* ---------- MoistureAugmenter ---------- */
typedef struct n4m_aug_moisture_handle_t n4m_aug_moisture_handle_t;
N4M_API n4m_status_t n4m_aug_moisture_create(
    n4m_aug_moisture_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double water_activity_delta,
    int    use_aw_range, double aw_low, double aw_high,
    double reference_water_activity,
    double free_water_fraction,
    double bound_water_shift,
    double moisture_content,
    int    enable_shift, int enable_intensity,
    const double* wavelengths, int64_t n_wavelengths);
N4M_API n4m_status_t n4m_aug_moisture_apply(
    const n4m_aug_moisture_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void         n4m_aug_moisture_destroy(
    n4m_aug_moisture_handle_t* handle);

/* ============================================================================
 * 28. Phase 18 — Augmenters: edge + splines + random (n4m_aug_*)
 * ============================================================================
 *
 * Twelve operators. Same universal triple as §25.
 *
 * The Spline_X_Simplification and Spline_Curve_Simplification operators are
 * v2-deferred (per the original Phase 0 plan): the `_apply` entry point
 * returns `N4M_ERR_NOT_IMPLEMENTED` until v2 lands the necessary
 * `rng.choice(replace=False)` primitive in the public surface.
 *
 * `EdgeArtifactsAugmenter` is a combined wrapper that applies any subset
 * of DetectorRollOff, StrayLight, EdgeCurvature, TruncatedPeak in sequence;
 * a single 4-bit flags field selects the sub-augmenters.
 *
 * `Random_X_Operation` selects an elementwise operator via the `op_kind`
 * integer: 0=multiply, 1=add, 2=subtract.
 */

/* --- DetectorRollOff ----------------------------------------------------- */
typedef struct n4m_aug_detector_rolloff_handle_t
    n4m_aug_detector_rolloff_handle_t;
N4M_API n4m_status_t n4m_aug_detector_rolloff_create(
    n4m_aug_detector_rolloff_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    int32_t detector_model,
    double  effect_strength,
    double  noise_amplification,
    int32_t include_baseline_distortion);
N4M_API n4m_status_t n4m_aug_detector_rolloff_apply(
    const n4m_aug_detector_rolloff_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t wavelengths,
    n4m_matrix_view_t out);
N4M_API void n4m_aug_detector_rolloff_destroy(
    n4m_aug_detector_rolloff_handle_t* handle);

/* --- StrayLight --------------------------------------------------------- */
typedef struct n4m_aug_stray_light_handle_t n4m_aug_stray_light_handle_t;
N4M_API n4m_status_t n4m_aug_stray_light_create(
    n4m_aug_stray_light_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double stray_light_fraction,
    double edge_enhancement,
    double edge_width,
    int32_t include_peak_truncation);
N4M_API n4m_status_t n4m_aug_stray_light_apply(
    const n4m_aug_stray_light_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t wavelengths,
    n4m_matrix_view_t out);
N4M_API void n4m_aug_stray_light_destroy(
    n4m_aug_stray_light_handle_t* handle);

/* --- EdgeCurvature ------------------------------------------------------ */
typedef struct n4m_aug_edge_curve_handle_t n4m_aug_edge_curve_handle_t;
N4M_API n4m_status_t n4m_aug_edge_curve_create(
    n4m_aug_edge_curve_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double curvature_strength,
    int32_t curvature_type,
    double asymmetry,
    double edge_focus);
N4M_API n4m_status_t n4m_aug_edge_curve_apply(
    const n4m_aug_edge_curve_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t wavelengths,
    n4m_matrix_view_t out);
N4M_API void n4m_aug_edge_curve_destroy(n4m_aug_edge_curve_handle_t* handle);

/* --- TruncatedPeak ------------------------------------------------------ */
typedef struct n4m_aug_truncated_peak_handle_t
    n4m_aug_truncated_peak_handle_t;
N4M_API n4m_status_t n4m_aug_truncated_peak_create(
    n4m_aug_truncated_peak_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double peak_probability,
    double amplitude_min, double amplitude_max,
    double width_min,     double width_max,
    int32_t left_edge,
    int32_t right_edge);
N4M_API n4m_status_t n4m_aug_truncated_peak_apply(
    const n4m_aug_truncated_peak_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t wavelengths,
    n4m_matrix_view_t out);
N4M_API void n4m_aug_truncated_peak_destroy(
    n4m_aug_truncated_peak_handle_t* handle);

/* --- EdgeArtifacts (combined wrapper) ----------------------------------- */
/* Flags:
 *   bit 0 (0x1)  : enable DetectorRollOff
 *   bit 1 (0x2)  : enable StrayLight
 *   bit 2 (0x4)  : enable EdgeCurvature
 *   bit 3 (0x8)  : enable TruncatedPeak
 * Default sequence mirrors the Python reference (truncated -> curve ->
 * stray -> detector). The overall_strength scales every sub-augmenter's
 * default strength. */
#define N4M_AUG_EDGE_ARTIFACTS_DETECTOR_ROLL_OFF 0x1
#define N4M_AUG_EDGE_ARTIFACTS_STRAY_LIGHT       0x2
#define N4M_AUG_EDGE_ARTIFACTS_EDGE_CURVATURE    0x4
#define N4M_AUG_EDGE_ARTIFACTS_TRUNCATED_PEAKS   0x8
typedef struct n4m_aug_edge_artifacts_handle_t
    n4m_aug_edge_artifacts_handle_t;
N4M_API n4m_status_t n4m_aug_edge_artifacts_create(
    n4m_aug_edge_artifacts_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    int32_t enabled_flags,
    double  overall_strength,
    int32_t detector_model);
N4M_API n4m_status_t n4m_aug_edge_artifacts_apply(
    const n4m_aug_edge_artifacts_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t wavelengths,
    n4m_matrix_view_t out);
N4M_API void n4m_aug_edge_artifacts_destroy(
    n4m_aug_edge_artifacts_handle_t* handle);

/* --- Spline_Smoothing --------------------------------------------------- */
typedef struct n4m_aug_spline_smooth_handle_t
    n4m_aug_spline_smooth_handle_t;
N4M_API n4m_status_t n4m_aug_spline_smooth_create(
    n4m_aug_spline_smooth_handle_t** out,
    n4m_rng_pcg64_state_t* rng);
N4M_API n4m_status_t n4m_aug_spline_smooth_apply(
    const n4m_aug_spline_smooth_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void n4m_aug_spline_smooth_destroy(
    n4m_aug_spline_smooth_handle_t* handle);

/* --- Spline_X_Perturbations -------------------------------------------- */
typedef struct n4m_aug_spline_x_perturb_handle_t
    n4m_aug_spline_x_perturb_handle_t;
N4M_API n4m_status_t n4m_aug_spline_x_perturb_create(
    n4m_aug_spline_x_perturb_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    int32_t spline_degree,
    double  perturbation_density,
    double  perturbation_range_min,
    double  perturbation_range_max);
N4M_API n4m_status_t n4m_aug_spline_x_perturb_apply(
    const n4m_aug_spline_x_perturb_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void n4m_aug_spline_x_perturb_destroy(
    n4m_aug_spline_x_perturb_handle_t* handle);

/* --- Spline_Y_Perturbations -------------------------------------------- */
typedef struct n4m_aug_spline_y_perturb_handle_t
    n4m_aug_spline_y_perturb_handle_t;
/* spline_points <= 0 means "use n_features / 2". */
N4M_API n4m_status_t n4m_aug_spline_y_perturb_create(
    n4m_aug_spline_y_perturb_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    int32_t spline_points,
    double  perturbation_intensity);
N4M_API n4m_status_t n4m_aug_spline_y_perturb_apply(
    const n4m_aug_spline_y_perturb_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void n4m_aug_spline_y_perturb_destroy(
    n4m_aug_spline_y_perturb_handle_t* handle);

/* --- Spline_X_Simplification (v2-deferred — apply returns
 * N4M_ERR_NOT_IMPLEMENTED) ----------------------------------------------- */
typedef struct n4m_aug_spline_x_simplify_handle_t
    n4m_aug_spline_x_simplify_handle_t;
N4M_API n4m_status_t n4m_aug_spline_x_simplify_create(
    n4m_aug_spline_x_simplify_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    int32_t spline_points,
    int32_t uniform);
N4M_API n4m_status_t n4m_aug_spline_x_simplify_apply(
    const n4m_aug_spline_x_simplify_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void n4m_aug_spline_x_simplify_destroy(
    n4m_aug_spline_x_simplify_handle_t* handle);

/* --- Spline_Curve_Simplification (v2-deferred — apply returns
 * N4M_ERR_NOT_IMPLEMENTED) ----------------------------------------------- */
typedef struct n4m_aug_spline_curve_simplify_handle_t
    n4m_aug_spline_curve_simplify_handle_t;
N4M_API n4m_status_t n4m_aug_spline_curve_simplify_create(
    n4m_aug_spline_curve_simplify_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    int32_t spline_points,
    int32_t uniform);
N4M_API n4m_status_t n4m_aug_spline_curve_simplify_apply(
    const n4m_aug_spline_curve_simplify_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void n4m_aug_spline_curve_simplify_destroy(
    n4m_aug_spline_curve_simplify_handle_t* handle);

/* --- Rotate_Translate --------------------------------------------------- */
typedef struct n4m_aug_rotate_translate_handle_t
    n4m_aug_rotate_translate_handle_t;
N4M_API n4m_status_t n4m_aug_rotate_translate_create(
    n4m_aug_rotate_translate_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double p_range,
    double y_factor);
N4M_API n4m_status_t n4m_aug_rotate_translate_apply(
    const n4m_aug_rotate_translate_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void n4m_aug_rotate_translate_destroy(
    n4m_aug_rotate_translate_handle_t* handle);

/* --- Random_X_Operation ------------------------------------------------- */
typedef struct n4m_aug_random_x_op_handle_t n4m_aug_random_x_op_handle_t;
N4M_API n4m_status_t n4m_aug_random_x_op_create(
    n4m_aug_random_x_op_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    int32_t op_kind,
    double  operator_range_min,
    double  operator_range_max);
N4M_API n4m_status_t n4m_aug_random_x_op_apply(
    const n4m_aug_random_x_op_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API void n4m_aug_random_x_op_destroy(
    n4m_aug_random_x_op_handle_t* handle);

/* ============================================================================
 * 28. Advanced chemometric operators
 * ========================================================================== */

typedef struct n4m_pp_direct_standardization_handle_t
    n4m_pp_direct_standardization_handle_t;
N4M_API n4m_status_t n4m_pp_direct_standardization_create(
    n4m_pp_direct_standardization_handle_t** out, int fit_intercept, double ridge);
N4M_API void n4m_pp_direct_standardization_destroy(
    n4m_pp_direct_standardization_handle_t* handle);
N4M_API n4m_status_t n4m_pp_direct_standardization_fit(
    n4m_pp_direct_standardization_handle_t* handle,
    n4m_matrix_view_t source, n4m_matrix_view_t target);
N4M_API n4m_status_t n4m_pp_direct_standardization_transform(
    const n4m_pp_direct_standardization_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_direct_standardization_is_fitted(
    const n4m_pp_direct_standardization_handle_t* handle, int* out_fitted);

typedef struct n4m_pp_robust_direct_standardization_handle_t
    n4m_pp_robust_direct_standardization_handle_t;
N4M_API n4m_status_t n4m_pp_robust_direct_standardization_create(
    n4m_pp_robust_direct_standardization_handle_t** out, int fit_intercept,
    double ridge, double trim_quantile, int32_t max_iter);
N4M_API void n4m_pp_robust_direct_standardization_destroy(
    n4m_pp_robust_direct_standardization_handle_t* handle);
N4M_API n4m_status_t n4m_pp_robust_direct_standardization_fit(
    n4m_pp_robust_direct_standardization_handle_t* handle,
    n4m_matrix_view_t source, n4m_matrix_view_t target);
N4M_API n4m_status_t n4m_pp_robust_direct_standardization_transform(
    const n4m_pp_robust_direct_standardization_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_robust_direct_standardization_is_fitted(
    const n4m_pp_robust_direct_standardization_handle_t* handle, int* out_fitted);

typedef struct n4m_pp_piecewise_direct_standardization_handle_t
    n4m_pp_piecewise_direct_standardization_handle_t;
N4M_API n4m_status_t n4m_pp_piecewise_direct_standardization_create(
    n4m_pp_piecewise_direct_standardization_handle_t** out,
    int32_t window_size, int fit_intercept, double ridge);
N4M_API void n4m_pp_piecewise_direct_standardization_destroy(
    n4m_pp_piecewise_direct_standardization_handle_t* handle);
N4M_API n4m_status_t n4m_pp_piecewise_direct_standardization_fit(
    n4m_pp_piecewise_direct_standardization_handle_t* handle,
    n4m_matrix_view_t source, n4m_matrix_view_t target);
N4M_API n4m_status_t n4m_pp_piecewise_direct_standardization_transform(
    const n4m_pp_piecewise_direct_standardization_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_piecewise_direct_standardization_is_fitted(
    const n4m_pp_piecewise_direct_standardization_handle_t* handle,
    int* out_fitted);

typedef struct n4m_pp_saps_handle_t n4m_pp_saps_handle_t;
N4M_API n4m_status_t n4m_pp_saps_create(
    n4m_pp_saps_handle_t** out, int32_t n_components, double score_weight,
    int fit_intercept, double ridge);
N4M_API void n4m_pp_saps_destroy(n4m_pp_saps_handle_t* handle);
N4M_API n4m_status_t n4m_pp_saps_fit(
    n4m_pp_saps_handle_t* handle, n4m_matrix_view_t source,
    n4m_matrix_view_t target);
N4M_API n4m_status_t n4m_pp_saps_transform(
    const n4m_pp_saps_handle_t* handle, n4m_matrix_view_t X,
    n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_saps_is_fitted(
    const n4m_pp_saps_handle_t* handle, int* out_fitted);

typedef struct n4m_pp_slope_bias_handle_t n4m_pp_slope_bias_handle_t;
N4M_API n4m_status_t n4m_pp_slope_bias_create(
    n4m_pp_slope_bias_handle_t** out);
N4M_API void n4m_pp_slope_bias_destroy(n4m_pp_slope_bias_handle_t* handle);
N4M_API n4m_status_t n4m_pp_slope_bias_fit(
    n4m_pp_slope_bias_handle_t* handle,
    const double* source, const double* target, int64_t n);
N4M_API n4m_status_t n4m_pp_slope_bias_transform(
    const n4m_pp_slope_bias_handle_t* handle,
    const double* y, int64_t n, double* out);
N4M_API n4m_status_t n4m_pp_slope_bias_is_fitted(
    const n4m_pp_slope_bias_handle_t* handle, int* out_fitted);

typedef struct n4m_pp_local_centering_handle_t n4m_pp_local_centering_handle_t;
N4M_API n4m_status_t n4m_pp_local_centering_create(
    n4m_pp_local_centering_handle_t** out);
N4M_API void n4m_pp_local_centering_destroy(
    n4m_pp_local_centering_handle_t* handle);
N4M_API n4m_status_t n4m_pp_local_centering_fit(
    n4m_pp_local_centering_handle_t* handle,
    n4m_matrix_view_t source, n4m_matrix_view_t target);
N4M_API n4m_status_t n4m_pp_local_centering_transform(
    const n4m_pp_local_centering_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_local_centering_is_fitted(
    const n4m_pp_local_centering_handle_t* handle, int* out_fitted);

typedef struct n4m_pp_weighted_snv_handle_t n4m_pp_weighted_snv_handle_t;
N4M_API n4m_status_t n4m_pp_weighted_snv_create(
    n4m_pp_weighted_snv_handle_t** out, const double* weights,
    int64_t n_weights, int32_t ddof, double eps);
N4M_API void n4m_pp_weighted_snv_destroy(n4m_pp_weighted_snv_handle_t* handle);
N4M_API n4m_status_t n4m_pp_weighted_snv_fit(
    n4m_pp_weighted_snv_handle_t* handle, n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_pp_weighted_snv_transform(
    const n4m_pp_weighted_snv_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_weighted_snv_is_fitted(
    const n4m_pp_weighted_snv_handle_t* handle, int* out_fitted);

typedef struct n4m_pp_vsn_handle_t n4m_pp_vsn_handle_t;
N4M_API n4m_status_t n4m_pp_vsn_create(n4m_pp_vsn_handle_t** out, double eps);
N4M_API void n4m_pp_vsn_destroy(n4m_pp_vsn_handle_t* handle);
N4M_API n4m_status_t n4m_pp_vsn_fit(n4m_pp_vsn_handle_t* handle,
                                    n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_pp_vsn_transform(
    const n4m_pp_vsn_handle_t* handle, n4m_matrix_view_t X,
    n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_vsn_is_fitted(
    const n4m_pp_vsn_handle_t* handle, int* out_fitted);

typedef struct n4m_pp_piecewise_snv_handle_t n4m_pp_piecewise_snv_handle_t;
N4M_API n4m_status_t n4m_pp_piecewise_snv_create(
    n4m_pp_piecewise_snv_handle_t** out, int32_t window,
    int32_t ddof, double eps);
N4M_API void n4m_pp_piecewise_snv_destroy(n4m_pp_piecewise_snv_handle_t* handle);
N4M_API n4m_status_t n4m_pp_piecewise_snv_fit(
    n4m_pp_piecewise_snv_handle_t* handle, n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_pp_piecewise_snv_transform(
    const n4m_pp_piecewise_snv_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_piecewise_snv_is_fitted(
    const n4m_pp_piecewise_snv_handle_t* handle, int* out_fitted);

typedef struct n4m_pp_piecewise_msc_handle_t n4m_pp_piecewise_msc_handle_t;
typedef struct n4m_pp_localized_msc_handle_t n4m_pp_localized_msc_handle_t;
N4M_API n4m_status_t n4m_pp_piecewise_msc_create(
    n4m_pp_piecewise_msc_handle_t** out, const double* reference,
    int64_t n_reference, int32_t window, double eps);
N4M_API void n4m_pp_piecewise_msc_destroy(
    n4m_pp_piecewise_msc_handle_t* handle);
N4M_API n4m_status_t n4m_pp_piecewise_msc_fit(
    n4m_pp_piecewise_msc_handle_t* handle, n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_pp_piecewise_msc_transform(
    const n4m_pp_piecewise_msc_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_piecewise_msc_is_fitted(
    const n4m_pp_piecewise_msc_handle_t* handle, int* out_fitted);
N4M_API n4m_status_t n4m_pp_localized_msc_create(
    n4m_pp_localized_msc_handle_t** out, const double* reference,
    int64_t n_reference, int32_t window, double eps);
N4M_API void n4m_pp_localized_msc_destroy(
    n4m_pp_localized_msc_handle_t* handle);
N4M_API n4m_status_t n4m_pp_localized_msc_fit(
    n4m_pp_localized_msc_handle_t* handle, n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_pp_localized_msc_transform(
    const n4m_pp_localized_msc_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_localized_msc_is_fitted(
    const n4m_pp_localized_msc_handle_t* handle, int* out_fitted);

typedef struct n4m_pp_xcorr_align_handle_t n4m_pp_xcorr_align_handle_t;
typedef struct n4m_pp_icoshift_align_handle_t n4m_pp_icoshift_align_handle_t;
typedef struct n4m_pp_dtw_align_handle_t n4m_pp_dtw_align_handle_t;
typedef struct n4m_pp_cow_align_handle_t n4m_pp_cow_align_handle_t;
N4M_API n4m_status_t n4m_pp_xcorr_align_create(
    n4m_pp_xcorr_align_handle_t** out, const double* reference,
    int64_t n_reference, int32_t interval_size, int32_t max_shift);
N4M_API void n4m_pp_xcorr_align_destroy(n4m_pp_xcorr_align_handle_t* handle);
N4M_API n4m_status_t n4m_pp_xcorr_align_fit(
    n4m_pp_xcorr_align_handle_t* handle, n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_pp_xcorr_align_transform(
    const n4m_pp_xcorr_align_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_xcorr_align_is_fitted(
    const n4m_pp_xcorr_align_handle_t* handle, int* out_fitted);
N4M_API n4m_status_t n4m_pp_icoshift_align_create(
    n4m_pp_icoshift_align_handle_t** out, const double* reference,
    int64_t n_reference, int32_t interval_size, int32_t max_shift);
N4M_API void n4m_pp_icoshift_align_destroy(
    n4m_pp_icoshift_align_handle_t* handle);
N4M_API n4m_status_t n4m_pp_icoshift_align_fit(
    n4m_pp_icoshift_align_handle_t* handle, n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_pp_icoshift_align_transform(
    const n4m_pp_icoshift_align_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_icoshift_align_is_fitted(
    const n4m_pp_icoshift_align_handle_t* handle, int* out_fitted);
N4M_API n4m_status_t n4m_pp_dtw_align_create(
    n4m_pp_dtw_align_handle_t** out, const double* reference,
    int64_t n_reference, int32_t interval_size, int32_t max_shift);
N4M_API void n4m_pp_dtw_align_destroy(n4m_pp_dtw_align_handle_t* handle);
N4M_API n4m_status_t n4m_pp_dtw_align_fit(
    n4m_pp_dtw_align_handle_t* handle, n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_pp_dtw_align_transform(
    const n4m_pp_dtw_align_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_dtw_align_is_fitted(
    const n4m_pp_dtw_align_handle_t* handle, int* out_fitted);
N4M_API n4m_status_t n4m_pp_cow_align_create(
    n4m_pp_cow_align_handle_t** out, const double* reference,
    int64_t n_reference, int32_t interval_size, int32_t max_shift);
N4M_API void n4m_pp_cow_align_destroy(n4m_pp_cow_align_handle_t* handle);
N4M_API n4m_status_t n4m_pp_cow_align_fit(
    n4m_pp_cow_align_handle_t* handle, n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_pp_cow_align_transform(
    const n4m_pp_cow_align_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_cow_align_is_fitted(
    const n4m_pp_cow_align_handle_t* handle, int* out_fitted);

typedef struct n4m_filter_variance_handle_t n4m_filter_variance_handle_t;
N4M_API n4m_status_t n4m_filter_variance_create(
    n4m_filter_variance_handle_t** out, double threshold, int32_t top_k);
N4M_API void n4m_filter_variance_destroy(n4m_filter_variance_handle_t* handle);
N4M_API n4m_status_t n4m_filter_variance_fit(
    n4m_filter_variance_handle_t* handle, n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_filter_variance_transform(
    const n4m_filter_variance_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_filter_variance_output_cols(
    const n4m_filter_variance_handle_t* handle, int64_t* out_cols);
N4M_API n4m_status_t n4m_filter_variance_is_fitted(
    const n4m_filter_variance_handle_t* handle, int* out_fitted);

typedef struct n4m_filter_correlation_handle_t n4m_filter_correlation_handle_t;
N4M_API n4m_status_t n4m_filter_correlation_create(
    n4m_filter_correlation_handle_t** out, double threshold, int32_t top_k);
N4M_API void n4m_filter_correlation_destroy(
    n4m_filter_correlation_handle_t* handle);
N4M_API n4m_status_t n4m_filter_correlation_fit(
    n4m_filter_correlation_handle_t* handle,
    n4m_matrix_view_t X, const double* y, int64_t n_y);
N4M_API n4m_status_t n4m_filter_correlation_transform(
    const n4m_filter_correlation_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_filter_correlation_output_cols(
    const n4m_filter_correlation_handle_t* handle, int64_t* out_cols);
N4M_API n4m_status_t n4m_filter_correlation_is_fitted(
    const n4m_filter_correlation_handle_t* handle, int* out_fitted);

typedef struct n4m_interval_generator_handle_t n4m_interval_generator_handle_t;
N4M_API n4m_status_t n4m_interval_generator_create(
    n4m_interval_generator_handle_t** out, int32_t interval_size, int32_t step);
N4M_API void n4m_interval_generator_destroy(
    n4m_interval_generator_handle_t* handle);
N4M_API n4m_status_t n4m_interval_generator_fit(
    n4m_interval_generator_handle_t* handle, n4m_matrix_view_t X);
N4M_API n4m_status_t n4m_interval_generator_transform(
    const n4m_interval_generator_handle_t* handle,
    n4m_matrix_view_t X, n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_interval_generator_output_cols(
    const n4m_interval_generator_handle_t* handle, int64_t* out_cols);
N4M_API n4m_status_t n4m_interval_generator_is_fitted(
    const n4m_interval_generator_handle_t* handle, int* out_fitted);

#ifdef __cplusplus
}  /* extern "C" */
#endif

/* ============================================================================
 * 29. ABI guard rails — fixed-size assertions on the C ABI shape
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

/* PLS-domain surface (model, config, pipeline, validation, AOM/POP,
 * variable selection, diagnostics, multi-block, transfer). Lives in a
 * dedicated header so the umbrella file's core surface stays focused;
 * included here unconditionally so that consumers continue to need only
 * a single `#include "n4m/n4m.h"`. */
#include "n4m/pls.h"

#endif /* N4M_N4M_H */
