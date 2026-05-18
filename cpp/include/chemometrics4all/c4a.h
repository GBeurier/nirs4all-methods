/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * chemometrics4all — public C ABI v1.7.0.
 *
 * Stability: experimental until v1.0.0. Every breaking change before that
 * version bumps the ABI MAJOR (see c4a_version.h). After v1.0.0 the ABI
 * follows strict semver: breaking changes bump MAJOR, additive changes bump
 * MINOR, bugfixes bump PATCH.
 *
 * This is the ONLY header consumers of libc4a are expected to include.
 *
 * Universal rules of the surface:
 *   - Every exported function is `noexcept` — no C++ exception ever crosses
 *     this boundary. Status-returning wrappers translate exceptions via
 *     c4a_status_t; void destroy/free wrappers catch and swallow after
 *     best-effort cleanup.
 *   - No STL types, no Eigen types, no C++ classes appear in this surface.
 *   - All multi-byte serialized integers are little-endian.
 *   - Floating-point quantities are IEEE 754 binary64 (`double`).
 *   - String pointers returned by const-`char*` getters point into
 *     library-owned storage; callers must NOT free them.
 *   - Error messages for failed calls live in a per-context 4096-byte buffer
 *     (C4A_ERROR_BUFFER_BYTES) that is invalidated by the next call on the
 *     same context — bindings must copy if they retain the value.
 *   - Memory ownership is symmetric: callers free what they allocated, the
 *     core frees what it allocated. The only core-allocated outputs are the
 *     opaque operator handles; each has an explicit `c4a_*_destroy` function.
 *
 * Phase 3 trim (May 2026): prior revisions of this header inherited a large
 * set of PLS-domain declarations from pls4all (config, pipeline, model,
 * operator bank, gating strategy, AOM, validation plan, method result,
 * c4a_array_t). None of those had implementations in the chemometrics4all
 * core; they were placeholders for an algorithm surface that has been
 * deferred to later phases (or removed entirely). The header has therefore
 * been pruned down to the canonical chemometrics4all surface: status codes,
 * matrix views, context lifecycle, RNG (Phase 1) and preprocessing operators
 * (Phases 2-3). Future phases append new sections in dedicated banners and
 * bump C4A_ABI_VERSION_MINOR.
 *
 * Status-code contracts (apply uniformly unless overridden in a function's
 * own doc comment):
 *
 *   - `_create(out, ...)` functions: return C4A_OK on success, C4A_ERR_NULL_POINTER
 *     when `out` is NULL, C4A_ERR_INVALID_ARGUMENT when constructor parameters
 *     are rejected, C4A_ERR_OUT_OF_MEMORY on allocation failure. `*out` is
 *     set to NULL on every failure.
 *   - `_destroy(handle)` functions: void, no-op on NULL.
 *   - For stateful operators (Phase 3): `_transform` returns C4A_ERR_NOT_FITTED
 *     when called before `_fit`. Calling `_fit` again replaces the prior
 *     state (sklearn-style refit semantics).
 *   - Functions that consume `c4a_matrix_view_t` may additionally return
 *     C4A_ERR_SHAPE_MISMATCH, C4A_ERR_DTYPE_MISMATCH, C4A_ERR_STRIDE_INVALID
 *     when the view fails validation.
 */
#ifndef CHEMOMETRICS4ALL_C4A_H
#define CHEMOMETRICS4ALL_C4A_H

#include <stddef.h>
#include <stdint.h>

#include "chemometrics4all/c4a_export.h"
#include "chemometrics4all/c4a_version.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 1. Status codes
 * ========================================================================== */

typedef enum c4a_status_t {
    C4A_OK                        =   0,
    C4A_ERR_INVALID_ARGUMENT      =   1,
    C4A_ERR_NULL_POINTER          =   2,
    C4A_ERR_SHAPE_MISMATCH        =   3,
    C4A_ERR_DTYPE_MISMATCH        =   4,
    C4A_ERR_STRIDE_INVALID        =   5,
    C4A_ERR_NOT_FITTED            =   6,
    C4A_ERR_NUMERICAL_FAILURE     =   7,
    C4A_ERR_CONVERGENCE_FAILED    =   8,
    C4A_ERR_OUT_OF_MEMORY         =   9,
    C4A_ERR_UNSUPPORTED           =  10,
    C4A_ERR_NOT_IMPLEMENTED       =  11,   /* deferred public surface */
    C4A_ERR_ABI_MISMATCH          =  12,
    C4A_ERR_IO                    =  13,
    C4A_ERR_CORRUPT_BUFFER        =  14,
    C4A_ERR_VERSION_INCOMPATIBLE  =  15,
    C4A_ERR_BACKEND_UNAVAILABLE   =  16,
    C4A_ERR_CANCELLED             =  17,
    C4A_ERR_INTERNAL              = 255
} c4a_status_t;

/* Returns a static NUL-terminated string describing the status code. Returns
 * "unknown status" for any integer outside the declared enum range; never
 * returns NULL. */
C4A_API const char* c4a_status_to_string(c4a_status_t s);

/* ============================================================================
 * 2. Backends and numerical dtypes
 * ==========================================================================
 *
 * Backends are runtime-selectable via c4a_context_set_backend. The reference
 * CPU backend is always available; everything else is built when the
 * corresponding CHEMOMETRICS4ALL_WITH_* CMake option is enabled at compile time.
 */
typedef enum c4a_backend_t {
    C4A_BACKEND_AUTO            = 0,   /* library picks the best available */
    C4A_BACKEND_REFERENCE_CPU   = 1,   /* always present, portable */
    C4A_BACKEND_NATIVE_CPU      = 2,   /* SIMD-tuned scalar paths */
    C4A_BACKEND_BLAS            = 3,
    C4A_BACKEND_OPENMP          = 4,   /* combinable with another backend */
    C4A_BACKEND_CUDA            = 5,
    C4A_BACKEND_OPENCL          = 6,   /* reserved, not implemented */
    C4A_BACKEND_METAL           = 7    /* reserved, not implemented */
} c4a_backend_t;

/* Returns 1 if the backend was compiled into this build of libc4a, 0
 * otherwise. AUTO is always available (it resolves to REFERENCE_CPU at
 * minimum). */
C4A_API int          c4a_backend_is_available(c4a_backend_t backend);
C4A_API const char*  c4a_backend_to_string(c4a_backend_t backend);

typedef enum c4a_dtype_t {
    C4A_DTYPE_UNKNOWN = 0,
    C4A_DTYPE_F64     = 1,
    C4A_DTYPE_F32     = 2,
    C4A_DTYPE_I32     = 3,    /* reserved for label / index views */
    C4A_DTYPE_I64     = 4
} c4a_dtype_t;

/* Returns the size in bytes of one element of the given dtype.
 * Returns 0 for C4A_DTYPE_UNKNOWN or any out-of-range value. */
C4A_API size_t       c4a_dtype_size(c4a_dtype_t dtype);
C4A_API const char*  c4a_dtype_to_string(c4a_dtype_t dtype);

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
 * Stride rules enforced by c4a_matrix_view_validate:
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
typedef struct c4a_matrix_view_t {
    void*        data;
    int64_t      rows;
    int64_t      cols;
    int64_t      row_stride;   /* elements between row i and i+1 */
    int64_t      col_stride;   /* elements between col j and j+1 */
    c4a_dtype_t  dtype;
    int32_t      reserved0;    /* keep struct size stable */
} c4a_matrix_view_t;

/* Initialise a row-major (C / NumPy default) view. row_stride = cols,
 * col_stride = 1. Returns C4A_ERR_NULL_POINTER if `out` is NULL or
 * `data` is NULL with rows*cols > 0; C4A_ERR_INVALID_ARGUMENT for negative
 * dimensions or C4A_DTYPE_UNKNOWN; C4A_OK on success. */
C4A_API c4a_status_t c4a_matrix_view_init_rowmajor(
    c4a_matrix_view_t* out, void* data,
    int64_t rows, int64_t cols, c4a_dtype_t dtype);

/* Initialise a column-major (Fortran / R / MATLAB default) view.
 * row_stride = 1, col_stride = rows. */
C4A_API c4a_status_t c4a_matrix_view_init_colmajor(
    c4a_matrix_view_t* out, void* data,
    int64_t rows, int64_t cols, c4a_dtype_t dtype);

/* Initialise a strided view with explicit positive strides. Useful for
 * slicing into a larger buffer or for transposed views. Strides must be
 * positive (>= 1) and ints; the caller is responsible for ensuring the
 * underlying buffer is large enough. */
C4A_API c4a_status_t c4a_matrix_view_init_strided(
    c4a_matrix_view_t* out, void* data,
    int64_t rows, int64_t cols,
    int64_t row_stride, int64_t col_stride,
    c4a_dtype_t dtype);

/* Validate that the view is well-formed. The rules are listed in §3 prelude
 * above. Returns:
 *   - C4A_OK if all rules hold.
 *   - C4A_ERR_NULL_POINTER  if `v` is NULL, or `v->data` is NULL with
 *                            rows*cols > 0.
 *   - C4A_ERR_STRIDE_INVALID if a stride is negative or zero where required.
 *   - C4A_ERR_INVALID_ARGUMENT for negative `rows` / `cols`, or dtype not
 *                              in {F64, F32, I32, I64}. */
C4A_API c4a_status_t c4a_matrix_view_validate(const c4a_matrix_view_t* v);

/* ============================================================================
 * 4. Context lifecycle
 * ==========================================================================
 *
 * One c4a_context_t represents an isolated execution environment. It owns:
 *   - the per-context 4 KiB error buffer,
 *   - the current backend selection,
 *   - the RNG seed used by stochastic algorithms,
 *   - the thread-count hint for parallel backends,
 *   - an optional user-pointer for binding-side bookkeeping.
 *
 * Threading: a single c4a_context_t* must not be used concurrently from two
 * threads — use one context per thread. Across contexts the library is
 * thread-safe.
 *
 * Signal safety: no c4a_* function is async-signal-safe. Do not call any
 * c4a_* function from a POSIX signal handler.
 */
typedef struct c4a_context_s c4a_context_t;

C4A_API c4a_status_t c4a_context_create(c4a_context_t** out_ctx);
C4A_API void         c4a_context_destroy(c4a_context_t* ctx);

C4A_API c4a_status_t c4a_context_set_seed(c4a_context_t* ctx, uint64_t seed);
C4A_API c4a_status_t c4a_context_get_seed(const c4a_context_t* ctx,
                                          uint64_t* out_seed);

C4A_API c4a_status_t c4a_context_set_backend(c4a_context_t* ctx,
                                             c4a_backend_t backend);
C4A_API c4a_status_t c4a_context_get_backend(const c4a_context_t* ctx,
                                             c4a_backend_t* out_backend);

/* `n_threads`: any int32_t is accepted. Values <= 0 are documented as
 * "library picks the default" (read by parallel backends later); positive
 * values pin the worker count. The setter does NOT validate the upper
 * bound — the caller is responsible for choosing a sensible cap. */
C4A_API c4a_status_t c4a_context_set_num_threads(c4a_context_t* ctx,
                                                 int32_t n_threads);
C4A_API c4a_status_t c4a_context_get_num_threads(const c4a_context_t* ctx,
                                                 int32_t* out_threads);

/* Returns a NUL-terminated UTF-8 string owned by the context. The pointer
 * remains valid until the next c4a_* call on this context — bindings must
 * copy if they need to retain the value. Never returns NULL: when no error
 * has occurred since context creation (or since the last clear_error),
 * returns the empty string "". If `ctx` is NULL, returns a static "" string
 * literal (still never NULL). The buffer capacity is fixed at
 * C4A_ERROR_BUFFER_BYTES (4096) — over-long messages are truncated at the
 * boundary, preserving the NUL terminator. */
C4A_API const char*  c4a_context_last_error(const c4a_context_t* ctx);
/* No-op when `ctx` is NULL. */
C4A_API void         c4a_context_clear_error(c4a_context_t* ctx);

/* Optional binding-side annotation. The library never reads or writes the
 * pointer's referent. `c4a_context_get_user_data(NULL)` returns NULL — this
 * is indistinguishable from a context that has had NULL stored explicitly;
 * if you need to distinguish, store a sentinel value. */
C4A_API c4a_status_t c4a_context_set_user_data(c4a_context_t* ctx, void* user);
C4A_API void*        c4a_context_get_user_data(const c4a_context_t* ctx);

/* ============================================================================
 * 5. ABI naming convention and operator contracts
 * ==========================================================================
 *
 * Reserved prefixes for the c4a_* namespace:
 *
 *   c4a_pp_*       preprocessing operators (Phases 2-7, 10)
 *   c4a_split_*    sample splitters       (Phase 11)
 *   c4a_filter_*   sample filters         (Phases 12-14)
 *   c4a_aug_*      data augmentations     (Phases 15-18)
 *   c4a_util_*     generic utilities      (cross-phase)
 *   c4a_rng_*      random number generators (Phase 1)
 *   c4a_metric_*   metrics and diagnostics (Phases 19-20)
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
 *         `_fit` returns C4A_ERR_NOT_FITTED.
 *       - `_inverse_transform(handle, X_view, out_view)` reverses the
 *         transform when the operator is invertible. Operators with no
 *         meaningful inverse omit this entry point.
 *       - `_is_fitted(handle, int* out)` reports the fitted state (1 if
 *         fitted, 0 otherwise).
 *       - `_destroy(handle)` releases the handle. NULL-safe.
 *
 * Matrix view inputs must be row-major contiguous F64 for the current
 * implementation; non-contiguous and non-F64 views return
 * C4A_ERR_STRIDE_INVALID or C4A_ERR_DTYPE_MISMATCH respectively.
 */

/* ============================================================================
 * 6. ABI version + build info
 * ========================================================================== */

C4A_API uint32_t     c4a_get_abi_version_major(void);
C4A_API uint32_t     c4a_get_abi_version_minor(void);
C4A_API uint32_t     c4a_get_abi_version_patch(void);
C4A_API uint32_t     c4a_get_abi_version_int(void);   /* MAJOR*10000 + MINOR*100 + PATCH */
C4A_API const char*  c4a_get_version_string(void);    /* e.g. "0.1.0+abi.1.3.0" */
C4A_API const char*  c4a_get_build_info(void);        /* compiler / flags / backends */
C4A_API const char*  c4a_get_git_revision(void);      /* git rev at build time, or "" */

/* Returns C4A_OK if the header's ABI MAJOR == library MAJOR and the header's
 * MINOR <= library MINOR (forward compat). Returns C4A_ERR_ABI_MISMATCH on
 * MAJOR mismatch, C4A_ERR_VERSION_INCOMPATIBLE on MINOR-too-high. */
C4A_API c4a_status_t c4a_check_abi_compatibility(
    uint32_t header_abi_major,
    uint32_t header_abi_minor);

/* ============================================================================
 * 7. Phase 1 — PCG64 RNG
 * ============================================================================
 *
 * Pseudo-random number generator providing bit-exact parity against
 * NumPy 1.26.4's `numpy.random.PCG64`. All subsequent stochastic operators
 * in chemometrics4all (augmenters, IsolationForest, KMeans, CARS, MCUVE)
 * consume a `c4a_rng_pcg64_state_t*` so that pipelines run on this library
 * yield identical results to their reference NumPy implementations.
 *
 * Algorithm: PCG-XSL-RR 128/64 with a 128-bit state and 128-bit increment.
 * Seeding from a single uint64 reproduces `np.random.default_rng(seed)`
 * via NumPy's SeedSequence (pool_size=4) mixer. Normal samples use the
 * 256-region Ziggurat algorithm with NumPy's vendored constants.
 *
 * Parity guarantees:
 *   - c4a_rng_pcg64_uint64_fill          ≡ np.random.default_rng(s)
 *                                            .bit_generator.random_raw(n)
 *     byte-for-byte exact.
 *   - c4a_rng_pcg64_standard_normal_fill ≡ np.random.default_rng(s)
 *                                            .standard_normal(n)
 *     bit-exact double-for-double.
 *
 * Lifecycle: `c4a_rng_pcg64_create` allocates + seeds the handle;
 * `c4a_rng_pcg64_destroy` releases it (NULL-safe, no-op on NULL).
 * Status semantics follow the universal rules at the top of this header.
 */
typedef struct c4a_rng_pcg64_state_t c4a_rng_pcg64_state_t;

C4A_API c4a_status_t c4a_rng_pcg64_create(uint64_t seed,
                                           c4a_rng_pcg64_state_t** out);

C4A_API void c4a_rng_pcg64_destroy(c4a_rng_pcg64_state_t* rng);

C4A_API c4a_status_t c4a_rng_pcg64_set_seed(c4a_rng_pcg64_state_t* rng,
                                             uint64_t seed);

C4A_API c4a_status_t c4a_rng_pcg64_uint64(c4a_rng_pcg64_state_t* rng,
                                           uint64_t* out);

C4A_API c4a_status_t c4a_rng_pcg64_uint64_fill(c4a_rng_pcg64_state_t* rng,
                                                uint64_t* out, size_t n);

C4A_API c4a_status_t c4a_rng_pcg64_standard_normal_fill(
    c4a_rng_pcg64_state_t* rng, double* out, size_t n);

C4A_API c4a_status_t c4a_rng_pcg64_advance(c4a_rng_pcg64_state_t* rng,
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
 * Matrix-view validation may additionally return C4A_ERR_DTYPE_MISMATCH
 * (non-F64), C4A_ERR_STRIDE_INVALID (non-contiguous row-major), or
 * C4A_ERR_SHAPE_MISMATCH (X and out shapes disagree).
 */

/* ---------- SNV (Standard Normal Variate) -------------------------------- */
typedef struct c4a_pp_snv_handle_t c4a_pp_snv_handle_t;
C4A_API c4a_status_t c4a_pp_snv_create(c4a_pp_snv_handle_t** out,
                                        int with_mean, int with_std, int ddof);
C4A_API void         c4a_pp_snv_destroy(c4a_pp_snv_handle_t* handle);
C4A_API c4a_status_t c4a_pp_snv_transform(const c4a_pp_snv_handle_t* handle,
                                           c4a_matrix_view_t X,
                                           c4a_matrix_view_t out);

/* ---------- LocalSNV (sliding-window SNV) -------------------------------- */
typedef struct c4a_pp_lsnv_handle_t c4a_pp_lsnv_handle_t;
typedef enum c4a_pp_lsnv_pad_mode_t {
    C4A_PP_LSNV_PAD_REFLECT  = 0,
    C4A_PP_LSNV_PAD_EDGE     = 1,
    C4A_PP_LSNV_PAD_CONSTANT = 2
} c4a_pp_lsnv_pad_mode_t;
/* `window` must be an odd integer >= 3. */
C4A_API c4a_status_t c4a_pp_lsnv_create(c4a_pp_lsnv_handle_t** out,
                                         int32_t window, int32_t pad_mode,
                                         double constant_value);
C4A_API void         c4a_pp_lsnv_destroy(c4a_pp_lsnv_handle_t* handle);
C4A_API c4a_status_t c4a_pp_lsnv_transform(const c4a_pp_lsnv_handle_t* handle,
                                            c4a_matrix_view_t X,
                                            c4a_matrix_view_t out);

/* ---------- RobustSNV (median + k * MAD) --------------------------------- */
typedef struct c4a_pp_rnv_handle_t c4a_pp_rnv_handle_t;
C4A_API c4a_status_t c4a_pp_rnv_create(c4a_pp_rnv_handle_t** out,
                                        int with_center, int with_scale,
                                        double k);
C4A_API void         c4a_pp_rnv_destroy(c4a_pp_rnv_handle_t* handle);
C4A_API c4a_status_t c4a_pp_rnv_transform(const c4a_pp_rnv_handle_t* handle,
                                           c4a_matrix_view_t X,
                                           c4a_matrix_view_t out);

/* ---------- AreaNormalization -------------------------------------------- */
typedef struct c4a_pp_area_handle_t c4a_pp_area_handle_t;
typedef enum c4a_pp_area_method_t {
    C4A_PP_AREA_SUM     = 0,
    C4A_PP_AREA_ABS_SUM = 1,
    C4A_PP_AREA_TRAPZ   = 2
} c4a_pp_area_method_t;
C4A_API c4a_status_t c4a_pp_area_create(c4a_pp_area_handle_t** out,
                                         int32_t method);
C4A_API void         c4a_pp_area_destroy(c4a_pp_area_handle_t* handle);
C4A_API c4a_status_t c4a_pp_area_transform(const c4a_pp_area_handle_t* handle,
                                            c4a_matrix_view_t X,
                                            c4a_matrix_view_t out);

/* ---------- Normalize (column-wise) -------------------------------------- */
typedef struct c4a_pp_normalize_handle_t c4a_pp_normalize_handle_t;
/* `(feature_min, feature_max) == (-1, 1)` selects linalg-norm mode; any
 * other pair selects user-defined-range mode. */
C4A_API c4a_status_t c4a_pp_normalize_create(c4a_pp_normalize_handle_t** out,
                                              double feature_min,
                                              double feature_max);
C4A_API void         c4a_pp_normalize_destroy(c4a_pp_normalize_handle_t* handle);
C4A_API c4a_status_t c4a_pp_normalize_transform(
    const c4a_pp_normalize_handle_t* handle,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out);

/* ---------- SimpleScale (column-wise min-max → [0, 1]) ------------------- */
typedef struct c4a_pp_simple_scale_handle_t c4a_pp_simple_scale_handle_t;
C4A_API c4a_status_t c4a_pp_simple_scale_create(
    c4a_pp_simple_scale_handle_t** out);
C4A_API void         c4a_pp_simple_scale_destroy(
    c4a_pp_simple_scale_handle_t* handle);
C4A_API c4a_status_t c4a_pp_simple_scale_transform(
    const c4a_pp_simple_scale_handle_t* handle,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out);

/* ---------- LogTransform ------------------------------------------------- */
typedef struct c4a_pp_log_handle_t c4a_pp_log_handle_t;
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
 *     caller must invoke ``c4a_pp_log_fit`` once on the training matrix.
 *     The fitted offset is cached on the handle and re-used by subsequent
 *     ``_transform`` calls. Calling ``_transform`` before ``_fit`` returns
 *     C4A_ERR_NOT_FITTED. Calling ``_fit`` again replaces the prior fit
 *     (sklearn-style refit semantics).
 *
 *   This preserves the pre-Phase-5b behaviour of ``auto_offset == 0`` while
 *   giving ``auto_offset != 0`` proper sklearn-style fit-once/transform-many
 *   semantics.
 */
C4A_API c4a_status_t c4a_pp_log_create(c4a_pp_log_handle_t** out,
                                        double base, double offset,
                                        int auto_offset, double min_value);
C4A_API void         c4a_pp_log_destroy(c4a_pp_log_handle_t* handle);
C4A_API c4a_status_t c4a_pp_log_fit(c4a_pp_log_handle_t* handle,
                                     c4a_matrix_view_t X);
C4A_API c4a_status_t c4a_pp_log_is_fitted(const c4a_pp_log_handle_t* handle,
                                           int* out_fitted);
C4A_API c4a_status_t c4a_pp_log_transform(const c4a_pp_log_handle_t* handle,
                                           c4a_matrix_view_t X,
                                           c4a_matrix_view_t out);

/* ============================================================================
 * 9. Phase 3 — Stateful scatter preprocessings
 * ============================================================================
 *
 * Four stateful operators implementing the `_create / _fit / _transform /
 * _destroy` ABI contract from §5. Each operator stores parameters learned
 * from a training matrix and replays them on subsequent calls. Calling
 * `_transform` (or `_inverse_transform`) before `_fit` returns
 * C4A_ERR_NOT_FITTED.
 *
 * Reference: nirs4all 0.8.x operators in `operators/transforms/{nirs,
 * signal,scalers}.py`.
 */

/* ---------- MSC (Multiplicative Scatter Correction) ---------------------- */
/*
 * Per-column scatter correction calibrated against the per-row mean of the
 * training matrix. For each column j of the training X:
 *   reference[i] = mean(X[i, :])              # length n_samples
 *   (a[j], b[j]) = polyfit(reference, X[:, j], deg=1)
 * Transform divides each column by its slope and subtracts the intercept:
 *   X'[:, j] = (X[:, j] - b[j]) / a[j]
 * Inverse:   X[:, j] = X'[:, j] * a[j] + b[j]
 *
 * Matches `nirs4all.operators.transforms.nirs.MultiplicativeScatterCorrection`
 * with `scale=False` (no pre-centering by column means before computing the
 * reference). The `scale=True` flavour is deferred — it adds a column-mean
 * pre-centering step that we do not yet need for the supported pipelines.
 *
 * `_fit` requires `X.rows >= 2` (least-squares fit needs >= 2 points).
 */
typedef struct c4a_pp_msc_handle_t c4a_pp_msc_handle_t;
C4A_API c4a_status_t c4a_pp_msc_create(c4a_pp_msc_handle_t** out);
C4A_API void         c4a_pp_msc_destroy(c4a_pp_msc_handle_t* handle);
C4A_API c4a_status_t c4a_pp_msc_fit(c4a_pp_msc_handle_t* handle,
                                     c4a_matrix_view_t X);
C4A_API c4a_status_t c4a_pp_msc_transform(const c4a_pp_msc_handle_t* handle,
                                           c4a_matrix_view_t X,
                                           c4a_matrix_view_t out);
C4A_API c4a_status_t c4a_pp_msc_inverse_transform(
    const c4a_pp_msc_handle_t* handle,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out);
C4A_API c4a_status_t c4a_pp_msc_is_fitted(const c4a_pp_msc_handle_t* handle,
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
typedef struct c4a_pp_emsc_handle_t c4a_pp_emsc_handle_t;
C4A_API c4a_status_t c4a_pp_emsc_create(c4a_pp_emsc_handle_t** out,
                                         int32_t degree);
C4A_API void         c4a_pp_emsc_destroy(c4a_pp_emsc_handle_t* handle);
C4A_API c4a_status_t c4a_pp_emsc_fit(c4a_pp_emsc_handle_t* handle,
                                      c4a_matrix_view_t X);
C4A_API c4a_status_t c4a_pp_emsc_transform(const c4a_pp_emsc_handle_t* handle,
                                            c4a_matrix_view_t X,
                                            c4a_matrix_view_t out);
C4A_API c4a_status_t c4a_pp_emsc_is_fitted(const c4a_pp_emsc_handle_t* handle,
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
typedef struct c4a_pp_baseline_handle_t c4a_pp_baseline_handle_t;
C4A_API c4a_status_t c4a_pp_baseline_create(c4a_pp_baseline_handle_t** out);
C4A_API void         c4a_pp_baseline_destroy(c4a_pp_baseline_handle_t* handle);
C4A_API c4a_status_t c4a_pp_baseline_fit(c4a_pp_baseline_handle_t* handle,
                                          c4a_matrix_view_t X);
C4A_API c4a_status_t c4a_pp_baseline_transform(
    const c4a_pp_baseline_handle_t* handle,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out);
C4A_API c4a_status_t c4a_pp_baseline_inverse_transform(
    const c4a_pp_baseline_handle_t* handle,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out);
C4A_API c4a_status_t c4a_pp_baseline_is_fitted(
    const c4a_pp_baseline_handle_t* handle,
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
 * C4A_ERR_NOT_FITTED; calling it with a column count different from the
 * fitted value returns C4A_ERR_SHAPE_MISMATCH. This lets bindings treat
 * Derivate uniformly with the other stateful preprocessings without a
 * special case.
 *
 * Use `c4a_pp_derivate_output_cols(order, input_cols)` to compute the
 * output column count expected by `_transform`. The helper returns 0 when
 * `order >= input_cols`.
 */
typedef struct c4a_pp_derivate_handle_t c4a_pp_derivate_handle_t;
C4A_API c4a_status_t c4a_pp_derivate_create(c4a_pp_derivate_handle_t** out,
                                             int32_t order, double delta);
C4A_API void         c4a_pp_derivate_destroy(c4a_pp_derivate_handle_t* handle);
C4A_API c4a_status_t c4a_pp_derivate_fit(c4a_pp_derivate_handle_t* handle,
                                          c4a_matrix_view_t X);
C4A_API c4a_status_t c4a_pp_derivate_transform(
    const c4a_pp_derivate_handle_t* handle,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out);
C4A_API int64_t      c4a_pp_derivate_output_cols(int32_t order,
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
 * Matrix-view validation may additionally return C4A_ERR_DTYPE_MISMATCH
 * (non-F64), C4A_ERR_STRIDE_INVALID (non-contiguous row-major), or
 * C4A_ERR_SHAPE_MISMATCH (X and out shapes disagree).
 */

/* ---------- SavitzkyGolay -------------------------------------------------
 *
 * Parameters:
 *   window_length : odd integer >= 1
 *   polyorder     : 0 <= polyorder < window_length
 *   deriv         : derivative order >= 0
 *   delta         : sample spacing (non-zero)
 *   mode          : boundary handling, see c4a_pp_savgol_mode_t below
 *   cval          : fill value used when mode == C4A_PP_SAVGOL_CONSTANT
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
typedef struct c4a_pp_savgol_handle_t c4a_pp_savgol_handle_t;
typedef enum c4a_pp_savgol_mode_t {
    C4A_PP_SAVGOL_MIRROR   = 0,
    C4A_PP_SAVGOL_CONSTANT = 1,
    C4A_PP_SAVGOL_NEAREST  = 2,
    C4A_PP_SAVGOL_WRAP     = 3,
    C4A_PP_SAVGOL_INTERP   = 4
} c4a_pp_savgol_mode_t;
C4A_API c4a_status_t c4a_pp_savgol_create(c4a_pp_savgol_handle_t** out,
                                           int32_t window_length,
                                           int32_t polyorder,
                                           int32_t deriv,
                                           double delta,
                                           c4a_pp_savgol_mode_t mode,
                                           double cval);
C4A_API void         c4a_pp_savgol_destroy(c4a_pp_savgol_handle_t* handle);
C4A_API c4a_status_t c4a_pp_savgol_transform(
    const c4a_pp_savgol_handle_t* handle,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out);

/* ---------- FirstDerivative -----------------------------------------------
 *
 * `np.gradient(X, delta, axis=1, edge_order)` with edge_order in {1, 2}.
 * Output shape matches input shape.  `delta` must be non-zero.
 */
typedef struct c4a_pp_first_derivative_handle_t
    c4a_pp_first_derivative_handle_t;
C4A_API c4a_status_t c4a_pp_first_derivative_create(
    c4a_pp_first_derivative_handle_t** out, double delta, int32_t edge_order);
C4A_API void         c4a_pp_first_derivative_destroy(
    c4a_pp_first_derivative_handle_t* handle);
C4A_API c4a_status_t c4a_pp_first_derivative_transform(
    const c4a_pp_first_derivative_handle_t* handle,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out);

/* ---------- SecondDerivative ----------------------------------------------
 *
 * Two passes of `np.gradient(X, delta, axis=1, edge_order)`.  Shape-preserving.
 */
typedef struct c4a_pp_second_derivative_handle_t
    c4a_pp_second_derivative_handle_t;
C4A_API c4a_status_t c4a_pp_second_derivative_create(
    c4a_pp_second_derivative_handle_t** out, double delta, int32_t edge_order);
C4A_API void         c4a_pp_second_derivative_destroy(
    c4a_pp_second_derivative_handle_t* handle);
C4A_API c4a_status_t c4a_pp_second_derivative_transform(
    const c4a_pp_second_derivative_handle_t* handle,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out);

/* ---------- NorrisWilliams -------------------------------------------------
 *
 * `segment` must be odd and >= 1 (1 disables smoothing).  `gap` must be >= 1.
 * `derivative_order` must be 1 or 2.  `delta` must be non-zero.  Shape-preserving.
 */
typedef struct c4a_pp_norris_williams_handle_t
    c4a_pp_norris_williams_handle_t;
C4A_API c4a_status_t c4a_pp_norris_williams_create(
    c4a_pp_norris_williams_handle_t** out,
    int32_t segment, int32_t gap, int32_t derivative_order, double delta);
C4A_API void         c4a_pp_norris_williams_destroy(
    c4a_pp_norris_williams_handle_t* handle);
C4A_API c4a_status_t c4a_pp_norris_williams_transform(
    const c4a_pp_norris_williams_handle_t* handle,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out);

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
typedef struct c4a_pp_gaussian_handle_t c4a_pp_gaussian_handle_t;
typedef enum c4a_pp_gaussian_mode_t {
    C4A_PP_GAUSSIAN_REFLECT  = 0,
    C4A_PP_GAUSSIAN_CONSTANT = 1,
    C4A_PP_GAUSSIAN_NEAREST  = 2,
    C4A_PP_GAUSSIAN_MIRROR   = 3,
    C4A_PP_GAUSSIAN_WRAP     = 4
} c4a_pp_gaussian_mode_t;
C4A_API c4a_status_t c4a_pp_gaussian_create(
    c4a_pp_gaussian_handle_t** out,
    double sigma, int32_t order,
    c4a_pp_gaussian_mode_t mode,
    double cval, double truncate);
C4A_API void         c4a_pp_gaussian_destroy(c4a_pp_gaussian_handle_t* handle);
C4A_API c4a_status_t c4a_pp_gaussian_transform(
    const c4a_pp_gaussian_handle_t* handle,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out);

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
 *   parity/python_generator/src/c4a_parity_pybaselines_ref/
 * validated once against pybaselines==1.1.4 (see parity/README.md).
 */

/* ---------- Detrend (polynomial baseline subtraction) ------------------- */
typedef struct c4a_pp_detrend_handle_t c4a_pp_detrend_handle_t;
/* `polyorder` >= 0. Default in pybaselines.polynomial: 1 (linear detrend). */
C4A_API c4a_status_t c4a_pp_detrend_create(c4a_pp_detrend_handle_t** out,
                                            int32_t polyorder);
C4A_API void         c4a_pp_detrend_destroy(c4a_pp_detrend_handle_t* handle);
C4A_API c4a_status_t c4a_pp_detrend_transform(
    const c4a_pp_detrend_handle_t* handle,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out);

/* ---------- AsLS (Eilers & Boelens 2005) -------------------------------- */
typedef struct c4a_pp_asls_handle_t c4a_pp_asls_handle_t;
/* `lam` > 0  (smoothing penalty; default 1e6).
 * `p`        (asymmetry, 0 < p < 1; default 1e-2).
 * `max_iter` >= 0 (default 50).
 * `tol`      >= 0 (default 1e-3, relative L2 weight change). */
C4A_API c4a_status_t c4a_pp_asls_create(c4a_pp_asls_handle_t** out,
                                         double lam, double p,
                                         int32_t max_iter, double tol);
C4A_API void         c4a_pp_asls_destroy(c4a_pp_asls_handle_t* handle);
C4A_API c4a_status_t c4a_pp_asls_transform(const c4a_pp_asls_handle_t* handle,
                                            c4a_matrix_view_t X,
                                            c4a_matrix_view_t out);

/* ---------- AirPLS (Zhang 2010) ----------------------------------------- */
typedef struct c4a_pp_airpls_handle_t c4a_pp_airpls_handle_t;
/* `lam`      > 0 (default 1e6).
 * `max_iter` >= 0 (default 50).
 * `tol`      >= 0 (default 1e-3, |sum(neg residuals)| / |y|_1). */
C4A_API c4a_status_t c4a_pp_airpls_create(c4a_pp_airpls_handle_t** out,
                                           double lam,
                                           int32_t max_iter, double tol);
C4A_API void         c4a_pp_airpls_destroy(c4a_pp_airpls_handle_t* handle);
C4A_API c4a_status_t c4a_pp_airpls_transform(
    const c4a_pp_airpls_handle_t* handle,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out);

/* ---------- ArPLS (Baek 2015) ------------------------------------------- */
typedef struct c4a_pp_arpls_handle_t c4a_pp_arpls_handle_t;
/* `lam`      > 0 (default 1e5).
 * `max_iter` >= 0 (default 50).
 * `tol`      >= 0 (default 1e-3, relative L2 weight change). */
C4A_API c4a_status_t c4a_pp_arpls_create(c4a_pp_arpls_handle_t** out,
                                          double lam,
                                          int32_t max_iter, double tol);
C4A_API void         c4a_pp_arpls_destroy(c4a_pp_arpls_handle_t* handle);
C4A_API c4a_status_t c4a_pp_arpls_transform(const c4a_pp_arpls_handle_t* handle,
                                             c4a_matrix_view_t X,
                                             c4a_matrix_view_t out);

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
 *   - BEADS         : Simplified Baseline Estimation And Denoising with
 *                     Sparsity (Ning & Selesnick 2014) using a banded
 *                     pentadiagonal LDLT with reweighted L2 sparsity surrogate.
 *                     The full BEADS algorithm (Chebyshev approximation of |.|
 *                     over a 7-diagonal system) is deferred to a later phase.
 *
 * All six operators implement the `_create / _transform / _destroy` ABI
 * contract from §5 (stateless — no `_fit`). Each operator subtracts the
 * estimated baseline and writes `out = X - baseline`. Convergence-failure
 * semantics match Phase 5a: silently returns the last iterate at max_iter
 * exhaustion.
 *
 * Parity reference: frozen NumPy ref under
 *   parity/python_generator/src/c4a_parity_pybaselines_ref/
 * validated once against pybaselines==1.1.4. The simplified BEADS reference
 * is documented separately in docs/algorithms/beads.md.
 */

/* ---------- ModPoly (Lieber & Mahadevan-Jansen 2003) -------------------- */
typedef struct c4a_pp_modpoly_handle_t c4a_pp_modpoly_handle_t;
/* `polyorder` >= 0 (default 2).
 * `max_iter`  >= 0 (default 250).
 * `tol`       >= 0 (default 1e-3, relative L2 of the baseline change). */
C4A_API c4a_status_t c4a_pp_modpoly_create(c4a_pp_modpoly_handle_t** out,
                                            int32_t polyorder,
                                            int32_t max_iter, double tol);
C4A_API void         c4a_pp_modpoly_destroy(c4a_pp_modpoly_handle_t* handle);
C4A_API c4a_status_t c4a_pp_modpoly_transform(
    const c4a_pp_modpoly_handle_t* handle,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out);

/* ---------- IModPoly (Gan, Ruan, Mo 2006) ------------------------------ */
typedef struct c4a_pp_imodpoly_handle_t c4a_pp_imodpoly_handle_t;
/* `polyorder` >= 0 (default 2).
 * `max_iter`  >= 0 (default 250).
 * `tol`       >= 0 (default 1e-3, relative change in residual stdev). */
C4A_API c4a_status_t c4a_pp_imodpoly_create(c4a_pp_imodpoly_handle_t** out,
                                             int32_t polyorder,
                                             int32_t max_iter, double tol);
C4A_API void         c4a_pp_imodpoly_destroy(c4a_pp_imodpoly_handle_t* handle);
C4A_API c4a_status_t c4a_pp_imodpoly_transform(
    const c4a_pp_imodpoly_handle_t* handle,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out);

/* ---------- SNIP (Ryan 1988, Morháč 1997) ------------------------------ */
typedef struct c4a_pp_snip_handle_t c4a_pp_snip_handle_t;
/* `max_half_window` >= 1 (default 20). The algorithm iterates window widths
 * w in [1, max_half_window] applying the LLS-transformed local-min clip. */
C4A_API c4a_status_t c4a_pp_snip_create(c4a_pp_snip_handle_t** out,
                                         int32_t max_half_window);
C4A_API void         c4a_pp_snip_destroy(c4a_pp_snip_handle_t* handle);
C4A_API c4a_status_t c4a_pp_snip_transform(const c4a_pp_snip_handle_t* handle,
                                            c4a_matrix_view_t X,
                                            c4a_matrix_view_t out);

/* ---------- RollingBall (Kneen & Annegarn 1996) ------------------------ */
typedef struct c4a_pp_rolling_ball_handle_t c4a_pp_rolling_ball_handle_t;
/* `half_window`        >= 1 (default 20). Window radius for the min-then-max
 *                            morphological filter.
 * `smooth_half_window` >= 0 (default 0). Optional moving-average smoothing of
 *                            the final baseline. 0 disables smoothing. */
C4A_API c4a_status_t c4a_pp_rolling_ball_create(
    c4a_pp_rolling_ball_handle_t** out,
    int32_t half_window, int32_t smooth_half_window);
C4A_API void         c4a_pp_rolling_ball_destroy(
    c4a_pp_rolling_ball_handle_t* handle);
C4A_API c4a_status_t c4a_pp_rolling_ball_transform(
    const c4a_pp_rolling_ball_handle_t* handle,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out);

/* ---------- IAsLS (He 2014) -------------------------------------------- */
typedef struct c4a_pp_iasls_handle_t c4a_pp_iasls_handle_t;
/* `lam`       > 0 (default 1e6).
 * `p`         (asymmetry, 0 < p < 1; default 1e-2).
 * `polyorder` >= 0 (default 2, prefit polynomial degree).
 * `max_iter`  >= 0 (default 50).
 * `tol`       >= 0 (default 1e-3, relative L2 weight change). */
C4A_API c4a_status_t c4a_pp_iasls_create(c4a_pp_iasls_handle_t** out,
                                          double lam, double p,
                                          int32_t polyorder,
                                          int32_t max_iter, double tol);
C4A_API void         c4a_pp_iasls_destroy(c4a_pp_iasls_handle_t* handle);
C4A_API c4a_status_t c4a_pp_iasls_transform(const c4a_pp_iasls_handle_t* handle,
                                             c4a_matrix_view_t X,
                                             c4a_matrix_view_t out);

/* ---------- BEADS (simplified) — Ning & Selesnick 2014 ----------------- */
typedef struct c4a_pp_beads_handle_t c4a_pp_beads_handle_t;
/* `lam_0`    > 0 (default 1e2, sparsity weight on baseline residual).
 * `lam_1`    > 0 (default 0.5, banded 1st-difference smoothness weight).
 * `lam_2`    > 0 (default 0.5, banded 2nd-difference smoothness weight).
 * `max_iter` >= 0 (default 50).
 * `tol`      >= 0 (default 1e-3, relative L2 baseline change).
 *
 * NOTE: this is the simplified BEADS variant using a reweighted-L2 sparsity
 * surrogate over the pentadiagonal D_2^T D_2 + scaled D_1^T D_1 system.
 * See docs/algorithms/beads.md for the simplification and what is deferred. */
C4A_API c4a_status_t c4a_pp_beads_create(c4a_pp_beads_handle_t** out,
                                          double lam_0, double lam_1,
                                          double lam_2,
                                          int32_t max_iter, double tol);
C4A_API void         c4a_pp_beads_destroy(c4a_pp_beads_handle_t* handle);
C4A_API c4a_status_t c4a_pp_beads_transform(const c4a_pp_beads_handle_t* handle,
                                             c4a_matrix_view_t X,
                                             c4a_matrix_view_t out);

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
typedef struct c4a_pp_to_absorbance_handle_t c4a_pp_to_absorbance_handle_t;
C4A_API c4a_status_t c4a_pp_to_absorbance_create(
    c4a_pp_to_absorbance_handle_t** out,
    int is_percent, double epsilon, int clip_negative);
C4A_API void         c4a_pp_to_absorbance_destroy(
    c4a_pp_to_absorbance_handle_t* handle);
C4A_API c4a_status_t c4a_pp_to_absorbance_transform(
    const c4a_pp_to_absorbance_handle_t* handle,
    c4a_matrix_view_t X, c4a_matrix_view_t out);

/* ---------- FromAbsorbance --------------------------------------------- */
typedef struct c4a_pp_from_absorbance_handle_t c4a_pp_from_absorbance_handle_t;
C4A_API c4a_status_t c4a_pp_from_absorbance_create(
    c4a_pp_from_absorbance_handle_t** out, int is_percent);
C4A_API void         c4a_pp_from_absorbance_destroy(
    c4a_pp_from_absorbance_handle_t* handle);
C4A_API c4a_status_t c4a_pp_from_absorbance_transform(
    const c4a_pp_from_absorbance_handle_t* handle,
    c4a_matrix_view_t X, c4a_matrix_view_t out);

/* ---------- PercentToFraction ----------------------------------------- */
typedef struct c4a_pp_pct_to_frac_handle_t c4a_pp_pct_to_frac_handle_t;
C4A_API c4a_status_t c4a_pp_pct_to_frac_create(
    c4a_pp_pct_to_frac_handle_t** out);
C4A_API void         c4a_pp_pct_to_frac_destroy(
    c4a_pp_pct_to_frac_handle_t* handle);
C4A_API c4a_status_t c4a_pp_pct_to_frac_transform(
    const c4a_pp_pct_to_frac_handle_t* handle,
    c4a_matrix_view_t X, c4a_matrix_view_t out);

/* ---------- FractionToPercent ----------------------------------------- */
typedef struct c4a_pp_frac_to_pct_handle_t c4a_pp_frac_to_pct_handle_t;
C4A_API c4a_status_t c4a_pp_frac_to_pct_create(
    c4a_pp_frac_to_pct_handle_t** out);
C4A_API void         c4a_pp_frac_to_pct_destroy(
    c4a_pp_frac_to_pct_handle_t* handle);
C4A_API c4a_status_t c4a_pp_frac_to_pct_transform(
    const c4a_pp_frac_to_pct_handle_t* handle,
    c4a_matrix_view_t X, c4a_matrix_view_t out);

/* ---------- KubelkaMunk ----------------------------------------------- */
typedef struct c4a_pp_kubelka_munk_handle_t c4a_pp_kubelka_munk_handle_t;
C4A_API c4a_status_t c4a_pp_kubelka_munk_create(
    c4a_pp_kubelka_munk_handle_t** out, int is_percent, double epsilon);
C4A_API void         c4a_pp_kubelka_munk_destroy(
    c4a_pp_kubelka_munk_handle_t* handle);
C4A_API c4a_status_t c4a_pp_kubelka_munk_transform(
    const c4a_pp_kubelka_munk_handle_t* handle,
    c4a_matrix_view_t X, c4a_matrix_view_t out);

/* ============================================================================
 * 14. Phase 8 — Orthogonalization (OSC, EPO)
 * ============================================================================
 *
 * Both operators expose the stateful `_create / _fit / _transform / _destroy
 * / _is_fitted` ABI contract. OSC additionally exposes `_inverse_transform`
 * which always returns C4A_ERR_UNSUPPORTED (the orthogonal-component
 * deflation is many-to-one). EPO has its own `_inverse_transform`.
 */

/* ---------- Orthogonal Signal Correction (Wold 1998) ------------------- */
typedef struct c4a_pp_osc_handle_t c4a_pp_osc_handle_t;
C4A_API c4a_status_t c4a_pp_osc_create(c4a_pp_osc_handle_t** out,
                                        int32_t n_components, int scale);
C4A_API void         c4a_pp_osc_destroy(c4a_pp_osc_handle_t* handle);
C4A_API c4a_status_t c4a_pp_osc_fit(c4a_pp_osc_handle_t* handle,
                                     c4a_matrix_view_t X,
                                     const double* y, int64_t y_len);
C4A_API c4a_status_t c4a_pp_osc_transform(const c4a_pp_osc_handle_t* handle,
                                           c4a_matrix_view_t X,
                                           c4a_matrix_view_t out);
C4A_API c4a_status_t c4a_pp_osc_inverse_transform(
    const c4a_pp_osc_handle_t* handle,
    c4a_matrix_view_t X, c4a_matrix_view_t out);
C4A_API c4a_status_t c4a_pp_osc_is_fitted(const c4a_pp_osc_handle_t* handle,
                                           int* out_fitted);

/* ---------- External Parameter Orthogonalisation (Roger 2003) ---------- */
typedef struct c4a_pp_epo_handle_t c4a_pp_epo_handle_t;
C4A_API c4a_status_t c4a_pp_epo_create(c4a_pp_epo_handle_t** out, int scale);
C4A_API void         c4a_pp_epo_destroy(c4a_pp_epo_handle_t* handle);
C4A_API c4a_status_t c4a_pp_epo_fit(c4a_pp_epo_handle_t* handle,
                                     c4a_matrix_view_t X,
                                     const double* d, int64_t d_len);
C4A_API c4a_status_t c4a_pp_epo_transform(const c4a_pp_epo_handle_t* handle,
                                           c4a_matrix_view_t X,
                                           c4a_matrix_view_t out);
C4A_API c4a_status_t c4a_pp_epo_inverse_transform(
    const c4a_pp_epo_handle_t* handle,
    c4a_matrix_view_t X, c4a_matrix_view_t out);
C4A_API c4a_status_t c4a_pp_epo_is_fitted(const c4a_pp_epo_handle_t* handle,
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
typedef struct c4a_pp_flex_pca_handle_t c4a_pp_flex_pca_handle_t;
C4A_API c4a_status_t c4a_pp_flex_pca_create(c4a_pp_flex_pca_handle_t** out,
                                             double n_components);
C4A_API void         c4a_pp_flex_pca_destroy(c4a_pp_flex_pca_handle_t* handle);
C4A_API c4a_status_t c4a_pp_flex_pca_fit(c4a_pp_flex_pca_handle_t* handle,
                                          c4a_matrix_view_t X);
C4A_API c4a_status_t c4a_pp_flex_pca_transform(
    const c4a_pp_flex_pca_handle_t* handle,
    c4a_matrix_view_t X, c4a_matrix_view_t out);
C4A_API c4a_status_t c4a_pp_flex_pca_is_fitted(
    const c4a_pp_flex_pca_handle_t* handle, int* out_fitted);
C4A_API c4a_status_t c4a_pp_flex_pca_output_cols(
    const c4a_pp_flex_pca_handle_t* handle, int64_t* out_cols);

/* ---------- FlexibleSVD ------------------------------------------------ */
typedef struct c4a_pp_flex_svd_handle_t c4a_pp_flex_svd_handle_t;
C4A_API c4a_status_t c4a_pp_flex_svd_create(c4a_pp_flex_svd_handle_t** out,
                                             double n_components);
C4A_API void         c4a_pp_flex_svd_destroy(c4a_pp_flex_svd_handle_t* handle);
C4A_API c4a_status_t c4a_pp_flex_svd_fit(c4a_pp_flex_svd_handle_t* handle,
                                          c4a_matrix_view_t X);
C4A_API c4a_status_t c4a_pp_flex_svd_transform(
    const c4a_pp_flex_svd_handle_t* handle,
    c4a_matrix_view_t X, c4a_matrix_view_t out);
C4A_API c4a_status_t c4a_pp_flex_svd_is_fitted(
    const c4a_pp_flex_svd_handle_t* handle, int* out_fitted);
C4A_API c4a_status_t c4a_pp_flex_svd_output_cols(
    const c4a_pp_flex_svd_handle_t* handle, int64_t* out_cols);

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
typedef struct c4a_pp_resampler_handle_t c4a_pp_resampler_handle_t;
C4A_API c4a_status_t c4a_pp_resampler_create(c4a_pp_resampler_handle_t** out,
                                              const double* target_wl,
                                              int64_t n_target,
                                              int32_t method,
                                              double crop_min, double crop_max,
                                              int use_crop, double fill_value,
                                              int bounds_error,
                                              int extrapolate);
C4A_API void         c4a_pp_resampler_destroy(c4a_pp_resampler_handle_t* h);
C4A_API c4a_status_t c4a_pp_resampler_fit(c4a_pp_resampler_handle_t* h,
                                           const double* source_wl,
                                           int64_t n_source);
C4A_API c4a_status_t c4a_pp_resampler_is_fitted(
    const c4a_pp_resampler_handle_t* h, int* out_fitted);
C4A_API int64_t      c4a_pp_resampler_output_cols(
    const c4a_pp_resampler_handle_t* h);
C4A_API c4a_status_t c4a_pp_resampler_transform(
    const c4a_pp_resampler_handle_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t out);

/* ---------- CropTransformer (stateless) ------------------------------- */
typedef struct c4a_pp_crop_handle_t c4a_pp_crop_handle_t;
C4A_API c4a_status_t c4a_pp_crop_create(c4a_pp_crop_handle_t** out,
                                         int64_t start, int64_t end);
C4A_API void         c4a_pp_crop_destroy(c4a_pp_crop_handle_t* h);
C4A_API int64_t      c4a_pp_crop_output_cols(const c4a_pp_crop_handle_t* h,
                                              int64_t input_cols);
C4A_API c4a_status_t c4a_pp_crop_transform(const c4a_pp_crop_handle_t* h,
                                            c4a_matrix_view_t X,
                                            c4a_matrix_view_t out);

/* ---------- ResampleTransformer (stateless) --------------------------- */
typedef struct c4a_pp_resample_handle_t c4a_pp_resample_handle_t;
C4A_API c4a_status_t c4a_pp_resample_create(c4a_pp_resample_handle_t** out,
                                             int64_t num_samples);
C4A_API void         c4a_pp_resample_destroy(c4a_pp_resample_handle_t* h);
C4A_API int64_t      c4a_pp_resample_output_cols(
    const c4a_pp_resample_handle_t* h, int64_t input_cols);
C4A_API c4a_status_t c4a_pp_resample_transform(
    const c4a_pp_resample_handle_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t out);

/* ---------- IntegerKBinsDiscretizer (stateful, int32 output) ---------- */
typedef struct c4a_pp_kbins_disc_handle_t c4a_pp_kbins_disc_handle_t;
C4A_API c4a_status_t c4a_pp_kbins_disc_create(c4a_pp_kbins_disc_handle_t** out,
                                               int32_t n_bins,
                                               int32_t strategy);
C4A_API void         c4a_pp_kbins_disc_destroy(c4a_pp_kbins_disc_handle_t* h);
C4A_API c4a_status_t c4a_pp_kbins_disc_fit(c4a_pp_kbins_disc_handle_t* h,
                                            c4a_matrix_view_t X);
C4A_API c4a_status_t c4a_pp_kbins_disc_is_fitted(
    const c4a_pp_kbins_disc_handle_t* h, int* out_fitted);
C4A_API c4a_status_t c4a_pp_kbins_disc_transform(
    const c4a_pp_kbins_disc_handle_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t out);

/* ---------- RangeDiscretizer (stateless, int32 output) ---------------- */
typedef struct c4a_pp_range_disc_handle_t c4a_pp_range_disc_handle_t;
C4A_API c4a_status_t c4a_pp_range_disc_create(c4a_pp_range_disc_handle_t** out,
                                               const double* bins,
                                               int64_t n_edges);
C4A_API void         c4a_pp_range_disc_destroy(c4a_pp_range_disc_handle_t* h);
C4A_API c4a_status_t c4a_pp_range_disc_transform(
    const c4a_pp_range_disc_handle_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t out);

/* ============================================================================
 * 17. Phase 11 — Splitters (c4a_split_*)
 * ============================================================================
 *
 * Nine train/test splitting strategies. Each splitter returns a
 * `c4a_split_result_t` containing two heap-allocated index arrays that the
 * caller releases via `c4a_split_result_destroy`.
 *
 *   - KennardStone, SPXY, SPXYFold, SPXYGFold (group-aware), KMeans,
 *     KBinsStratified, BinnedStratifiedGroupKFold, SystematicCircular,
 *     SPlit (data twinning).
 */

/* ---------- Shared result type ----------------------------------------- */
typedef struct c4a_split_result_t {
    int64_t* train_idx;
    int64_t  n_train;
    int64_t* test_idx;
    int64_t  n_test;
    void*    _owner;
} c4a_split_result_t;

C4A_API void c4a_split_result_destroy(c4a_split_result_t* r);

/* ---------- KennardStone ----------------------------------------------- */
typedef struct c4a_split_kennard_stone_t c4a_split_kennard_stone_t;
C4A_API c4a_status_t c4a_split_kennard_stone_create(
    c4a_split_kennard_stone_t** out, double test_size);
C4A_API void c4a_split_kennard_stone_destroy(c4a_split_kennard_stone_t* h);
C4A_API c4a_status_t c4a_split_kennard_stone_split(
    const c4a_split_kennard_stone_t* h,
    c4a_matrix_view_t X, c4a_split_result_t* out);

/* ---------- SPXY (single train/test, joint X-Y) ----------------------- */
typedef struct c4a_split_spxy_t c4a_split_spxy_t;
C4A_API c4a_status_t c4a_split_spxy_create(c4a_split_spxy_t** out,
                                            double test_size);
C4A_API void c4a_split_spxy_destroy(c4a_split_spxy_t* h);
C4A_API c4a_status_t c4a_split_spxy_split(const c4a_split_spxy_t* h,
                                           c4a_matrix_view_t X,
                                           c4a_matrix_view_t Y,
                                           c4a_split_result_t* out);

/* ---------- SPXYFold (k-fold) ---------------------------------------- */
typedef struct c4a_split_spxy_fold_t c4a_split_spxy_fold_t;
C4A_API c4a_status_t c4a_split_spxy_fold_create(c4a_split_spxy_fold_t** out,
                                                 int32_t n_splits,
                                                 int32_t y_metric);
C4A_API void c4a_split_spxy_fold_destroy(c4a_split_spxy_fold_t* h);
C4A_API c4a_status_t c4a_split_spxy_fold_n_splits(
    const c4a_split_spxy_fold_t* h, int32_t* out);
C4A_API c4a_status_t c4a_split_spxy_fold_split_fold(
    const c4a_split_spxy_fold_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t Y,
    int32_t fold_idx, c4a_split_result_t* out);

/* ---------- SPXYGFold (group-aware k-fold) --------------------------- */
typedef struct c4a_split_spxy_g_fold_t c4a_split_spxy_g_fold_t;
C4A_API c4a_status_t c4a_split_spxy_g_fold_create(c4a_split_spxy_g_fold_t** out,
                                                    int32_t n_splits,
                                                    int32_t y_metric,
                                                    int32_t aggregation);
C4A_API void c4a_split_spxy_g_fold_destroy(c4a_split_spxy_g_fold_t* h);
C4A_API c4a_status_t c4a_split_spxy_g_fold_n_splits(
    const c4a_split_spxy_g_fold_t* h, int32_t* out);
C4A_API c4a_status_t c4a_split_spxy_g_fold_split_fold(
    const c4a_split_spxy_g_fold_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t Y,
    const int64_t* groups, int64_t groups_len,
    int32_t fold_idx, c4a_split_result_t* out);

/* ---------- KMeans (k-means++) ---------------------------------------- */
typedef struct c4a_split_kmeans_t c4a_split_kmeans_t;
C4A_API c4a_status_t c4a_split_kmeans_create(c4a_split_kmeans_t** out,
                                               double test_size, uint64_t seed,
                                               int32_t max_iter);
C4A_API void c4a_split_kmeans_destroy(c4a_split_kmeans_t* h);
C4A_API c4a_status_t c4a_split_kmeans_split(const c4a_split_kmeans_t* h,
                                              c4a_matrix_view_t X,
                                              c4a_split_result_t* out);

/* ---------- KBinsStratified ------------------------------------------ */
typedef struct c4a_split_kbins_stratified_t c4a_split_kbins_stratified_t;
C4A_API c4a_status_t c4a_split_kbins_stratified_create(
    c4a_split_kbins_stratified_t** out,
    double test_size, uint64_t seed,
    int32_t n_bins, int32_t strategy);
C4A_API void c4a_split_kbins_stratified_destroy(c4a_split_kbins_stratified_t* h);
C4A_API c4a_status_t c4a_split_kbins_stratified_split(
    const c4a_split_kbins_stratified_t* h,
    c4a_matrix_view_t Y, c4a_split_result_t* out);

/* ---------- BinnedStratifiedGroupKFold ------------------------------- */
typedef struct c4a_split_binned_strat_group_kfold_t
    c4a_split_binned_strat_group_kfold_t;
C4A_API c4a_status_t c4a_split_binned_strat_group_kfold_create(
    c4a_split_binned_strat_group_kfold_t** out,
    int32_t n_splits, int32_t n_bins, int32_t strategy,
    int32_t shuffle, uint64_t seed);
C4A_API void c4a_split_binned_strat_group_kfold_destroy(
    c4a_split_binned_strat_group_kfold_t* h);
C4A_API c4a_status_t c4a_split_binned_strat_group_kfold_n_splits(
    const c4a_split_binned_strat_group_kfold_t* h, int32_t* out);
C4A_API c4a_status_t c4a_split_binned_strat_group_kfold_split_fold(
    const c4a_split_binned_strat_group_kfold_t* h,
    c4a_matrix_view_t Y,
    const int64_t* groups, int64_t groups_len,
    int32_t fold_idx, c4a_split_result_t* out);

/* ---------- SystematicCircular --------------------------------------- */
typedef struct c4a_split_systematic_circular_t c4a_split_systematic_circular_t;
C4A_API c4a_status_t c4a_split_systematic_circular_create(
    c4a_split_systematic_circular_t** out, double test_size, uint64_t seed);
C4A_API void c4a_split_systematic_circular_destroy(
    c4a_split_systematic_circular_t* h);
C4A_API c4a_status_t c4a_split_systematic_circular_split(
    const c4a_split_systematic_circular_t* h,
    c4a_matrix_view_t Y, c4a_split_result_t* out);

/* ---------- SPlit (data twinning) ------------------------------------ */
typedef struct c4a_split_split_splitter_t c4a_split_split_splitter_t;
C4A_API c4a_status_t c4a_split_split_splitter_create(
    c4a_split_split_splitter_t** out, double test_size, uint64_t seed);
C4A_API void c4a_split_split_splitter_destroy(c4a_split_split_splitter_t* h);
C4A_API c4a_status_t c4a_split_split_splitter_split(
    const c4a_split_split_splitter_t* h,
    c4a_matrix_view_t X, c4a_split_result_t* out);

/* ============================================================================
 * 18. Phase 12 — Sample filters: Y-based outliers (c4a_filter_*)
 * ============================================================================
 *
 * Sample-level filters return a per-row keep-mask (`uint8_t`, 1 = keep,
 * 0 = exclude) plus a `c4a_filter_stats_t` summary. Mask + stats buffers
 * are caller-allocated.
 *
 * Phase 12 introduces the `c4a_filter_stats_t` struct shared with later
 * filter phases (P14, future P13).
 */

/* ---------- Shared filter result type --------------------------------- */
typedef struct c4a_filter_stats_t {
    int64_t n_samples;        /* total samples seen by the filter         */
    int64_t n_kept;           /* count of mask[i] == 1                    */
    int64_t n_excluded;       /* n_samples - n_kept                       */
    double  exclusion_rate;   /* n_excluded / n_samples; 0.0 when n == 0  */
} c4a_filter_stats_t;

/* ---------- Y-outlier method enum ------------------------------------- */
typedef enum c4a_y_outlier_method_t {
    C4A_Y_OUTLIER_IQR        = 0,
    C4A_Y_OUTLIER_ZSCORE     = 1,
    C4A_Y_OUTLIER_PERCENTILE = 2,
    C4A_Y_OUTLIER_MAD        = 3
} c4a_y_outlier_method_t;

/* ---------- YOutlierFilter -------------------------------------------- */
typedef struct c4a_filter_y_outlier_handle_t c4a_filter_y_outlier_handle_t;
C4A_API c4a_status_t c4a_filter_y_outlier_create(
    c4a_filter_y_outlier_handle_t** out,
    c4a_y_outlier_method_t method,
    double threshold, double lower_pct, double upper_pct);
C4A_API void         c4a_filter_y_outlier_destroy(
    c4a_filter_y_outlier_handle_t* handle);
C4A_API c4a_status_t c4a_filter_y_outlier_fit_get_mask(
    const c4a_filter_y_outlier_handle_t* handle,
    const double* y, int64_t n,
    uint8_t* mask_out, c4a_filter_stats_t* stats_out);

/* ============================================================================
 * 19. Phase 14 — Leverage / Quality / Composite filters (c4a_filter_*)
 * ============================================================================
 *
 * Three meta-filter operators that reuse the §18 `c4a_filter_stats_t`.
 *
 *   - HighLeverageFilter   : hat-matrix or PCA score-space leverage.
 *   - SpectralQualityFilter : six all-or-nothing thresholds on each row.
 *   - CompositeFilter      : combines sub-filter keep-masks via ANY or ALL.
 */

/* ---------- HighLeverageFilter (stateful) ----------------------------- */
typedef struct c4a_filter_leverage_handle_t c4a_filter_leverage_handle_t;
C4A_API c4a_status_t c4a_filter_leverage_create(
    c4a_filter_leverage_handle_t** out,
    int32_t method,
    double  threshold_multiplier,
    int     use_absolute_threshold,
    double  absolute_threshold,
    int32_t n_components,
    int     center);
C4A_API void         c4a_filter_leverage_destroy(
    c4a_filter_leverage_handle_t* handle);
C4A_API c4a_status_t c4a_filter_leverage_fit(
    c4a_filter_leverage_handle_t* handle, c4a_matrix_view_t X);
C4A_API c4a_status_t c4a_filter_leverage_is_fitted(
    const c4a_filter_leverage_handle_t* handle, int* out_fitted);
C4A_API c4a_status_t c4a_filter_leverage_apply(
    const c4a_filter_leverage_handle_t* handle,
    c4a_matrix_view_t X,
    uint8_t* mask_out, c4a_filter_stats_t* stats_out);
C4A_API double       c4a_filter_leverage_threshold(
    const c4a_filter_leverage_handle_t* handle);

/* ---------- SpectralQualityFilter (stateless) ------------------------- */
typedef struct c4a_filter_quality_handle_t c4a_filter_quality_handle_t;
C4A_API c4a_status_t c4a_filter_quality_create(
    c4a_filter_quality_handle_t** out,
    double max_nan_ratio, double max_zero_ratio,
    double min_variance,
    int use_max, double max_value,
    int use_min, double min_value,
    int check_inf);
C4A_API void         c4a_filter_quality_destroy(
    c4a_filter_quality_handle_t* handle);
C4A_API c4a_status_t c4a_filter_quality_apply(
    const c4a_filter_quality_handle_t* handle,
    c4a_matrix_view_t X,
    uint8_t* mask_out, c4a_filter_stats_t* stats_out);

/* ---------- CompositeFilter ------------------------------------------- */
typedef enum c4a_composite_mode_t {
    C4A_COMPOSITE_ANY = 0,    /* exclude if ANY sub-filter excludes */
    C4A_COMPOSITE_ALL = 1     /* exclude only if ALL sub-filters exclude */
} c4a_composite_mode_t;

typedef struct c4a_filter_composite_handle_t c4a_filter_composite_handle_t;
C4A_API c4a_status_t c4a_filter_composite_create(
    c4a_filter_composite_handle_t** out, c4a_composite_mode_t mode);
C4A_API void         c4a_filter_composite_destroy(
    c4a_filter_composite_handle_t* handle);
C4A_API c4a_status_t c4a_filter_composite_add_leverage(
    c4a_filter_composite_handle_t* handle,
    c4a_filter_leverage_handle_t* sub);
C4A_API c4a_status_t c4a_filter_composite_add_quality(
    c4a_filter_composite_handle_t* handle,
    c4a_filter_quality_handle_t* sub);
C4A_API c4a_status_t c4a_filter_composite_apply(
    const c4a_filter_composite_handle_t* handle,
    c4a_matrix_view_t X,
    uint8_t* mask_out, c4a_filter_stats_t* stats_out);

/* ============================================================================
 * 20. Phase 19 — Signal type detection, NIRS metrics, T² / Q residuals
 * ============================================================================
 *
 * Three groups of free-function utilities:
 *
 *   - c4a_signal_* : auto-detect spectrum type (absorbance / reflectance /
 *                    transmittance / Kubelka-Munk / log(1/R) / log(1/T))
 *                    plus the percent-scaled variants.
 *   - c4a_metric_* : eight closed-form regression metrics (RMSE, MAE,
 *                    bias, SEP, RPD, RPIQ, R², NRMSE).
 *   - c4a_util_*   : multivariate outlier statistics (Hotelling T²,
 *                    Q-residuals) with their analytical upper control limits.
 */

/* ---------- 20.1 SignalTypeDetector ----------------------------------- */
typedef enum c4a_signal_type_t {
    C4A_SIGNAL_UNKNOWN               = 0,
    C4A_SIGNAL_ABSORBANCE            = 1,
    C4A_SIGNAL_REFLECTANCE           = 2,
    C4A_SIGNAL_REFLECTANCE_PERCENT   = 3,
    C4A_SIGNAL_TRANSMITTANCE         = 4,
    C4A_SIGNAL_TRANSMITTANCE_PERCENT = 5,
    C4A_SIGNAL_KUBELKA_MUNK          = 6,
    C4A_SIGNAL_LOG_1_R               = 7,
    C4A_SIGNAL_LOG_1_T               = 8
} c4a_signal_type_t;

C4A_API c4a_status_t c4a_signal_detect(c4a_matrix_view_t X,
                                        const double* wavelengths_optional,
                                        int64_t wl_length,
                                        double confidence_threshold,
                                        c4a_signal_type_t* type_out,
                                        double* confidence_out,
                                        char reason_buf[256]);

/* ---------- 20.2 NIRS regression metrics ------------------------------ */
C4A_API c4a_status_t c4a_metric_rmse (const double* y_true,
                                       const double* y_pred,
                                       int64_t n, double* out);
C4A_API c4a_status_t c4a_metric_mae  (const double* y_true,
                                       const double* y_pred,
                                       int64_t n, double* out);
C4A_API c4a_status_t c4a_metric_bias (const double* y_true,
                                       const double* y_pred,
                                       int64_t n, double* out);
C4A_API c4a_status_t c4a_metric_sep  (const double* y_true,
                                       const double* y_pred,
                                       int64_t n, double* out);
C4A_API c4a_status_t c4a_metric_rpd  (const double* y_true,
                                       const double* y_pred,
                                       int64_t n, double* out);
C4A_API c4a_status_t c4a_metric_rpiq (const double* y_true,
                                       const double* y_pred,
                                       int64_t n, double* out);
C4A_API c4a_status_t c4a_metric_r2   (const double* y_true,
                                       const double* y_pred,
                                       int64_t n, double* out);
C4A_API c4a_status_t c4a_metric_nrmse(const double* y_true,
                                       const double* y_pred,
                                       int64_t n, double* out);

/* ---------- 20.3 Multivariate outlier statistics ---------------------- */
C4A_API c4a_status_t c4a_util_hotelling_t2(c4a_matrix_view_t X,
                                            int32_t n_components,
                                            double alpha,
                                            double* t2_per_sample,
                                            int64_t n_samples,
                                            double* ucl_out);
C4A_API c4a_status_t c4a_util_q_residuals(c4a_matrix_view_t X,
                                           int32_t n_components,
                                           double alpha,
                                           double* q_per_sample,
                                           int64_t n_samples,
                                           double* ucl_out);

/* ============================================================================
 * 21. Phase 20 — Transfer metrics utility (c4a_transfer_*)
 * ============================================================================
 *
 * `c4a_transfer_metrics_compute` computes nine alignment metrics between two
 * datasets (source and target), as defined in
 * nirs4all/analysis/transfer_metrics.py. The struct is plain-data — every
 * field is a `double`. NaN encodes "metric not applicable" (e.g. Grassmann
 * when the two datasets do not share a feature count).
 */
typedef struct c4a_transfer_metrics_t {
    double centroid_distance;
    double cka_similarity;
    double grassmann_distance;
    double rv_coefficient;
    double procrustes_disparity;
    double trustworthiness;
    double spread_distance;
    double evr_source;
    double evr_target;
} c4a_transfer_metrics_t;

C4A_API c4a_status_t c4a_transfer_metrics_compute(
    c4a_matrix_view_t X_source,
    c4a_matrix_view_t X_target,
    int32_t n_components,
    int32_t k_neighbors,
    uint64_t seed,
    c4a_transfer_metrics_t* out);

/* ============================================================================
 * 22. Phase 21 — FCK static transformer (c4a_pp_fck_static_*)
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
typedef struct c4a_pp_fck_static_handle_t c4a_pp_fck_static_handle_t;
C4A_API c4a_status_t c4a_pp_fck_static_create(
    c4a_pp_fck_static_handle_t** out,
    int32_t kernel_size,
    const double* filter_orders, int32_t n_orders,
    const double* filter_scales, int32_t n_scales);
C4A_API void c4a_pp_fck_static_destroy(c4a_pp_fck_static_handle_t* handle);
C4A_API c4a_status_t c4a_pp_fck_static_transform(
    const c4a_pp_fck_static_handle_t* handle,
    c4a_matrix_view_t X, c4a_matrix_view_t out);
C4A_API c4a_status_t c4a_pp_fck_static_output_cols(int32_t n_kernels,
                                                    int32_t n_features,
                                                    int32_t* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

/* ============================================================================
 * 23. ABI guard rails — fixed-size assertions on the C ABI shape
 * ==========================================================================
 *
 * Compilers may shrink enums under non-default flags (e.g. -fshort-enums on
 * gcc/clang). Because enums appear in function signatures and in the
 * c4a_matrix_view_t struct, that would silently break the ABI. We assert at
 * compile time that every public enum is sizeof(int) = 4 bytes, and that
 * c4a_matrix_view_t has its documented 48-byte LP64 layout.
 *
 * If any of these fail, the build fails fast — and the caller knows to use
 * a default-enum-size toolchain.
 */
#ifdef __cplusplus
#  define C4A_STATIC_ASSERT(cond, msg) static_assert((cond), msg)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#  define C4A_STATIC_ASSERT(cond, msg) _Static_assert((cond), msg)
#else
#  define C4A_STATIC_ASSERT(cond, msg) typedef char c4a_sa_##__LINE__[(cond) ? 1 : -1]
#endif

C4A_STATIC_ASSERT(sizeof(c4a_status_t)         == 4, "c4a_status_t must be 4 bytes");
C4A_STATIC_ASSERT(sizeof(c4a_backend_t)        == 4, "c4a_backend_t must be 4 bytes");
C4A_STATIC_ASSERT(sizeof(c4a_dtype_t)          == 4, "c4a_dtype_t must be 4 bytes");
C4A_STATIC_ASSERT(sizeof(c4a_pp_lsnv_pad_mode_t) == 4,
                  "c4a_pp_lsnv_pad_mode_t must be 4 bytes");
C4A_STATIC_ASSERT(sizeof(c4a_pp_area_method_t) == 4,
                  "c4a_pp_area_method_t must be 4 bytes");
C4A_STATIC_ASSERT(sizeof(c4a_pp_savgol_mode_t) == 4,
                  "c4a_pp_savgol_mode_t must be 4 bytes");
C4A_STATIC_ASSERT(sizeof(c4a_pp_gaussian_mode_t) == 4,
                  "c4a_pp_gaussian_mode_t must be 4 bytes");

/* c4a_matrix_view_t must be 48 bytes on LP64 / LLP64. ILP32 is not
 * supported; on ILP32 platforms the layout would differ (pointer is 4 bytes),
 * which is why we exclude armeabi-v7a. */
#if defined(__LP64__) || defined(_WIN64)
C4A_STATIC_ASSERT(sizeof(c4a_matrix_view_t) == 48,
                  "c4a_matrix_view_t layout must be 48 bytes on LP64/LLP64");
#endif

#endif /* CHEMOMETRICS4ALL_C4A_H */
