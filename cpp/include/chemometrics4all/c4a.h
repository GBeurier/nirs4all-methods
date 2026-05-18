/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * chemometrics4all — public C ABI v1.4.0.
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
 * When `auto_offset != 0`, the offset that makes the post-offset minimum
 * equal to `min_value` is recomputed on EVERY call to `_transform` against
 * the data seen by that call (mirroring nirs4all's `fit_transform`). This
 * means calling `_transform` twice with different X values may yield
 * outputs that don't share a common baseline. For sklearn-style fit-once /
 * transform-many semantics, prefer `auto_offset = 0` with an explicit
 * `offset`. A proper `_fit/_transform` split for LogTransform is deferred
 * to Phase 4+ along with the rest of the scaling family. */
C4A_API c4a_status_t c4a_pp_log_create(c4a_pp_log_handle_t** out,
                                        double base, double offset,
                                        int auto_offset, double min_value);
C4A_API void         c4a_pp_log_destroy(c4a_pp_log_handle_t* handle);
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
 * `_fit` is a no-op (the operator carries no learned state) but is still
 * required by the stateful contract; calling `_transform` before `_fit`
 * returns C4A_ERR_NOT_FITTED. This lets bindings treat Derivate uniformly
 * with the other stateful preprocessings without a special case.
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

#ifdef __cplusplus
}  /* extern "C" */
#endif

/* ============================================================================
 * 11. ABI guard rails — fixed-size assertions on the C ABI shape
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
