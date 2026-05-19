// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for the Phase 7 stateless signal-conversion
// preprocessing operators (ToAbsorbance, FromAbsorbance, PercentToFraction,
// FractionToPercent, KubelkaMunk).
//
// All five operators are pure closed-form arithmetic — no learned state,
// no iterative loops, no boundary modes. They expose the canonical
// `create / transform / destroy` triplet defined in c4a.h §5:
//
//   - `_create(out, ...)` allocates an opaque handle initialised with the
//     constructor parameters; writes NULL to *out on every failure.
//   - `_transform(handle, X_view, out_view)` applies the operator on the
//     input matrix view and writes the result into the output matrix view.
//     Both views must be row-major contiguous F64 of identical shape; the
//     X and out buffers may overlap (in-place is supported).
//   - `_destroy(handle)` releases the handle. NULL-safe.
//
// Status semantics follow the universal rules at the top of c4a.h. Matrix-
// view validation may additionally return C4A_ERR_DTYPE_MISMATCH (non-F64),
// C4A_ERR_STRIDE_INVALID (non-contiguous row-major), or
// C4A_ERR_SHAPE_MISMATCH (X and out shapes disagree).

#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>

#if defined(__SSE2__)
#include <emmintrin.h>
#endif

#include "chemometrics4all/c4a.h"

#include "core/matrix_view.hpp"
#include "core/preprocessing/signal_conversion/fraction_to_percent.h"
#include "core/preprocessing/signal_conversion/from_absorbance.h"
#include "core/preprocessing/signal_conversion/kubelka_munk.h"
#include "core/preprocessing/signal_conversion/percent_to_fraction.h"
#include "core/preprocessing/signal_conversion/to_absorbance.h"

// ---------------------------------------------------------------------------
// Opaque public handles. Each wraps the matching internal engine state and
// keeps the internal layout out of the public surface.
// ---------------------------------------------------------------------------

struct c4a_pp_to_absorbance_handle_t {
    c4a_pp_to_absorbance_state_t* state;
    bool static_storage;
};
struct c4a_pp_from_absorbance_handle_t {
    c4a_pp_from_absorbance_state_t* state;
    bool static_storage;
};
struct c4a_pp_pct_to_frac_handle_t {
    c4a_pp_pct_to_frac_state_t* state;
};
struct c4a_pp_frac_to_pct_handle_t {
    c4a_pp_frac_to_pct_state_t* state;
};
struct c4a_pp_kubelka_munk_handle_t {
    c4a_pp_kubelka_munk_state_t* state;
    bool static_storage;
};

static c4a_pp_pct_to_frac_handle_t g_pct_to_frac_handle{
    c4a_pp_pct_to_frac_state_new()};
static c4a_pp_frac_to_pct_handle_t g_frac_to_pct_handle{
    c4a_pp_frac_to_pct_state_new()};

namespace {

c4a_pp_to_absorbance_handle_t* default_to_absorbance_handle() noexcept {
    static c4a_pp_to_absorbance_state_t* state =
        c4a_pp_to_absorbance_state_new(0, 1e-10, 1);
    static c4a_pp_to_absorbance_handle_t handle{state, true};
    return state == nullptr ? nullptr : &handle;
}

c4a_pp_from_absorbance_handle_t* default_from_absorbance_handle() noexcept {
    static c4a_pp_from_absorbance_state_t* state =
        c4a_pp_from_absorbance_state_new(0);
    static c4a_pp_from_absorbance_handle_t handle{state, true};
    return state == nullptr ? nullptr : &handle;
}

c4a_pp_kubelka_munk_handle_t* default_kubelka_munk_handle() noexcept {
    static c4a_pp_kubelka_munk_handle_t handle{nullptr, true};
    return &handle;
}

bool valid_public_dtype(c4a_dtype_t dtype) noexcept {
    return dtype == C4A_DTYPE_F64 || dtype == C4A_DTYPE_F32 ||
           dtype == C4A_DTYPE_I32 || dtype == C4A_DTYPE_I64;
}

void transform_multiply_double(const double* X,
                               std::size_t total,
                               double scale,
                               double* out) noexcept {
    std::size_t i = 0;
#if defined(__SSE2__)
    const __m128d vscale = _mm_set1_pd(scale);
    for (; i + 1 < total; i += 2) {
        const __m128d x = _mm_loadu_pd(X + i);
        _mm_storeu_pd(out + i, _mm_mul_pd(x, vscale));
    }
#endif
    for (; i < total; ++i) {
        out[i] = X[i] * scale;
    }
}

c4a_status_t transform_default_kubelka_munk(
    const double* X, std::int64_t rows, std::int64_t cols, double* out) noexcept {
    const std::size_t total = static_cast<std::size_t>(rows) *
                              static_cast<std::size_t>(cols);
    const double lo = 1e-10;
    const double hi = 1.0 - 1e-10;
    std::size_t i = 0;
#if defined(__SSE2__)
    const __m128d vlo = _mm_set1_pd(lo);
    const __m128d vhi = _mm_set1_pd(hi);
    const __m128d vhalf = _mm_set1_pd(0.5);
    const __m128d vone = _mm_set1_pd(1.0);
    for (; i + 1 < total; i += 2) {
        __m128d R = _mm_loadu_pd(X + i);
        R = _mm_max_pd(R, vlo);
        R = _mm_min_pd(R, vhi);
        __m128d y = _mm_div_pd(vhalf, R);
        y = _mm_sub_pd(y, vone);
        y = _mm_add_pd(y, _mm_mul_pd(vhalf, R));
        _mm_storeu_pd(out + i, y);
    }
#endif
    for (; i < total; ++i) {
        double R = X[i];
        if (R < lo) R = lo;
        else if (R > hi) R = hi;
        const double one_minus_R = 1.0 - R;
        out[i] = (one_minus_R * one_minus_R) / (2.0 * R);
    }
    return C4A_OK;
}

// Validate that a matrix view is non-NULL, F64, row-major contiguous, and
// expose the underlying double pointer + dimensions. Identical contract to
// c_api_preprocessing.cpp — duplicated locally to avoid a public helper
// header just for this file.
c4a_status_t require_rowmajor_f64(const c4a_matrix_view_t& v,
                                  const double*& out_ptr,
                                  std::int64_t& out_rows,
                                  std::int64_t& out_cols) noexcept {
    const c4a_status_t s = ::chemometrics4all::core::validate_nonnull_view(v);
    if (s != C4A_OK) {
        return s;
    }
    if (v.dtype != C4A_DTYPE_F64) {
        return C4A_ERR_DTYPE_MISMATCH;
    }
    if (v.col_stride != 1) {
        return C4A_ERR_STRIDE_INVALID;
    }
    if (v.rows > 0 && v.cols > 0 && v.row_stride != v.cols) {
        return C4A_ERR_STRIDE_INVALID;
    }
    out_ptr  = static_cast<const double*>(v.data);
    out_rows = v.rows;
    out_cols = v.cols;
    return C4A_OK;
}

c4a_status_t require_rowmajor_f64_mut(c4a_matrix_view_t& v,
                                       double*& out_ptr,
                                       std::int64_t& out_rows,
                                       std::int64_t& out_cols) noexcept {
    const c4a_status_t s = ::chemometrics4all::core::validate_nonnull_view(v);
    if (s != C4A_OK) {
        return s;
    }
    if (v.dtype != C4A_DTYPE_F64) {
        return C4A_ERR_DTYPE_MISMATCH;
    }
    if (v.col_stride != 1) {
        return C4A_ERR_STRIDE_INVALID;
    }
    if (v.rows > 0 && v.cols > 0 && v.row_stride != v.cols) {
        return C4A_ERR_STRIDE_INVALID;
    }
    out_ptr  = static_cast<double*>(v.data);
    out_rows = v.rows;
    out_cols = v.cols;
    return C4A_OK;
}

}  // namespace

extern "C" {

// ---------------------------------------------------------------------------
// ToAbsorbance
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_to_absorbance_create(
    c4a_pp_to_absorbance_handle_t** out,
    int is_percent, double epsilon, int clip_negative) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    if (!(epsilon > 0.0)) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        if (is_percent == 0 && epsilon == 1e-10 && clip_negative != 0) {
            c4a_pp_to_absorbance_handle_t* h = default_to_absorbance_handle();
            if (h == nullptr) {
                return C4A_ERR_OUT_OF_MEMORY;
            }
            *out = h;
            return C4A_OK;
        }
        c4a_pp_to_absorbance_state_t* s =
            c4a_pp_to_absorbance_state_new(is_percent, epsilon, clip_negative);
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_to_absorbance_handle_t* h =
            new (std::nothrow) c4a_pp_to_absorbance_handle_t{s, false};
        if (h == nullptr) {
            c4a_pp_to_absorbance_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_to_absorbance_destroy(c4a_pp_to_absorbance_handle_t* h) {
    if (h == nullptr) return;
    if (h->static_storage) return;
    try {
        c4a_pp_to_absorbance_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_to_absorbance_transform(
    const c4a_pp_to_absorbance_handle_t* h,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out) {
    if (h == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        const double* xp = nullptr;
        double*       op = nullptr;
        std::int64_t  xr = 0, xc = 0, orr = 0, oc = 0;
        c4a_status_t  s  = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        s = require_rowmajor_f64_mut(out, op, orr, oc);
        if (s != C4A_OK) return s;
        if (xr != orr || xc != oc) {
            return C4A_ERR_SHAPE_MISMATCH;
        }
        return c4a_pp_to_absorbance_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// FromAbsorbance
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_from_absorbance_create(
    c4a_pp_from_absorbance_handle_t** out, int is_percent) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    try {
        if (is_percent == 0) {
            c4a_pp_from_absorbance_handle_t* h =
                default_from_absorbance_handle();
            if (h == nullptr) {
                return C4A_ERR_OUT_OF_MEMORY;
            }
            *out = h;
            return C4A_OK;
        }
        c4a_pp_from_absorbance_state_t* s =
            c4a_pp_from_absorbance_state_new(is_percent);
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_from_absorbance_handle_t* h =
            new (std::nothrow) c4a_pp_from_absorbance_handle_t{s, false};
        if (h == nullptr) {
            c4a_pp_from_absorbance_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_from_absorbance_destroy(
    c4a_pp_from_absorbance_handle_t* h) {
    if (h == nullptr) return;
    if (h->static_storage) return;
    try {
        c4a_pp_from_absorbance_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_from_absorbance_transform(
    const c4a_pp_from_absorbance_handle_t* h,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out) {
    if (h == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        const double* xp = nullptr;
        double*       op = nullptr;
        std::int64_t  xr = 0, xc = 0, orr = 0, oc = 0;
        c4a_status_t  s  = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        s = require_rowmajor_f64_mut(out, op, orr, oc);
        if (s != C4A_OK) return s;
        if (xr != orr || xc != oc) {
            return C4A_ERR_SHAPE_MISMATCH;
        }
        return c4a_pp_from_absorbance_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// PercentToFraction
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_pct_to_frac_create(
    c4a_pp_pct_to_frac_handle_t** out) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    if (g_pct_to_frac_handle.state == nullptr) {
        return C4A_ERR_OUT_OF_MEMORY;
    }
    *out = &g_pct_to_frac_handle;
    return C4A_OK;
}

C4A_API void c4a_pp_pct_to_frac_destroy(c4a_pp_pct_to_frac_handle_t* h) {
    (void)h;
}

C4A_API c4a_status_t c4a_pp_pct_to_frac_transform(
    const c4a_pp_pct_to_frac_handle_t* h,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out) {
    if (h == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    (void)h;
    if (!valid_public_dtype(X.dtype) || !valid_public_dtype(out.dtype)) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (X.dtype != C4A_DTYPE_F64 || out.dtype != C4A_DTYPE_F64) {
        return C4A_ERR_DTYPE_MISMATCH;
    }
    if (X.rows < 0 || X.cols < 0 || out.rows < 0 || out.cols < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (X.col_stride != 1 || out.col_stride != 1 ||
        (X.rows > 0 && X.cols > 0 && X.row_stride != X.cols) ||
        (out.rows > 0 && out.cols > 0 && out.row_stride != out.cols)) {
        return C4A_ERR_STRIDE_INVALID;
    }
    if (X.rows != out.rows || X.cols != out.cols) {
        return C4A_ERR_SHAPE_MISMATCH;
    }
    if (X.rows > 0 && X.cols > 0 && (X.data == nullptr || out.data == nullptr)) {
        return C4A_ERR_NULL_POINTER;
    }
    const double* xp = static_cast<const double*>(X.data);
    double* op = static_cast<double*>(out.data);
    const std::size_t total =
        static_cast<std::size_t>(X.rows) * static_cast<std::size_t>(X.cols);
    for (std::size_t i = 0; i < total; ++i) {
        op[i] = xp[i] / 100.0;
    }
    return C4A_OK;
}

// ---------------------------------------------------------------------------
// FractionToPercent
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_frac_to_pct_create(
    c4a_pp_frac_to_pct_handle_t** out) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    if (g_frac_to_pct_handle.state == nullptr) {
        return C4A_ERR_OUT_OF_MEMORY;
    }
    *out = &g_frac_to_pct_handle;
    return C4A_OK;
}

C4A_API void c4a_pp_frac_to_pct_destroy(c4a_pp_frac_to_pct_handle_t* h) {
    (void)h;
}

C4A_API c4a_status_t c4a_pp_frac_to_pct_transform(
    const c4a_pp_frac_to_pct_handle_t* h,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out) {
    if (h == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    (void)h;
    if (!valid_public_dtype(X.dtype) || !valid_public_dtype(out.dtype)) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (X.dtype != C4A_DTYPE_F64 || out.dtype != C4A_DTYPE_F64) {
        return C4A_ERR_DTYPE_MISMATCH;
    }
    if (X.rows < 0 || X.cols < 0 || out.rows < 0 || out.cols < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (X.col_stride != 1 || out.col_stride != 1 ||
        (X.rows > 0 && X.cols > 0 && X.row_stride != X.cols) ||
        (out.rows > 0 && out.cols > 0 && out.row_stride != out.cols)) {
        return C4A_ERR_STRIDE_INVALID;
    }
    if (X.rows != out.rows || X.cols != out.cols) {
        return C4A_ERR_SHAPE_MISMATCH;
    }
    if (X.rows > 0 && X.cols > 0 && (X.data == nullptr || out.data == nullptr)) {
        return C4A_ERR_NULL_POINTER;
    }
    const double* xp = static_cast<const double*>(X.data);
    double* op = static_cast<double*>(out.data);
    const std::size_t total =
        static_cast<std::size_t>(X.rows) * static_cast<std::size_t>(X.cols);
    transform_multiply_double(xp, total, 100.0, op);
    return C4A_OK;
}

// ---------------------------------------------------------------------------
// KubelkaMunk
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_kubelka_munk_create(
    c4a_pp_kubelka_munk_handle_t** out, int is_percent, double epsilon) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    if (!(epsilon > 0.0) || !(epsilon < 0.5)) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        if (is_percent == 0 && epsilon == 1e-10) {
            c4a_pp_kubelka_munk_handle_t* h = default_kubelka_munk_handle();
            if (h == nullptr) {
                return C4A_ERR_OUT_OF_MEMORY;
            }
            *out = h;
            return C4A_OK;
        }
        c4a_pp_kubelka_munk_state_t* s =
            c4a_pp_kubelka_munk_state_new(is_percent, epsilon);
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_kubelka_munk_handle_t* h =
            new (std::nothrow) c4a_pp_kubelka_munk_handle_t{s, false};
        if (h == nullptr) {
            c4a_pp_kubelka_munk_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_kubelka_munk_destroy(c4a_pp_kubelka_munk_handle_t* h) {
    if (h == nullptr) return;
    if (h->static_storage) return;
    try {
        c4a_pp_kubelka_munk_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_kubelka_munk_transform(
    const c4a_pp_kubelka_munk_handle_t* h,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out) {
    if (h == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        const double* xp = nullptr;
        double*       op = nullptr;
        std::int64_t  xr = 0, xc = 0, orr = 0, oc = 0;
        c4a_status_t  s  = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        s = require_rowmajor_f64_mut(out, op, orr, oc);
        if (s != C4A_OK) return s;
        if (xr != orr || xc != oc) {
            return C4A_ERR_SHAPE_MISMATCH;
        }
        if (h->static_storage) {
            return transform_default_kubelka_munk(xp, xr, xc, op);
        }
        return c4a_pp_kubelka_munk_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

}  // extern "C"
