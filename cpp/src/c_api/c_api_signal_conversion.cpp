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
};
struct c4a_pp_from_absorbance_handle_t {
    c4a_pp_from_absorbance_state_t* state;
};
struct c4a_pp_pct_to_frac_handle_t {
    c4a_pp_pct_to_frac_state_t* state;
};
struct c4a_pp_frac_to_pct_handle_t {
    c4a_pp_frac_to_pct_state_t* state;
};
struct c4a_pp_kubelka_munk_handle_t {
    c4a_pp_kubelka_munk_state_t* state;
};

namespace {

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
        c4a_pp_to_absorbance_state_t* s =
            c4a_pp_to_absorbance_state_new(is_percent, epsilon, clip_negative);
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_to_absorbance_handle_t* h =
            new (std::nothrow) c4a_pp_to_absorbance_handle_t{s};
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
        c4a_pp_from_absorbance_state_t* s =
            c4a_pp_from_absorbance_state_new(is_percent);
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_from_absorbance_handle_t* h =
            new (std::nothrow) c4a_pp_from_absorbance_handle_t{s};
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
    try {
        c4a_pp_pct_to_frac_state_t* s = c4a_pp_pct_to_frac_state_new();
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_pct_to_frac_handle_t* h =
            new (std::nothrow) c4a_pp_pct_to_frac_handle_t{s};
        if (h == nullptr) {
            c4a_pp_pct_to_frac_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_pct_to_frac_destroy(c4a_pp_pct_to_frac_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_pct_to_frac_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_pct_to_frac_transform(
    const c4a_pp_pct_to_frac_handle_t* h,
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
        return c4a_pp_pct_to_frac_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
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
    try {
        c4a_pp_frac_to_pct_state_t* s = c4a_pp_frac_to_pct_state_new();
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_frac_to_pct_handle_t* h =
            new (std::nothrow) c4a_pp_frac_to_pct_handle_t{s};
        if (h == nullptr) {
            c4a_pp_frac_to_pct_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_frac_to_pct_destroy(c4a_pp_frac_to_pct_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_frac_to_pct_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_frac_to_pct_transform(
    const c4a_pp_frac_to_pct_handle_t* h,
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
        return c4a_pp_frac_to_pct_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
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
        c4a_pp_kubelka_munk_state_t* s =
            c4a_pp_kubelka_munk_state_new(is_percent, epsilon);
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_kubelka_munk_handle_t* h =
            new (std::nothrow) c4a_pp_kubelka_munk_handle_t{s};
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
        return c4a_pp_kubelka_munk_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

}  // extern "C"
