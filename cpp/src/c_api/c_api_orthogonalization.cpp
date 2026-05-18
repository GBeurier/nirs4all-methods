// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for the Phase 8 orthogonalization preprocessing
// operators (OSC, EPO). Both operators expose the
// `create / fit / transform / destroy / is_fitted` stateful ABI contract
// from §5 of c4a.h. OSC additionally exposes `inverse_transform` returning
// C4A_ERR_UNSUPPORTED (the orthogonal-component deflation is many-to-one
// and cannot be inverted from the post-transform state alone). EPO does
// not expose inverse_transform.
//
// The signatures for §16 of c4a.h are reported in the phase-8 end-of-work
// report; the public header is updated centrally. Until that integration
// lands, the opaque handle typedefs are declared locally below so the
// translation unit compiles. The exported symbol names (c4a_pp_osc_*,
// c4a_pp_epo_*) match the version script's c4a_* glob.

#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>

#include "chemometrics4all/c4a.h"

#include "core/matrix_view.hpp"
#include "core/preprocessing/orthogonalization/epo.h"
#include "core/preprocessing/orthogonalization/osc.h"

// ---------------------------------------------------------------------------
// Opaque public handle typedefs (will live in c4a.h §16 after central
// integration; declared locally here so this TU compiles in isolation).
// ---------------------------------------------------------------------------

extern "C" {

typedef struct c4a_pp_osc_handle_t c4a_pp_osc_handle_t;
typedef struct c4a_pp_epo_handle_t c4a_pp_epo_handle_t;

C4A_API c4a_status_t c4a_pp_osc_create(c4a_pp_osc_handle_t** out,
                                        int32_t n_components, int scale);
C4A_API void         c4a_pp_osc_destroy(c4a_pp_osc_handle_t* h);
C4A_API c4a_status_t c4a_pp_osc_fit(c4a_pp_osc_handle_t* h,
                                     c4a_matrix_view_t X,
                                     const double* y, int64_t y_len);
C4A_API c4a_status_t c4a_pp_osc_transform(const c4a_pp_osc_handle_t* h,
                                            c4a_matrix_view_t X,
                                            c4a_matrix_view_t out);
C4A_API c4a_status_t c4a_pp_osc_inverse_transform(
    const c4a_pp_osc_handle_t* h,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out);
C4A_API c4a_status_t c4a_pp_osc_is_fitted(const c4a_pp_osc_handle_t* h,
                                           int* out_fitted);

C4A_API c4a_status_t c4a_pp_epo_create(c4a_pp_epo_handle_t** out, int scale);
C4A_API void         c4a_pp_epo_destroy(c4a_pp_epo_handle_t* h);
C4A_API c4a_status_t c4a_pp_epo_fit(c4a_pp_epo_handle_t* h,
                                     c4a_matrix_view_t X,
                                     const double* d, int64_t d_len);
C4A_API c4a_status_t c4a_pp_epo_transform(const c4a_pp_epo_handle_t* h,
                                            c4a_matrix_view_t X,
                                            c4a_matrix_view_t out);
C4A_API c4a_status_t c4a_pp_epo_inverse_transform(
    const c4a_pp_epo_handle_t* h,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out);
C4A_API c4a_status_t c4a_pp_epo_is_fitted(const c4a_pp_epo_handle_t* h,
                                           int* out_fitted);

}  // extern "C"

struct c4a_pp_osc_handle_t {
    c4a_pp_osc_state_t* state;
};
struct c4a_pp_epo_handle_t {
    c4a_pp_epo_state_t* state;
};

namespace {

// Validate that a matrix view is non-NULL, F64, row-major contiguous.
// Mirrors the helper used in c_api_preprocessing.cpp (duplicated here to
// keep the orthogonalization TU self-contained — both helpers reduce to a
// few lines of view-poking and there is no shared internal header for them).
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

// ---------------------------------------------------------------------------
// OSC (Phase 8 stateful)
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_osc_create(c4a_pp_osc_handle_t** out,
                                        int32_t n_components, int scale) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    if (n_components < 1) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        c4a_pp_osc_state_t* s = c4a_pp_osc_state_new(n_components, scale);
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_osc_handle_t* h = new (std::nothrow) c4a_pp_osc_handle_t{s};
        if (h == nullptr) {
            c4a_pp_osc_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_osc_destroy(c4a_pp_osc_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_osc_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_osc_fit(c4a_pp_osc_handle_t* h,
                                     c4a_matrix_view_t X,
                                     const double* y, int64_t y_len) {
    if (h == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    if (y == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        const double* xp = nullptr;
        std::int64_t  xr = 0, xc = 0;
        const c4a_status_t s = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        if (y_len != xr) {
            return C4A_ERR_SHAPE_MISMATCH;
        }
        return c4a_pp_osc_state_fit(h->state, xp, y, xr, xc);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_pp_osc_transform(const c4a_pp_osc_handle_t* h,
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
        return c4a_pp_osc_state_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_pp_osc_inverse_transform(
    const c4a_pp_osc_handle_t* h,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out) {
    (void)X;
    (void)out;
    if (h == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    /* OSC's orthogonal-component subtraction is many-to-one: the post-
     * transform value lives in the subspace orthogonal to the stored
     * orthogonal components, so the inverse is not unique. Return the
     * documented "this operator has no meaningful inverse" status. */
    return C4A_ERR_UNSUPPORTED;
}

C4A_API c4a_status_t c4a_pp_osc_is_fitted(const c4a_pp_osc_handle_t* h,
                                           int* out_fitted) {
    if (h == nullptr || out_fitted == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        *out_fitted = c4a_pp_osc_state_is_fitted(h->state);
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// EPO (Phase 8 stateful)
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_epo_create(c4a_pp_epo_handle_t** out, int scale) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    try {
        c4a_pp_epo_state_t* s = c4a_pp_epo_state_new(scale);
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_epo_handle_t* h = new (std::nothrow) c4a_pp_epo_handle_t{s};
        if (h == nullptr) {
            c4a_pp_epo_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_epo_destroy(c4a_pp_epo_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_epo_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_epo_fit(c4a_pp_epo_handle_t* h,
                                     c4a_matrix_view_t X,
                                     const double* d, int64_t d_len) {
    if (h == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    if (d == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        const double* xp = nullptr;
        std::int64_t  xr = 0, xc = 0;
        const c4a_status_t s = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        if (d_len != xr) {
            return C4A_ERR_SHAPE_MISMATCH;
        }
        return c4a_pp_epo_state_fit(h->state, xp, d, xr, xc);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_pp_epo_transform(const c4a_pp_epo_handle_t* h,
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
        return c4a_pp_epo_state_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_pp_epo_inverse_transform(
    const c4a_pp_epo_handle_t* h,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out) {
    (void)X;
    (void)out;
    if (h == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    /* EPO's projection is by design lossy (removes the d-correlated
     * subspace), so there is no meaningful inverse. */
    return C4A_ERR_UNSUPPORTED;
}

C4A_API c4a_status_t c4a_pp_epo_is_fitted(const c4a_pp_epo_handle_t* h,
                                           int* out_fitted) {
    if (h == nullptr || out_fitted == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        *out_fitted = c4a_pp_epo_state_is_fitted(h->state);
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}
