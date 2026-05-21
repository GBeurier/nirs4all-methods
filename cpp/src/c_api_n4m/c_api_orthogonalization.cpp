// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for the Phase 8 orthogonalization preprocessing
// operators (OSC, EPO). Both operators expose the
// `create / fit / transform / destroy / is_fitted` stateful ABI contract
// from §5 of n4m.h. OSC additionally exposes `inverse_transform` returning
// N4M_ERR_UNSUPPORTED (the orthogonal-component deflation is many-to-one
// and cannot be inverted from the post-transform state alone). EPO does
// not expose inverse_transform.
//
// The signatures for §16 of n4m.h are reported in the phase-8 end-of-work
// report; the public header is updated centrally. Until that integration
// lands, the opaque handle typedefs are declared locally below so the
// translation unit compiles. The exported symbol names (n4m_pp_osc_*,
// n4m_pp_epo_*) match the version script's n4m_* glob.

#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>

#include "n4m/n4m.h"

#include "core/common/matrix_view.hpp"
#include "core/preprocessing/orthogonalization/epo.h"
#include "core/preprocessing/orthogonalization/osc.h"

// ---------------------------------------------------------------------------
// Opaque public handle typedefs (will live in n4m.h §16 after central
// integration; declared locally here so this TU compiles in isolation).
// ---------------------------------------------------------------------------

extern "C" {

typedef struct n4m_pp_osc_handle_t n4m_pp_osc_handle_t;
typedef struct n4m_pp_epo_handle_t n4m_pp_epo_handle_t;

N4M_API n4m_status_t n4m_pp_osc_create(n4m_pp_osc_handle_t** out,
                                        int32_t n_components, int scale);
N4M_API void         n4m_pp_osc_destroy(n4m_pp_osc_handle_t* h);
N4M_API n4m_status_t n4m_pp_osc_fit(n4m_pp_osc_handle_t* h,
                                     n4m_matrix_view_t X,
                                     const double* y, int64_t y_len);
N4M_API n4m_status_t n4m_pp_osc_transform(const n4m_pp_osc_handle_t* h,
                                            n4m_matrix_view_t X,
                                            n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_osc_inverse_transform(
    const n4m_pp_osc_handle_t* h,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_osc_is_fitted(const n4m_pp_osc_handle_t* h,
                                           int* out_fitted);

N4M_API n4m_status_t n4m_pp_epo_create(n4m_pp_epo_handle_t** out, int scale);
N4M_API void         n4m_pp_epo_destroy(n4m_pp_epo_handle_t* h);
N4M_API n4m_status_t n4m_pp_epo_fit(n4m_pp_epo_handle_t* h,
                                     n4m_matrix_view_t X,
                                     const double* d, int64_t d_len);
N4M_API n4m_status_t n4m_pp_epo_transform(const n4m_pp_epo_handle_t* h,
                                            n4m_matrix_view_t X,
                                            n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_epo_transform_with_d(
    const n4m_pp_epo_handle_t* h,
    n4m_matrix_view_t X,
    const double* d, int64_t d_len,
    n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_epo_inverse_transform(
    const n4m_pp_epo_handle_t* h,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out);
N4M_API n4m_status_t n4m_pp_epo_is_fitted(const n4m_pp_epo_handle_t* h,
                                           int* out_fitted);

}  // extern "C"

struct n4m_pp_osc_handle_t {
    n4m_pp_osc_state_t* state;
};
struct n4m_pp_epo_handle_t {
    n4m_pp_epo_state_t* state;
};

namespace {

// Validate that a matrix view is non-NULL, F64, row-major contiguous.
// Mirrors the helper used in c_api_preprocessing.cpp (duplicated here to
// keep the orthogonalization TU self-contained — both helpers reduce to a
// few lines of view-poking and there is no shared internal header for them).
n4m_status_t require_rowmajor_f64(const n4m_matrix_view_t& v,
                                  const double*& out_ptr,
                                  std::int64_t& out_rows,
                                  std::int64_t& out_cols) noexcept {
    const n4m_status_t s = ::n4m::core::validate_nonnull_view(v);
    if (s != N4M_OK) {
        return s;
    }
    if (v.dtype != N4M_DTYPE_F64) {
        return N4M_ERR_DTYPE_MISMATCH;
    }
    if (v.col_stride != 1) {
        return N4M_ERR_STRIDE_INVALID;
    }
    if (v.rows > 0 && v.cols > 0 && v.row_stride != v.cols) {
        return N4M_ERR_STRIDE_INVALID;
    }
    out_ptr  = static_cast<const double*>(v.data);
    out_rows = v.rows;
    out_cols = v.cols;
    return N4M_OK;
}

n4m_status_t require_rowmajor_f64_mut(n4m_matrix_view_t& v,
                                       double*& out_ptr,
                                       std::int64_t& out_rows,
                                       std::int64_t& out_cols) noexcept {
    const n4m_status_t s = ::n4m::core::validate_nonnull_view(v);
    if (s != N4M_OK) {
        return s;
    }
    if (v.dtype != N4M_DTYPE_F64) {
        return N4M_ERR_DTYPE_MISMATCH;
    }
    if (v.col_stride != 1) {
        return N4M_ERR_STRIDE_INVALID;
    }
    if (v.rows > 0 && v.cols > 0 && v.row_stride != v.cols) {
        return N4M_ERR_STRIDE_INVALID;
    }
    out_ptr  = static_cast<double*>(v.data);
    out_rows = v.rows;
    out_cols = v.cols;
    return N4M_OK;
}

}  // namespace

// ---------------------------------------------------------------------------
// OSC (Phase 8 stateful)
// ---------------------------------------------------------------------------

N4M_API n4m_status_t n4m_pp_osc_create(n4m_pp_osc_handle_t** out,
                                        int32_t n_components, int scale) {
    if (out == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    *out = nullptr;
    if (n_components < 1) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    try {
        n4m_pp_osc_state_t* s = n4m_pp_osc_state_new(n_components, scale);
        if (s == nullptr) {
            return N4M_ERR_OUT_OF_MEMORY;
        }
        n4m_pp_osc_handle_t* h = new (std::nothrow) n4m_pp_osc_handle_t{s};
        if (h == nullptr) {
            n4m_pp_osc_state_free(s);
            return N4M_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API void n4m_pp_osc_destroy(n4m_pp_osc_handle_t* h) {
    if (h == nullptr) return;
    try {
        n4m_pp_osc_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

N4M_API n4m_status_t n4m_pp_osc_fit(n4m_pp_osc_handle_t* h,
                                     n4m_matrix_view_t X,
                                     const double* y, int64_t y_len) {
    if (h == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    if (y == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    try {
        const double* xp = nullptr;
        std::int64_t  xr = 0, xc = 0;
        const n4m_status_t s = require_rowmajor_f64(X, xp, xr, xc);
        if (s != N4M_OK) return s;
        if (y_len != xr) {
            return N4M_ERR_SHAPE_MISMATCH;
        }
        return n4m_pp_osc_state_fit(h->state, xp, y, xr, xc);
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_pp_osc_transform(const n4m_pp_osc_handle_t* h,
                                            n4m_matrix_view_t X,
                                            n4m_matrix_view_t out) {
    if (h == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    try {
        const double* xp = nullptr;
        double*       op = nullptr;
        std::int64_t  xr = 0, xc = 0, orr = 0, oc = 0;
        n4m_status_t  s  = require_rowmajor_f64(X, xp, xr, xc);
        if (s != N4M_OK) return s;
        s = require_rowmajor_f64_mut(out, op, orr, oc);
        if (s != N4M_OK) return s;
        if (xr != orr || xc != oc) {
            return N4M_ERR_SHAPE_MISMATCH;
        }
        return n4m_pp_osc_state_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_pp_osc_inverse_transform(
    const n4m_pp_osc_handle_t* h,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out) {
    (void)X;
    (void)out;
    if (h == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    /* OSC's orthogonal-component subtraction is many-to-one: the post-
     * transform value lives in the subspace orthogonal to the stored
     * orthogonal components, so the inverse is not unique. Return the
     * documented "this operator has no meaningful inverse" status. */
    return N4M_ERR_UNSUPPORTED;
}

N4M_API n4m_status_t n4m_pp_osc_is_fitted(const n4m_pp_osc_handle_t* h,
                                           int* out_fitted) {
    if (h == nullptr || out_fitted == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    try {
        *out_fitted = n4m_pp_osc_state_is_fitted(h->state);
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// EPO (Phase 8 stateful)
// ---------------------------------------------------------------------------

N4M_API n4m_status_t n4m_pp_epo_create(n4m_pp_epo_handle_t** out, int scale) {
    if (out == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    *out = nullptr;
    try {
        n4m_pp_epo_state_t* s = n4m_pp_epo_state_new(scale);
        if (s == nullptr) {
            return N4M_ERR_OUT_OF_MEMORY;
        }
        n4m_pp_epo_handle_t* h = new (std::nothrow) n4m_pp_epo_handle_t{s};
        if (h == nullptr) {
            n4m_pp_epo_state_free(s);
            return N4M_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API void n4m_pp_epo_destroy(n4m_pp_epo_handle_t* h) {
    if (h == nullptr) return;
    try {
        n4m_pp_epo_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

N4M_API n4m_status_t n4m_pp_epo_fit(n4m_pp_epo_handle_t* h,
                                     n4m_matrix_view_t X,
                                     const double* d, int64_t d_len) {
    if (h == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    if (d == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    try {
        const double* xp = nullptr;
        std::int64_t  xr = 0, xc = 0;
        const n4m_status_t s = require_rowmajor_f64(X, xp, xr, xc);
        if (s != N4M_OK) return s;
        if (d_len != xr) {
            return N4M_ERR_SHAPE_MISMATCH;
        }
        return n4m_pp_epo_state_fit(h->state, xp, d, xr, xc);
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_pp_epo_transform(const n4m_pp_epo_handle_t* h,
                                            n4m_matrix_view_t X,
                                            n4m_matrix_view_t out) {
    if (h == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    try {
        const double* xp = nullptr;
        double*       op = nullptr;
        std::int64_t  xr = 0, xc = 0, orr = 0, oc = 0;
        n4m_status_t  s  = require_rowmajor_f64(X, xp, xr, xc);
        if (s != N4M_OK) return s;
        s = require_rowmajor_f64_mut(out, op, orr, oc);
        if (s != N4M_OK) return s;
        if (xr != orr || xc != oc) {
            return N4M_ERR_SHAPE_MISMATCH;
        }
        return n4m_pp_epo_state_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_pp_epo_transform_with_d(
    const n4m_pp_epo_handle_t* h,
    n4m_matrix_view_t X,
    const double* d, int64_t d_len,
    n4m_matrix_view_t out) {
    if (h == nullptr || d == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    try {
        const double* xp = nullptr;
        double*       op = nullptr;
        std::int64_t  xr = 0, xc = 0, orr = 0, oc = 0;
        n4m_status_t  s  = require_rowmajor_f64(X, xp, xr, xc);
        if (s != N4M_OK) return s;
        s = require_rowmajor_f64_mut(out, op, orr, oc);
        if (s != N4M_OK) return s;
        if (xr != orr || xc != oc || d_len != xr) {
            return N4M_ERR_SHAPE_MISMATCH;
        }
        return n4m_pp_epo_state_apply_with_d(h->state, xp, d, xr, xc, op);
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_pp_epo_inverse_transform(
    const n4m_pp_epo_handle_t* h,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out) {
    (void)X;
    (void)out;
    if (h == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    /* EPO's projection is by design lossy (removes the d-correlated
     * subspace), so there is no meaningful inverse. */
    return N4M_ERR_UNSUPPORTED;
}

N4M_API n4m_status_t n4m_pp_epo_is_fitted(const n4m_pp_epo_handle_t* h,
                                           int* out_fitted) {
    if (h == nullptr || out_fitted == nullptr) {
        return N4M_ERR_NULL_POINTER;
    }
    try {
        *out_fitted = n4m_pp_epo_state_is_fitted(h->state);
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}
