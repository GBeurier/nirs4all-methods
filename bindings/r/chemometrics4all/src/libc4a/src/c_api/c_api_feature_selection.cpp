// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for the Phase 9 stateful feature-selection /
// dimensionality-reduction operators (FlexiblePCA, FlexibleSVD).
//
// The Phase 9 brief excludes CARS and MCUVE, which require an internal
// PLS callback that will be wired in a later phase. Only the two
// SVD-based dimensionality-reduction operators are implemented here.
//
// Both operators expose a `create / fit / transform / destroy` quartet
// with companion `is_fitted` and `output_cols` helpers — the same
// stateful contract as MSC / EMSC / Baseline / Derivate in Phase 3.
// Output column count is data-dependent (driven by either the
// constructor's integer count or the variance-ratio threshold), so
// `output_cols` is exposed as a runtime getter on the handle (analogous
// to Derivate's helper, but reading the learned value off the fitted
// state).
//
// The public symbol prefix is `c4a_pp_flex_pca_*` / `c4a_pp_flex_svd_*`.
// Until c4a.h is updated at integration time, the new declarations live
// here behind `extern "C"` so the library exports them under
// `CHEMOMETRICS4ALL_1` (via the existing version script wildcard).

#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>

#include "chemometrics4all/c4a.h"

#include "core/matrix_view.hpp"
#include "core/preprocessing/feature_selection/flexible_pca.h"
#include "core/preprocessing/feature_selection/flexible_svd.h"

// ---------------------------------------------------------------------------
// Public opaque types and entry points. These declarations are colocated with
// their implementations until c4a.h §9.x is opened at integration time.
// ---------------------------------------------------------------------------

extern "C" {

typedef struct c4a_pp_flex_pca_handle_t c4a_pp_flex_pca_handle_t;
typedef struct c4a_pp_flex_svd_handle_t c4a_pp_flex_svd_handle_t;

C4A_API c4a_status_t c4a_pp_flex_pca_create(c4a_pp_flex_pca_handle_t** out,
                                             double n_components);
C4A_API void         c4a_pp_flex_pca_destroy(c4a_pp_flex_pca_handle_t* handle);
C4A_API c4a_status_t c4a_pp_flex_pca_fit(c4a_pp_flex_pca_handle_t* handle,
                                          c4a_matrix_view_t X);
C4A_API c4a_status_t c4a_pp_flex_pca_transform(
    const c4a_pp_flex_pca_handle_t* handle,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out);
C4A_API c4a_status_t c4a_pp_flex_pca_is_fitted(
    const c4a_pp_flex_pca_handle_t* handle,
    int* out_fitted);
C4A_API c4a_status_t c4a_pp_flex_pca_output_cols(
    const c4a_pp_flex_pca_handle_t* handle,
    int64_t* out_cols);

C4A_API c4a_status_t c4a_pp_flex_svd_create(c4a_pp_flex_svd_handle_t** out,
                                             double n_components);
C4A_API void         c4a_pp_flex_svd_destroy(c4a_pp_flex_svd_handle_t* handle);
C4A_API c4a_status_t c4a_pp_flex_svd_fit(c4a_pp_flex_svd_handle_t* handle,
                                          c4a_matrix_view_t X);
C4A_API c4a_status_t c4a_pp_flex_svd_transform(
    const c4a_pp_flex_svd_handle_t* handle,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out);
C4A_API c4a_status_t c4a_pp_flex_svd_is_fitted(
    const c4a_pp_flex_svd_handle_t* handle,
    int* out_fitted);
C4A_API c4a_status_t c4a_pp_flex_svd_output_cols(
    const c4a_pp_flex_svd_handle_t* handle,
    int64_t* out_cols);

}  // extern "C"

// ---------------------------------------------------------------------------
// Opaque public handles. Each one wraps the matching internal engine state.
// ---------------------------------------------------------------------------

struct c4a_pp_flex_pca_handle_t {
    c4a_pp_flex_pca_state_t* state;
};
struct c4a_pp_flex_svd_handle_t {
    c4a_pp_flex_svd_state_t* state;
};

// ---------------------------------------------------------------------------
// Shared matrix-view validators (duplicates of the ones in
// c_api_preprocessing.cpp; kept local here to keep the two translation
// units independent).
// ---------------------------------------------------------------------------

namespace {

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
// FlexiblePCA
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_flex_pca_create(c4a_pp_flex_pca_handle_t** out,
                                             double n_components) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    /* n_components must be > 0 and not NaN. */
    if (!(n_components > 0.0)) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        c4a_pp_flex_pca_state_t* s =
            c4a_pp_flex_pca_state_new(n_components);
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_flex_pca_handle_t* h =
            new (std::nothrow) c4a_pp_flex_pca_handle_t{s};
        if (h == nullptr) {
            c4a_pp_flex_pca_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_flex_pca_destroy(c4a_pp_flex_pca_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_flex_pca_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_flex_pca_fit(c4a_pp_flex_pca_handle_t* h,
                                          c4a_matrix_view_t X) {
    if (h == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        const double* xp = nullptr;
        std::int64_t  xr = 0, xc = 0;
        const c4a_status_t s = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        return c4a_pp_flex_pca_state_fit(h->state, xp, xr, xc);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_pp_flex_pca_transform(
    const c4a_pp_flex_pca_handle_t* h,
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
        if (xr != orr) {
            return C4A_ERR_SHAPE_MISMATCH;
        }
        return c4a_pp_flex_pca_state_apply(h->state, xp, xr, xc, oc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_pp_flex_pca_is_fitted(
    const c4a_pp_flex_pca_handle_t* h, int* out_fitted) {
    if (h == nullptr || out_fitted == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        *out_fitted = c4a_pp_flex_pca_state_is_fitted(h->state);
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_pp_flex_pca_output_cols(
    const c4a_pp_flex_pca_handle_t* h, int64_t* out_cols) {
    if (h == nullptr || out_cols == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        if (!c4a_pp_flex_pca_state_is_fitted(h->state)) {
            return C4A_ERR_NOT_FITTED;
        }
        *out_cols = c4a_pp_flex_pca_state_n_components(h->state);
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// FlexibleSVD
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_flex_svd_create(c4a_pp_flex_svd_handle_t** out,
                                             double n_components) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    if (!(n_components > 0.0)) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        c4a_pp_flex_svd_state_t* s =
            c4a_pp_flex_svd_state_new(n_components);
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_flex_svd_handle_t* h =
            new (std::nothrow) c4a_pp_flex_svd_handle_t{s};
        if (h == nullptr) {
            c4a_pp_flex_svd_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_flex_svd_destroy(c4a_pp_flex_svd_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_flex_svd_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_flex_svd_fit(c4a_pp_flex_svd_handle_t* h,
                                          c4a_matrix_view_t X) {
    if (h == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        const double* xp = nullptr;
        std::int64_t  xr = 0, xc = 0;
        const c4a_status_t s = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        return c4a_pp_flex_svd_state_fit(h->state, xp, xr, xc);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_pp_flex_svd_transform(
    const c4a_pp_flex_svd_handle_t* h,
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
        if (xr != orr) {
            return C4A_ERR_SHAPE_MISMATCH;
        }
        return c4a_pp_flex_svd_state_apply(h->state, xp, xr, xc, oc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_pp_flex_svd_is_fitted(
    const c4a_pp_flex_svd_handle_t* h, int* out_fitted) {
    if (h == nullptr || out_fitted == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        *out_fitted = c4a_pp_flex_svd_state_is_fitted(h->state);
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_pp_flex_svd_output_cols(
    const c4a_pp_flex_svd_handle_t* h, int64_t* out_cols) {
    if (h == nullptr || out_cols == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        if (!c4a_pp_flex_svd_state_is_fitted(h->state)) {
            return C4A_ERR_NOT_FITTED;
        }
        *out_cols = c4a_pp_flex_svd_state_n_components(h->state);
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

}  // extern "C"
