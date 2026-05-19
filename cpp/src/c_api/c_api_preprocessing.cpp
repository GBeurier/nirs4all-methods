// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for the Phase 2 stateless preprocessing operators
// (SNV, LocalSNV, RobustSNV, AreaNormalization, Normalize, SimpleScale,
// LogTransform) and the Phase 3 stateful operators (MSC, EMSC, Baseline
// column-centering, Derivate).
//
// Stateless operators expose a `create / transform / destroy` triplet;
// stateful operators expose `create / fit / transform / destroy` with
// optional `inverse_transform` and `is_fitted`. The bodies delegate to the
// internal C engines under cpp/src/core/preprocessing/{scatter,scaling,
// derivatives}/.
//
// Universal rules of the wrappers (mirrored from c_api_rng.cpp):
//   - Every entry point is wrapped in try/catch so no C++ exception ever
//     crosses the boundary.
//   - The opaque public type is a thin struct {engine_state*} that owns the
//     internal allocation; this decouples the public name from the internal
//     layout.
//   - `_create` takes a pointer-to-pointer out-arg; it returns
//     C4A_ERR_NULL_POINTER if `out` is NULL and C4A_ERR_INVALID_ARGUMENT for
//     invalid constructor parameters that the engine rejects. It writes
//     NULL to *out on every failure.
//   - `_destroy` is NULL-safe and never throws.
//   - `_transform` takes a non-NULL state, a c4a_matrix_view_t for input,
//     and a c4a_matrix_view_t for output. It validates both views, requires
//     row-major contiguous F64, and writes into the output buffer.
//   - Stateful `_transform` and `_inverse_transform` return
//     C4A_ERR_NOT_FITTED when the underlying state has not yet been fit.

#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>

#include "chemometrics4all/c4a.h"

#include "core/matrix_view.hpp"
#include "core/preprocessing/baselines/airpls.h"
#include "core/preprocessing/baselines/arpls.h"
#include "core/preprocessing/baselines/asls.h"
#include "core/preprocessing/baselines/beads.h"
#include "core/preprocessing/baselines/detrend.h"
#include "core/preprocessing/baselines/iasls.h"
#include "core/preprocessing/baselines/imodpoly.h"
#include "core/preprocessing/baselines/modpoly.h"
#include "core/preprocessing/baselines/rolling_ball.h"
#include "core/preprocessing/baselines/snip.h"
#include "core/preprocessing/derivatives/derivate.h"
#include "core/preprocessing/derivatives/first_derivative.h"
#include "core/preprocessing/derivatives/norris_williams.h"
#include "core/preprocessing/derivatives/savitzky_golay.h"
#include "core/preprocessing/derivatives/second_derivative.h"
#include "core/preprocessing/scaling/baseline.h"
#include "core/preprocessing/scaling/log_transform.h"
#include "core/preprocessing/scaling/normalize.h"
#include "core/preprocessing/scaling/simple_scale.h"
#include "core/preprocessing/scatter/area_normalization.h"
#include "core/preprocessing/scatter/emsc.h"
#include "core/preprocessing/scatter/local_snv.h"
#include "core/preprocessing/scatter/msc.h"
#include "core/preprocessing/scatter/robust_snv.h"
#include "core/preprocessing/scatter/snv.h"
#include "core/preprocessing/smoothing/gaussian.h"

// ---------------------------------------------------------------------------
// Opaque public handles. Each one wraps the matching internal engine state.
// Defining them here keeps the internal layout out of the public surface.
// ---------------------------------------------------------------------------

struct c4a_pp_snv_handle_t {
    c4a_pp_snv_state_t* state;
};
struct c4a_pp_lsnv_handle_t {
    c4a_pp_lsnv_state_t* state;
};
struct c4a_pp_rnv_handle_t {
    c4a_pp_rnv_state_t* state;
};
struct c4a_pp_area_handle_t {
    bool                 static_handle;
    c4a_pp_area_method_t method;
};
struct c4a_pp_normalize_handle_t {
    bool   static_handle;
    double feature_min;
    double feature_max;
};
struct c4a_pp_simple_scale_handle_t {
    bool                         static_handle;
    c4a_pp_simple_scale_state_t* state;
};
struct c4a_pp_log_handle_t {
    c4a_pp_log_state_t* state;
};
struct c4a_pp_msc_handle_t {
    c4a_pp_msc_state_t* state;
};
struct c4a_pp_emsc_handle_t {
    c4a_pp_emsc_state_t* state;
};
struct c4a_pp_baseline_handle_t {
    c4a_pp_baseline_state_t* state;
};
struct c4a_pp_derivate_handle_t {
    c4a_pp_derivate_state_t* state;
};
struct c4a_pp_savgol_handle_t {
    c4a_pp_savgol_state_t* state;
};
struct c4a_pp_first_derivative_handle_t {
    c4a_pp_first_derivative_state_t* state;
};
struct c4a_pp_second_derivative_handle_t {
    c4a_pp_second_derivative_state_t* state;
};
struct c4a_pp_norris_williams_handle_t {
    c4a_pp_norris_williams_state_t* state;
};
struct c4a_pp_gaussian_handle_t {
    c4a_pp_gaussian_state_t* state;
};
struct c4a_pp_detrend_handle_t {
    c4a_pp_detrend_state_t* state;
};

namespace {
c4a_pp_area_handle_t k_area_sum_handle{true, C4A_PP_AREA_SUM};
c4a_pp_area_handle_t k_area_abs_sum_handle{true, C4A_PP_AREA_ABS_SUM};
c4a_pp_area_handle_t k_area_trapz_handle{true, C4A_PP_AREA_TRAPZ};
c4a_pp_normalize_handle_t k_normalize_default_handle{true, -1.0, 1.0};

c4a_pp_simple_scale_handle_t* simple_scale_default_handle() noexcept {
    static c4a_pp_simple_scale_state_t* state =
        c4a_pp_simple_scale_state_new();
    static c4a_pp_simple_scale_handle_t handle{true, state};
    return state == nullptr ? nullptr : &handle;
}
}  // namespace
struct c4a_pp_asls_handle_t {
    c4a_pp_asls_state_t* state;
};
struct c4a_pp_airpls_handle_t {
    c4a_pp_airpls_state_t* state;
};
struct c4a_pp_arpls_handle_t {
    c4a_pp_arpls_state_t* state;
};
struct c4a_pp_modpoly_handle_t {
    c4a_pp_modpoly_state_t* state;
};
struct c4a_pp_imodpoly_handle_t {
    c4a_pp_imodpoly_state_t* state;
};
struct c4a_pp_snip_handle_t {
    c4a_pp_snip_state_t* state;
};
struct c4a_pp_rolling_ball_handle_t {
    c4a_pp_rolling_ball_state_t* state;
};
struct c4a_pp_iasls_handle_t {
    c4a_pp_iasls_state_t* state;
};
struct c4a_pp_beads_handle_t {
    c4a_pp_beads_state_t* state;
};

namespace {

// Validate that a matrix view is non-NULL, F64, row-major contiguous (the
// only layout supported by the Phase 2 kernels), and return the underlying
// double pointer + dimensions.
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

c4a_status_t require_rowmajor_f64_pair(const c4a_matrix_view_t& X,
                                       c4a_matrix_view_t& out,
                                       const double*& xp,
                                       double*& op,
                                       std::int64_t& rows,
                                       std::int64_t& cols) noexcept {
    if (X.dtype != C4A_DTYPE_F64 || out.dtype != C4A_DTYPE_F64) {
        return C4A_ERR_DTYPE_MISMATCH;
    }
    if (X.rows < 0 || X.cols < 0 || out.rows < 0 || out.cols < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (X.row_stride < 0 || X.col_stride < 0 ||
        out.row_stride < 0 || out.col_stride < 0) {
        return C4A_ERR_STRIDE_INVALID;
    }
    if (X.col_stride != 1 || out.col_stride != 1) {
        return C4A_ERR_STRIDE_INVALID;
    }
    if (X.rows > 0 && X.cols > 0 && X.row_stride != X.cols) {
        return C4A_ERR_STRIDE_INVALID;
    }
    if (out.rows > 0 && out.cols > 0 && out.row_stride != out.cols) {
        return C4A_ERR_STRIDE_INVALID;
    }
    if (X.rows > 0 && X.cols > 0 && X.data == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    if (out.rows > 0 && out.cols > 0 && out.data == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    if (X.rows != out.rows || X.cols != out.cols) {
        return C4A_ERR_SHAPE_MISMATCH;
    }
    xp = static_cast<const double*>(X.data);
    op = static_cast<double*>(out.data);
    rows = X.rows;
    cols = X.cols;
    return C4A_OK;
}

}  // namespace

extern "C" {

// ---------------------------------------------------------------------------
// SNV
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_snv_create(c4a_pp_snv_handle_t** out,
                                        int with_mean, int with_std, int ddof) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    if (ddof < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        c4a_pp_snv_state_t* s = c4a_pp_snv_state_new(with_mean, with_std, ddof);
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_snv_handle_t* h = new (std::nothrow) c4a_pp_snv_handle_t{s};
        if (h == nullptr) {
            c4a_pp_snv_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_snv_destroy(c4a_pp_snv_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_snv_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_snv_transform(const c4a_pp_snv_handle_t* h,
                                           c4a_matrix_view_t X,
                                           c4a_matrix_view_t out) {
    if (h == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        const double* xp = nullptr;
        double*       op = nullptr;
        std::int64_t  xr = 0, xc = 0;
        c4a_status_t  s  = require_rowmajor_f64_pair(X, out, xp, op, xr, xc);
        if (s != C4A_OK) return s;
        return c4a_pp_snv_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// LocalSNV
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_lsnv_create(c4a_pp_lsnv_handle_t** out,
                                        int32_t window, int32_t pad_mode,
                                        double constant_value) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    if (window < 3 || (window % 2) == 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (pad_mode != C4A_PP_LSNV_PAD_REFLECT &&
        pad_mode != C4A_PP_LSNV_PAD_EDGE &&
        pad_mode != C4A_PP_LSNV_PAD_CONSTANT) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        c4a_pp_lsnv_state_t* s = c4a_pp_lsnv_state_new(
            window, static_cast<c4a_pp_lsnv_pad_mode_t>(pad_mode), constant_value);
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_lsnv_handle_t* h = new (std::nothrow) c4a_pp_lsnv_handle_t{s};
        if (h == nullptr) {
            c4a_pp_lsnv_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_lsnv_destroy(c4a_pp_lsnv_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_lsnv_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_lsnv_transform(const c4a_pp_lsnv_handle_t* h,
                                            c4a_matrix_view_t X,
                                            c4a_matrix_view_t out) {
    if (h == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        const double* xp = nullptr;
        double*       op = nullptr;
        std::int64_t  xr = 0, xc = 0;
        c4a_status_t  s  = require_rowmajor_f64_pair(X, out, xp, op, xr, xc);
        if (s != C4A_OK) return s;
        return c4a_pp_lsnv_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// RobustSNV
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_rnv_create(c4a_pp_rnv_handle_t** out,
                                        int with_center, int with_scale,
                                        double k) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    try {
        c4a_pp_rnv_state_t* s = c4a_pp_rnv_state_new(with_center, with_scale, k);
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_rnv_handle_t* h = new (std::nothrow) c4a_pp_rnv_handle_t{s};
        if (h == nullptr) {
            c4a_pp_rnv_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_rnv_destroy(c4a_pp_rnv_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_rnv_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_rnv_transform(const c4a_pp_rnv_handle_t* h,
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
        return c4a_pp_rnv_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// AreaNormalization
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_area_create(c4a_pp_area_handle_t** out,
                                         int32_t method) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    if (method != C4A_PP_AREA_SUM &&
        method != C4A_PP_AREA_ABS_SUM &&
        method != C4A_PP_AREA_TRAPZ) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    switch (method) {
        case C4A_PP_AREA_SUM:
            *out = &k_area_sum_handle;
            return C4A_OK;
        case C4A_PP_AREA_ABS_SUM:
            *out = &k_area_abs_sum_handle;
            return C4A_OK;
        case C4A_PP_AREA_TRAPZ:
            *out = &k_area_trapz_handle;
            return C4A_OK;
        default:
            return C4A_ERR_INVALID_ARGUMENT;
    }
}

C4A_API void c4a_pp_area_destroy(c4a_pp_area_handle_t* h) {
    if (h == nullptr) return;
    try {
        if (h->static_handle) return;
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_area_transform(const c4a_pp_area_handle_t* h,
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
        return c4a_pp_area_apply_method(h->method, xp, xr, xc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// Normalize
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_normalize_create(c4a_pp_normalize_handle_t** out,
                                              double feature_min,
                                              double feature_max) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    if (feature_min == -1.0 && feature_max == 1.0) {
        *out = &k_normalize_default_handle;
        return C4A_OK;
    }
    try {
        c4a_pp_normalize_handle_t* h =
            new (std::nothrow) c4a_pp_normalize_handle_t{false, feature_min,
                                                         feature_max};
        if (h == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_normalize_destroy(c4a_pp_normalize_handle_t* h) {
    if (h == nullptr) return;
    try {
        if (h->static_handle) return;
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_normalize_transform(
    const c4a_pp_normalize_handle_t* h,
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
        return c4a_pp_normalize_apply_params(h->feature_min, h->feature_max,
                                             xp, xr, xc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// SimpleScale
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_simple_scale_create(
    c4a_pp_simple_scale_handle_t** out) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    c4a_pp_simple_scale_handle_t* h = simple_scale_default_handle();
    if (h == nullptr) {
        *out = nullptr;
        return C4A_ERR_OUT_OF_MEMORY;
    }
    *out = h;
    return C4A_OK;
}

C4A_API void c4a_pp_simple_scale_destroy(c4a_pp_simple_scale_handle_t* h) {
    if (h == nullptr) return;
    try {
        if (h->static_handle) return;
        c4a_pp_simple_scale_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_simple_scale_transform(
    const c4a_pp_simple_scale_handle_t* h,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out) {
    if (h == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        const double* xp = nullptr;
        double*       op = nullptr;
        std::int64_t  xr = 0, xc = 0;
        c4a_status_t  s  = require_rowmajor_f64_pair(X, out, xp, op, xr, xc);
        if (s != C4A_OK) return s;
        return c4a_pp_simple_scale_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// LogTransform
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_log_create(c4a_pp_log_handle_t** out,
                                        double base, double offset,
                                        int auto_offset, double min_value) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    /* `base == 0.0` is the sentinel for "natural log"; any positive base
     * other than 0 and 1 is acceptable. Reject base <= 0 except for the
     * sentinel, and reject base == 1 (log(1) == 0 would divide-by-zero). */
    if (base != 0.0 && (base <= 0.0 || base == 1.0)) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (min_value <= 0.0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        c4a_pp_log_state_t* s = c4a_pp_log_state_new(base, offset, auto_offset,
                                                       min_value);
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_log_handle_t* h = new (std::nothrow) c4a_pp_log_handle_t{s};
        if (h == nullptr) {
            c4a_pp_log_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_log_destroy(c4a_pp_log_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_log_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_log_fit(c4a_pp_log_handle_t* h,
                                     c4a_matrix_view_t X) {
    if (h == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        const double* xp = nullptr;
        std::int64_t  xr = 0, xc = 0;
        c4a_status_t  s  = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        return c4a_pp_log_state_fit(h->state, xp, xr, xc);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_pp_log_is_fitted(const c4a_pp_log_handle_t* h,
                                           int* out_fitted) {
    if (h == nullptr || out_fitted == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        *out_fitted = c4a_pp_log_state_is_fitted(h->state);
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_pp_log_transform(const c4a_pp_log_handle_t* h,
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
        return c4a_pp_log_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// MSC (Phase 3 stateful)
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_msc_create(c4a_pp_msc_handle_t** out) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    try {
        c4a_pp_msc_state_t* s = c4a_pp_msc_state_new();
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_msc_handle_t* h = new (std::nothrow) c4a_pp_msc_handle_t{s};
        if (h == nullptr) {
            c4a_pp_msc_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_msc_destroy(c4a_pp_msc_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_msc_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_msc_fit(c4a_pp_msc_handle_t* h,
                                     c4a_matrix_view_t X) {
    if (h == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        const double* xp = nullptr;
        std::int64_t  xr = 0, xc = 0;
        const c4a_status_t s = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        return c4a_pp_msc_state_fit(h->state, xp, xr, xc);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_pp_msc_transform(const c4a_pp_msc_handle_t* h,
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
        return c4a_pp_msc_state_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_pp_msc_inverse_transform(
    const c4a_pp_msc_handle_t* h,
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
        return c4a_pp_msc_state_inverse_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_pp_msc_is_fitted(const c4a_pp_msc_handle_t* h,
                                           int* out_fitted) {
    if (h == nullptr || out_fitted == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        *out_fitted = c4a_pp_msc_state_is_fitted(h->state);
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// EMSC (Phase 3 stateful)
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_emsc_create(c4a_pp_emsc_handle_t** out,
                                         int32_t degree) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    if (degree < 1) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        c4a_pp_emsc_state_t* s = c4a_pp_emsc_state_new(degree);
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_emsc_handle_t* h = new (std::nothrow) c4a_pp_emsc_handle_t{s};
        if (h == nullptr) {
            c4a_pp_emsc_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_emsc_destroy(c4a_pp_emsc_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_emsc_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_emsc_fit(c4a_pp_emsc_handle_t* h,
                                      c4a_matrix_view_t X) {
    if (h == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        const double* xp = nullptr;
        std::int64_t  xr = 0, xc = 0;
        const c4a_status_t s = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        return c4a_pp_emsc_state_fit(h->state, xp, xr, xc);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_pp_emsc_transform(const c4a_pp_emsc_handle_t* h,
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
        return c4a_pp_emsc_state_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_pp_emsc_is_fitted(const c4a_pp_emsc_handle_t* h,
                                            int* out_fitted) {
    if (h == nullptr || out_fitted == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        *out_fitted = c4a_pp_emsc_state_is_fitted(h->state);
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// Baseline (column-mean centering, Phase 3 stateful)
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_baseline_create(c4a_pp_baseline_handle_t** out) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    try {
        c4a_pp_baseline_state_t* s = c4a_pp_baseline_state_new();
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_baseline_handle_t* h =
            new (std::nothrow) c4a_pp_baseline_handle_t{s};
        if (h == nullptr) {
            c4a_pp_baseline_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_baseline_destroy(c4a_pp_baseline_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_baseline_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_baseline_fit(c4a_pp_baseline_handle_t* h,
                                          c4a_matrix_view_t X) {
    if (h == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        const double* xp = nullptr;
        std::int64_t  xr = 0, xc = 0;
        const c4a_status_t s = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        return c4a_pp_baseline_state_fit(h->state, xp, xr, xc);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_pp_baseline_transform(
    const c4a_pp_baseline_handle_t* h,
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
        return c4a_pp_baseline_state_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_pp_baseline_inverse_transform(
    const c4a_pp_baseline_handle_t* h,
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
        return c4a_pp_baseline_state_inverse_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_pp_baseline_is_fitted(
    const c4a_pp_baseline_handle_t* h, int* out_fitted) {
    if (h == nullptr || out_fitted == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        *out_fitted = c4a_pp_baseline_state_is_fitted(h->state);
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// Derivate (Phase 3 stateful)
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_derivate_create(c4a_pp_derivate_handle_t** out,
                                             int32_t order, double delta) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    if (order < 1 || delta == 0.0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        c4a_pp_derivate_state_t* s = c4a_pp_derivate_state_new(order, delta);
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_derivate_handle_t* h =
            new (std::nothrow) c4a_pp_derivate_handle_t{s};
        if (h == nullptr) {
            c4a_pp_derivate_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_derivate_destroy(c4a_pp_derivate_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_derivate_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_derivate_fit(c4a_pp_derivate_handle_t* h,
                                          c4a_matrix_view_t X) {
    if (h == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    try {
        const double* xp = nullptr;
        std::int64_t  xr = 0, xc = 0;
        const c4a_status_t s = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        return c4a_pp_derivate_state_fit(h->state, xp, xr, xc);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_pp_derivate_transform(
    const c4a_pp_derivate_handle_t* h,
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
        return c4a_pp_derivate_state_apply(h->state, xp, xr, xc, oc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API int64_t c4a_pp_derivate_output_cols(int32_t order,
                                              int64_t input_cols) {
    try {
        return c4a_pp_derivate_output_cols_helper(order, input_cols);
    } catch (...) {
        return 0;
    }
}

// ---------------------------------------------------------------------------
// SavitzkyGolay (Phase 4 stateless)
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_savgol_create(c4a_pp_savgol_handle_t** out,
                                           int32_t window_length,
                                           int32_t polyorder,
                                           int32_t deriv,
                                           double delta,
                                           c4a_pp_savgol_mode_t mode,
                                           double cval) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    if (window_length < 1 || (window_length % 2) == 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (polyorder < 0 || polyorder >= window_length) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (deriv < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (delta == 0.0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (mode != C4A_PP_SAVGOL_MIRROR &&
        mode != C4A_PP_SAVGOL_CONSTANT &&
        mode != C4A_PP_SAVGOL_NEAREST &&
        mode != C4A_PP_SAVGOL_WRAP &&
        mode != C4A_PP_SAVGOL_INTERP) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        c4a_pp_savgol_state_t* s =
            c4a_pp_savgol_state_new(window_length, polyorder, deriv, delta,
                                     mode, cval);
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_savgol_handle_t* h =
            new (std::nothrow) c4a_pp_savgol_handle_t{s};
        if (h == nullptr) {
            c4a_pp_savgol_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_savgol_destroy(c4a_pp_savgol_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_savgol_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_savgol_transform(const c4a_pp_savgol_handle_t* h,
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
        return c4a_pp_savgol_state_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// FirstDerivative (Phase 4 stateless)
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_first_derivative_create(
    c4a_pp_first_derivative_handle_t** out, double delta, int32_t edge_order) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    if (delta == 0.0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (edge_order != 1 && edge_order != 2) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        c4a_pp_first_derivative_state_t* s =
            c4a_pp_first_derivative_state_new(delta, edge_order);
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_first_derivative_handle_t* h =
            new (std::nothrow) c4a_pp_first_derivative_handle_t{s};
        if (h == nullptr) {
            c4a_pp_first_derivative_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_first_derivative_destroy(
    c4a_pp_first_derivative_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_first_derivative_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_first_derivative_transform(
    const c4a_pp_first_derivative_handle_t* h,
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
        return c4a_pp_first_derivative_state_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// SecondDerivative (Phase 4 stateless)
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_second_derivative_create(
    c4a_pp_second_derivative_handle_t** out, double delta, int32_t edge_order) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    if (delta == 0.0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (edge_order != 1 && edge_order != 2) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        c4a_pp_second_derivative_state_t* s =
            c4a_pp_second_derivative_state_new(delta, edge_order);
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_second_derivative_handle_t* h =
            new (std::nothrow) c4a_pp_second_derivative_handle_t{s};
        if (h == nullptr) {
            c4a_pp_second_derivative_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_second_derivative_destroy(
    c4a_pp_second_derivative_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_second_derivative_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_second_derivative_transform(
    const c4a_pp_second_derivative_handle_t* h,
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
        return c4a_pp_second_derivative_state_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// NorrisWilliams (Phase 4 stateless)
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_norris_williams_create(
    c4a_pp_norris_williams_handle_t** out,
    int32_t segment, int32_t gap, int32_t derivative_order, double delta) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    if (segment < 1 || (segment % 2) == 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (gap < 1) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (derivative_order != 1 && derivative_order != 2) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (delta == 0.0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        c4a_pp_norris_williams_state_t* s = c4a_pp_norris_williams_state_new(
            segment, gap, derivative_order, delta);
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_norris_williams_handle_t* h =
            new (std::nothrow) c4a_pp_norris_williams_handle_t{s};
        if (h == nullptr) {
            c4a_pp_norris_williams_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_norris_williams_destroy(
    c4a_pp_norris_williams_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_norris_williams_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_norris_williams_transform(
    const c4a_pp_norris_williams_handle_t* h,
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
        return c4a_pp_norris_williams_state_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// Gaussian (Phase 4 stateless)
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_gaussian_create(
    c4a_pp_gaussian_handle_t** out,
    double sigma, int32_t order,
    c4a_pp_gaussian_mode_t mode,
    double cval, double truncate) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    if (!(sigma > 0.0)) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (order < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (!(truncate >= 0.0)) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (mode != C4A_PP_GAUSSIAN_REFLECT &&
        mode != C4A_PP_GAUSSIAN_CONSTANT &&
        mode != C4A_PP_GAUSSIAN_NEAREST &&
        mode != C4A_PP_GAUSSIAN_MIRROR &&
        mode != C4A_PP_GAUSSIAN_WRAP) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        c4a_pp_gaussian_state_t* s =
            c4a_pp_gaussian_state_new(sigma, order, mode, cval, truncate);
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_gaussian_handle_t* h =
            new (std::nothrow) c4a_pp_gaussian_handle_t{s};
        if (h == nullptr) {
            c4a_pp_gaussian_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_gaussian_destroy(c4a_pp_gaussian_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_gaussian_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_gaussian_transform(
    const c4a_pp_gaussian_handle_t* h,
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
        return c4a_pp_gaussian_state_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// Detrend (Phase 5a stateless)
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_detrend_create(c4a_pp_detrend_handle_t** out,
                                            int32_t polyorder) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    if (polyorder < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        c4a_pp_detrend_state_t* s = c4a_pp_detrend_state_new(polyorder);
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_detrend_handle_t* h =
            new (std::nothrow) c4a_pp_detrend_handle_t{s};
        if (h == nullptr) {
            c4a_pp_detrend_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_detrend_destroy(c4a_pp_detrend_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_detrend_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_detrend_transform(
    const c4a_pp_detrend_handle_t* h,
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
        return c4a_pp_detrend_state_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// AsLS (Phase 5a stateless)
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_asls_create(c4a_pp_asls_handle_t** out,
                                         double lam, double p,
                                         int32_t max_iter, double tol) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    if (!(lam > 0.0) || !(p > 0.0 && p < 1.0) || max_iter < 0 ||
        !(tol >= 0.0)) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        c4a_pp_asls_state_t* s =
            c4a_pp_asls_state_new(lam, p, max_iter, tol);
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_asls_handle_t* h =
            new (std::nothrow) c4a_pp_asls_handle_t{s};
        if (h == nullptr) {
            c4a_pp_asls_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_asls_destroy(c4a_pp_asls_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_asls_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_asls_transform(const c4a_pp_asls_handle_t* h,
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
        return c4a_pp_asls_state_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// AirPLS (Phase 5a stateless)
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_airpls_create(c4a_pp_airpls_handle_t** out,
                                           double lam,
                                           int32_t max_iter, double tol) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    if (!(lam > 0.0) || max_iter < 0 || !(tol >= 0.0)) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        c4a_pp_airpls_state_t* s =
            c4a_pp_airpls_state_new(lam, max_iter, tol);
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_airpls_handle_t* h =
            new (std::nothrow) c4a_pp_airpls_handle_t{s};
        if (h == nullptr) {
            c4a_pp_airpls_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_airpls_destroy(c4a_pp_airpls_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_airpls_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_airpls_transform(
    const c4a_pp_airpls_handle_t* h,
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
        return c4a_pp_airpls_state_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// ArPLS (Phase 5a stateless)
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_arpls_create(c4a_pp_arpls_handle_t** out,
                                          double lam,
                                          int32_t max_iter, double tol) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    if (!(lam > 0.0) || max_iter < 0 || !(tol >= 0.0)) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        c4a_pp_arpls_state_t* s =
            c4a_pp_arpls_state_new(lam, max_iter, tol);
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_arpls_handle_t* h =
            new (std::nothrow) c4a_pp_arpls_handle_t{s};
        if (h == nullptr) {
            c4a_pp_arpls_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_arpls_destroy(c4a_pp_arpls_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_arpls_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_arpls_transform(const c4a_pp_arpls_handle_t* h,
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
        return c4a_pp_arpls_state_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// ModPoly (Phase 5b stateless)
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_modpoly_create(c4a_pp_modpoly_handle_t** out,
                                            int32_t polyorder,
                                            int32_t max_iter, double tol) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    if (polyorder < 0 || max_iter < 0 || !(tol >= 0.0)) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        c4a_pp_modpoly_state_t* s =
            c4a_pp_modpoly_state_new(polyorder, max_iter, tol);
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_modpoly_handle_t* h =
            new (std::nothrow) c4a_pp_modpoly_handle_t{s};
        if (h == nullptr) {
            c4a_pp_modpoly_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_modpoly_destroy(c4a_pp_modpoly_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_modpoly_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_modpoly_transform(const c4a_pp_modpoly_handle_t* h,
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
        return c4a_pp_modpoly_state_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// IModPoly (Phase 5b stateless)
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_imodpoly_create(c4a_pp_imodpoly_handle_t** out,
                                             int32_t polyorder,
                                             int32_t max_iter, double tol) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    if (polyorder < 0 || max_iter < 0 || !(tol >= 0.0)) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        c4a_pp_imodpoly_state_t* s =
            c4a_pp_imodpoly_state_new(polyorder, max_iter, tol);
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_imodpoly_handle_t* h =
            new (std::nothrow) c4a_pp_imodpoly_handle_t{s};
        if (h == nullptr) {
            c4a_pp_imodpoly_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_imodpoly_destroy(c4a_pp_imodpoly_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_imodpoly_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_imodpoly_transform(
    const c4a_pp_imodpoly_handle_t* h,
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
        return c4a_pp_imodpoly_state_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// SNIP (Phase 5b stateless)
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_snip_create(c4a_pp_snip_handle_t** out,
                                         int32_t max_half_window) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    if (max_half_window < 1) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        c4a_pp_snip_state_t* s = c4a_pp_snip_state_new(max_half_window);
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_snip_handle_t* h =
            new (std::nothrow) c4a_pp_snip_handle_t{s};
        if (h == nullptr) {
            c4a_pp_snip_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_snip_destroy(c4a_pp_snip_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_snip_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_snip_transform(const c4a_pp_snip_handle_t* h,
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
        return c4a_pp_snip_state_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// RollingBall (Phase 5b stateless)
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_rolling_ball_create(
    c4a_pp_rolling_ball_handle_t** out,
    int32_t half_window, int32_t smooth_half_window) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    if (half_window < 1 || smooth_half_window < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        c4a_pp_rolling_ball_state_t* s =
            c4a_pp_rolling_ball_state_new(half_window, smooth_half_window);
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_rolling_ball_handle_t* h =
            new (std::nothrow) c4a_pp_rolling_ball_handle_t{s};
        if (h == nullptr) {
            c4a_pp_rolling_ball_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_rolling_ball_destroy(c4a_pp_rolling_ball_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_rolling_ball_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_rolling_ball_transform(
    const c4a_pp_rolling_ball_handle_t* h,
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
        return c4a_pp_rolling_ball_state_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// IAsLS (Phase 5b stateless)
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_iasls_create(c4a_pp_iasls_handle_t** out,
                                          double lam, double p,
                                          int32_t polyorder,
                                          int32_t max_iter, double tol) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    if (!(lam > 0.0) || !(p > 0.0 && p < 1.0) || polyorder < 0 ||
        max_iter < 0 || !(tol >= 0.0)) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        c4a_pp_iasls_state_t* s =
            c4a_pp_iasls_state_new(lam, p, polyorder, max_iter, tol);
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_iasls_handle_t* h =
            new (std::nothrow) c4a_pp_iasls_handle_t{s};
        if (h == nullptr) {
            c4a_pp_iasls_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_iasls_destroy(c4a_pp_iasls_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_iasls_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_iasls_transform(const c4a_pp_iasls_handle_t* h,
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
        return c4a_pp_iasls_state_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// BEADS (Phase 5b stateless, simplified pentadiagonal variant)
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_beads_create(c4a_pp_beads_handle_t** out,
                                          double lam_0, double lam_1,
                                          double lam_2,
                                          int32_t max_iter, double tol) {
    if (out == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    *out = nullptr;
    if (!(lam_0 > 0.0) || !(lam_1 > 0.0) || !(lam_2 > 0.0) ||
        max_iter < 0 || !(tol >= 0.0)) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        c4a_pp_beads_state_t* s =
            c4a_pp_beads_state_new(lam_0, lam_1, lam_2, max_iter, tol);
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_beads_handle_t* h =
            new (std::nothrow) c4a_pp_beads_handle_t{s};
        if (h == nullptr) {
            c4a_pp_beads_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_beads_destroy(c4a_pp_beads_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_beads_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

C4A_API c4a_status_t c4a_pp_beads_transform(const c4a_pp_beads_handle_t* h,
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
        return c4a_pp_beads_state_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

}  // extern "C"
