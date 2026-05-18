// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for the Phase 2 stateless preprocessing operators:
// SNV, LocalSNV, RobustSNV, AreaNormalization, Normalize, SimpleScale,
// LogTransform. Each operator exposes a create / transform / destroy
// triplet on the public C ABI; the bodies delegate to the internal C
// engines under cpp/src/core/preprocessing/{scatter,scaling}/.
//
// Universal rules of the wrappers (mirrored from c_api_rng.cpp):
//   - Every entry point is wrapped in try/catch so no C++ exception ever
//     crosses the boundary.
//   - The opaque public type is a thin struct {engine_state*} that owns the
//     internal allocation; this decouples the public name from the internal
//     layout.
//   - `_create` takes a pointer-to-pointer out-arg; it returns
//     C4A_ERR_NULL_POINTER if `out` is NULL and C4A_ERR_INVALID_ARGUMENT for
//     invalid constructor parameters that the engine rejects (e.g.
//     non-odd LSNV window). It writes NULL to *out on every failure.
//   - `_destroy` is NULL-safe and never throws.
//   - `_transform` takes a non-NULL state, a c4a_matrix_view_t for input,
//     and a c4a_matrix_view_t for output. It validates both views, checks
//     shape equality, requires row-major contiguous F64, and writes
//     in-place into the output buffer.

#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>

#include "chemometrics4all/c4a.h"

#include "core/matrix_view.hpp"
#include "core/preprocessing/scaling/log_transform.h"
#include "core/preprocessing/scaling/normalize.h"
#include "core/preprocessing/scaling/simple_scale.h"
#include "core/preprocessing/scatter/area_normalization.h"
#include "core/preprocessing/scatter/local_snv.h"
#include "core/preprocessing/scatter/robust_snv.h"
#include "core/preprocessing/scatter/snv.h"

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
    c4a_pp_area_state_t* state;
};
struct c4a_pp_normalize_handle_t {
    c4a_pp_normalize_state_t* state;
};
struct c4a_pp_simple_scale_handle_t {
    c4a_pp_simple_scale_state_t* state;
};
struct c4a_pp_log_handle_t {
    c4a_pp_log_state_t* state;
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
        std::int64_t  xr = 0, xc = 0, orr = 0, oc = 0;
        c4a_status_t  s  = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        s = require_rowmajor_f64_mut(out, op, orr, oc);
        if (s != C4A_OK) return s;
        if (xr != orr || xc != oc) {
            return C4A_ERR_SHAPE_MISMATCH;
        }
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
        std::int64_t  xr = 0, xc = 0, orr = 0, oc = 0;
        c4a_status_t  s  = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        s = require_rowmajor_f64_mut(out, op, orr, oc);
        if (s != C4A_OK) return s;
        if (xr != orr || xc != oc) {
            return C4A_ERR_SHAPE_MISMATCH;
        }
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
    try {
        c4a_pp_area_state_t* s = c4a_pp_area_state_new(
            static_cast<c4a_pp_area_method_t>(method));
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_area_handle_t* h = new (std::nothrow) c4a_pp_area_handle_t{s};
        if (h == nullptr) {
            c4a_pp_area_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_area_destroy(c4a_pp_area_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_area_state_free(h->state);
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
        return c4a_pp_area_apply(h->state, xp, xr, xc, op);
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
    try {
        c4a_pp_normalize_state_t* s =
            c4a_pp_normalize_state_new(feature_min, feature_max);
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_normalize_handle_t* h =
            new (std::nothrow) c4a_pp_normalize_handle_t{s};
        if (h == nullptr) {
            c4a_pp_normalize_state_free(s);
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
        c4a_pp_normalize_state_free(h->state);
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
        return c4a_pp_normalize_apply(h->state, xp, xr, xc, op);
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
    *out = nullptr;
    try {
        c4a_pp_simple_scale_state_t* s = c4a_pp_simple_scale_state_new();
        if (s == nullptr) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        c4a_pp_simple_scale_handle_t* h =
            new (std::nothrow) c4a_pp_simple_scale_handle_t{s};
        if (h == nullptr) {
            c4a_pp_simple_scale_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_simple_scale_destroy(c4a_pp_simple_scale_handle_t* h) {
    if (h == nullptr) return;
    try {
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
        std::int64_t  xr = 0, xc = 0, orr = 0, oc = 0;
        c4a_status_t  s  = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        s = require_rowmajor_f64_mut(out, op, orr, oc);
        if (s != C4A_OK) return s;
        if (xr != orr || xc != oc) {
            return C4A_ERR_SHAPE_MISMATCH;
        }
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

}  // extern "C"
