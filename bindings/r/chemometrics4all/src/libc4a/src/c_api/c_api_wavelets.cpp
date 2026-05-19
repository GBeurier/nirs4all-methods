// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for the Phase 6 wavelet preprocessing operators.
//
// Six operators are exposed:
//
//   - Wavelet           (stateless, single-level DWT)        c4a_pp_wavelet_*
//   - Haar              (stateless, single-level haar+per)   c4a_pp_haar_*
//   - WaveletDenoise    (stateless, multi-level denoise)     c4a_pp_wavelet_denoise_*
//   - WaveletFeatures   (stateless, statistical features)    c4a_pp_wavelet_features_*
//   - WaveletPCA        (stateful, DWT-flatten + PCA)        c4a_pp_wavelet_pca_*
//   - WaveletSVD        (stateful, DWT-flatten + SVD)        c4a_pp_wavelet_svd_*
//
// The stateless operators expose a `create / transform / destroy`
// triplet plus an `output_cols` helper that returns the row width as a
// function of the input width.  The stateful operators expose the same
// `create / fit / transform / destroy` quartet as FlexiblePCA /
// FlexibleSVD plus `is_fitted` and `output_cols` getters.
//
// All public declarations are colocated here until c4a.h is updated at
// integration time; the library exports them under CHEMOMETRICS4ALL_1
// via the existing version-script wildcard (`c4a_*`).

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <new>

#include "chemometrics4all/c4a.h"

#include "core/matrix_view.hpp"
#include "core/common/wavelet_kernels.h"
#include "core/preprocessing/wavelets/haar.h"
#include "core/preprocessing/wavelets/wavelet.h"
#include "core/preprocessing/wavelets/wavelet_denoise.h"
#include "core/preprocessing/wavelets/wavelet_features.h"
#include "core/preprocessing/wavelets/wavelet_pca.h"
#include "core/preprocessing/wavelets/wavelet_svd.h"

// ---------------------------------------------------------------------------
// Public opaque types wrapping the internal C engine states.
// Declarations and enums for the Phase 6 wavelet operators live in
// c4a.h §23 (enums + opaque typedefs + function prototypes).
// ---------------------------------------------------------------------------

struct c4a_pp_wavelet_handle_t {
    c4a_pp_wavelet_state_t* state;
};
struct c4a_pp_haar_handle_t {
    c4a_pp_haar_state_t* state;
};
struct c4a_pp_wavelet_denoise_handle_t {
    c4a_pp_wavelet_denoise_state_t* state;
};
struct c4a_pp_wavelet_features_handle_t {
    c4a_pp_wavelet_features_state_t* state;
};
struct c4a_pp_wavelet_pca_handle_t {
    c4a_pp_wavelet_pca_state_t* state;
};
struct c4a_pp_wavelet_svd_handle_t {
    c4a_pp_wavelet_svd_state_t* state;
};

// ---------------------------------------------------------------------------
// Helper: row-major F64 view validation (duplicates the same helper in
// c_api_feature_selection.cpp; kept local to keep TUs independent).
// ---------------------------------------------------------------------------

namespace {

c4a_status_t require_rowmajor_f64(const c4a_matrix_view_t& v,
                                  const double*& out_ptr,
                                  std::int64_t& out_rows,
                                  std::int64_t& out_cols) noexcept {
    const c4a_status_t s = ::chemometrics4all::core::validate_nonnull_view(v);
    if (s != C4A_OK) return s;
    if (v.dtype != C4A_DTYPE_F64)      return C4A_ERR_DTYPE_MISMATCH;
    if (v.col_stride != 1)             return C4A_ERR_STRIDE_INVALID;
    if (v.rows > 0 && v.cols > 0 && v.row_stride != v.cols)
        return C4A_ERR_STRIDE_INVALID;
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
    if (s != C4A_OK) return s;
    if (v.dtype != C4A_DTYPE_F64)      return C4A_ERR_DTYPE_MISMATCH;
    if (v.col_stride != 1)             return C4A_ERR_STRIDE_INVALID;
    if (v.rows > 0 && v.cols > 0 && v.row_stride != v.cols)
        return C4A_ERR_STRIDE_INVALID;
    out_ptr  = static_cast<double*>(v.data);
    out_rows = v.rows;
    out_cols = v.cols;
    return C4A_OK;
}

bool family_is_valid(c4a_pp_wavelet_family_t f) {
    return f == C4A_PP_WAVELET_HAAR  || f == C4A_PP_WAVELET_DB4 ||
           f == C4A_PP_WAVELET_SYM4  || f == C4A_PP_WAVELET_COIF1;
}

bool mode_is_valid(c4a_pp_wavelet_boundary_t m) {
    return m == C4A_PP_WAVELET_BOUNDARY_PERIODIZATION ||
           m == C4A_PP_WAVELET_BOUNDARY_SYMMETRIC     ||
           m == C4A_PP_WAVELET_BOUNDARY_ZERO;
}

bool entropy_is_valid(c4a_pp_wavelet_features_entropy_t e) {
    return e == C4A_PP_WAVELET_FEATURES_ENTROPY_ENERGY ||
           e == C4A_PP_WAVELET_FEATURES_ENTROPY_HISTOGRAM;
}

}  // namespace

extern "C" {

// ---------------------------------------------------------------------------
// Wavelet (stateless single-level DWT)
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_wavelet_create(c4a_pp_wavelet_handle_t** out,
                                            c4a_pp_wavelet_family_t   family,
                                            c4a_pp_wavelet_boundary_t mode) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (!family_is_valid(family) || !mode_is_valid(mode)) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        c4a_pp_wavelet_state_t* s = c4a_pp_wavelet_state_new(
            static_cast<c4a_wavelet_family_t>(family),
            static_cast<c4a_wavelet_mode_t>(mode));
        if (s == nullptr) return C4A_ERR_OUT_OF_MEMORY;
        auto* h = new (std::nothrow) c4a_pp_wavelet_handle_t{s};
        if (h == nullptr) {
            c4a_pp_wavelet_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_wavelet_destroy(c4a_pp_wavelet_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_wavelet_state_free(h->state);
        delete h;
    } catch (...) { /* swallow */ }
}

C4A_API c4a_status_t c4a_pp_wavelet_output_cols(
    const c4a_pp_wavelet_handle_t* h,
    int64_t input_cols, int64_t* out_cols) {
    if (h == nullptr || out_cols == nullptr) return C4A_ERR_NULL_POINTER;
    if (input_cols < 1) return C4A_ERR_INVALID_ARGUMENT;
    const int64_t m = c4a_wavelet_dwt_output_length(
        input_cols,
        c4a_pp_wavelet_state_family(h->state),
        c4a_pp_wavelet_state_mode(h->state));
    *out_cols = 2 * m;
    return C4A_OK;
}

C4A_API c4a_status_t c4a_pp_wavelet_transform(
    const c4a_pp_wavelet_handle_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t out) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;
        double*       op = nullptr;
        std::int64_t  xr = 0, xc = 0, orr = 0, oc = 0;
        c4a_status_t  s  = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        s = require_rowmajor_f64_mut(out, op, orr, oc);
        if (s != C4A_OK) return s;
        if (xr != orr) return C4A_ERR_SHAPE_MISMATCH;
        return c4a_pp_wavelet_state_apply(h->state, xp, xr, xc, oc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// Haar (convenience wrapper)
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_haar_create(c4a_pp_haar_handle_t** out) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    try {
        c4a_pp_haar_state_t* s = c4a_pp_haar_state_new();
        if (s == nullptr) return C4A_ERR_OUT_OF_MEMORY;
        auto* h = new (std::nothrow) c4a_pp_haar_handle_t{s};
        if (h == nullptr) {
            c4a_pp_haar_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_haar_destroy(c4a_pp_haar_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_haar_state_free(h->state);
        delete h;
    } catch (...) { /* swallow */ }
}

C4A_API c4a_status_t c4a_pp_haar_output_cols(int64_t input_cols,
                                              int64_t* out_cols) {
    if (out_cols == nullptr) return C4A_ERR_NULL_POINTER;
    if (input_cols < 1) return C4A_ERR_INVALID_ARGUMENT;
    const int64_t m = c4a_wavelet_dwt_output_length(
        input_cols, C4A_WAVELET_HAAR, C4A_WAVELET_MODE_PERIODIZATION);
    *out_cols = 2 * m;
    return C4A_OK;
}

C4A_API c4a_status_t c4a_pp_haar_transform(const c4a_pp_haar_handle_t* h,
                                            c4a_matrix_view_t X,
                                            c4a_matrix_view_t out) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;
        double*       op = nullptr;
        std::int64_t  xr = 0, xc = 0, orr = 0, oc = 0;
        c4a_status_t  s  = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        s = require_rowmajor_f64_mut(out, op, orr, oc);
        if (s != C4A_OK) return s;
        if (xr != orr) return C4A_ERR_SHAPE_MISMATCH;
        return c4a_pp_haar_state_apply(h->state, xp, xr, xc, oc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// WaveletDenoise (multi-level + thresholding + reconstruction)
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_wavelet_denoise_create(
    c4a_pp_wavelet_denoise_handle_t** out,
    c4a_pp_wavelet_family_t   family,
    c4a_pp_wavelet_boundary_t mode,
    int32_t                   level,
    c4a_pp_wavelet_threshold_t threshold_mode,
    c4a_pp_wavelet_noise_t     noise_estimator) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (!family_is_valid(family) || !mode_is_valid(mode) || level < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (threshold_mode != C4A_PP_WAVELET_THRESHOLD_SOFT &&
        threshold_mode != C4A_PP_WAVELET_THRESHOLD_HARD) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (noise_estimator != C4A_PP_WAVELET_NOISE_MEDIAN &&
        noise_estimator != C4A_PP_WAVELET_NOISE_STD) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        c4a_pp_wavelet_denoise_state_t* s = c4a_pp_wavelet_denoise_state_new(
            static_cast<c4a_wavelet_family_t>(family),
            static_cast<c4a_wavelet_mode_t>(mode),
            level,
            static_cast<c4a_pp_wavelet_threshold_mode_t>(threshold_mode),
            static_cast<c4a_pp_wavelet_noise_estimator_t>(noise_estimator));
        if (s == nullptr) return C4A_ERR_OUT_OF_MEMORY;
        auto* h = new (std::nothrow) c4a_pp_wavelet_denoise_handle_t{s};
        if (h == nullptr) {
            c4a_pp_wavelet_denoise_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_wavelet_denoise_destroy(c4a_pp_wavelet_denoise_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_wavelet_denoise_state_free(h->state);
        delete h;
    } catch (...) { /* swallow */ }
}

C4A_API c4a_status_t c4a_pp_wavelet_denoise_transform(
    const c4a_pp_wavelet_denoise_handle_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t out) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;
        double*       op = nullptr;
        std::int64_t  xr = 0, xc = 0, orr = 0, oc = 0;
        c4a_status_t  s  = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        s = require_rowmajor_f64_mut(out, op, orr, oc);
        if (s != C4A_OK) return s;
        if (xr != orr || xc != oc) return C4A_ERR_SHAPE_MISMATCH;
        return c4a_pp_wavelet_denoise_state_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// WaveletFeatures (multi-level + per-band statistics)
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_wavelet_features_create(
    c4a_pp_wavelet_features_handle_t** out,
    c4a_pp_wavelet_family_t   family,
    c4a_pp_wavelet_boundary_t mode,
    int32_t                   max_level) {
    return c4a_pp_wavelet_features_create_ex(
        out, family, mode, max_level, C4A_PP_WAVELET_FEATURES_ENTROPY_ENERGY);
}

C4A_API c4a_status_t c4a_pp_wavelet_features_create_ex(
    c4a_pp_wavelet_features_handle_t** out,
    c4a_pp_wavelet_family_t           family,
    c4a_pp_wavelet_boundary_t         mode,
    int32_t                           max_level,
    c4a_pp_wavelet_features_entropy_t entropy_mode) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (!family_is_valid(family) || !mode_is_valid(mode) || max_level < 0 ||
        !entropy_is_valid(entropy_mode)) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        c4a_pp_wavelet_features_state_t* s = c4a_pp_wavelet_features_state_new_ex(
            static_cast<c4a_wavelet_family_t>(family),
            static_cast<c4a_wavelet_mode_t>(mode),
            max_level,
            static_cast<c4a_pp_wavelet_features_entropy_t>(entropy_mode));
        if (s == nullptr) return C4A_ERR_OUT_OF_MEMORY;
        auto* h = new (std::nothrow) c4a_pp_wavelet_features_handle_t{s};
        if (h == nullptr) {
            c4a_pp_wavelet_features_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_wavelet_features_destroy(c4a_pp_wavelet_features_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_wavelet_features_state_free(h->state);
        delete h;
    } catch (...) { /* swallow */ }
}

C4A_API c4a_status_t c4a_pp_wavelet_features_output_cols(
    const c4a_pp_wavelet_features_handle_t* h,
    int64_t input_cols, int64_t* out_cols) {
    if (h == nullptr || out_cols == nullptr) return C4A_ERR_NULL_POINTER;
    if (input_cols < 1) return C4A_ERR_INVALID_ARGUMENT;
    *out_cols = c4a_pp_wavelet_features_output_size(h->state, input_cols);
    return C4A_OK;
}

C4A_API c4a_status_t c4a_pp_wavelet_features_transform(
    const c4a_pp_wavelet_features_handle_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t out) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;
        double*       op = nullptr;
        std::int64_t  xr = 0, xc = 0, orr = 0, oc = 0;
        c4a_status_t  s  = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        s = require_rowmajor_f64_mut(out, op, orr, oc);
        if (s != C4A_OK) return s;
        if (xr != orr) return C4A_ERR_SHAPE_MISMATCH;
        return c4a_pp_wavelet_features_state_apply(h->state, xp, xr, xc, oc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// WaveletPCA
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_wavelet_pca_create(
    c4a_pp_wavelet_pca_handle_t** out,
    c4a_pp_wavelet_family_t   family,
    c4a_pp_wavelet_boundary_t mode,
    int32_t                   max_level,
    double                    n_components) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (!family_is_valid(family) || !mode_is_valid(mode) ||
        max_level < 0 || !(n_components > 0.0)) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        c4a_pp_wavelet_pca_state_t* s = c4a_pp_wavelet_pca_state_new(
            static_cast<c4a_wavelet_family_t>(family),
            static_cast<c4a_wavelet_mode_t>(mode),
            max_level, n_components);
        if (s == nullptr) return C4A_ERR_OUT_OF_MEMORY;
        auto* h = new (std::nothrow) c4a_pp_wavelet_pca_handle_t{s};
        if (h == nullptr) {
            c4a_pp_wavelet_pca_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_wavelet_pca_destroy(c4a_pp_wavelet_pca_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_wavelet_pca_state_free(h->state);
        delete h;
    } catch (...) { /* swallow */ }
}

C4A_API c4a_status_t c4a_pp_wavelet_pca_fit(c4a_pp_wavelet_pca_handle_t* h,
                                             c4a_matrix_view_t X) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;
        std::int64_t  xr = 0, xc = 0;
        const c4a_status_t s = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        return c4a_pp_wavelet_pca_state_fit(h->state, xp, xr, xc);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_pp_wavelet_pca_transform(
    const c4a_pp_wavelet_pca_handle_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t out) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;
        double*       op = nullptr;
        std::int64_t  xr = 0, xc = 0, orr = 0, oc = 0;
        c4a_status_t  s  = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        s = require_rowmajor_f64_mut(out, op, orr, oc);
        if (s != C4A_OK) return s;
        if (xr != orr) return C4A_ERR_SHAPE_MISMATCH;
        return c4a_pp_wavelet_pca_state_apply(h->state, xp, xr, xc, oc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_pp_wavelet_pca_is_fitted(
    const c4a_pp_wavelet_pca_handle_t* h, int* out_fitted) {
    if (h == nullptr || out_fitted == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        *out_fitted = c4a_pp_wavelet_pca_state_is_fitted(h->state);
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_pp_wavelet_pca_output_cols(
    const c4a_pp_wavelet_pca_handle_t* h, int64_t* out_cols) {
    if (h == nullptr || out_cols == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        if (!c4a_pp_wavelet_pca_state_is_fitted(h->state)) {
            return C4A_ERR_NOT_FITTED;
        }
        *out_cols = c4a_pp_wavelet_pca_state_n_components(h->state);
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// WaveletSVD
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_pp_wavelet_svd_create(
    c4a_pp_wavelet_svd_handle_t** out,
    c4a_pp_wavelet_family_t   family,
    c4a_pp_wavelet_boundary_t mode,
    int32_t                   max_level,
    double                    n_components) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (!family_is_valid(family) || !mode_is_valid(mode) ||
        max_level < 0 || !(n_components > 0.0)) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        c4a_pp_wavelet_svd_state_t* s = c4a_pp_wavelet_svd_state_new(
            static_cast<c4a_wavelet_family_t>(family),
            static_cast<c4a_wavelet_mode_t>(mode),
            max_level, n_components);
        if (s == nullptr) return C4A_ERR_OUT_OF_MEMORY;
        auto* h = new (std::nothrow) c4a_pp_wavelet_svd_handle_t{s};
        if (h == nullptr) {
            c4a_pp_wavelet_svd_state_free(s);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_pp_wavelet_svd_destroy(c4a_pp_wavelet_svd_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_pp_wavelet_svd_state_free(h->state);
        delete h;
    } catch (...) { /* swallow */ }
}

C4A_API c4a_status_t c4a_pp_wavelet_svd_fit(c4a_pp_wavelet_svd_handle_t* h,
                                             c4a_matrix_view_t X) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;
        std::int64_t  xr = 0, xc = 0;
        const c4a_status_t s = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        return c4a_pp_wavelet_svd_state_fit(h->state, xp, xr, xc);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_pp_wavelet_svd_transform(
    const c4a_pp_wavelet_svd_handle_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t out) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;
        double*       op = nullptr;
        std::int64_t  xr = 0, xc = 0, orr = 0, oc = 0;
        c4a_status_t  s  = require_rowmajor_f64(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        s = require_rowmajor_f64_mut(out, op, orr, oc);
        if (s != C4A_OK) return s;
        if (xr != orr) return C4A_ERR_SHAPE_MISMATCH;
        return c4a_pp_wavelet_svd_state_apply(h->state, xp, xr, xc, oc, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_pp_wavelet_svd_is_fitted(
    const c4a_pp_wavelet_svd_handle_t* h, int* out_fitted) {
    if (h == nullptr || out_fitted == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        *out_fitted = c4a_pp_wavelet_svd_state_is_fitted(h->state);
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_pp_wavelet_svd_output_cols(
    const c4a_pp_wavelet_svd_handle_t* h, int64_t* out_cols) {
    if (h == nullptr || out_cols == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        if (!c4a_pp_wavelet_svd_state_is_fitted(h->state)) {
            return C4A_ERR_NOT_FITTED;
        }
        *out_cols = c4a_pp_wavelet_svd_state_n_components(h->state);
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

}  // extern "C"
