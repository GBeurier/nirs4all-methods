// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for the Phase 6 wavelet preprocessing operators.
//
// Six operators are exposed:
//
//   - Wavelet           (stateless, single-level DWT)        n4m_pp_wavelet_*
//   - Haar              (stateless, single-level haar+per)   n4m_pp_haar_*
//   - WaveletDenoise    (stateless, multi-level denoise)     n4m_pp_wavelet_denoise_*
//   - WaveletFeatures   (stateless, statistical features)    n4m_pp_wavelet_features_*
//   - WaveletPCA        (stateful, DWT-flatten + PCA)        n4m_pp_wavelet_pca_*
//   - WaveletSVD        (stateful, DWT-flatten + SVD)        n4m_pp_wavelet_svd_*
//
// The stateless operators expose a `create / transform / destroy`
// triplet plus an `output_cols` helper that returns the row width as a
// function of the input width.  The stateful operators expose the same
// `create / fit / transform / destroy` quartet as FlexiblePCA /
// FlexibleSVD plus `is_fitted` and `output_cols` getters.
//
// All public declarations are colocated here until n4m.h is updated at
// integration time; the library exports them under N4M_1
// via the existing version-script wildcard (`n4m_*`).

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <new>

#include "n4m/n4m.h"

#include "core/common/matrix_view.hpp"
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
// n4m.h §23 (enums + opaque typedefs + function prototypes).
// ---------------------------------------------------------------------------

struct n4m_pp_wavelet_handle_t {
    n4m_pp_wavelet_state_t* state;
};
struct n4m_pp_haar_handle_t {
    n4m_pp_haar_state_t* state;
};
struct n4m_pp_wavelet_denoise_handle_t {
    n4m_pp_wavelet_denoise_state_t* state;
};
struct n4m_pp_wavelet_features_handle_t {
    n4m_pp_wavelet_features_state_t* state;
};
struct n4m_pp_wavelet_pca_handle_t {
    n4m_pp_wavelet_pca_state_t* state;
};
struct n4m_pp_wavelet_svd_handle_t {
    n4m_pp_wavelet_svd_state_t* state;
};

// ---------------------------------------------------------------------------
// Helper: row-major F64 view validation (duplicates the same helper in
// c_api_feature_selection.cpp; kept local to keep TUs independent).
// ---------------------------------------------------------------------------

namespace {

n4m_status_t require_rowmajor_f64(const n4m_matrix_view_t& v,
                                  const double*& out_ptr,
                                  std::int64_t& out_rows,
                                  std::int64_t& out_cols) noexcept {
    const n4m_status_t s = ::n4m::core::validate_nonnull_view(v);
    if (s != N4M_OK) return s;
    if (v.dtype != N4M_DTYPE_F64)      return N4M_ERR_DTYPE_MISMATCH;
    if (v.col_stride != 1)             return N4M_ERR_STRIDE_INVALID;
    if (v.rows > 0 && v.cols > 0 && v.row_stride != v.cols)
        return N4M_ERR_STRIDE_INVALID;
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
    if (s != N4M_OK) return s;
    if (v.dtype != N4M_DTYPE_F64)      return N4M_ERR_DTYPE_MISMATCH;
    if (v.col_stride != 1)             return N4M_ERR_STRIDE_INVALID;
    if (v.rows > 0 && v.cols > 0 && v.row_stride != v.cols)
        return N4M_ERR_STRIDE_INVALID;
    out_ptr  = static_cast<double*>(v.data);
    out_rows = v.rows;
    out_cols = v.cols;
    return N4M_OK;
}

bool family_is_valid(n4m_pp_wavelet_family_t f) {
    return f == N4M_PP_WAVELET_HAAR  || f == N4M_PP_WAVELET_DB4 ||
           f == N4M_PP_WAVELET_SYM4  || f == N4M_PP_WAVELET_COIF1;
}

bool mode_is_valid(n4m_pp_wavelet_boundary_t m) {
    return m == N4M_PP_WAVELET_BOUNDARY_PERIODIZATION ||
           m == N4M_PP_WAVELET_BOUNDARY_SYMMETRIC     ||
           m == N4M_PP_WAVELET_BOUNDARY_ZERO;
}

bool entropy_is_valid(n4m_pp_wavelet_features_entropy_t e) {
    return e == N4M_PP_WAVELET_FEATURES_ENTROPY_ENERGY ||
           e == N4M_PP_WAVELET_FEATURES_ENTROPY_HISTOGRAM;
}

}  // namespace

extern "C" {

// ---------------------------------------------------------------------------
// Wavelet (stateless single-level DWT)
// ---------------------------------------------------------------------------

N4M_API n4m_status_t n4m_pp_wavelet_create(n4m_pp_wavelet_handle_t** out,
                                            n4m_pp_wavelet_family_t   family,
                                            n4m_pp_wavelet_boundary_t mode) {
    if (out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = nullptr;
    if (!family_is_valid(family) || !mode_is_valid(mode)) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    try {
        n4m_pp_wavelet_state_t* s = n4m_pp_wavelet_state_new(
            static_cast<n4m_wavelet_family_t>(family),
            static_cast<n4m_wavelet_mode_t>(mode));
        if (s == nullptr) return N4M_ERR_OUT_OF_MEMORY;
        auto* h = new (std::nothrow) n4m_pp_wavelet_handle_t{s};
        if (h == nullptr) {
            n4m_pp_wavelet_state_free(s);
            return N4M_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API void n4m_pp_wavelet_destroy(n4m_pp_wavelet_handle_t* h) {
    if (h == nullptr) return;
    try {
        n4m_pp_wavelet_state_free(h->state);
        delete h;
    } catch (...) { /* swallow */ }
}

N4M_API n4m_status_t n4m_pp_wavelet_output_cols(
    const n4m_pp_wavelet_handle_t* h,
    int64_t input_cols, int64_t* out_cols) {
    if (h == nullptr || out_cols == nullptr) return N4M_ERR_NULL_POINTER;
    if (input_cols < 1) return N4M_ERR_INVALID_ARGUMENT;
    const int64_t m = n4m_wavelet_dwt_output_length(
        input_cols,
        n4m_pp_wavelet_state_family(h->state),
        n4m_pp_wavelet_state_mode(h->state));
    *out_cols = 2 * m;
    return N4M_OK;
}

N4M_API n4m_status_t n4m_pp_wavelet_transform(
    const n4m_pp_wavelet_handle_t* h,
    n4m_matrix_view_t X, n4m_matrix_view_t out) {
    if (h == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;
        double*       op = nullptr;
        std::int64_t  xr = 0, xc = 0, orr = 0, oc = 0;
        n4m_status_t  s  = require_rowmajor_f64(X, xp, xr, xc);
        if (s != N4M_OK) return s;
        s = require_rowmajor_f64_mut(out, op, orr, oc);
        if (s != N4M_OK) return s;
        if (xr != orr) return N4M_ERR_SHAPE_MISMATCH;
        return n4m_pp_wavelet_state_apply(h->state, xp, xr, xc, oc, op);
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// Haar (convenience wrapper)
// ---------------------------------------------------------------------------

N4M_API n4m_status_t n4m_pp_haar_create(n4m_pp_haar_handle_t** out) {
    if (out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = nullptr;
    try {
        n4m_pp_haar_state_t* s = n4m_pp_haar_state_new();
        if (s == nullptr) return N4M_ERR_OUT_OF_MEMORY;
        auto* h = new (std::nothrow) n4m_pp_haar_handle_t{s};
        if (h == nullptr) {
            n4m_pp_haar_state_free(s);
            return N4M_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API void n4m_pp_haar_destroy(n4m_pp_haar_handle_t* h) {
    if (h == nullptr) return;
    try {
        n4m_pp_haar_state_free(h->state);
        delete h;
    } catch (...) { /* swallow */ }
}

N4M_API n4m_status_t n4m_pp_haar_output_cols(int64_t input_cols,
                                              int64_t* out_cols) {
    if (out_cols == nullptr) return N4M_ERR_NULL_POINTER;
    if (input_cols < 1) return N4M_ERR_INVALID_ARGUMENT;
    const int64_t m = n4m_wavelet_dwt_output_length(
        input_cols, N4M_WAVELET_HAAR, N4M_WAVELET_MODE_PERIODIZATION);
    *out_cols = 2 * m;
    return N4M_OK;
}

N4M_API n4m_status_t n4m_pp_haar_transform(const n4m_pp_haar_handle_t* h,
                                            n4m_matrix_view_t X,
                                            n4m_matrix_view_t out) {
    if (h == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;
        double*       op = nullptr;
        std::int64_t  xr = 0, xc = 0, orr = 0, oc = 0;
        n4m_status_t  s  = require_rowmajor_f64(X, xp, xr, xc);
        if (s != N4M_OK) return s;
        s = require_rowmajor_f64_mut(out, op, orr, oc);
        if (s != N4M_OK) return s;
        if (xr != orr) return N4M_ERR_SHAPE_MISMATCH;
        return n4m_pp_haar_state_apply(h->state, xp, xr, xc, oc, op);
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// WaveletDenoise (multi-level + thresholding + reconstruction)
// ---------------------------------------------------------------------------

N4M_API n4m_status_t n4m_pp_wavelet_denoise_create(
    n4m_pp_wavelet_denoise_handle_t** out,
    n4m_pp_wavelet_family_t   family,
    n4m_pp_wavelet_boundary_t mode,
    int32_t                   level,
    n4m_pp_wavelet_threshold_t threshold_mode,
    n4m_pp_wavelet_noise_t     noise_estimator) {
    if (out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = nullptr;
    if (!family_is_valid(family) || !mode_is_valid(mode) || level < 0) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (threshold_mode != N4M_PP_WAVELET_THRESHOLD_SOFT &&
        threshold_mode != N4M_PP_WAVELET_THRESHOLD_HARD) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (noise_estimator != N4M_PP_WAVELET_NOISE_MEDIAN &&
        noise_estimator != N4M_PP_WAVELET_NOISE_STD) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    try {
        n4m_pp_wavelet_denoise_state_t* s = n4m_pp_wavelet_denoise_state_new(
            static_cast<n4m_wavelet_family_t>(family),
            static_cast<n4m_wavelet_mode_t>(mode),
            level,
            static_cast<n4m_pp_wavelet_threshold_mode_t>(threshold_mode),
            static_cast<n4m_pp_wavelet_noise_estimator_t>(noise_estimator));
        if (s == nullptr) return N4M_ERR_OUT_OF_MEMORY;
        auto* h = new (std::nothrow) n4m_pp_wavelet_denoise_handle_t{s};
        if (h == nullptr) {
            n4m_pp_wavelet_denoise_state_free(s);
            return N4M_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API void n4m_pp_wavelet_denoise_destroy(n4m_pp_wavelet_denoise_handle_t* h) {
    if (h == nullptr) return;
    try {
        n4m_pp_wavelet_denoise_state_free(h->state);
        delete h;
    } catch (...) { /* swallow */ }
}

N4M_API n4m_status_t n4m_pp_wavelet_denoise_transform(
    const n4m_pp_wavelet_denoise_handle_t* h,
    n4m_matrix_view_t X, n4m_matrix_view_t out) {
    if (h == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;
        double*       op = nullptr;
        std::int64_t  xr = 0, xc = 0, orr = 0, oc = 0;
        n4m_status_t  s  = require_rowmajor_f64(X, xp, xr, xc);
        if (s != N4M_OK) return s;
        s = require_rowmajor_f64_mut(out, op, orr, oc);
        if (s != N4M_OK) return s;
        if (xr != orr || xc != oc) return N4M_ERR_SHAPE_MISMATCH;
        return n4m_pp_wavelet_denoise_state_apply(h->state, xp, xr, xc, op);
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// WaveletFeatures (multi-level + per-band statistics)
// ---------------------------------------------------------------------------

N4M_API n4m_status_t n4m_pp_wavelet_features_create(
    n4m_pp_wavelet_features_handle_t** out,
    n4m_pp_wavelet_family_t   family,
    n4m_pp_wavelet_boundary_t mode,
    int32_t                   max_level) {
    return n4m_pp_wavelet_features_create_ex(
        out, family, mode, max_level, N4M_PP_WAVELET_FEATURES_ENTROPY_ENERGY);
}

N4M_API n4m_status_t n4m_pp_wavelet_features_create_ex(
    n4m_pp_wavelet_features_handle_t** out,
    n4m_pp_wavelet_family_t           family,
    n4m_pp_wavelet_boundary_t         mode,
    int32_t                           max_level,
    n4m_pp_wavelet_features_entropy_t entropy_mode) {
    if (out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = nullptr;
    if (!family_is_valid(family) || !mode_is_valid(mode) || max_level < 0 ||
        !entropy_is_valid(entropy_mode)) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    try {
        n4m_pp_wavelet_features_state_t* s = n4m_pp_wavelet_features_state_new_ex(
            static_cast<n4m_wavelet_family_t>(family),
            static_cast<n4m_wavelet_mode_t>(mode),
            max_level,
            static_cast<n4m_pp_wavelet_features_entropy_t>(entropy_mode));
        if (s == nullptr) return N4M_ERR_OUT_OF_MEMORY;
        auto* h = new (std::nothrow) n4m_pp_wavelet_features_handle_t{s};
        if (h == nullptr) {
            n4m_pp_wavelet_features_state_free(s);
            return N4M_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API void n4m_pp_wavelet_features_destroy(n4m_pp_wavelet_features_handle_t* h) {
    if (h == nullptr) return;
    try {
        n4m_pp_wavelet_features_state_free(h->state);
        delete h;
    } catch (...) { /* swallow */ }
}

N4M_API n4m_status_t n4m_pp_wavelet_features_output_cols(
    const n4m_pp_wavelet_features_handle_t* h,
    int64_t input_cols, int64_t* out_cols) {
    if (h == nullptr || out_cols == nullptr) return N4M_ERR_NULL_POINTER;
    if (input_cols < 1) return N4M_ERR_INVALID_ARGUMENT;
    *out_cols = n4m_pp_wavelet_features_output_size(h->state, input_cols);
    return N4M_OK;
}

N4M_API n4m_status_t n4m_pp_wavelet_features_transform(
    const n4m_pp_wavelet_features_handle_t* h,
    n4m_matrix_view_t X, n4m_matrix_view_t out) {
    if (h == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;
        double*       op = nullptr;
        std::int64_t  xr = 0, xc = 0, orr = 0, oc = 0;
        n4m_status_t  s  = require_rowmajor_f64(X, xp, xr, xc);
        if (s != N4M_OK) return s;
        s = require_rowmajor_f64_mut(out, op, orr, oc);
        if (s != N4M_OK) return s;
        if (xr != orr) return N4M_ERR_SHAPE_MISMATCH;
        return n4m_pp_wavelet_features_state_apply(h->state, xp, xr, xc, oc, op);
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// WaveletPCA
// ---------------------------------------------------------------------------

N4M_API n4m_status_t n4m_pp_wavelet_pca_create(
    n4m_pp_wavelet_pca_handle_t** out,
    n4m_pp_wavelet_family_t   family,
    n4m_pp_wavelet_boundary_t mode,
    int32_t                   max_level,
    double                    n_components) {
    if (out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = nullptr;
    if (!family_is_valid(family) || !mode_is_valid(mode) ||
        max_level < 0 || !(n_components > 0.0)) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    try {
        n4m_pp_wavelet_pca_state_t* s = n4m_pp_wavelet_pca_state_new(
            static_cast<n4m_wavelet_family_t>(family),
            static_cast<n4m_wavelet_mode_t>(mode),
            max_level, n_components);
        if (s == nullptr) return N4M_ERR_OUT_OF_MEMORY;
        auto* h = new (std::nothrow) n4m_pp_wavelet_pca_handle_t{s};
        if (h == nullptr) {
            n4m_pp_wavelet_pca_state_free(s);
            return N4M_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API void n4m_pp_wavelet_pca_destroy(n4m_pp_wavelet_pca_handle_t* h) {
    if (h == nullptr) return;
    try {
        n4m_pp_wavelet_pca_state_free(h->state);
        delete h;
    } catch (...) { /* swallow */ }
}

N4M_API n4m_status_t n4m_pp_wavelet_pca_fit(n4m_pp_wavelet_pca_handle_t* h,
                                             n4m_matrix_view_t X) {
    if (h == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;
        std::int64_t  xr = 0, xc = 0;
        const n4m_status_t s = require_rowmajor_f64(X, xp, xr, xc);
        if (s != N4M_OK) return s;
        return n4m_pp_wavelet_pca_state_fit(h->state, xp, xr, xc);
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_pp_wavelet_pca_transform(
    const n4m_pp_wavelet_pca_handle_t* h,
    n4m_matrix_view_t X, n4m_matrix_view_t out) {
    if (h == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;
        double*       op = nullptr;
        std::int64_t  xr = 0, xc = 0, orr = 0, oc = 0;
        n4m_status_t  s  = require_rowmajor_f64(X, xp, xr, xc);
        if (s != N4M_OK) return s;
        s = require_rowmajor_f64_mut(out, op, orr, oc);
        if (s != N4M_OK) return s;
        if (xr != orr) return N4M_ERR_SHAPE_MISMATCH;
        return n4m_pp_wavelet_pca_state_apply(h->state, xp, xr, xc, oc, op);
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_pp_wavelet_pca_is_fitted(
    const n4m_pp_wavelet_pca_handle_t* h, int* out_fitted) {
    if (h == nullptr || out_fitted == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        *out_fitted = n4m_pp_wavelet_pca_state_is_fitted(h->state);
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_pp_wavelet_pca_output_cols(
    const n4m_pp_wavelet_pca_handle_t* h, int64_t* out_cols) {
    if (h == nullptr || out_cols == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        if (!n4m_pp_wavelet_pca_state_is_fitted(h->state)) {
            return N4M_ERR_NOT_FITTED;
        }
        *out_cols = n4m_pp_wavelet_pca_state_n_components(h->state);
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// WaveletSVD
// ---------------------------------------------------------------------------

N4M_API n4m_status_t n4m_pp_wavelet_svd_create(
    n4m_pp_wavelet_svd_handle_t** out,
    n4m_pp_wavelet_family_t   family,
    n4m_pp_wavelet_boundary_t mode,
    int32_t                   max_level,
    double                    n_components) {
    if (out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = nullptr;
    if (!family_is_valid(family) || !mode_is_valid(mode) ||
        max_level < 0 || !(n_components > 0.0)) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    try {
        n4m_pp_wavelet_svd_state_t* s = n4m_pp_wavelet_svd_state_new(
            static_cast<n4m_wavelet_family_t>(family),
            static_cast<n4m_wavelet_mode_t>(mode),
            max_level, n_components);
        if (s == nullptr) return N4M_ERR_OUT_OF_MEMORY;
        auto* h = new (std::nothrow) n4m_pp_wavelet_svd_handle_t{s};
        if (h == nullptr) {
            n4m_pp_wavelet_svd_state_free(s);
            return N4M_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API void n4m_pp_wavelet_svd_destroy(n4m_pp_wavelet_svd_handle_t* h) {
    if (h == nullptr) return;
    try {
        n4m_pp_wavelet_svd_state_free(h->state);
        delete h;
    } catch (...) { /* swallow */ }
}

N4M_API n4m_status_t n4m_pp_wavelet_svd_fit(n4m_pp_wavelet_svd_handle_t* h,
                                             n4m_matrix_view_t X) {
    if (h == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;
        std::int64_t  xr = 0, xc = 0;
        const n4m_status_t s = require_rowmajor_f64(X, xp, xr, xc);
        if (s != N4M_OK) return s;
        return n4m_pp_wavelet_svd_state_fit(h->state, xp, xr, xc);
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_pp_wavelet_svd_transform(
    const n4m_pp_wavelet_svd_handle_t* h,
    n4m_matrix_view_t X, n4m_matrix_view_t out) {
    if (h == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;
        double*       op = nullptr;
        std::int64_t  xr = 0, xc = 0, orr = 0, oc = 0;
        n4m_status_t  s  = require_rowmajor_f64(X, xp, xr, xc);
        if (s != N4M_OK) return s;
        s = require_rowmajor_f64_mut(out, op, orr, oc);
        if (s != N4M_OK) return s;
        if (xr != orr) return N4M_ERR_SHAPE_MISMATCH;
        return n4m_pp_wavelet_svd_state_apply(h->state, xp, xr, xc, oc, op);
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_pp_wavelet_svd_is_fitted(
    const n4m_pp_wavelet_svd_handle_t* h, int* out_fitted) {
    if (h == nullptr || out_fitted == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        *out_fitted = n4m_pp_wavelet_svd_state_is_fitted(h->state);
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API n4m_status_t n4m_pp_wavelet_svd_output_cols(
    const n4m_pp_wavelet_svd_handle_t* h, int64_t* out_cols) {
    if (h == nullptr || out_cols == nullptr) return N4M_ERR_NULL_POINTER;
    try {
        if (!n4m_pp_wavelet_svd_state_is_fitted(h->state)) {
            return N4M_ERR_NOT_FITTED;
        }
        *out_cols = n4m_pp_wavelet_svd_state_n_components(h->state);
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

}  // extern "C"
