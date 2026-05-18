// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for the Phase 16 wavelength + spectral augmenters.
// Implements the 30 public symbols declared in
// c_api_augmenters_wavelength_spectral.h on top of the engines under
// core/augmentations/{wavelength,spectral}/.
//
// Universal rules (mirroring c_api_filters_meta.cpp):
//   - Every entry point is wrapped in try/catch so no C++ exception ever
//     crosses the boundary.
//   - The opaque public types are thin structs {engine_state*} that own the
//     internal allocation.
//   - `_create` takes pointer-to-pointer out-arg; returns
//     C4A_ERR_NULL_POINTER if `out` is NULL and
//     C4A_ERR_INVALID_ARGUMENT for invalid constructor parameters.
//     Writes NULL to *out on every failure path.
//   - `_destroy` is NULL-safe and never throws.
//
// The RNG handle is stored by reference; the augmenter does NOT own it.
// The caller must keep the RNG alive for the augmenter's lifetime.
//
// `c4a_rng_pcg64_state_t` is opaque to the public ABI but locally defined
// in c_api_rng.cpp as a thin wrapper carrying `c4a_rng_pcg64 engine`.
// We need access to `engine` here, so the struct definition is repeated
// in an anonymous namespace below — the layout is binary-compatible by
// virtue of containing the same single member.

#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>

#include "chemometrics4all/c4a.h"

#include "core/augmentations/spectral/band_mask.h"
#include "core/augmentations/spectral/band_perturb.h"
#include "core/augmentations/spectral/channel_dropout.h"
#include "core/augmentations/spectral/gauss_jitter.h"
#include "core/augmentations/spectral/local_clip.h"
#include "core/augmentations/spectral/magnitude_warp.h"
#include "core/augmentations/spectral/unsharp_mask.h"
#include "core/augmentations/wavelength/local_warp.h"
#include "core/augmentations/wavelength/wavelength_shift.h"
#include "core/augmentations/wavelength/wavelength_stretch.h"
#include "core/matrix_view.hpp"

#include "rng_state_internal.hpp"

// Public opaque handles for the ten Phase 16 augmenters. Each wraps an
// engine-state pointer.
struct c4a_aug_wavelength_shift_handle_t {
    c4a_aug_wavelength_shift_state_t* state;
};
struct c4a_aug_wavelength_stretch_handle_t {
    c4a_aug_wavelength_stretch_state_t* state;
};
struct c4a_aug_local_warp_handle_t {
    c4a_aug_local_warp_state_t* state;
};
struct c4a_aug_band_perturb_handle_t {
    c4a_aug_band_perturb_state_t* state;
};
struct c4a_aug_band_mask_handle_t {
    c4a_aug_band_mask_state_t* state;
};
struct c4a_aug_channel_dropout_handle_t {
    c4a_aug_channel_dropout_state_t* state;
};
struct c4a_aug_gauss_jitter_handle_t {
    c4a_aug_gauss_jitter_state_t* state;
};
struct c4a_aug_unsharp_mask_handle_t {
    c4a_aug_unsharp_mask_state_t* state;
};
struct c4a_aug_magnitude_warp_handle_t {
    c4a_aug_magnitude_warp_state_t* state;
};
struct c4a_aug_local_clip_handle_t {
    c4a_aug_local_clip_state_t* state;
};

namespace {

// Validate that a matrix view is non-NULL, F64, row-major contiguous.
c4a_status_t require_rowmajor_f64(const c4a_matrix_view_t& v,
                                   double*& out_ptr,
                                   std::int64_t& out_rows,
                                   std::int64_t& out_cols) noexcept {
    const c4a_status_t s = ::chemometrics4all::core::validate_nonnull_view(v);
    if (s != C4A_OK) return s;
    if (v.dtype != C4A_DTYPE_F64)         return C4A_ERR_DTYPE_MISMATCH;
    if (v.col_stride != 1)                return C4A_ERR_STRIDE_INVALID;
    if (v.rows > 0 && v.cols > 0 && v.row_stride != v.cols) {
        return C4A_ERR_STRIDE_INVALID;
    }
    out_ptr  = static_cast<double*>(v.data);
    out_rows = v.rows;
    out_cols = v.cols;
    return C4A_OK;
}

// Validate that two matrix views agree in shape + dtype.
c4a_status_t require_same_shape(const c4a_matrix_view_t& X,
                                 const c4a_matrix_view_t& out,
                                 const double*& xp, double*& yp,
                                 std::int64_t& rows,
                                 std::int64_t& cols) noexcept {
    double* xp_nonconst = nullptr;
    double* yp_nonconst = nullptr;
    std::int64_t xr = 0, xc = 0, yr = 0, yc = 0;
    c4a_status_t s = require_rowmajor_f64(X, xp_nonconst, xr, xc);
    if (s != C4A_OK) return s;
    s = require_rowmajor_f64(out, yp_nonconst, yr, yc);
    if (s != C4A_OK) return s;
    if (xr != yr || xc != yc) return C4A_ERR_SHAPE_MISMATCH;
    xp = xp_nonconst;
    yp = yp_nonconst;
    rows = xr;
    cols = xc;
    return C4A_OK;
}

}  // namespace

extern "C" {

// ===========================================================================
// WavelengthShift
// ===========================================================================

C4A_API c4a_status_t c4a_aug_wavelength_shift_create(
    c4a_aug_wavelength_shift_handle_t** out,
    c4a_rng_pcg64_state_t* rng,
    double shift_lo, double shift_hi,
    const double* wavelengths, int64_t n_wavelengths) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        c4a_aug_wavelength_shift_state_t* st =
            c4a_aug_wavelength_shift_state_new(
                &rng->engine, shift_lo, shift_hi, wavelengths, n_wavelengths);
        if (st == nullptr) return C4A_ERR_INVALID_ARGUMENT;
        auto* h = new (std::nothrow) c4a_aug_wavelength_shift_handle_t{st};
        if (h == nullptr) {
            c4a_aug_wavelength_shift_state_free(st);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_aug_wavelength_shift_apply(
    const c4a_aug_wavelength_shift_handle_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t out) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;
        double*       yp = nullptr;
        std::int64_t  r = 0, c = 0;
        const c4a_status_t s = require_same_shape(X, out, xp, yp, r, c);
        if (s != C4A_OK) return s;
        return c4a_aug_wavelength_shift_state_apply(h->state, xp, r, c, yp);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_aug_wavelength_shift_destroy(
    c4a_aug_wavelength_shift_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_aug_wavelength_shift_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

// ===========================================================================
// WavelengthStretch
// ===========================================================================

C4A_API c4a_status_t c4a_aug_wavelength_stretch_create(
    c4a_aug_wavelength_stretch_handle_t** out,
    c4a_rng_pcg64_state_t* rng,
    double stretch_lo, double stretch_hi,
    const double* wavelengths, int64_t n_wavelengths) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        c4a_aug_wavelength_stretch_state_t* st =
            c4a_aug_wavelength_stretch_state_new(
                &rng->engine, stretch_lo, stretch_hi,
                wavelengths, n_wavelengths);
        if (st == nullptr) return C4A_ERR_INVALID_ARGUMENT;
        auto* h = new (std::nothrow) c4a_aug_wavelength_stretch_handle_t{st};
        if (h == nullptr) {
            c4a_aug_wavelength_stretch_state_free(st);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_aug_wavelength_stretch_apply(
    const c4a_aug_wavelength_stretch_handle_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t out) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;
        double*       yp = nullptr;
        std::int64_t  r = 0, c = 0;
        const c4a_status_t s = require_same_shape(X, out, xp, yp, r, c);
        if (s != C4A_OK) return s;
        return c4a_aug_wavelength_stretch_state_apply(h->state, xp, r, c, yp);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_aug_wavelength_stretch_destroy(
    c4a_aug_wavelength_stretch_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_aug_wavelength_stretch_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

// ===========================================================================
// LocalWavelengthWarp
// ===========================================================================

C4A_API c4a_status_t c4a_aug_local_warp_create(
    c4a_aug_local_warp_handle_t** out,
    c4a_rng_pcg64_state_t* rng,
    int32_t n_control_points,
    double  max_shift,
    const double* wavelengths, int64_t n_wavelengths) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        c4a_aug_local_warp_state_t* st = c4a_aug_local_warp_state_new(
            &rng->engine, n_control_points, max_shift,
            wavelengths, n_wavelengths);
        if (st == nullptr) return C4A_ERR_INVALID_ARGUMENT;
        auto* h = new (std::nothrow) c4a_aug_local_warp_handle_t{st};
        if (h == nullptr) {
            c4a_aug_local_warp_state_free(st);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_aug_local_warp_apply(
    const c4a_aug_local_warp_handle_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t out) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;
        double*       yp = nullptr;
        std::int64_t  r = 0, c = 0;
        const c4a_status_t s = require_same_shape(X, out, xp, yp, r, c);
        if (s != C4A_OK) return s;
        return c4a_aug_local_warp_state_apply(h->state, xp, r, c, yp);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_aug_local_warp_destroy(c4a_aug_local_warp_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_aug_local_warp_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

// ===========================================================================
// BandPerturbation
// ===========================================================================

C4A_API c4a_status_t c4a_aug_band_perturb_create(
    c4a_aug_band_perturb_handle_t** out,
    c4a_rng_pcg64_state_t* rng,
    int32_t n_bands,
    int32_t bw_lo, int32_t bw_hi,
    double  gain_lo, double  gain_hi,
    double  offset_lo, double offset_hi) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        c4a_aug_band_perturb_state_t* st = c4a_aug_band_perturb_state_new(
            &rng->engine, n_bands, bw_lo, bw_hi,
            gain_lo, gain_hi, offset_lo, offset_hi);
        if (st == nullptr) return C4A_ERR_INVALID_ARGUMENT;
        auto* h = new (std::nothrow) c4a_aug_band_perturb_handle_t{st};
        if (h == nullptr) {
            c4a_aug_band_perturb_state_free(st);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_aug_band_perturb_apply(
    const c4a_aug_band_perturb_handle_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t out) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;
        double*       yp = nullptr;
        std::int64_t  r = 0, c = 0;
        const c4a_status_t s = require_same_shape(X, out, xp, yp, r, c);
        if (s != C4A_OK) return s;
        return c4a_aug_band_perturb_state_apply(h->state, xp, r, c, yp);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_aug_band_perturb_destroy(c4a_aug_band_perturb_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_aug_band_perturb_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

// ===========================================================================
// BandMasking
// ===========================================================================

C4A_API c4a_status_t c4a_aug_band_mask_create(
    c4a_aug_band_mask_handle_t** out,
    c4a_rng_pcg64_state_t* rng,
    int32_t n_bands_lo, int32_t n_bands_hi,
    int32_t bw_lo, int32_t bw_hi,
    int32_t mode) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr) return C4A_ERR_NULL_POINTER;
    if (mode != static_cast<int32_t>(C4A_AUG_BAND_MASK_ZERO) &&
        mode != static_cast<int32_t>(C4A_AUG_BAND_MASK_INTERP)) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        c4a_aug_band_mask_state_t* st = c4a_aug_band_mask_state_new(
            &rng->engine, n_bands_lo, n_bands_hi, bw_lo, bw_hi,
            static_cast<c4a_aug_band_mask_mode_t>(mode));
        if (st == nullptr) return C4A_ERR_INVALID_ARGUMENT;
        auto* h = new (std::nothrow) c4a_aug_band_mask_handle_t{st};
        if (h == nullptr) {
            c4a_aug_band_mask_state_free(st);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_aug_band_mask_apply(
    const c4a_aug_band_mask_handle_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t out) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;
        double*       yp = nullptr;
        std::int64_t  r = 0, c = 0;
        const c4a_status_t s = require_same_shape(X, out, xp, yp, r, c);
        if (s != C4A_OK) return s;
        return c4a_aug_band_mask_state_apply(h->state, xp, r, c, yp);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_aug_band_mask_destroy(c4a_aug_band_mask_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_aug_band_mask_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

// ===========================================================================
// ChannelDropout
// ===========================================================================

C4A_API c4a_status_t c4a_aug_channel_dropout_create(
    c4a_aug_channel_dropout_handle_t** out,
    c4a_rng_pcg64_state_t* rng,
    double  dropout_prob,
    int32_t mode) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr) return C4A_ERR_NULL_POINTER;
    if (mode != static_cast<int32_t>(C4A_AUG_CHANNEL_DROPOUT_ZERO) &&
        mode != static_cast<int32_t>(C4A_AUG_CHANNEL_DROPOUT_INTERP)) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    try {
        c4a_aug_channel_dropout_state_t* st =
            c4a_aug_channel_dropout_state_new(
                &rng->engine, dropout_prob,
                static_cast<c4a_aug_channel_dropout_mode_t>(mode));
        if (st == nullptr) return C4A_ERR_INVALID_ARGUMENT;
        auto* h = new (std::nothrow) c4a_aug_channel_dropout_handle_t{st};
        if (h == nullptr) {
            c4a_aug_channel_dropout_state_free(st);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_aug_channel_dropout_apply(
    const c4a_aug_channel_dropout_handle_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t out) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;
        double*       yp = nullptr;
        std::int64_t  r = 0, c = 0;
        const c4a_status_t s = require_same_shape(X, out, xp, yp, r, c);
        if (s != C4A_OK) return s;
        return c4a_aug_channel_dropout_state_apply(h->state, xp, r, c, yp);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_aug_channel_dropout_destroy(
    c4a_aug_channel_dropout_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_aug_channel_dropout_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

// ===========================================================================
// GaussianSmoothingJitter
// ===========================================================================

C4A_API c4a_status_t c4a_aug_gauss_jitter_create(
    c4a_aug_gauss_jitter_handle_t** out,
    c4a_rng_pcg64_state_t* rng,
    double  sigma_lo, double sigma_hi,
    int32_t kernel_width) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        c4a_aug_gauss_jitter_state_t* st = c4a_aug_gauss_jitter_state_new(
            &rng->engine, sigma_lo, sigma_hi, kernel_width);
        if (st == nullptr) return C4A_ERR_INVALID_ARGUMENT;
        auto* h = new (std::nothrow) c4a_aug_gauss_jitter_handle_t{st};
        if (h == nullptr) {
            c4a_aug_gauss_jitter_state_free(st);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_aug_gauss_jitter_apply(
    const c4a_aug_gauss_jitter_handle_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t out) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;
        double*       yp = nullptr;
        std::int64_t  r = 0, c = 0;
        const c4a_status_t s = require_same_shape(X, out, xp, yp, r, c);
        if (s != C4A_OK) return s;
        return c4a_aug_gauss_jitter_state_apply(h->state, xp, r, c, yp);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_aug_gauss_jitter_destroy(c4a_aug_gauss_jitter_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_aug_gauss_jitter_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

// ===========================================================================
// UnsharpSpectralMask
// ===========================================================================

C4A_API c4a_status_t c4a_aug_unsharp_mask_create(
    c4a_aug_unsharp_mask_handle_t** out,
    c4a_rng_pcg64_state_t* rng,
    double  amount_lo, double amount_hi,
    double  sigma, int32_t kernel_width) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        c4a_aug_unsharp_mask_state_t* st = c4a_aug_unsharp_mask_state_new(
            &rng->engine, amount_lo, amount_hi, sigma, kernel_width);
        if (st == nullptr) return C4A_ERR_INVALID_ARGUMENT;
        auto* h = new (std::nothrow) c4a_aug_unsharp_mask_handle_t{st};
        if (h == nullptr) {
            c4a_aug_unsharp_mask_state_free(st);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_aug_unsharp_mask_apply(
    const c4a_aug_unsharp_mask_handle_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t out) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;
        double*       yp = nullptr;
        std::int64_t  r = 0, c = 0;
        const c4a_status_t s = require_same_shape(X, out, xp, yp, r, c);
        if (s != C4A_OK) return s;
        return c4a_aug_unsharp_mask_state_apply(h->state, xp, r, c, yp);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_aug_unsharp_mask_destroy(c4a_aug_unsharp_mask_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_aug_unsharp_mask_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

// ===========================================================================
// SmoothMagnitudeWarp
// ===========================================================================

C4A_API c4a_status_t c4a_aug_magnitude_warp_create(
    c4a_aug_magnitude_warp_handle_t** out,
    c4a_rng_pcg64_state_t* rng,
    int32_t n_control_points,
    double  gain_lo, double gain_hi,
    const double* wavelengths, int64_t n_wavelengths) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        c4a_aug_magnitude_warp_state_t* st = c4a_aug_magnitude_warp_state_new(
            &rng->engine, n_control_points, gain_lo, gain_hi,
            wavelengths, n_wavelengths);
        if (st == nullptr) return C4A_ERR_INVALID_ARGUMENT;
        auto* h = new (std::nothrow) c4a_aug_magnitude_warp_handle_t{st};
        if (h == nullptr) {
            c4a_aug_magnitude_warp_state_free(st);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_aug_magnitude_warp_apply(
    const c4a_aug_magnitude_warp_handle_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t out) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;
        double*       yp = nullptr;
        std::int64_t  r = 0, c = 0;
        const c4a_status_t s = require_same_shape(X, out, xp, yp, r, c);
        if (s != C4A_OK) return s;
        return c4a_aug_magnitude_warp_state_apply(h->state, xp, r, c, yp);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_aug_magnitude_warp_destroy(
    c4a_aug_magnitude_warp_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_aug_magnitude_warp_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

// ===========================================================================
// LocalClipping
// ===========================================================================

C4A_API c4a_status_t c4a_aug_local_clip_create(
    c4a_aug_local_clip_handle_t** out,
    c4a_rng_pcg64_state_t* rng,
    int32_t n_regions,
    int32_t width_lo, int32_t width_hi) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        c4a_aug_local_clip_state_t* st = c4a_aug_local_clip_state_new(
            &rng->engine, n_regions, width_lo, width_hi);
        if (st == nullptr) return C4A_ERR_INVALID_ARGUMENT;
        auto* h = new (std::nothrow) c4a_aug_local_clip_handle_t{st};
        if (h == nullptr) {
            c4a_aug_local_clip_state_free(st);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_aug_local_clip_apply(
    const c4a_aug_local_clip_handle_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t out) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;
        double*       yp = nullptr;
        std::int64_t  r = 0, c = 0;
        const c4a_status_t s = require_same_shape(X, out, xp, yp, r, c);
        if (s != C4A_OK) return s;
        return c4a_aug_local_clip_state_apply(h->state, xp, r, c, yp);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_aug_local_clip_destroy(c4a_aug_local_clip_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_aug_local_clip_state_free(h->state);
        delete h;
    } catch (...) {
        // swallow
    }
}

}  // extern "C"
