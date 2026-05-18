// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for the Phase 18 augmenter slice: 12 operators
// (DetectorRollOff, StrayLight, EdgeCurvature, TruncatedPeak, EdgeArtifacts,
// Spline_Smoothing, Spline_X_Perturbations, Spline_Y_Perturbations,
// Spline_X_Simplification, Spline_Curve_Simplification, Rotate_Translate,
// Random_X_Operation). All twelve share the universal `c4a_aug_*` ABI shape
// locked in roadmap/phase-15-18-augmenters-abi-contract.md:
//
//   c4a_aug_<name>_create(out, rng, ...);
//   c4a_aug_<name>_apply (handle, X, [wavelengths,] out);
//   c4a_aug_<name>_destroy(handle);
//
// The RNG is stored by reference; the augmenter does NOT own it (caller
// keeps it alive). Each wrapper is wrapped in try/catch so no C++ exception
// ever crosses the boundary.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <new>

#include "chemometrics4all/c4a.h"

#include "core/augmentations/edge_artifacts/detector_rolloff.h"
#include "core/augmentations/edge_artifacts/edge_curvature.h"
#include "core/augmentations/edge_artifacts/stray_light.h"
#include "core/augmentations/edge_artifacts/truncated_peak.h"
#include "core/augmentations/random/random_x_op.h"
#include "core/augmentations/random/rotate_translate.h"
#include "core/augmentations/splines/spline_curve_simplification.h"
#include "core/augmentations/splines/spline_smoothing.h"
#include "core/augmentations/splines/spline_x_perturbations.h"
#include "core/augmentations/splines/spline_x_simplification.h"
#include "core/augmentations/splines/spline_y_perturbations.h"
#include "core/matrix_view.hpp"

#include "rng_state_internal.hpp"

namespace {

// Resolve the internal engine pointer from a public RNG state pointer.
c4a_rng_pcg64* rng_engine(c4a_rng_pcg64_state_t* state) noexcept {
    return (state == nullptr) ? nullptr : &state->engine;
}

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

c4a_status_t require_rowmajor_f64_const(const c4a_matrix_view_t& v,
                                         const double*& out_ptr,
                                         std::int64_t& out_rows,
                                         std::int64_t& out_cols) noexcept {
    const c4a_status_t s = ::chemometrics4all::core::validate_nonnull_view(v);
    if (s != C4A_OK) return s;
    if (v.dtype != C4A_DTYPE_F64)         return C4A_ERR_DTYPE_MISMATCH;
    if (v.col_stride != 1)                return C4A_ERR_STRIDE_INVALID;
    if (v.rows > 0 && v.cols > 0 && v.row_stride != v.cols) {
        return C4A_ERR_STRIDE_INVALID;
    }
    out_ptr  = static_cast<const double*>(v.data);
    out_rows = v.rows;
    out_cols = v.cols;
    return C4A_OK;
}

// Validate that input/output matrix views match in shape.
c4a_status_t require_matching_shape(std::int64_t Xr, std::int64_t Xc,
                                     std::int64_t Or, std::int64_t Oc) noexcept {
    if (Xr != Or || Xc != Oc) return C4A_ERR_SHAPE_MISMATCH;
    return C4A_OK;
}

// Wavelength is a (1, cols) row-major view OR (cols, 1) column. Accept
// the canonical (1, cols) form.
c4a_status_t require_wavelengths_match(const c4a_matrix_view_t& wl,
                                        std::int64_t X_cols,
                                        const double*& out_ptr) noexcept {
    const c4a_status_t s = ::chemometrics4all::core::validate_nonnull_view(wl);
    if (s != C4A_OK) return s;
    if (wl.dtype != C4A_DTYPE_F64)         return C4A_ERR_DTYPE_MISMATCH;
    if (wl.col_stride != 1)                return C4A_ERR_STRIDE_INVALID;
    if (wl.rows != 1 || wl.cols != X_cols) return C4A_ERR_SHAPE_MISMATCH;
    out_ptr = static_cast<const double*>(wl.data);
    return C4A_OK;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public opaque handles
// ---------------------------------------------------------------------------

struct c4a_aug_detector_rolloff_handle_t {
    c4a_aug_detector_rolloff_state_t* state;
    c4a_rng_pcg64_state_t*            rng;     // borrowed
};
struct c4a_aug_stray_light_handle_t {
    c4a_aug_stray_light_state_t* state;
    c4a_rng_pcg64_state_t*       rng;
};
struct c4a_aug_edge_curve_handle_t {
    c4a_aug_edge_curvature_state_t* state;
    c4a_rng_pcg64_state_t*          rng;
};
struct c4a_aug_truncated_peak_handle_t {
    c4a_aug_truncated_peak_state_t* state;
    c4a_rng_pcg64_state_t*          rng;
};
struct c4a_aug_edge_artifacts_handle_t {
    int32_t                            enabled_flags;
    double                             overall_strength;
    c4a_aug_detector_rolloff_state_t*  detector_state;
    c4a_aug_stray_light_state_t*       stray_state;
    c4a_aug_edge_curvature_state_t*    curve_state;
    c4a_aug_truncated_peak_state_t*    truncated_state;
    c4a_rng_pcg64_state_t*             rng;
};
struct c4a_aug_spline_smooth_handle_t {
    c4a_aug_spline_smooth_state_t* state;
    c4a_rng_pcg64_state_t*         rng;
};
struct c4a_aug_spline_x_perturb_handle_t {
    c4a_aug_spline_x_perturb_state_t* state;
    c4a_rng_pcg64_state_t*            rng;
};
struct c4a_aug_spline_y_perturb_handle_t {
    c4a_aug_spline_y_perturb_state_t* state;
    c4a_rng_pcg64_state_t*            rng;
};
struct c4a_aug_spline_x_simplify_handle_t {
    c4a_aug_spline_x_simplify_state_t* state;
    c4a_rng_pcg64_state_t*             rng;
};
struct c4a_aug_spline_curve_simplify_handle_t {
    c4a_aug_spline_curve_simplify_state_t* state;
    c4a_rng_pcg64_state_t*                 rng;
};
struct c4a_aug_rotate_translate_handle_t {
    c4a_aug_rotate_translate_state_t* state;
    c4a_rng_pcg64_state_t*            rng;
};
struct c4a_aug_random_x_op_handle_t {
    c4a_aug_random_x_op_state_t* state;
    c4a_rng_pcg64_state_t*       rng;
};

extern "C" {

// ---------------------------------------------------------------------------
// DetectorRollOff
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_aug_detector_rolloff_create(
    c4a_aug_detector_rolloff_handle_t** out,
    c4a_rng_pcg64_state_t* rng,
    int32_t detector_model,
    double effect_strength,
    double noise_amplification,
    int32_t include_baseline_distortion) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        auto* st = c4a_aug_detector_rolloff_state_new(
            detector_model, effect_strength, noise_amplification,
            include_baseline_distortion);
        if (st == nullptr) return C4A_ERR_INVALID_ARGUMENT;
        auto* h = new (std::nothrow) c4a_aug_detector_rolloff_handle_t{st, rng};
        if (h == nullptr) {
            c4a_aug_detector_rolloff_state_free(st);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API void c4a_aug_detector_rolloff_destroy(
    c4a_aug_detector_rolloff_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_aug_detector_rolloff_state_free(h->state);
        delete h;
    } catch (...) {}
}

C4A_API c4a_status_t c4a_aug_detector_rolloff_apply(
    const c4a_aug_detector_rolloff_handle_t* h,
    c4a_matrix_view_t X,
    c4a_matrix_view_t wavelengths,
    c4a_matrix_view_t out) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;  std::int64_t xr = 0, xc = 0;
        double*       op = nullptr;  std::int64_t orw = 0, oc = 0;
        c4a_status_t s = require_rowmajor_f64_const(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        s = require_rowmajor_f64(out, op, orw, oc);
        if (s != C4A_OK) return s;
        s = require_matching_shape(xr, xc, orw, oc);
        if (s != C4A_OK) return s;
        const double* wlp = nullptr;
        s = require_wavelengths_match(wavelengths, xc, wlp);
        if (s != C4A_OK) return s;
        return c4a_aug_detector_rolloff_state_apply(
            h->state, rng_engine(h->rng), xp, xr, xc, wlp, op);
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// StrayLight
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_aug_stray_light_create(
    c4a_aug_stray_light_handle_t** out,
    c4a_rng_pcg64_state_t* rng,
    double stray_light_fraction,
    double edge_enhancement,
    double edge_width,
    int32_t include_peak_truncation) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        auto* st = c4a_aug_stray_light_state_new(
            stray_light_fraction, edge_enhancement, edge_width,
            include_peak_truncation);
        if (st == nullptr) return C4A_ERR_INVALID_ARGUMENT;
        auto* h = new (std::nothrow) c4a_aug_stray_light_handle_t{st, rng};
        if (h == nullptr) {
            c4a_aug_stray_light_state_free(st);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API void c4a_aug_stray_light_destroy(c4a_aug_stray_light_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_aug_stray_light_state_free(h->state);
        delete h;
    } catch (...) {}
}

C4A_API c4a_status_t c4a_aug_stray_light_apply(
    const c4a_aug_stray_light_handle_t* h,
    c4a_matrix_view_t X,
    c4a_matrix_view_t wavelengths,
    c4a_matrix_view_t out) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;  std::int64_t xr = 0, xc = 0;
        double*       op = nullptr;  std::int64_t orw = 0, oc = 0;
        c4a_status_t s = require_rowmajor_f64_const(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        s = require_rowmajor_f64(out, op, orw, oc);
        if (s != C4A_OK) return s;
        s = require_matching_shape(xr, xc, orw, oc);
        if (s != C4A_OK) return s;
        const double* wlp = nullptr;
        s = require_wavelengths_match(wavelengths, xc, wlp);
        if (s != C4A_OK) return s;
        return c4a_aug_stray_light_state_apply(
            h->state, rng_engine(h->rng), xp, xr, xc, wlp, op);
    } catch (...) { return C4A_ERR_INTERNAL; }
}

// ---------------------------------------------------------------------------
// EdgeCurvature
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_aug_edge_curve_create(
    c4a_aug_edge_curve_handle_t** out,
    c4a_rng_pcg64_state_t* rng,
    double curvature_strength,
    int32_t curvature_type,
    double asymmetry,
    double edge_focus) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        auto* st = c4a_aug_edge_curvature_state_new(
            curvature_strength, curvature_type, asymmetry, edge_focus);
        if (st == nullptr) return C4A_ERR_INVALID_ARGUMENT;
        auto* h = new (std::nothrow) c4a_aug_edge_curve_handle_t{st, rng};
        if (h == nullptr) {
            c4a_aug_edge_curvature_state_free(st);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API void c4a_aug_edge_curve_destroy(c4a_aug_edge_curve_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_aug_edge_curvature_state_free(h->state);
        delete h;
    } catch (...) {}
}

C4A_API c4a_status_t c4a_aug_edge_curve_apply(
    const c4a_aug_edge_curve_handle_t* h,
    c4a_matrix_view_t X,
    c4a_matrix_view_t wavelengths,
    c4a_matrix_view_t out) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;  std::int64_t xr = 0, xc = 0;
        double*       op = nullptr;  std::int64_t orw = 0, oc = 0;
        c4a_status_t s = require_rowmajor_f64_const(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        s = require_rowmajor_f64(out, op, orw, oc);
        if (s != C4A_OK) return s;
        s = require_matching_shape(xr, xc, orw, oc);
        if (s != C4A_OK) return s;
        const double* wlp = nullptr;
        s = require_wavelengths_match(wavelengths, xc, wlp);
        if (s != C4A_OK) return s;
        return c4a_aug_edge_curvature_state_apply(
            h->state, rng_engine(h->rng), xp, xr, xc, wlp, op);
    } catch (...) { return C4A_ERR_INTERNAL; }
}

// ---------------------------------------------------------------------------
// TruncatedPeak
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_aug_truncated_peak_create(
    c4a_aug_truncated_peak_handle_t** out,
    c4a_rng_pcg64_state_t* rng,
    double peak_probability,
    double amplitude_min, double amplitude_max,
    double width_min,     double width_max,
    int32_t left_edge,
    int32_t right_edge) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        auto* st = c4a_aug_truncated_peak_state_new(
            peak_probability, amplitude_min, amplitude_max,
            width_min, width_max, left_edge, right_edge);
        if (st == nullptr) return C4A_ERR_INVALID_ARGUMENT;
        auto* h = new (std::nothrow) c4a_aug_truncated_peak_handle_t{st, rng};
        if (h == nullptr) {
            c4a_aug_truncated_peak_state_free(st);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API void c4a_aug_truncated_peak_destroy(
    c4a_aug_truncated_peak_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_aug_truncated_peak_state_free(h->state);
        delete h;
    } catch (...) {}
}

C4A_API c4a_status_t c4a_aug_truncated_peak_apply(
    const c4a_aug_truncated_peak_handle_t* h,
    c4a_matrix_view_t X,
    c4a_matrix_view_t wavelengths,
    c4a_matrix_view_t out) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;  std::int64_t xr = 0, xc = 0;
        double*       op = nullptr;  std::int64_t orw = 0, oc = 0;
        c4a_status_t s = require_rowmajor_f64_const(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        s = require_rowmajor_f64(out, op, orw, oc);
        if (s != C4A_OK) return s;
        s = require_matching_shape(xr, xc, orw, oc);
        if (s != C4A_OK) return s;
        const double* wlp = nullptr;
        s = require_wavelengths_match(wavelengths, xc, wlp);
        if (s != C4A_OK) return s;
        return c4a_aug_truncated_peak_state_apply(
            h->state, rng_engine(h->rng), xp, xr, xc, wlp, op);
    } catch (...) { return C4A_ERR_INTERNAL; }
}

// ---------------------------------------------------------------------------
// EdgeArtifacts (combined)
// ---------------------------------------------------------------------------
//
// Applies enabled sub-augmenters in fixed order (matches the Python
// reference): truncated_peaks -> edge_curvature -> stray_light ->
// detector_rolloff. The intermediate buffer is held in `out`; the engine
// alternates between X (or out) and a heap-allocated temporary depending
// on how many sub-augmenters are enabled.

C4A_API c4a_status_t c4a_aug_edge_artifacts_create(
    c4a_aug_edge_artifacts_handle_t** out,
    c4a_rng_pcg64_state_t* rng,
    int32_t enabled_flags,
    double overall_strength,
    int32_t detector_model) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        auto* h = new (std::nothrow) c4a_aug_edge_artifacts_handle_t{};
        if (h == nullptr) return C4A_ERR_OUT_OF_MEMORY;
        h->enabled_flags     = enabled_flags;
        h->overall_strength  = overall_strength;
        h->detector_state    = nullptr;
        h->stray_state       = nullptr;
        h->curve_state       = nullptr;
        h->truncated_state   = nullptr;
        h->rng               = rng;
        // Build sub-states with Python-reference defaults scaled by
        // overall_strength.
        if (enabled_flags & C4A_AUG_EDGE_ARTIFACTS_DETECTOR_ROLL_OFF) {
            h->detector_state = c4a_aug_detector_rolloff_state_new(
                detector_model, overall_strength,
                /*noise_amp=*/0.02, /*baseline=*/1);
            if (h->detector_state == nullptr) {
                delete h;
                return C4A_ERR_INVALID_ARGUMENT;
            }
        }
        if (enabled_flags & C4A_AUG_EDGE_ARTIFACTS_STRAY_LIGHT) {
            h->stray_state = c4a_aug_stray_light_state_new(
                /*fraction=*/0.001 * overall_strength,
                /*edge_enh=*/2.0, /*edge_width=*/0.1,
                /*peak_trunc=*/1);
            if (h->stray_state == nullptr) {
                c4a_aug_detector_rolloff_state_free(h->detector_state);
                delete h;
                return C4A_ERR_INVALID_ARGUMENT;
            }
        }
        if (enabled_flags & C4A_AUG_EDGE_ARTIFACTS_EDGE_CURVATURE) {
            h->curve_state = c4a_aug_edge_curvature_state_new(
                /*strength=*/0.02 * overall_strength,
                /*type=*/0, /*asym=*/0.0, /*focus=*/0.7);
            if (h->curve_state == nullptr) {
                c4a_aug_detector_rolloff_state_free(h->detector_state);
                c4a_aug_stray_light_state_free(h->stray_state);
                delete h;
                return C4A_ERR_INVALID_ARGUMENT;
            }
        }
        if (enabled_flags & C4A_AUG_EDGE_ARTIFACTS_TRUNCATED_PEAKS) {
            h->truncated_state = c4a_aug_truncated_peak_state_new(
                /*prob=*/0.3,
                /*amp=*/0.01 * overall_strength, 0.1 * overall_strength,
                /*width=*/50.0, 200.0,
                /*left=*/1, /*right=*/1);
            if (h->truncated_state == nullptr) {
                c4a_aug_detector_rolloff_state_free(h->detector_state);
                c4a_aug_stray_light_state_free(h->stray_state);
                c4a_aug_edge_curvature_state_free(h->curve_state);
                delete h;
                return C4A_ERR_INVALID_ARGUMENT;
            }
        }
        *out = h;
        return C4A_OK;
    } catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API void c4a_aug_edge_artifacts_destroy(
    c4a_aug_edge_artifacts_handle_t* h) {
    if (h == nullptr) return;
    try {
        c4a_aug_detector_rolloff_state_free(h->detector_state);
        c4a_aug_stray_light_state_free(h->stray_state);
        c4a_aug_edge_curvature_state_free(h->curve_state);
        c4a_aug_truncated_peak_state_free(h->truncated_state);
        delete h;
    } catch (...) {}
}

C4A_API c4a_status_t c4a_aug_edge_artifacts_apply(
    const c4a_aug_edge_artifacts_handle_t* h,
    c4a_matrix_view_t X,
    c4a_matrix_view_t wavelengths,
    c4a_matrix_view_t out) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr;  std::int64_t xr = 0, xc = 0;
        double*       op = nullptr;  std::int64_t orw = 0, oc = 0;
        c4a_status_t s = require_rowmajor_f64_const(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        s = require_rowmajor_f64(out, op, orw, oc);
        if (s != C4A_OK) return s;
        s = require_matching_shape(xr, xc, orw, oc);
        if (s != C4A_OK) return s;
        const double* wlp = nullptr;
        s = require_wavelengths_match(wavelengths, xc, wlp);
        if (s != C4A_OK) return s;
        const std::int64_t total = xr * xc;
        // Stage X -> out. Subsequent stages operate in-place on out.
        if (op != xp) {
            std::memcpy(op, xp, (size_t)total * sizeof(double));
        }
        if (h->truncated_state != nullptr) {
            const c4a_status_t st = c4a_aug_truncated_peak_state_apply(
                h->truncated_state, rng_engine(h->rng), op, xr, xc, wlp, op);
            if (st != C4A_OK) return st;
        }
        if (h->curve_state != nullptr) {
            const c4a_status_t st = c4a_aug_edge_curvature_state_apply(
                h->curve_state, rng_engine(h->rng), op, xr, xc, wlp, op);
            if (st != C4A_OK) return st;
        }
        if (h->stray_state != nullptr) {
            const c4a_status_t st = c4a_aug_stray_light_state_apply(
                h->stray_state, rng_engine(h->rng), op, xr, xc, wlp, op);
            if (st != C4A_OK) return st;
        }
        if (h->detector_state != nullptr) {
            const c4a_status_t st = c4a_aug_detector_rolloff_state_apply(
                h->detector_state, rng_engine(h->rng), op, xr, xc, wlp, op);
            if (st != C4A_OK) return st;
        }
        return C4A_OK;
    } catch (...) { return C4A_ERR_INTERNAL; }
}

// ---------------------------------------------------------------------------
// Spline_Smoothing
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_aug_spline_smooth_create(
    c4a_aug_spline_smooth_handle_t** out,
    c4a_rng_pcg64_state_t* rng) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        auto* st = c4a_aug_spline_smooth_state_new();
        if (st == nullptr) return C4A_ERR_OUT_OF_MEMORY;
        auto* h = new (std::nothrow) c4a_aug_spline_smooth_handle_t{st, rng};
        if (h == nullptr) {
            c4a_aug_spline_smooth_state_free(st);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API void c4a_aug_spline_smooth_destroy(
    c4a_aug_spline_smooth_handle_t* h) {
    if (h == nullptr) return;
    try { c4a_aug_spline_smooth_state_free(h->state); delete h; }
    catch (...) {}
}

C4A_API c4a_status_t c4a_aug_spline_smooth_apply(
    const c4a_aug_spline_smooth_handle_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t out) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr; std::int64_t xr = 0, xc = 0;
        double*       op = nullptr; std::int64_t orw = 0, oc = 0;
        c4a_status_t s = require_rowmajor_f64_const(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        s = require_rowmajor_f64(out, op, orw, oc);
        if (s != C4A_OK) return s;
        s = require_matching_shape(xr, xc, orw, oc);
        if (s != C4A_OK) return s;
        return c4a_aug_spline_smooth_state_apply(
            h->state, rng_engine(h->rng), xp, xr, xc, op);
    } catch (...) { return C4A_ERR_INTERNAL; }
}

// ---------------------------------------------------------------------------
// Spline_X_Perturbations
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_aug_spline_x_perturb_create(
    c4a_aug_spline_x_perturb_handle_t** out,
    c4a_rng_pcg64_state_t* rng,
    int32_t spline_degree,
    double perturbation_density,
    double perturbation_range_min,
    double perturbation_range_max) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        auto* st = c4a_aug_spline_x_perturb_state_new(
            spline_degree, perturbation_density,
            perturbation_range_min, perturbation_range_max);
        if (st == nullptr) return C4A_ERR_INVALID_ARGUMENT;
        auto* h = new (std::nothrow) c4a_aug_spline_x_perturb_handle_t{st, rng};
        if (h == nullptr) {
            c4a_aug_spline_x_perturb_state_free(st);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API void c4a_aug_spline_x_perturb_destroy(
    c4a_aug_spline_x_perturb_handle_t* h) {
    if (h == nullptr) return;
    try { c4a_aug_spline_x_perturb_state_free(h->state); delete h; }
    catch (...) {}
}

C4A_API c4a_status_t c4a_aug_spline_x_perturb_apply(
    const c4a_aug_spline_x_perturb_handle_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t out) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr; std::int64_t xr = 0, xc = 0;
        double*       op = nullptr; std::int64_t orw = 0, oc = 0;
        c4a_status_t s = require_rowmajor_f64_const(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        s = require_rowmajor_f64(out, op, orw, oc);
        if (s != C4A_OK) return s;
        s = require_matching_shape(xr, xc, orw, oc);
        if (s != C4A_OK) return s;
        return c4a_aug_spline_x_perturb_state_apply(
            h->state, rng_engine(h->rng), xp, xr, xc, op);
    } catch (...) { return C4A_ERR_INTERNAL; }
}

// ---------------------------------------------------------------------------
// Spline_Y_Perturbations
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_aug_spline_y_perturb_create(
    c4a_aug_spline_y_perturb_handle_t** out,
    c4a_rng_pcg64_state_t* rng,
    int32_t spline_points,
    double perturbation_intensity) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        auto* st = c4a_aug_spline_y_perturb_state_new(
            spline_points, perturbation_intensity);
        if (st == nullptr) return C4A_ERR_OUT_OF_MEMORY;
        auto* h = new (std::nothrow) c4a_aug_spline_y_perturb_handle_t{st, rng};
        if (h == nullptr) {
            c4a_aug_spline_y_perturb_state_free(st);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API void c4a_aug_spline_y_perturb_destroy(
    c4a_aug_spline_y_perturb_handle_t* h) {
    if (h == nullptr) return;
    try { c4a_aug_spline_y_perturb_state_free(h->state); delete h; }
    catch (...) {}
}

C4A_API c4a_status_t c4a_aug_spline_y_perturb_apply(
    const c4a_aug_spline_y_perturb_handle_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t out) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr; std::int64_t xr = 0, xc = 0;
        double*       op = nullptr; std::int64_t orw = 0, oc = 0;
        c4a_status_t s = require_rowmajor_f64_const(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        s = require_rowmajor_f64(out, op, orw, oc);
        if (s != C4A_OK) return s;
        s = require_matching_shape(xr, xc, orw, oc);
        if (s != C4A_OK) return s;
        return c4a_aug_spline_y_perturb_state_apply(
            h->state, rng_engine(h->rng), xp, xr, xc, op);
    } catch (...) { return C4A_ERR_INTERNAL; }
}

// ---------------------------------------------------------------------------
// Spline_X_Simplification (v2-deferred stub)
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_aug_spline_x_simplify_create(
    c4a_aug_spline_x_simplify_handle_t** out,
    c4a_rng_pcg64_state_t* rng,
    int32_t spline_points,
    int32_t uniform_flag) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        auto* st = c4a_aug_spline_x_simplify_state_new(
            spline_points, uniform_flag);
        if (st == nullptr) return C4A_ERR_OUT_OF_MEMORY;
        auto* h = new (std::nothrow) c4a_aug_spline_x_simplify_handle_t{st, rng};
        if (h == nullptr) {
            c4a_aug_spline_x_simplify_state_free(st);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API void c4a_aug_spline_x_simplify_destroy(
    c4a_aug_spline_x_simplify_handle_t* h) {
    if (h == nullptr) return;
    try { c4a_aug_spline_x_simplify_state_free(h->state); delete h; }
    catch (...) {}
}

C4A_API c4a_status_t c4a_aug_spline_x_simplify_apply(
    const c4a_aug_spline_x_simplify_handle_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t out) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr; std::int64_t xr = 0, xc = 0;
        double*       op = nullptr; std::int64_t orw = 0, oc = 0;
        c4a_status_t s = require_rowmajor_f64_const(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        s = require_rowmajor_f64(out, op, orw, oc);
        if (s != C4A_OK) return s;
        s = require_matching_shape(xr, xc, orw, oc);
        if (s != C4A_OK) return s;
        return c4a_aug_spline_x_simplify_state_apply(
            h->state, rng_engine(h->rng), xp, xr, xc, op);
    } catch (...) { return C4A_ERR_INTERNAL; }
}

// ---------------------------------------------------------------------------
// Spline_Curve_Simplification (v2-deferred stub)
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_aug_spline_curve_simplify_create(
    c4a_aug_spline_curve_simplify_handle_t** out,
    c4a_rng_pcg64_state_t* rng,
    int32_t spline_points,
    int32_t uniform_flag) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        auto* st = c4a_aug_spline_curve_simplify_state_new(
            spline_points, uniform_flag);
        if (st == nullptr) return C4A_ERR_OUT_OF_MEMORY;
        auto* h = new (std::nothrow)
            c4a_aug_spline_curve_simplify_handle_t{st, rng};
        if (h == nullptr) {
            c4a_aug_spline_curve_simplify_state_free(st);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API void c4a_aug_spline_curve_simplify_destroy(
    c4a_aug_spline_curve_simplify_handle_t* h) {
    if (h == nullptr) return;
    try { c4a_aug_spline_curve_simplify_state_free(h->state); delete h; }
    catch (...) {}
}

C4A_API c4a_status_t c4a_aug_spline_curve_simplify_apply(
    const c4a_aug_spline_curve_simplify_handle_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t out) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr; std::int64_t xr = 0, xc = 0;
        double*       op = nullptr; std::int64_t orw = 0, oc = 0;
        c4a_status_t s = require_rowmajor_f64_const(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        s = require_rowmajor_f64(out, op, orw, oc);
        if (s != C4A_OK) return s;
        s = require_matching_shape(xr, xc, orw, oc);
        if (s != C4A_OK) return s;
        return c4a_aug_spline_curve_simplify_state_apply(
            h->state, rng_engine(h->rng), xp, xr, xc, op);
    } catch (...) { return C4A_ERR_INTERNAL; }
}

// ---------------------------------------------------------------------------
// Rotate_Translate
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_aug_rotate_translate_create(
    c4a_aug_rotate_translate_handle_t** out,
    c4a_rng_pcg64_state_t* rng,
    double p_range,
    double y_factor) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        auto* st = c4a_aug_rotate_translate_state_new(p_range, y_factor);
        if (st == nullptr) return C4A_ERR_INVALID_ARGUMENT;
        auto* h = new (std::nothrow) c4a_aug_rotate_translate_handle_t{st, rng};
        if (h == nullptr) {
            c4a_aug_rotate_translate_state_free(st);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API void c4a_aug_rotate_translate_destroy(
    c4a_aug_rotate_translate_handle_t* h) {
    if (h == nullptr) return;
    try { c4a_aug_rotate_translate_state_free(h->state); delete h; }
    catch (...) {}
}

C4A_API c4a_status_t c4a_aug_rotate_translate_apply(
    const c4a_aug_rotate_translate_handle_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t out) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr; std::int64_t xr = 0, xc = 0;
        double*       op = nullptr; std::int64_t orw = 0, oc = 0;
        c4a_status_t s = require_rowmajor_f64_const(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        s = require_rowmajor_f64(out, op, orw, oc);
        if (s != C4A_OK) return s;
        s = require_matching_shape(xr, xc, orw, oc);
        if (s != C4A_OK) return s;
        return c4a_aug_rotate_translate_state_apply(
            h->state, rng_engine(h->rng), xp, xr, xc, op);
    } catch (...) { return C4A_ERR_INTERNAL; }
}

// ---------------------------------------------------------------------------
// Random_X_Operation
// ---------------------------------------------------------------------------

C4A_API c4a_status_t c4a_aug_random_x_op_create(
    c4a_aug_random_x_op_handle_t** out,
    c4a_rng_pcg64_state_t* rng,
    int32_t op_kind,
    double operator_range_min,
    double operator_range_max) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        auto* st = c4a_aug_random_x_op_state_new(
            op_kind, operator_range_min, operator_range_max);
        if (st == nullptr) return C4A_ERR_INVALID_ARGUMENT;
        auto* h = new (std::nothrow) c4a_aug_random_x_op_handle_t{st, rng};
        if (h == nullptr) {
            c4a_aug_random_x_op_state_free(st);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return C4A_OK;
    } catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API void c4a_aug_random_x_op_destroy(c4a_aug_random_x_op_handle_t* h) {
    if (h == nullptr) return;
    try { c4a_aug_random_x_op_state_free(h->state); delete h; }
    catch (...) {}
}

C4A_API c4a_status_t c4a_aug_random_x_op_apply(
    const c4a_aug_random_x_op_handle_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t out) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        const double* xp = nullptr; std::int64_t xr = 0, xc = 0;
        double*       op = nullptr; std::int64_t orw = 0, oc = 0;
        c4a_status_t s = require_rowmajor_f64_const(X, xp, xr, xc);
        if (s != C4A_OK) return s;
        s = require_rowmajor_f64(out, op, orw, oc);
        if (s != C4A_OK) return s;
        s = require_matching_shape(xr, xc, orw, oc);
        if (s != C4A_OK) return s;
        return c4a_aug_random_x_op_state_apply(
            h->state, rng_engine(h->rng), xp, xr, xc, op);
    } catch (...) { return C4A_ERR_INTERNAL; }
}

}  // extern "C"
