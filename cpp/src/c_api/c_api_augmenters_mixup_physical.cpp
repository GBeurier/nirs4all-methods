// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for the Phase 17 augmenters (mixup, scattering,
// environmental). Each wrapper has a try/catch around its full body so no
// C++ exception ever crosses the boundary. Opaque handles wrap the
// internal C engine in core/augmentations/.
//
// Universal ABI shape (per the Phase 15-18 contract document):
//
//   c4a_aug_<NAME>_create(handle_t** out,
//                         c4a_rng_pcg64_state_t* rng,
//                         /* operator-specific params */);
//   c4a_aug_<NAME>_apply(const handle_t* handle,
//                        c4a_matrix_view_t X, c4a_matrix_view_t out);
//   c4a_aug_<NAME>_destroy(handle_t* handle);
//
// The RNG handle is stored BY REFERENCE on the augmenter handle — the
// augmenter does NOT own it. The caller must keep the RNG alive for the
// augmenter's lifetime.
//
// For augmenters that additionally need wavelength information, the
// wavelengths are provided as a per-handle array passed at create time
// (length n_features, copied into the handle).

#include <cstring>
#include <exception>
#include <new>

#include "chemometrics4all/c4a.h"

#include "core/augmentations/mixup/mixup.h"
#include "core/augmentations/mixup/local_mixup.h"
#include "core/augmentations/scattering/scatter_sim_msc.h"
#include "core/augmentations/scattering/particle_size.h"
#include "core/augmentations/scattering/emsc_distort.h"
#include "core/augmentations/scattering/batch_effect.h"
#include "core/augmentations/scattering/instrument_broaden.h"
#include "core/augmentations/scattering/dead_band.h"
#include "core/augmentations/environmental/temperature.h"
#include "core/augmentations/environmental/moisture.h"

#include "rng_state_internal.hpp"

namespace {

// Validate a 2-D F64 row-major contiguous matrix view. Returns the status
// code on failure; sets `rows` and `cols` (and `data`) on success.
template <typename T>
c4a_status_t validate_view(c4a_matrix_view_t v, int64_t& rows, int64_t& cols,
                            T*& data) {
    if (v.dtype != C4A_DTYPE_F64) {
        return C4A_ERR_DTYPE_MISMATCH;
    }
    if (v.rows < 0 || v.cols < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    // Row-major contiguous: row_stride == cols, col_stride == 1
    // (degenerate dimensions tolerated).
    if (v.cols >= 2 && v.col_stride != 1) {
        return C4A_ERR_STRIDE_INVALID;
    }
    if (v.rows >= 2 && v.row_stride != v.cols) {
        return C4A_ERR_STRIDE_INVALID;
    }
    rows = v.rows;
    cols = v.cols;
    data = static_cast<T*>(v.data);
    if (rows > 0 && cols > 0 && data == nullptr) {
        return C4A_ERR_NULL_POINTER;
    }
    return C4A_OK;
}

c4a_status_t validate_shapes(c4a_matrix_view_t X, c4a_matrix_view_t Y) {
    if (X.rows != Y.rows || X.cols != Y.cols) {
        return C4A_ERR_SHAPE_MISMATCH;
    }
    return C4A_OK;
}

}  // namespace

// ============================================================================
// MixupAugmenter
// ============================================================================
struct c4a_aug_mixup_handle_t {
    c4a_aug_mixup_state_t* state;
    c4a_rng_pcg64_state_t* rng;
};

extern "C" {

C4A_API c4a_status_t c4a_aug_mixup_create(c4a_aug_mixup_handle_t** out,
                                           c4a_rng_pcg64_state_t* rng,
                                           double alpha) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        c4a_aug_mixup_state_t* s = c4a_aug_mixup_state_new(alpha);
        if (s == nullptr) return C4A_ERR_INVALID_ARGUMENT;
        auto* h = new c4a_aug_mixup_handle_t{s, rng};
        *out = h;
        return C4A_OK;
    } catch (const std::bad_alloc&) {
        return C4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return C4A_ERR_INTERNAL;
    }
}

C4A_API c4a_status_t c4a_aug_mixup_apply(const c4a_aug_mixup_handle_t* h,
                                          c4a_matrix_view_t X,
                                          c4a_matrix_view_t out) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    int64_t rows = 0, cols = 0;
    const double* Xp = nullptr;
    double* Yp = nullptr;
    c4a_status_t st = validate_view<const double>(X, rows, cols, Xp);
    if (st != C4A_OK) return st;
    int64_t r2 = 0, c2 = 0;
    st = validate_view<double>(out, r2, c2, Yp);
    if (st != C4A_OK) return st;
    st = validate_shapes(X, out);
    if (st != C4A_OK) return st;
    try {
        return c4a_aug_mixup_apply_impl(h->state, &h->rng->engine,
                                         Xp, rows, cols, Yp);
    } catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API void c4a_aug_mixup_destroy(c4a_aug_mixup_handle_t* h) {
    try {
        if (h != nullptr) {
            c4a_aug_mixup_state_free(h->state);
            delete h;
        }
    } catch (...) {}
}

// ============================================================================
// LocalMixupAugmenter
// ============================================================================
struct c4a_aug_local_mixup_handle_t {
    c4a_aug_local_mixup_state_t* state;
    c4a_rng_pcg64_state_t* rng;
};

C4A_API c4a_status_t c4a_aug_local_mixup_create(c4a_aug_local_mixup_handle_t** out,
                                                 c4a_rng_pcg64_state_t* rng,
                                                 double alpha,
                                                 int32_t k_neighbors) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        auto* s = c4a_aug_local_mixup_state_new(alpha, k_neighbors);
        if (s == nullptr) return C4A_ERR_INVALID_ARGUMENT;
        auto* h = new c4a_aug_local_mixup_handle_t{s, rng};
        *out = h;
        return C4A_OK;
    } catch (const std::bad_alloc&) { return C4A_ERR_OUT_OF_MEMORY; }
      catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API c4a_status_t c4a_aug_local_mixup_apply(const c4a_aug_local_mixup_handle_t* h,
                                                c4a_matrix_view_t X,
                                                c4a_matrix_view_t out) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    int64_t rows = 0, cols = 0;
    const double* Xp = nullptr; double* Yp = nullptr;
    c4a_status_t st = validate_view<const double>(X, rows, cols, Xp);
    if (st != C4A_OK) return st;
    int64_t r2 = 0, c2 = 0;
    st = validate_view<double>(out, r2, c2, Yp);
    if (st != C4A_OK) return st;
    if ((st = validate_shapes(X, out)) != C4A_OK) return st;
    try {
        return c4a_aug_local_mixup_apply_impl(h->state, &h->rng->engine,
                                               Xp, rows, cols, Yp);
    } catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API void c4a_aug_local_mixup_destroy(c4a_aug_local_mixup_handle_t* h) {
    try {
        if (h != nullptr) {
            c4a_aug_local_mixup_state_free(h->state);
            delete h;
        }
    } catch (...) {}
}

// ============================================================================
// ScatterSimulationMSC
// ============================================================================
struct c4a_aug_scatter_sim_handle_t {
    c4a_aug_scatter_sim_state_t* state;
    c4a_rng_pcg64_state_t* rng;
};

C4A_API c4a_status_t c4a_aug_scatter_sim_create(c4a_aug_scatter_sim_handle_t** out,
                                                 c4a_rng_pcg64_state_t* rng,
                                                 double a_low, double a_high,
                                                 double b_low, double b_high) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        auto* s = c4a_aug_scatter_sim_state_new(a_low, a_high, b_low, b_high);
        if (s == nullptr) return C4A_ERR_INVALID_ARGUMENT;
        auto* h = new c4a_aug_scatter_sim_handle_t{s, rng};
        *out = h;
        return C4A_OK;
    } catch (const std::bad_alloc&) { return C4A_ERR_OUT_OF_MEMORY; }
      catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API c4a_status_t c4a_aug_scatter_sim_apply(const c4a_aug_scatter_sim_handle_t* h,
                                                c4a_matrix_view_t X,
                                                c4a_matrix_view_t out) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    int64_t rows = 0, cols = 0;
    const double* Xp = nullptr; double* Yp = nullptr;
    c4a_status_t st = validate_view<const double>(X, rows, cols, Xp);
    if (st != C4A_OK) return st;
    int64_t r2 = 0, c2 = 0;
    st = validate_view<double>(out, r2, c2, Yp);
    if (st != C4A_OK) return st;
    if ((st = validate_shapes(X, out)) != C4A_OK) return st;
    try {
        return c4a_aug_scatter_sim_apply_impl(h->state, &h->rng->engine,
                                               Xp, rows, cols, Yp);
    } catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API void c4a_aug_scatter_sim_destroy(c4a_aug_scatter_sim_handle_t* h) {
    try {
        if (h != nullptr) {
            c4a_aug_scatter_sim_state_free(h->state);
            delete h;
        }
    } catch (...) {}
}

// ============================================================================
// ParticleSizeAugmenter — wavelengths captured at create time
// ============================================================================
struct c4a_aug_particle_size_handle_t {
    c4a_aug_particle_size_state_t* state;
    c4a_rng_pcg64_state_t* rng;
    double* wavelengths;
    int64_t n_wavelengths;
};

C4A_API c4a_status_t c4a_aug_particle_size_create(
    c4a_aug_particle_size_handle_t** out,
    c4a_rng_pcg64_state_t* rng,
    double mean_size_um, double size_variation_um,
    int    use_size_range, double size_range_low_um, double size_range_high_um,
    double reference_size_um, double wavelength_exponent,
    double size_effect_strength,
    int    include_path_length, double path_length_sensitivity,
    const double* wavelengths, int64_t n_wavelengths) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr || wavelengths == nullptr) return C4A_ERR_NULL_POINTER;
    if (n_wavelengths <= 0) return C4A_ERR_INVALID_ARGUMENT;
    try {
        auto* s = c4a_aug_particle_size_state_new(
            mean_size_um, size_variation_um,
            use_size_range, size_range_low_um, size_range_high_um,
            reference_size_um, wavelength_exponent,
            size_effect_strength,
            include_path_length, path_length_sensitivity);
        if (s == nullptr) return C4A_ERR_INVALID_ARGUMENT;
        double* wl_copy = new double[static_cast<size_t>(n_wavelengths)];
        std::memcpy(wl_copy, wavelengths,
                    static_cast<size_t>(n_wavelengths) * sizeof(double));
        auto* h = new c4a_aug_particle_size_handle_t{s, rng, wl_copy, n_wavelengths};
        *out = h;
        return C4A_OK;
    } catch (const std::bad_alloc&) { return C4A_ERR_OUT_OF_MEMORY; }
      catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API c4a_status_t c4a_aug_particle_size_apply(
    const c4a_aug_particle_size_handle_t* h,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    int64_t rows = 0, cols = 0;
    const double* Xp = nullptr; double* Yp = nullptr;
    c4a_status_t st = validate_view<const double>(X, rows, cols, Xp);
    if (st != C4A_OK) return st;
    int64_t r2 = 0, c2 = 0;
    st = validate_view<double>(out, r2, c2, Yp);
    if (st != C4A_OK) return st;
    if ((st = validate_shapes(X, out)) != C4A_OK) return st;
    if (cols != h->n_wavelengths) return C4A_ERR_SHAPE_MISMATCH;
    try {
        return c4a_aug_particle_size_apply_impl(
            h->state, &h->rng->engine, Xp, rows, cols, h->wavelengths, Yp);
    } catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API void c4a_aug_particle_size_destroy(c4a_aug_particle_size_handle_t* h) {
    try {
        if (h != nullptr) {
            c4a_aug_particle_size_state_free(h->state);
            delete[] h->wavelengths;
            delete h;
        }
    } catch (...) {}
}

// ============================================================================
// EMSCDistortionAugmenter
// ============================================================================
struct c4a_aug_emsc_distort_handle_t {
    c4a_aug_emsc_distort_state_t* state;
    c4a_rng_pcg64_state_t* rng;
    double* wavelengths;
    int64_t n_wavelengths;
};

C4A_API c4a_status_t c4a_aug_emsc_distort_create(
    c4a_aug_emsc_distort_handle_t** out,
    c4a_rng_pcg64_state_t* rng,
    double mult_low, double mult_high,
    double add_low,  double add_high,
    int32_t polynomial_order,
    double  polynomial_strength,
    double  correlation,
    const double* wavelengths, int64_t n_wavelengths) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr || wavelengths == nullptr) return C4A_ERR_NULL_POINTER;
    if (n_wavelengths <= 0) return C4A_ERR_INVALID_ARGUMENT;
    try {
        auto* s = c4a_aug_emsc_distort_state_new(
            mult_low, mult_high, add_low, add_high,
            polynomial_order, polynomial_strength, correlation);
        if (s == nullptr) return C4A_ERR_INVALID_ARGUMENT;
        double* wl_copy = new double[static_cast<size_t>(n_wavelengths)];
        std::memcpy(wl_copy, wavelengths,
                    static_cast<size_t>(n_wavelengths) * sizeof(double));
        auto* h = new c4a_aug_emsc_distort_handle_t{s, rng, wl_copy, n_wavelengths};
        *out = h;
        return C4A_OK;
    } catch (const std::bad_alloc&) { return C4A_ERR_OUT_OF_MEMORY; }
      catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API c4a_status_t c4a_aug_emsc_distort_apply(
    const c4a_aug_emsc_distort_handle_t* h,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    int64_t rows = 0, cols = 0;
    const double* Xp = nullptr; double* Yp = nullptr;
    c4a_status_t st = validate_view<const double>(X, rows, cols, Xp);
    if (st != C4A_OK) return st;
    int64_t r2 = 0, c2 = 0;
    st = validate_view<double>(out, r2, c2, Yp);
    if (st != C4A_OK) return st;
    if ((st = validate_shapes(X, out)) != C4A_OK) return st;
    if (cols != h->n_wavelengths) return C4A_ERR_SHAPE_MISMATCH;
    try {
        return c4a_aug_emsc_distort_apply_impl(
            h->state, &h->rng->engine, Xp, rows, cols, h->wavelengths, Yp);
    } catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API void c4a_aug_emsc_distort_destroy(c4a_aug_emsc_distort_handle_t* h) {
    try {
        if (h != nullptr) {
            c4a_aug_emsc_distort_state_free(h->state);
            delete[] h->wavelengths;
            delete h;
        }
    } catch (...) {}
}

// ============================================================================
// BatchEffectAugmenter — wavelengths are optional (NULL allowed)
// ============================================================================
struct c4a_aug_batch_effect_handle_t {
    c4a_aug_batch_effect_state_t* state;
    c4a_rng_pcg64_state_t* rng;
    double* wavelengths;    // may be nullptr
    int64_t n_wavelengths;
};

C4A_API c4a_status_t c4a_aug_batch_effect_create(
    c4a_aug_batch_effect_handle_t** out,
    c4a_rng_pcg64_state_t* rng,
    double offset_std, double slope_std, double gain_std,
    int32_t variation_scope,
    const double* wavelengths, int64_t n_wavelengths) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        auto* s = c4a_aug_batch_effect_state_new(
            offset_std, slope_std, gain_std, variation_scope);
        if (s == nullptr) return C4A_ERR_INVALID_ARGUMENT;
        double* wl_copy = nullptr;
        if (wavelengths != nullptr && n_wavelengths > 0) {
            wl_copy = new double[static_cast<size_t>(n_wavelengths)];
            std::memcpy(wl_copy, wavelengths,
                        static_cast<size_t>(n_wavelengths) * sizeof(double));
        }
        auto* h = new c4a_aug_batch_effect_handle_t{s, rng, wl_copy, n_wavelengths};
        *out = h;
        return C4A_OK;
    } catch (const std::bad_alloc&) { return C4A_ERR_OUT_OF_MEMORY; }
      catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API c4a_status_t c4a_aug_batch_effect_apply(
    const c4a_aug_batch_effect_handle_t* h,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    int64_t rows = 0, cols = 0;
    const double* Xp = nullptr; double* Yp = nullptr;
    c4a_status_t st = validate_view<const double>(X, rows, cols, Xp);
    if (st != C4A_OK) return st;
    int64_t r2 = 0, c2 = 0;
    st = validate_view<double>(out, r2, c2, Yp);
    if (st != C4A_OK) return st;
    if ((st = validate_shapes(X, out)) != C4A_OK) return st;
    /* If wavelengths were given at create time, cols must match. */
    if (h->wavelengths != nullptr && cols != h->n_wavelengths) {
        return C4A_ERR_SHAPE_MISMATCH;
    }
    try {
        return c4a_aug_batch_effect_apply_impl(
            h->state, &h->rng->engine, Xp, rows, cols, h->wavelengths, Yp);
    } catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API void c4a_aug_batch_effect_destroy(c4a_aug_batch_effect_handle_t* h) {
    try {
        if (h != nullptr) {
            c4a_aug_batch_effect_state_free(h->state);
            delete[] h->wavelengths;
            delete h;
        }
    } catch (...) {}
}

// ============================================================================
// InstrumentalBroadeningAugmenter — wavelengths optional (NULL allowed)
// ============================================================================
struct c4a_aug_instrument_broaden_handle_t {
    c4a_aug_instrument_broaden_state_t* state;
    c4a_rng_pcg64_state_t* rng;
    double* wavelengths;
    int64_t n_wavelengths;
};

C4A_API c4a_status_t c4a_aug_instrument_broaden_create(
    c4a_aug_instrument_broaden_handle_t** out,
    c4a_rng_pcg64_state_t* rng,
    double fwhm,
    int    use_fwhm_range, double fwhm_low, double fwhm_high,
    int32_t variation_scope,
    const double* wavelengths, int64_t n_wavelengths) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        auto* s = c4a_aug_instrument_broaden_state_new(
            fwhm, use_fwhm_range, fwhm_low, fwhm_high, variation_scope);
        if (s == nullptr) return C4A_ERR_INVALID_ARGUMENT;
        double* wl_copy = nullptr;
        if (wavelengths != nullptr && n_wavelengths > 0) {
            wl_copy = new double[static_cast<size_t>(n_wavelengths)];
            std::memcpy(wl_copy, wavelengths,
                        static_cast<size_t>(n_wavelengths) * sizeof(double));
        }
        auto* h = new c4a_aug_instrument_broaden_handle_t{s, rng, wl_copy, n_wavelengths};
        *out = h;
        return C4A_OK;
    } catch (const std::bad_alloc&) { return C4A_ERR_OUT_OF_MEMORY; }
      catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API c4a_status_t c4a_aug_instrument_broaden_apply(
    const c4a_aug_instrument_broaden_handle_t* h,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    int64_t rows = 0, cols = 0;
    const double* Xp = nullptr; double* Yp = nullptr;
    c4a_status_t st = validate_view<const double>(X, rows, cols, Xp);
    if (st != C4A_OK) return st;
    int64_t r2 = 0, c2 = 0;
    st = validate_view<double>(out, r2, c2, Yp);
    if (st != C4A_OK) return st;
    if ((st = validate_shapes(X, out)) != C4A_OK) return st;
    if (h->wavelengths != nullptr && cols != h->n_wavelengths) {
        return C4A_ERR_SHAPE_MISMATCH;
    }
    try {
        return c4a_aug_instrument_broaden_apply_impl(
            h->state, &h->rng->engine, Xp, rows, cols, h->wavelengths, Yp);
    } catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API void c4a_aug_instrument_broaden_destroy(
    c4a_aug_instrument_broaden_handle_t* h) {
    try {
        if (h != nullptr) {
            c4a_aug_instrument_broaden_state_free(h->state);
            delete[] h->wavelengths;
            delete h;
        }
    } catch (...) {}
}

// ============================================================================
// DeadBandAugmenter
// ============================================================================
struct c4a_aug_dead_band_handle_t {
    c4a_aug_dead_band_state_t* state;
    c4a_rng_pcg64_state_t* rng;
};

C4A_API c4a_status_t c4a_aug_dead_band_create(
    c4a_aug_dead_band_handle_t** out,
    c4a_rng_pcg64_state_t* rng,
    int32_t n_bands,
    int32_t width_low, int32_t width_high,
    double  noise_std, double probability,
    int32_t variation_scope) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr) return C4A_ERR_NULL_POINTER;
    try {
        auto* s = c4a_aug_dead_band_state_new(
            n_bands, width_low, width_high, noise_std, probability,
            variation_scope);
        if (s == nullptr) return C4A_ERR_INVALID_ARGUMENT;
        auto* h = new c4a_aug_dead_band_handle_t{s, rng};
        *out = h;
        return C4A_OK;
    } catch (const std::bad_alloc&) { return C4A_ERR_OUT_OF_MEMORY; }
      catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API c4a_status_t c4a_aug_dead_band_apply(
    const c4a_aug_dead_band_handle_t* h,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    int64_t rows = 0, cols = 0;
    const double* Xp = nullptr; double* Yp = nullptr;
    c4a_status_t st = validate_view<const double>(X, rows, cols, Xp);
    if (st != C4A_OK) return st;
    int64_t r2 = 0, c2 = 0;
    st = validate_view<double>(out, r2, c2, Yp);
    if (st != C4A_OK) return st;
    if ((st = validate_shapes(X, out)) != C4A_OK) return st;
    try {
        return c4a_aug_dead_band_apply_impl(h->state, &h->rng->engine,
                                             Xp, rows, cols, Yp);
    } catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API void c4a_aug_dead_band_destroy(c4a_aug_dead_band_handle_t* h) {
    try {
        if (h != nullptr) {
            c4a_aug_dead_band_state_free(h->state);
            delete h;
        }
    } catch (...) {}
}

// ============================================================================
// TemperatureAugmenter
// ============================================================================
struct c4a_aug_temperature_handle_t {
    c4a_aug_temperature_state_t* state;
    c4a_rng_pcg64_state_t* rng;
    double* wavelengths;
    int64_t n_wavelengths;
};

C4A_API c4a_status_t c4a_aug_temperature_create(
    c4a_aug_temperature_handle_t** out,
    c4a_rng_pcg64_state_t* rng,
    double temperature_delta,
    int    use_temp_range, double temp_low, double temp_high,
    int    enable_shift, int enable_intensity, int enable_broadening,
    int    region_specific,
    const double* wavelengths, int64_t n_wavelengths) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr || wavelengths == nullptr) return C4A_ERR_NULL_POINTER;
    if (n_wavelengths <= 0) return C4A_ERR_INVALID_ARGUMENT;
    try {
        auto* s = c4a_aug_temperature_state_new(
            temperature_delta, use_temp_range, temp_low, temp_high,
            enable_shift, enable_intensity, enable_broadening, region_specific);
        if (s == nullptr) return C4A_ERR_INVALID_ARGUMENT;
        double* wl_copy = new double[static_cast<size_t>(n_wavelengths)];
        std::memcpy(wl_copy, wavelengths,
                    static_cast<size_t>(n_wavelengths) * sizeof(double));
        auto* h = new c4a_aug_temperature_handle_t{s, rng, wl_copy, n_wavelengths};
        *out = h;
        return C4A_OK;
    } catch (const std::bad_alloc&) { return C4A_ERR_OUT_OF_MEMORY; }
      catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API c4a_status_t c4a_aug_temperature_apply(
    const c4a_aug_temperature_handle_t* h,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    int64_t rows = 0, cols = 0;
    const double* Xp = nullptr; double* Yp = nullptr;
    c4a_status_t st = validate_view<const double>(X, rows, cols, Xp);
    if (st != C4A_OK) return st;
    int64_t r2 = 0, c2 = 0;
    st = validate_view<double>(out, r2, c2, Yp);
    if (st != C4A_OK) return st;
    if ((st = validate_shapes(X, out)) != C4A_OK) return st;
    if (cols != h->n_wavelengths) return C4A_ERR_SHAPE_MISMATCH;
    try {
        return c4a_aug_temperature_apply_impl(
            h->state, &h->rng->engine, Xp, rows, cols, h->wavelengths, Yp);
    } catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API void c4a_aug_temperature_destroy(c4a_aug_temperature_handle_t* h) {
    try {
        if (h != nullptr) {
            c4a_aug_temperature_state_free(h->state);
            delete[] h->wavelengths;
            delete h;
        }
    } catch (...) {}
}

// ============================================================================
// MoistureAugmenter
// ============================================================================
struct c4a_aug_moisture_handle_t {
    c4a_aug_moisture_state_t* state;
    c4a_rng_pcg64_state_t* rng;
    double* wavelengths;
    int64_t n_wavelengths;
};

C4A_API c4a_status_t c4a_aug_moisture_create(
    c4a_aug_moisture_handle_t** out,
    c4a_rng_pcg64_state_t* rng,
    double water_activity_delta,
    int    use_aw_range, double aw_low, double aw_high,
    double reference_water_activity,
    double free_water_fraction,
    double bound_water_shift,
    double moisture_content,
    int    enable_shift, int enable_intensity,
    const double* wavelengths, int64_t n_wavelengths) {
    if (out == nullptr) return C4A_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr || wavelengths == nullptr) return C4A_ERR_NULL_POINTER;
    if (n_wavelengths <= 0) return C4A_ERR_INVALID_ARGUMENT;
    try {
        auto* s = c4a_aug_moisture_state_new(
            water_activity_delta, use_aw_range, aw_low, aw_high,
            reference_water_activity, free_water_fraction, bound_water_shift,
            moisture_content, enable_shift, enable_intensity);
        if (s == nullptr) return C4A_ERR_INVALID_ARGUMENT;
        double* wl_copy = new double[static_cast<size_t>(n_wavelengths)];
        std::memcpy(wl_copy, wavelengths,
                    static_cast<size_t>(n_wavelengths) * sizeof(double));
        auto* h = new c4a_aug_moisture_handle_t{s, rng, wl_copy, n_wavelengths};
        *out = h;
        return C4A_OK;
    } catch (const std::bad_alloc&) { return C4A_ERR_OUT_OF_MEMORY; }
      catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API c4a_status_t c4a_aug_moisture_apply(
    const c4a_aug_moisture_handle_t* h,
    c4a_matrix_view_t X,
    c4a_matrix_view_t out) {
    if (h == nullptr) return C4A_ERR_NULL_POINTER;
    int64_t rows = 0, cols = 0;
    const double* Xp = nullptr; double* Yp = nullptr;
    c4a_status_t st = validate_view<const double>(X, rows, cols, Xp);
    if (st != C4A_OK) return st;
    int64_t r2 = 0, c2 = 0;
    st = validate_view<double>(out, r2, c2, Yp);
    if (st != C4A_OK) return st;
    if ((st = validate_shapes(X, out)) != C4A_OK) return st;
    if (cols != h->n_wavelengths) return C4A_ERR_SHAPE_MISMATCH;
    try {
        return c4a_aug_moisture_apply_impl(
            h->state, &h->rng->engine, Xp, rows, cols, h->wavelengths, Yp);
    } catch (...) { return C4A_ERR_INTERNAL; }
}

C4A_API void c4a_aug_moisture_destroy(c4a_aug_moisture_handle_t* h) {
    try {
        if (h != nullptr) {
            c4a_aug_moisture_state_free(h->state);
            delete[] h->wavelengths;
            delete h;
        }
    } catch (...) {}
}

}  // extern "C"
