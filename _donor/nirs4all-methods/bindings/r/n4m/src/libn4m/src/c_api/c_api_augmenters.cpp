// SPDX-License-Identifier: CECILL-2.1
//
// extern "C" wrappers for the Phase 15 noise + drift augmenters
// (new ABI category `n4m_aug_*`). The 7 operators landed in Phase 15:
//
//   * GaussianAdditiveNoise         -> n4m_aug_gaussian_noise_*
//   * MultiplicativeNoise           -> n4m_aug_multiplicative_noise_*
//   * SpikeNoise                    -> n4m_aug_spike_noise_*
//   * HeteroscedasticNoiseAugmenter -> n4m_aug_hetero_noise_*
//   * LinearBaselineDrift           -> n4m_aug_linear_drift_*
//   * PolynomialBaselineDrift       -> n4m_aug_poly_drift_*
//   * PathLengthAugmenter           -> n4m_aug_path_length_*
//
// Each augmenter exposes exactly three symbols:
//
//   n4m_status_t n4m_aug_<NAME>_create(n4m_aug_<NAME>_handle_t** out,
//                                       n4m_rng_pcg64_state_t* rng,
//                                       /* op-specific params */);
//   n4m_status_t n4m_aug_<NAME>_apply(const n4m_aug_<NAME>_handle_t* handle,
//                                      n4m_matrix_view_t X,
//                                      n4m_matrix_view_t out);
//   void         n4m_aug_<NAME>_destroy(n4m_aug_<NAME>_handle_t* handle);
//
// The RNG handle is the FIRST argument of every `_create` and the augmenter
// stores it by reference — the caller owns the RNG lifetime. The contract
// is captured in `roadmap/phase-15-18-augmenters-abi-contract.md`.
//
// Universal rules of the wrappers (mirrored from c_api_filters.cpp):
//   - Every entry point is wrapped in try/catch so no C++ exception ever
//     crosses the boundary.
//   - The opaque public type is a thin struct {engine_state*} that owns the
//     internal allocation; this decouples the public name from the internal
//     layout.
//   - `_create` takes a pointer-to-pointer out-arg; it returns
//     N4M_ERR_NULL_POINTER if `out` or `rng` is NULL and N4M_ERR_INVALID_ARGUMENT
//     for invalid constructor parameters that the engine rejects. It writes
//     NULL to *out on every failure.
//   - `_destroy` is NULL-safe and never throws.
//   - `_apply` validates the X / out matrix views to row-major contiguous F64
//     and matching shapes before delegating to the engine.

#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>

#include "n4m/n4m.h"

#include "core/augmentations/drift/linear_drift.h"
#include "core/augmentations/drift/path_length.h"
#include "core/augmentations/drift/poly_drift.h"
#include "core/augmentations/noise/gaussian_noise.h"
#include "core/augmentations/noise/hetero_noise.h"
#include "core/augmentations/noise/multiplicative_noise.h"
#include "core/augmentations/noise/spike_noise.h"
#include "core/matrix_view.hpp"

#include "rng_state_internal.hpp"

// ---------------------------------------------------------------------------
// Opaque public handles (one per augmenter). These names are part of the C
// ABI but are NOT declared in n4m.h — they are forward-declared here and
// used by ABI consumers as opaque types. The contract is locked in
// `roadmap/phase-15-18-augmenters-abi-contract.md`.
// ---------------------------------------------------------------------------

struct n4m_aug_gaussian_noise_handle_t {
    n4m_aug_gaussian_noise_state_t* state;
};
struct n4m_aug_multiplicative_noise_handle_t {
    n4m_aug_multiplicative_noise_state_t* state;
};
struct n4m_aug_spike_noise_handle_t {
    n4m_aug_spike_noise_state_t* state;
};
struct n4m_aug_hetero_noise_handle_t {
    n4m_aug_hetero_noise_state_t* state;
};
struct n4m_aug_linear_drift_handle_t {
    n4m_aug_linear_drift_state_t* state;
};
struct n4m_aug_poly_drift_handle_t {
    n4m_aug_poly_drift_state_t* state;
};
struct n4m_aug_path_length_handle_t {
    n4m_aug_path_length_state_t* state;
};

// ---------------------------------------------------------------------------
// Forward declarations of the public ABI symbols defined below.
//
// These are normally pulled in from `n4m/n4m.h`, but Phase 15
// does not modify the public header (per the locked ABI contract: the
// augmenter declarations live exclusively in the C ABI symbol table for the
// time being). The forward declarations exist solely so the toolchain's
// `-Wmissing-declarations` does not fire on the definitions further down;
// they have no effect on the emitted symbols.
// ---------------------------------------------------------------------------

extern "C" {

N4M_API n4m_status_t n4m_aug_gaussian_noise_create(
    n4m_aug_gaussian_noise_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double sigma);
N4M_API void n4m_aug_gaussian_noise_destroy(n4m_aug_gaussian_noise_handle_t* h);
N4M_API n4m_status_t n4m_aug_gaussian_noise_apply(
    const n4m_aug_gaussian_noise_handle_t* h,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

N4M_API n4m_status_t n4m_aug_multiplicative_noise_create(
    n4m_aug_multiplicative_noise_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double sigma_gain);
N4M_API void n4m_aug_multiplicative_noise_destroy(
    n4m_aug_multiplicative_noise_handle_t* h);
N4M_API n4m_status_t n4m_aug_multiplicative_noise_apply(
    const n4m_aug_multiplicative_noise_handle_t* h,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

N4M_API n4m_status_t n4m_aug_spike_noise_create(
    n4m_aug_spike_noise_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    int32_t n_spikes_min, int32_t n_spikes_max,
    double amplitude_min, double amplitude_max);
N4M_API void n4m_aug_spike_noise_destroy(n4m_aug_spike_noise_handle_t* h);
N4M_API n4m_status_t n4m_aug_spike_noise_apply(
    const n4m_aug_spike_noise_handle_t* h,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

N4M_API n4m_status_t n4m_aug_hetero_noise_create(
    n4m_aug_hetero_noise_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double noise_base, double noise_signal_dep);
N4M_API void n4m_aug_hetero_noise_destroy(n4m_aug_hetero_noise_handle_t* h);
N4M_API n4m_status_t n4m_aug_hetero_noise_apply(
    const n4m_aug_hetero_noise_handle_t* h,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

N4M_API n4m_status_t n4m_aug_linear_drift_create(
    n4m_aug_linear_drift_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double offset_min, double offset_max,
    double slope_min,  double slope_max);
N4M_API void n4m_aug_linear_drift_destroy(n4m_aug_linear_drift_handle_t* h);
N4M_API n4m_status_t n4m_aug_linear_drift_apply(
    const n4m_aug_linear_drift_handle_t* h,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

N4M_API n4m_status_t n4m_aug_poly_drift_create(
    n4m_aug_poly_drift_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    int32_t degree,
    const double* coeff_min, const double* coeff_max);
N4M_API void n4m_aug_poly_drift_destroy(n4m_aug_poly_drift_handle_t* h);
N4M_API n4m_status_t n4m_aug_poly_drift_apply(
    const n4m_aug_poly_drift_handle_t* h,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

N4M_API n4m_status_t n4m_aug_path_length_create(
    n4m_aug_path_length_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double path_length_std, double min_path_length);
N4M_API void n4m_aug_path_length_destroy(n4m_aug_path_length_handle_t* h);
N4M_API n4m_status_t n4m_aug_path_length_apply(
    const n4m_aug_path_length_handle_t* h,
    n4m_matrix_view_t X, n4m_matrix_view_t out);

}  // extern "C"

namespace {

// Validate that a matrix view is non-NULL, F64, row-major contiguous.
n4m_status_t require_rowmajor_f64(const n4m_matrix_view_t& v,
                                   double*& out_ptr,
                                   std::int64_t& out_rows,
                                   std::int64_t& out_cols) noexcept {
    const n4m_status_t s = ::n4m::core::validate_nonnull_view(v);
    if (s != N4M_OK) return s;
    if (v.dtype != N4M_DTYPE_F64)         return N4M_ERR_DTYPE_MISMATCH;
    if (v.col_stride != 1)                return N4M_ERR_STRIDE_INVALID;
    if (v.rows > 0 && v.cols > 0 && v.row_stride != v.cols) {
        return N4M_ERR_STRIDE_INVALID;
    }
    out_ptr  = static_cast<double*>(v.data);
    out_rows = v.rows;
    out_cols = v.cols;
    return N4M_OK;
}

n4m_status_t require_rowmajor_f64_const(const n4m_matrix_view_t& v,
                                         const double*& out_ptr,
                                         std::int64_t& out_rows,
                                         std::int64_t& out_cols) noexcept {
    double* p = nullptr;
    const n4m_status_t s = require_rowmajor_f64(v, p, out_rows, out_cols);
    out_ptr = p;
    return s;
}

n4m_status_t validate_apply_views(const n4m_matrix_view_t& X,
                                   const n4m_matrix_view_t& out,
                                   const double*& X_ptr, double*& out_ptr,
                                   std::int64_t& rows, std::int64_t& cols) noexcept {
    n4m_status_t s = require_rowmajor_f64_const(X, X_ptr, rows, cols);
    if (s != N4M_OK) return s;
    std::int64_t out_rows = 0, out_cols = 0;
    s = require_rowmajor_f64(out, out_ptr, out_rows, out_cols);
    if (s != N4M_OK) return s;
    if (out_rows != rows || out_cols != cols) return N4M_ERR_SHAPE_MISMATCH;
    return N4M_OK;
}

}  // namespace

extern "C" {

// ===========================================================================
// 1. GaussianAdditiveNoise
// ===========================================================================

N4M_API n4m_status_t n4m_aug_gaussian_noise_create(
    n4m_aug_gaussian_noise_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double sigma) {
    if (out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr) return N4M_ERR_NULL_POINTER;
    if (!(sigma >= 0.0)) return N4M_ERR_INVALID_ARGUMENT;
    try {
        n4m_aug_gaussian_noise_state_t* s =
            n4m_aug_gaussian_noise_state_new(&rng->engine, sigma);
        if (s == nullptr) return N4M_ERR_OUT_OF_MEMORY;
        n4m_aug_gaussian_noise_handle_t* h =
            new (std::nothrow) n4m_aug_gaussian_noise_handle_t{s};
        if (h == nullptr) {
            n4m_aug_gaussian_noise_state_free(s);
            return N4M_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API void n4m_aug_gaussian_noise_destroy(
    n4m_aug_gaussian_noise_handle_t* h) {
    if (h == nullptr) return;
    try {
        n4m_aug_gaussian_noise_state_free(h->state);
        delete h;
    } catch (...) {}
}

N4M_API n4m_status_t n4m_aug_gaussian_noise_apply(
    const n4m_aug_gaussian_noise_handle_t* h,
    n4m_matrix_view_t X, n4m_matrix_view_t out) {
    if (h == nullptr) return N4M_ERR_NULL_POINTER;
    const double* X_ptr = nullptr;
    double* out_ptr = nullptr;
    std::int64_t rows = 0, cols = 0;
    const n4m_status_t s = validate_apply_views(X, out, X_ptr, out_ptr, rows, cols);
    if (s != N4M_OK) return s;
    try {
        return n4m_aug_gaussian_noise_state_apply(h->state, X_ptr, rows, cols, out_ptr);
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

// ===========================================================================
// 2. MultiplicativeNoise
// ===========================================================================

N4M_API n4m_status_t n4m_aug_multiplicative_noise_create(
    n4m_aug_multiplicative_noise_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double sigma_gain) {
    if (out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr) return N4M_ERR_NULL_POINTER;
    if (!(sigma_gain >= 0.0)) return N4M_ERR_INVALID_ARGUMENT;
    try {
        n4m_aug_multiplicative_noise_state_t* s =
            n4m_aug_multiplicative_noise_state_new(&rng->engine, sigma_gain);
        if (s == nullptr) return N4M_ERR_OUT_OF_MEMORY;
        n4m_aug_multiplicative_noise_handle_t* h =
            new (std::nothrow) n4m_aug_multiplicative_noise_handle_t{s};
        if (h == nullptr) {
            n4m_aug_multiplicative_noise_state_free(s);
            return N4M_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API void n4m_aug_multiplicative_noise_destroy(
    n4m_aug_multiplicative_noise_handle_t* h) {
    if (h == nullptr) return;
    try {
        n4m_aug_multiplicative_noise_state_free(h->state);
        delete h;
    } catch (...) {}
}

N4M_API n4m_status_t n4m_aug_multiplicative_noise_apply(
    const n4m_aug_multiplicative_noise_handle_t* h,
    n4m_matrix_view_t X, n4m_matrix_view_t out) {
    if (h == nullptr) return N4M_ERR_NULL_POINTER;
    const double* X_ptr = nullptr;
    double* out_ptr = nullptr;
    std::int64_t rows = 0, cols = 0;
    const n4m_status_t s = validate_apply_views(X, out, X_ptr, out_ptr, rows, cols);
    if (s != N4M_OK) return s;
    try {
        return n4m_aug_multiplicative_noise_state_apply(h->state, X_ptr, rows, cols, out_ptr);
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

// ===========================================================================
// 3. SpikeNoise
// ===========================================================================

N4M_API n4m_status_t n4m_aug_spike_noise_create(
    n4m_aug_spike_noise_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    int32_t n_spikes_min,
    int32_t n_spikes_max,
    double  amplitude_min,
    double  amplitude_max) {
    if (out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr) return N4M_ERR_NULL_POINTER;
    if (n_spikes_min < 0 || n_spikes_max < n_spikes_min) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (!(amplitude_max >= amplitude_min)) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    try {
        n4m_aug_spike_noise_state_t* s = n4m_aug_spike_noise_state_new(
            &rng->engine, n_spikes_min, n_spikes_max,
            amplitude_min, amplitude_max);
        if (s == nullptr) return N4M_ERR_OUT_OF_MEMORY;
        n4m_aug_spike_noise_handle_t* h =
            new (std::nothrow) n4m_aug_spike_noise_handle_t{s};
        if (h == nullptr) {
            n4m_aug_spike_noise_state_free(s);
            return N4M_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API void n4m_aug_spike_noise_destroy(n4m_aug_spike_noise_handle_t* h) {
    if (h == nullptr) return;
    try {
        n4m_aug_spike_noise_state_free(h->state);
        delete h;
    } catch (...) {}
}

N4M_API n4m_status_t n4m_aug_spike_noise_apply(
    const n4m_aug_spike_noise_handle_t* h,
    n4m_matrix_view_t X, n4m_matrix_view_t out) {
    if (h == nullptr) return N4M_ERR_NULL_POINTER;
    const double* X_ptr = nullptr;
    double* out_ptr = nullptr;
    std::int64_t rows = 0, cols = 0;
    const n4m_status_t s = validate_apply_views(X, out, X_ptr, out_ptr, rows, cols);
    if (s != N4M_OK) return s;
    try {
        return n4m_aug_spike_noise_state_apply(h->state, X_ptr, rows, cols, out_ptr);
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

// ===========================================================================
// 4. HeteroscedasticNoiseAugmenter
// ===========================================================================

N4M_API n4m_status_t n4m_aug_hetero_noise_create(
    n4m_aug_hetero_noise_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double noise_base,
    double noise_signal_dep) {
    if (out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr) return N4M_ERR_NULL_POINTER;
    if (!(noise_base >= 0.0))       return N4M_ERR_INVALID_ARGUMENT;
    if (!(noise_signal_dep >= 0.0)) return N4M_ERR_INVALID_ARGUMENT;
    try {
        n4m_aug_hetero_noise_state_t* s = n4m_aug_hetero_noise_state_new(
            &rng->engine, noise_base, noise_signal_dep);
        if (s == nullptr) return N4M_ERR_OUT_OF_MEMORY;
        n4m_aug_hetero_noise_handle_t* h =
            new (std::nothrow) n4m_aug_hetero_noise_handle_t{s};
        if (h == nullptr) {
            n4m_aug_hetero_noise_state_free(s);
            return N4M_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API void n4m_aug_hetero_noise_destroy(n4m_aug_hetero_noise_handle_t* h) {
    if (h == nullptr) return;
    try {
        n4m_aug_hetero_noise_state_free(h->state);
        delete h;
    } catch (...) {}
}

N4M_API n4m_status_t n4m_aug_hetero_noise_apply(
    const n4m_aug_hetero_noise_handle_t* h,
    n4m_matrix_view_t X, n4m_matrix_view_t out) {
    if (h == nullptr) return N4M_ERR_NULL_POINTER;
    const double* X_ptr = nullptr;
    double* out_ptr = nullptr;
    std::int64_t rows = 0, cols = 0;
    const n4m_status_t s = validate_apply_views(X, out, X_ptr, out_ptr, rows, cols);
    if (s != N4M_OK) return s;
    try {
        return n4m_aug_hetero_noise_state_apply(h->state, X_ptr, rows, cols, out_ptr);
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

// ===========================================================================
// 5. LinearBaselineDrift
// ===========================================================================

N4M_API n4m_status_t n4m_aug_linear_drift_create(
    n4m_aug_linear_drift_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double offset_min, double offset_max,
    double slope_min,  double slope_max) {
    if (out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr) return N4M_ERR_NULL_POINTER;
    if (!(offset_max >= offset_min)) return N4M_ERR_INVALID_ARGUMENT;
    if (!(slope_max  >= slope_min )) return N4M_ERR_INVALID_ARGUMENT;
    try {
        n4m_aug_linear_drift_state_t* s = n4m_aug_linear_drift_state_new(
            &rng->engine, offset_min, offset_max, slope_min, slope_max);
        if (s == nullptr) return N4M_ERR_OUT_OF_MEMORY;
        n4m_aug_linear_drift_handle_t* h =
            new (std::nothrow) n4m_aug_linear_drift_handle_t{s};
        if (h == nullptr) {
            n4m_aug_linear_drift_state_free(s);
            return N4M_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API void n4m_aug_linear_drift_destroy(n4m_aug_linear_drift_handle_t* h) {
    if (h == nullptr) return;
    try {
        n4m_aug_linear_drift_state_free(h->state);
        delete h;
    } catch (...) {}
}

N4M_API n4m_status_t n4m_aug_linear_drift_apply(
    const n4m_aug_linear_drift_handle_t* h,
    n4m_matrix_view_t X, n4m_matrix_view_t out) {
    if (h == nullptr) return N4M_ERR_NULL_POINTER;
    const double* X_ptr = nullptr;
    double* out_ptr = nullptr;
    std::int64_t rows = 0, cols = 0;
    const n4m_status_t s = validate_apply_views(X, out, X_ptr, out_ptr, rows, cols);
    if (s != N4M_OK) return s;
    try {
        return n4m_aug_linear_drift_state_apply(h->state, X_ptr, rows, cols, out_ptr);
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

// ===========================================================================
// 6. PolynomialBaselineDrift
// ===========================================================================

N4M_API n4m_status_t n4m_aug_poly_drift_create(
    n4m_aug_poly_drift_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    int32_t degree,
    const double* coeff_min,
    const double* coeff_max) {
    if (out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr) return N4M_ERR_NULL_POINTER;
    if (degree < 0) return N4M_ERR_INVALID_ARGUMENT;
    if (coeff_min == nullptr || coeff_max == nullptr) return N4M_ERR_NULL_POINTER;
    for (int32_t k = 0; k <= degree; ++k) {
        if (!(coeff_max[k] >= coeff_min[k])) return N4M_ERR_INVALID_ARGUMENT;
    }
    try {
        n4m_aug_poly_drift_state_t* s = n4m_aug_poly_drift_state_new(
            &rng->engine, degree, coeff_min, coeff_max);
        if (s == nullptr) return N4M_ERR_OUT_OF_MEMORY;
        n4m_aug_poly_drift_handle_t* h =
            new (std::nothrow) n4m_aug_poly_drift_handle_t{s};
        if (h == nullptr) {
            n4m_aug_poly_drift_state_free(s);
            return N4M_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API void n4m_aug_poly_drift_destroy(n4m_aug_poly_drift_handle_t* h) {
    if (h == nullptr) return;
    try {
        n4m_aug_poly_drift_state_free(h->state);
        delete h;
    } catch (...) {}
}

N4M_API n4m_status_t n4m_aug_poly_drift_apply(
    const n4m_aug_poly_drift_handle_t* h,
    n4m_matrix_view_t X, n4m_matrix_view_t out) {
    if (h == nullptr) return N4M_ERR_NULL_POINTER;
    const double* X_ptr = nullptr;
    double* out_ptr = nullptr;
    std::int64_t rows = 0, cols = 0;
    const n4m_status_t s = validate_apply_views(X, out, X_ptr, out_ptr, rows, cols);
    if (s != N4M_OK) return s;
    try {
        return n4m_aug_poly_drift_state_apply(h->state, X_ptr, rows, cols, out_ptr);
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

// ===========================================================================
// 7. PathLengthAugmenter
// ===========================================================================

N4M_API n4m_status_t n4m_aug_path_length_create(
    n4m_aug_path_length_handle_t** out,
    n4m_rng_pcg64_state_t* rng,
    double path_length_std,
    double min_path_length) {
    if (out == nullptr) return N4M_ERR_NULL_POINTER;
    *out = nullptr;
    if (rng == nullptr) return N4M_ERR_NULL_POINTER;
    if (!(path_length_std >= 0.0)) return N4M_ERR_INVALID_ARGUMENT;
    try {
        n4m_aug_path_length_state_t* s = n4m_aug_path_length_state_new(
            &rng->engine, path_length_std, min_path_length);
        if (s == nullptr) return N4M_ERR_OUT_OF_MEMORY;
        n4m_aug_path_length_handle_t* h =
            new (std::nothrow) n4m_aug_path_length_handle_t{s};
        if (h == nullptr) {
            n4m_aug_path_length_state_free(s);
            return N4M_ERR_OUT_OF_MEMORY;
        }
        *out = h;
        return N4M_OK;
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

N4M_API void n4m_aug_path_length_destroy(n4m_aug_path_length_handle_t* h) {
    if (h == nullptr) return;
    try {
        n4m_aug_path_length_state_free(h->state);
        delete h;
    } catch (...) {}
}

N4M_API n4m_status_t n4m_aug_path_length_apply(
    const n4m_aug_path_length_handle_t* h,
    n4m_matrix_view_t X, n4m_matrix_view_t out) {
    if (h == nullptr) return N4M_ERR_NULL_POINTER;
    const double* X_ptr = nullptr;
    double* out_ptr = nullptr;
    std::int64_t rows = 0, cols = 0;
    const n4m_status_t s = validate_apply_views(X, out, X_ptr, out_ptr, rows, cols);
    if (s != N4M_OK) return s;
    try {
        return n4m_aug_path_length_state_apply(h->state, X_ptr, rows, cols, out_ptr);
    } catch (...) {
        return N4M_ERR_INTERNAL;
    }
}

}  // extern "C"
