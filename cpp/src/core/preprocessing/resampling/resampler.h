/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for Resampler — wavelength-grid interpolation between
 * a fitted source axis and a configured target axis. The public C ABI lives
 * in c4a.h and is implemented in c_api/c_api_resampling.cpp.
 *
 * Reference: nirs4all.operators.transforms.resampler.Resampler.
 *
 * Lifecycle:
 *   _create(target_wl[n_target], n_target, method, crop_min, crop_max,
 *            use_crop, fill_value, bounds_error, extrapolate)
 *   _fit(source_wl[n_source])    — learns source axis + crop mask
 *   _transform(X, rows, in_cols) — per-row interpolation onto target axis
 *
 * `n_target == -1` selects identity mode (target == source); the operator
 * still requires fit to know the source axis but acts as a pass-through
 * when applied.
 *
 * `method`:
 *   0 — linear  (scipy interp1d 'linear')
 *   1 — nearest (scipy interp1d 'nearest')
 *   2 — cubic   (scipy interp1d 'cubic', not-a-knot natural spline)
 *
 * `use_crop != 0` applies `[crop_min, crop_max]` filter to the source
 * wavelengths before interpolation.
 *
 * `bounds_error != 0` rejects queries outside [src_min, src_max] via
 * C4A_ERR_NUMERICAL_FAILURE. When `bounds_error == 0` and the query is
 * out of range, the engine either uses `fill_value` (if `extrapolate == 0`)
 * or extrapolates per the interpolation method (`extrapolate != 0`).
 */
#ifndef CHEMOMETRICS4ALL_CORE_PP_RESAMPLING_RESAMPLER_H
#define CHEMOMETRICS4ALL_CORE_PP_RESAMPLING_RESAMPLER_H

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_pp_resampler_state_t c4a_pp_resampler_state_t;

/* Allocate a resampler state.
 *   target_wl       — target wavelength array of length n_target. May be NULL
 *                     iff n_target == -1 (identity).
 *   n_target        — target length; -1 selects identity (pass-through).
 *   method          — interpolation method: 0 linear, 1 nearest, 2 cubic.
 *   crop_min/max    — wavelength crop range (inclusive).
 *   use_crop        — 1 enables cropping, 0 disables.
 *   fill_value      — value used for out-of-bounds target wavelengths when
 *                     `bounds_error == 0` and `extrapolate == 0`.
 *   bounds_error    — 1 rejects out-of-range queries.
 *   extrapolate     — 1 uses the interpolation method to extrapolate
 *                     out-of-range queries; 0 uses `fill_value`. */
c4a_pp_resampler_state_t* c4a_pp_resampler_state_new(
    const double* target_wl, int64_t n_target,
    int32_t method,
    double crop_min, double crop_max, int use_crop,
    double fill_value, int bounds_error, int extrapolate);

void c4a_pp_resampler_state_free(c4a_pp_resampler_state_t* state);

int c4a_pp_resampler_state_is_fitted(const c4a_pp_resampler_state_t* state);

/* Fit on the source wavelength axis (length `n_source`). Must be strictly
 * monotonic (ascending). When `use_crop` is set the engine builds an index
 * mask of the source columns that fall inside [crop_min, crop_max]. */
c4a_status_t c4a_pp_resampler_state_fit(c4a_pp_resampler_state_t* state,
                                         const double* source_wl,
                                         int64_t n_source);

/* Expected input column count for `_apply` (post-crop, this is n_source; the
 * crop mask is applied internally during apply). The matrix-view check
 * verifies the raw input has n_source_input == n_source. */
int64_t c4a_pp_resampler_state_input_cols(const c4a_pp_resampler_state_t* s);

/* Output column count for `_apply`. For identity mode this equals the
 * (post-crop) source column count. */
int64_t c4a_pp_resampler_state_output_cols(const c4a_pp_resampler_state_t* s);

c4a_status_t c4a_pp_resampler_state_apply(
    const c4a_pp_resampler_state_t* state,
    const double* X,
    int64_t rows, int64_t cols,
    int64_t out_cols,
    double* out);

#ifdef __cplusplus
}
#endif

#endif /* CHEMOMETRICS4ALL_CORE_PP_RESAMPLING_RESAMPLER_H */
