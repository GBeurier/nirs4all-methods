/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Shared implementation for the two spline-simplification augmenters
 * (Spline_X_Simplification and Spline_Curve_Simplification). Both reduce each
 * spectrum to `nb_points` control points and refit a cubic interpolating
 * B-spline (scipy.interpolate.splrep(s=0, k=3) -> the not-a-knot surrogate in
 * bspline.h), evaluated at every wavelength with extrapolate=False.
 *
 * The two operators are identical in the random (non-uniform) path; they
 * differ ONLY in the uniform path, where Spline_Curve_Simplification applies a
 * final np.unique to the linspace control indices and Spline_X_Simplification
 * does not (`unique_in_uniform`). INTERNAL.
 */
#ifndef N4M_CORE_AUG_SPLINE_SIMPLIFY_COMMON_H
#define N4M_CORE_AUG_SPLINE_SIMPLIFY_COMMON_H

#include <stdint.h>

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Apply the simplification to a (rows x cols) row-major matrix `X`, writing the
 * (rows x cols) result to `out`. `rng_void` is an n4m_rng_pcg64* (consumed only
 * in the non-uniform path). `spline_points <= 0` selects the reference default
 * of cols / 4. `uniform` picks evenly-spaced control points; `unique_in_uniform`
 * applies np.unique to them (Curve) or not (X). Returns N4M_OK on success. */
n4m_status_t n4m_spline_simplify_apply_impl(void* rng_void, const double* X,
                                            int64_t rows, int64_t cols,
                                            int32_t spline_points, int uniform,
                                            int unique_in_uniform, double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_AUG_SPLINE_SIMPLIFY_COMMON_H */
