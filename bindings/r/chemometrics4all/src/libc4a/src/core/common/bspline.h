/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal cubic B-spline helpers for chemometrics4all.
 *
 * Vendored minimal implementation of the cubic B-spline interpolation needed
 * by the Phase 18 Spline_* augmenters. SciPy's `interpolate.splrep` /
 * `interpolate.BSpline` form the public Python reference; the routines below
 * cover only the very narrow shape we need:
 *
 *   - splrep_natural_cubic(x, y, n, knots_out, coef_out)
 *       Build a natural cubic spline through (x[i], y[i]) on a strictly
 *       monotonically increasing x. The output is a (knots, coefficients)
 *       pair compatible with the De Boor evaluation in bspline_eval(). Knots
 *       are augmented at each end with three additional clamped copies so
 *       the boundary condition matches the "not-a-knot" cubic produced by
 *       scipy.interpolate.splrep(..., s=0, k=3) for the same inputs.
 *
 *   - bspline_eval(...)
 *       Evaluate the cubic spline at an arbitrary point. Returns 0 for
 *       points outside [x[0], x[n-1]] when extrapolate=0 (mirrors
 *       BSpline(extrapolate=False) used by Spline_Y_Perturbations and the
 *       simplification augmenters), or clamps to the nearest interior
 *       interval when extrapolate=1.
 *
 *   - bspline_eval_array(...)
 *       Convenience wrapper that calls bspline_eval() for every output
 *       sample.
 *
 * Pure C, no external dependencies. INTERNAL.
 */
#ifndef CHEMOMETRICS4ALL_CORE_COMMON_BSPLINE_H
#define CHEMOMETRICS4ALL_CORE_COMMON_BSPLINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Build a natural cubic spline through (x[i], y[i]) for i in 0..n-1.
 *
 * On success:
 *   - knots_out[0..n+5] holds the augmented knot vector (length n + 6).
 *     The first 4 knots equal x[0] and the last 4 knots equal x[n-1] for
 *     the natural endpoint condition.
 *   - coef_out[0..n+1] holds the B-spline coefficients (length n + 2).
 *
 * Both buffers must be sized by the caller as documented. Returns 0 on
 * success, non-zero on failure (n < 4, non-monotonic x, or NULL pointer). */
int c4a_bspline_build_natural(const double* x, const double* y, int32_t n,
                              double* knots_out, double* coef_out);

/* Evaluate the cubic B-spline at point xq.
 *
 * Parameters:
 *   knots    — length n + 6 augmented knot vector returned by
 *              c4a_bspline_build_natural()
 *   coef     — length n + 2 coefficient vector returned by
 *              c4a_bspline_build_natural()
 *   n        — original number of input samples
 *   xq       — query point
 *   extrapolate — 0: return 0.0 outside [x[0], x[n-1]]
 *                  1: clamp xq into [x[0], x[n-1]]
 *
 * Returns the interpolated value. */
double c4a_bspline_eval(const double* knots, const double* coef, int32_t n,
                        double xq, int extrapolate);

/* Evaluate the spline at every xq[i] and write into out[i]. */
void c4a_bspline_eval_array(const double* knots, const double* coef,
                            int32_t n, const double* xq, int32_t nq,
                            int extrapolate, double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_COMMON_BSPLINE_H */
