/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * C translation of the FITPACK (P. Dierckx) routines behind
 * scipy.interpolate.UnivariateSpline: curfit (smoothing, iopt 0/1, s > 0)
 * and splev. Translated from SciPy 1.17.1 scipy/interpolate/fitpack/;
 * see LICENSE.scipy.txt in this directory. INTERNAL.
 */
#ifndef N4M_CORE_COMMON_VENDORED_FITPACK_H
#define N4M_CORE_COMMON_VENDORED_FITPACK_H

#ifdef __cplusplus
extern "C" {
#endif

/* curfit with iopt in {0, 1} and s > 0. The caller guarantees FITPACK's input
 * restrictions: 1 <= k <= 5, m > k, 2k+2 <= nest, xb <= x[0] <= ... <=
 * x[m-1] <= xe. wrk holds m*(k+1) + nest*(7+3k) doubles and iwrk nest ints;
 * both, t and n carry state from an iopt=0 call into a following iopt=1 call.
 * Returns FITPACK's ier. */
int n4m_fitpack_curfit(int iopt, int m, const double* x, const double* y,
                       const double* w, double xb, double xe, int k, double s,
                       int nest, int* n, double* t, double* c, double* fp,
                       double* wrk, int* iwrk);

/* splev with e = 0 (points outside the knot span are extrapolated). */
void n4m_fitpack_splev(const double* t, int n, const double* c, int k,
                       const double* x, double* y, int m);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_COMMON_VENDORED_FITPACK_H */
