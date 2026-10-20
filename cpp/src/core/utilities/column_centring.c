/* SPDX-License-Identifier: CECILL-2.1 */
/* Corrected two-pass column centring at one power-of-two scale
 * (column_centring.h). */

#include "column_centring.h"

#include <math.h>
#include <stddef.h>

n4m_status_t n4m_util_centre_columns_pow2(const double* X, int64_t n, int64_t p, double* Xc,
                                          int* scale_exp) {
    const size_t un = (size_t)n, up = (size_t)p;
    int g = 0;
    int varying = 0;
    for (size_t j = 0; j < up; ++j) {
        double maxabs = 0.0;
        for (size_t i = 0; i < un; ++i) {
            const double x = X[i * up + j];
            if (!isfinite(x)) return N4M_ERR_NUMERICAL_FAILURE;
            if (fabs(x) > maxabs) maxabs = fabs(x);
        }
        int e = 0;
        if (maxabs > 0.0) (void)frexp(maxabs, &e);
        double s = 0.0;
        for (size_t i = 0; i < un; ++i) {
            Xc[i * up + j] = ldexp(X[i * up + j], -e);
            s += Xc[i * up + j];
        }
        const double m = s / (double)n;
        double sr = 0.0;
        for (size_t i = 0; i < un; ++i) {
            Xc[i * up + j] -= m;
            sr += Xc[i * up + j];
        }
        const double r = sr / (double)n;
        double cmax = 0.0;
        for (size_t i = 0; i < un; ++i) {
            Xc[i * up + j] -= r;
            if (fabs(Xc[i * up + j]) > cmax) cmax = fabs(Xc[i * up + j]);
        }
        if (cmax > 0.0) {
            int ce = 0;
            (void)frexp(cmax, &ce);
            if (!varying || e + ce > g) g = e + ce;
            varying = 1;
        }
    }
    /* Second pass: column j, centred at scale 2^e_j, goes to the common 2^g.
     * e_j is recomputed from the input, which needs no scratch allocation. */
    for (size_t j = 0; j < up; ++j) {
        double maxabs = 0.0;
        for (size_t i = 0; i < un; ++i)
            if (fabs(X[i * up + j]) > maxabs) maxabs = fabs(X[i * up + j]);
        int e = 0;
        if (maxabs > 0.0) (void)frexp(maxabs, &e);
        for (size_t i = 0; i < un; ++i) Xc[i * up + j] = ldexp(Xc[i * up + j], e - g);
    }
    *scale_exp = g;
    return N4M_OK;
}
