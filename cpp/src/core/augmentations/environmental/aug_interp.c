/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * 1-D linear interpolation matching np.interp default behavior.
 */

#include "aug_interp.h"

/* Binary search: returns the largest k such that xp[k] <= q. If q < xp[0]
 * returns -1. If q >= xp[n_xp - 1] returns n_xp - 1. */
static int64_t bsearch_xp(const double* xp, int64_t n_xp, double q) {
    if (q < xp[0]) return -1;
    if (q >= xp[n_xp - 1]) return n_xp - 1;
    int64_t lo = 0, hi = n_xp - 1;
    while (hi - lo > 1) {
        const int64_t mid = (lo + hi) / 2;
        if (xp[mid] <= q) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

void c4a_aug_np_interp(const double* x,  int64_t n_x,
                       const double* xp, const double* fp, int64_t n_xp,
                       double* out) {
    if (n_xp <= 0) {
        for (int64_t i = 0; i < n_x; ++i) {
            out[i] = 0.0;
        }
        return;
    }
    if (n_xp == 1) {
        for (int64_t i = 0; i < n_x; ++i) {
            out[i] = fp[0];
        }
        return;
    }
    for (int64_t i = 0; i < n_x; ++i) {
        const double q = x[i];
        const int64_t k = bsearch_xp(xp, n_xp, q);
        if (k < 0) {
            out[i] = fp[0];
        } else if (k == n_xp - 1) {
            out[i] = fp[n_xp - 1];
        } else {
            const double x0 = xp[k];
            const double x1 = xp[k + 1];
            const double dx = x1 - x0;
            if (dx == 0.0) {
                /* Coincident knots: NumPy returns fp[k]. */
                out[i] = fp[k];
            } else {
                const double t = (q - x0) / dx;
                out[i] = fp[k] + t * (fp[k + 1] - fp[k]);
            }
        }
    }
}
