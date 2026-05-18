/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Reference implementation of c4a_fck_kernel_1d. Mirrors
 * nirs4all.operators.models.sklearn.fckpls.fractional_kernel_1d step-for-step
 * so the output is bit-identical when the inputs match.
 *
 * One subtlety worth noting:
 *   - `sign(0) == 0` (numpy convention). The centre index has |idx| == 0, so
 *     even without the `frac[center] = 0` guard the centre is multiplied by
 *     zero. The explicit guard exists in numpy because `0 ** alpha` may be
 *     either 0.0, 1.0, or nan depending on `alpha`; we add the equivalent
 *     `if |idx| < 1e-10 then frac = 0` clamp to stay bit-identical.
 */

#include "fck_kernel.h"

#include <math.h>
#include <stddef.h>

static const double kFckAlphaZeroEpsilon = 1e-10;   /* alpha-is-zero cutoff */
static const double kFckIdxEpsilon       = 1e-10;   /* centre-index cutoff */
static const double kFckAlphaMeanCutoff  = 0.1;     /* zero-mean threshold */
static const double kFckSigmaFloor       = 1e-6;
static const double kFckNormFloor        = 1e-12;

C4A_API int c4a_fck_kernel_1d(double alpha, double sigma, int32_t kernel_size,
                               double* out) {
    if (out == NULL || kernel_size <= 0) {
        return -1;
    }

    const double sigma_safe = (sigma < kFckSigmaFloor) ? kFckSigmaFloor : sigma;
    const double inv_two_sigma2 = 0.5 / (sigma_safe * sigma_safe);
    const double half = (double)(kernel_size - 1) * 0.5;

    if (alpha < kFckAlphaZeroEpsilon) {
        /* Pure Gaussian envelope. */
        for (int32_t i = 0; i < kernel_size; ++i) {
            const double idx = (double)i - half;
            out[i] = exp(-inv_two_sigma2 * idx * idx);
        }
    } else {
        /* Gaussian * sign(idx) * |idx|^alpha (with idx ~ 0 clamped to 0). */
        for (int32_t i = 0; i < kernel_size; ++i) {
            const double idx       = (double)i - half;
            const double gaussian  = exp(-inv_two_sigma2 * idx * idx);
            const double abs_idx   = fabs(idx);
            double frac;
            if (abs_idx < kFckIdxEpsilon) {
                frac = 0.0;
            } else {
                frac = pow(abs_idx, alpha);
            }
            double sign;
            if (idx > 0.0)      sign =  1.0;
            else if (idx < 0.0) sign = -1.0;
            else                sign =  0.0;
            out[i] = gaussian * frac * sign;
        }
    }

    /* Zero-mean correction for alpha > 0.1 (matches Python guard). */
    if (alpha > kFckAlphaMeanCutoff) {
        double mean = 0.0;
        for (int32_t i = 0; i < kernel_size; ++i) {
            mean += out[i];
        }
        mean /= (double)kernel_size;
        for (int32_t i = 0; i < kernel_size; ++i) {
            out[i] -= mean;
        }
    }

    /* L1 normalisation (skip if too small to avoid numerical blow-up). */
    double norm = 0.0;
    for (int32_t i = 0; i < kernel_size; ++i) {
        norm += fabs(out[i]);
    }
    if (norm > kFckNormFloor) {
        const double inv_norm = 1.0 / norm;
        for (int32_t i = 0; i < kernel_size; ++i) {
            out[i] *= inv_norm;
        }
    }

    return 0;
}
