/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Reference implementation of n4m_fck_kernel_1d. Mirrors
 * nirs4all.operators.transforms.fck_static._build_kernel step-for-step so
 * the output is bit-identical when the inputs match.
 */

#include "fck_kernel.h"

#include <math.h>
#include <stddef.h>

static const double kFckAlphaGaussianCutoff = 0.1;
static const double kFckPowerEpsilon        = 1e-8;
static const double kFckSigmaFloor       = 1e-6;

int n4m_fck_kernel_1d(double alpha, double scale, double sigma,
                       int32_t kernel_size, double* out) {
    if (out == NULL || kernel_size < 3 || (kernel_size % 2) == 0 ||
        !(scale > 0.0)) {
        return -1;
    }

    const double sigma_safe = (sigma < kFckSigmaFloor) ? kFckSigmaFloor : sigma;
    const double inv_two_sigma2 = 0.5 / (sigma_safe * sigma_safe);
    const int32_t half = kernel_size / 2;

    if (alpha < kFckAlphaGaussianCutoff) {
        for (int32_t i = 0; i < kernel_size; ++i) {
            const double idx = ((double)i - (double)half) * scale;
            out[i] = exp(-inv_two_sigma2 * idx * idx);
        }
    } else {
        double mean = 0.0;
        for (int32_t i = 0; i < kernel_size; ++i) {
            const double idx       = ((double)i - (double)half) * scale;
            const double gaussian  = exp(-inv_two_sigma2 * idx * idx);
            double sign;
            if (idx > 0.0)      sign =  1.0;
            else if (idx < 0.0) sign = -1.0;
            else                sign =  0.0;
            const double frac = sign * pow(fabs(idx) + kFckPowerEpsilon, alpha);
            out[i] = gaussian * frac;
            mean += out[i];
        }
        mean /= (double)kernel_size;
        for (int32_t i = 0; i < kernel_size; ++i) {
            out[i] -= mean;
        }
    }

    double norm = 0.0;
    for (int32_t i = 0; i < kernel_size; ++i) {
        norm += fabs(out[i]);
    }
    norm += kFckPowerEpsilon;
    const double inv_norm = 1.0 / norm;
    for (int32_t i = 0; i < kernel_size; ++i) {
        out[i] *= inv_norm;
    }

    return 0;
}
