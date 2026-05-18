/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Shared random utilities for c4a_aug_* augmenters.
 *
 * All primitives consume the PCG64 handle's stream and advance it. The
 * algorithms below follow NumPy 1.26.4 reference semantics as closely as
 * practical given the public bit_generator surface — bit-exact parity is
 * available for uniform, standard_normal, uint64; the Beta sampler uses
 * Cheng's BB/BC algorithm which matches NumPy's `legacy_beta` in
 * numpy/random/src/distributions/distributions.c.
 */

#include "aug_rng_utils.h"

#include <math.h>

double c4a_aug_rng_next_double(c4a_rng_pcg64* rng) {
    return c4a_pcg64_engine_next_double(rng);
}

uint64_t c4a_aug_rng_uint64(c4a_rng_pcg64* rng) {
    return c4a_pcg64_engine_next_uint64(rng);
}

double c4a_aug_rng_uniform(c4a_rng_pcg64* rng, double low, double high) {
    const double u = c4a_pcg64_engine_next_double(rng);
    return low + (high - low) * u;
}

void c4a_aug_rng_uniform_fill(c4a_rng_pcg64* rng,
                              double low, double high,
                              double* out, size_t n) {
    const double range = high - low;
    for (size_t i = 0; i < n; ++i) {
        out[i] = low + range * c4a_pcg64_engine_next_double(rng);
    }
}

double c4a_aug_rng_standard_normal(c4a_rng_pcg64* rng) {
    double v;
    c4a_pcg64_engine_standard_normal_fill(rng, &v, 1u);
    return v;
}

void c4a_aug_rng_normal_fill(c4a_rng_pcg64* rng,
                             double loc, double scale,
                             double* out, size_t n) {
    c4a_pcg64_engine_standard_normal_fill(rng, out, n);
    if (loc == 0.0 && scale == 1.0) {
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        out[i] = loc + scale * out[i];
    }
}

/* NumPy `Generator.integers(0, n)` rejection algorithm
 * (Lemire's masked rejection, see
 *  numpy/random/src/distributions/distributions.c :: random_bounded_uint64).
 *
 * For our augmenters all sample-count ranges fit comfortably below 2^32 so
 * we use the standard Lemire-with-mask path. */
uint64_t c4a_aug_rng_integers(c4a_rng_pcg64* rng, uint64_t n) {
    if (n <= 1u) {
        return 0u;
    }
    /* mask = smallest 2^k - 1 >= n - 1 */
    uint64_t mask = n - 1u;
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    mask |= mask >> 32;
    for (;;) {
        uint64_t r = c4a_pcg64_engine_next_uint64(rng) & mask;
        if (r < n) {
            return r;
        }
    }
}

/* Beta sampler — Cheng's BB algorithm for min(a, b) > 1, BC for min < 1.
 * Matches NumPy's `legacy_beta` (numpy/random/src/distributions/distributions.c),
 * which is the algorithm invoked by `Generator.beta`. The implementation
 * advances the PCG64 state via two `next_double()` draws per inner loop. */
double c4a_aug_rng_beta(c4a_rng_pcg64* rng, double a, double b) {
    if (a <= 0.0 || b <= 0.0) {
        return 0.5;  /* defensive — caller should validate */
    }
    if (a <= 1.0 && b <= 1.0) {
        /* Joehnk's algorithm (BC variant). */
        for (;;) {
            const double u = c4a_pcg64_engine_next_double(rng);
            const double v = c4a_pcg64_engine_next_double(rng);
            const double x = pow(u, 1.0 / a);
            const double y = pow(v, 1.0 / b);
            const double s = x + y;
            if (s <= 1.0 && s > 0.0) {
                /* If x+y is a sub-normal or zero, retry to avoid div-by-zero. */
                if (s >= 1e-300) {
                    return x / s;
                }
                /* very rare path: log-space alternative */
                const double lu = log(u) / a;
                const double lv = log(v) / b;
                const double m  = (lu > lv) ? lu : lv;
                const double el = exp(lu - m) + exp(lv - m);
                return exp(lu - m - log(el));
            }
        }
    }
    /* Cheng's BB for min(a, b) > 1. */
    double aa, bb;
    int swap = 0;
    if (a < b) { aa = a; bb = b; }
    else { aa = b; bb = a; swap = 1; }
    const double alpha  = aa + bb;
    const double beta   = sqrt((alpha - 2.0) / (2.0 * aa * bb - alpha));
    const double gamma  = aa + 1.0 / beta;
    for (;;) {
        const double u1 = c4a_pcg64_engine_next_double(rng);
        const double u2 = c4a_pcg64_engine_next_double(rng);
        const double v  = beta * log(u1 / (1.0 - u1));
        const double w  = aa * exp(v);
        const double z  = u1 * u1 * u2;
        const double r  = gamma * v - 1.3862944;
        const double s  = aa + r - w;
        if (s + 2.609438 >= 5.0 * z) {
            const double result = w / (bb + w);
            return swap ? 1.0 - result : result;
        }
        const double t = log(z);
        if (s >= t) {
            const double result = w / (bb + w);
            return swap ? 1.0 - result : result;
        }
        if (r + alpha * log(alpha / (bb + w)) >= t) {
            const double result = w / (bb + w);
            return swap ? 1.0 - result : result;
        }
    }
}

void c4a_aug_rng_permutation(c4a_rng_pcg64* rng, int64_t* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = i;
    }
    /* NumPy default_rng().permutation uses Fisher-Yates with integers(0, i+1)
     * iterating downward. The exact NumPy implementation walks `i` from
     * (n-1) down to 1 and swaps out[i] with out[integers(0, i+1)]. */
    for (int64_t i = n - 1; i > 0; --i) {
        const uint64_t j = c4a_aug_rng_integers(rng, (uint64_t)(i + 1));
        const int64_t  tmp = out[i];
        out[i]   = out[(int64_t)j];
        out[(int64_t)j] = tmp;
    }
}
