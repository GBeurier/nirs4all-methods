/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Shared random utilities for n4m_aug_* augmenters.
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
#include <stdlib.h>
#include <string.h>

double n4m_aug_rng_next_double(n4m_rng_pcg64* rng) {
    return n4m_pcg64_engine_next_double(rng);
}

uint64_t n4m_aug_rng_uint64(n4m_rng_pcg64* rng) {
    return n4m_pcg64_engine_next_uint64(rng);
}

double n4m_aug_rng_uniform(n4m_rng_pcg64* rng, double low, double high) {
    const double u = n4m_pcg64_engine_next_double(rng);
    return low + (high - low) * u;
}

void n4m_aug_rng_uniform_fill(n4m_rng_pcg64* rng,
                              double low, double high,
                              double* out, size_t n) {
    const double range = high - low;
    for (size_t i = 0; i < n; ++i) {
        out[i] = low + range * n4m_pcg64_engine_next_double(rng);
    }
}

double n4m_aug_rng_standard_normal(n4m_rng_pcg64* rng) {
    double v;
    n4m_pcg64_engine_standard_normal_fill(rng, &v, 1u);
    return v;
}

void n4m_aug_rng_normal_fill(n4m_rng_pcg64* rng,
                             double loc, double scale,
                             double* out, size_t n) {
    n4m_pcg64_engine_standard_normal_fill(rng, out, n);
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
uint64_t n4m_aug_rng_integers(n4m_rng_pcg64* rng, uint64_t n) {
    if (n <= 1u) {
        return 0u;
    }
    /* [0, n) == [0, n-1] inclusive. Delegate to the parity-correct bounded
     * sampler (buffered uint32 Lemire) so this matches numpy's
     * Generator.integers(0, n) bit-for-bit; the earlier masked-rejection
     * implementation used the low bits and did NOT match numpy. */
    return n4m_aug_rng_bounded(rng, n - 1u);
}

/* Beta sampler — Cheng's BB algorithm for min(a, b) > 1, BC for min < 1.
 * Matches NumPy's `legacy_beta` (numpy/random/src/distributions/distributions.c),
 * which is the algorithm invoked by `Generator.beta`. The implementation
 * advances the PCG64 state via two `next_double()` draws per inner loop. */
double n4m_aug_rng_beta(n4m_rng_pcg64* rng, double a, double b) {
    if (a <= 0.0 || b <= 0.0) {
        return 0.5;  /* defensive — caller should validate */
    }
    if (a <= 1.0 && b <= 1.0) {
        /* Joehnk's algorithm (BC variant). */
        for (;;) {
            const double u = n4m_pcg64_engine_next_double(rng);
            const double v = n4m_pcg64_engine_next_double(rng);
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
        const double u1 = n4m_pcg64_engine_next_double(rng);
        const double u2 = n4m_pcg64_engine_next_double(rng);
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

void n4m_aug_rng_permutation(n4m_rng_pcg64* rng, int64_t* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = i;
    }
    /* Uniform reverse Fisher-Yates over the parity-correct bounded sampler.
     * This is a correct uniform shuffle but NOT bit-identical to numpy's
     * Generator.permutation (which uses a different internal shuffle). */
    for (int64_t i = n - 1; i > 0; --i) {
        const uint64_t j = n4m_aug_rng_integers(rng, (uint64_t)(i + 1));
        const int64_t  tmp = out[i];
        out[i]   = out[(int64_t)j];
        out[(int64_t)j] = tmp;
    }
}

uint64_t n4m_aug_rng_bounded(n4m_rng_pcg64* rng, uint64_t rng_incl) {
    if (rng_incl == 0u) {
        return 0u;
    }
    /* Buffered uint32 Lemire — NumPy's bounded_lemire_uint32. m = number of
     * representable values; result is the high 32 bits of (uint32 * m). */
    const uint64_t m = rng_incl + 1u;            /* <= 2^32 for our inputs */
    uint64_t prod = (uint64_t)n4m_pcg64_engine_next_uint32(rng) * m;
    uint32_t leftover = (uint32_t)prod;
    if ((uint64_t)leftover < m) {
        /* threshold = (2^32 - m) % m, computed in 64-bit to avoid overflow. */
        const uint32_t threshold = (uint32_t)(((uint64_t)1u << 32) % m);
        while ((uint64_t)leftover < (uint64_t)threshold) {
            prod = (uint64_t)n4m_pcg64_engine_next_uint32(rng) * m;
            leftover = (uint32_t)prod;
        }
    }
    return prod >> 32;
}

/* smallest 2^k - 1 >= x — NumPy's _gen_mask. */
static uint64_t n4m_aug_gen_mask(uint64_t x) {
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return x;
}

/* Reverse partial Fisher-Yates over data[first .. n), drawing each swap index
 * via random_bounded_uint64(bitgen, 0, i, 0, 0) == n4m_aug_rng_integers(i + 1).
 * Mirrors NumPy's _shuffle_int (with first=1 the whole array is shuffled). */
static void n4m_aug_shuffle_int(n4m_rng_pcg64* rng, int64_t n, int64_t first,
                                int64_t* data) {
    for (int64_t i = n - 1; i >= first; --i) {
        const int64_t j = (int64_t)n4m_aug_rng_bounded(rng, (uint64_t)i);
        const int64_t tmp = data[j];
        data[j] = data[i];
        data[i] = tmp;
    }
}

int n4m_aug_rng_choice_no_replace(n4m_rng_pcg64* rng, int64_t pop_size,
                                  int64_t size, int64_t* out) {
    if (rng == NULL || out == NULL) return -1;
    if (size < 0 || pop_size < 0 || size > pop_size) return -1;
    if (size == 0) return 0;

    /* Large-population tail-shuffle path (cutoff = 50 for the default
     * shuffle=True). Shuffle the tail `size` slots of a full arange, then
     * take the last `size`. */
    if (pop_size > 10000 && size > pop_size / 50) {
        int64_t* idx = (int64_t*)malloc((size_t)pop_size * sizeof(int64_t));
        if (idx == NULL) return -1;
        for (int64_t i = 0; i < pop_size; ++i) idx[i] = i;
        int64_t first = pop_size - size;
        if (first < 1) first = 1;
        n4m_aug_shuffle_int(rng, pop_size, first, idx);
        memcpy(out, idx + (pop_size - size), (size_t)size * sizeof(int64_t));
        free(idx);
        return 0;
    }

    /* Floyd's algorithm with an open-addressed hash set. set_size is the
     * smallest power of two strictly larger than 1.2 * size. */
    const uint64_t EMPTY = (uint64_t)-1;
    uint64_t set_size = (uint64_t)(1.2 * (double)size);
    const uint64_t mask = n4m_aug_gen_mask(set_size);
    set_size = 1u + mask;
    uint64_t* hash_set = (uint64_t*)malloc((size_t)set_size * sizeof(uint64_t));
    if (hash_set == NULL) return -1;
    for (uint64_t i = 0; i < set_size; ++i) hash_set[i] = EMPTY;

    for (int64_t j = pop_size - size; j < pop_size; ++j) {
        const uint64_t val = n4m_aug_rng_bounded(rng, (uint64_t)j); /* [0, j] */
        uint64_t loc = val & mask;
        while (hash_set[loc] != EMPTY && hash_set[loc] != val) {
            loc = (loc + 1u) & mask;
        }
        if (hash_set[loc] == EMPTY) {
            hash_set[loc] = val;
            out[j - pop_size + size] = (int64_t)val;
        } else {
            /* val already chosen — insert j instead (Floyd's substitution). */
            loc = (uint64_t)j & mask;
            while (hash_set[loc] != EMPTY) {
                loc = (loc + 1u) & mask;
            }
            hash_set[loc] = (uint64_t)j;
            out[j - pop_size + size] = j;
        }
    }
    free(hash_set);

    /* Final shuffle (shuffle=True). */
    n4m_aug_shuffle_int(rng, size, 1, out);
    return 0;
}
