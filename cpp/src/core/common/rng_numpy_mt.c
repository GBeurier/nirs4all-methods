/* SPDX-License-Identifier: CECILL-2.1 */
/* numpy.random.RandomState-compatible MT19937. See rng_numpy_mt.h. */
#include "rng_numpy_mt.h"

#include <math.h>

#define NPMT_N 624
#define NPMT_M 397
#define NPMT_MATRIX_A   0x9908b0dfUL
#define NPMT_UPPER_MASK 0x80000000UL
#define NPMT_LOWER_MASK 0x7fffffffUL

/* init_genrand(s) — Knuth 2002 1812433253 LCG fill (numpy/Matsumoto). */
static void npmt_init_genrand(n4m_rng_numpy_mt* r, uint32_t s) {
    r->mt[0] = s;
    for (int i = 1; i < NPMT_N; i++) {
        r->mt[i] = (uint32_t)(1812433253UL * (r->mt[i - 1] ^ (r->mt[i - 1] >> 30)) + (uint32_t)i);
    }
    r->mti = NPMT_N;
}

/* init_by_array(key, key_length) — the numpy/Matsumoto array seeding. */
static void npmt_init_by_array(n4m_rng_numpy_mt* r, const uint32_t* key, int key_length) {
    npmt_init_genrand(r, 19650218UL);
    int i = 1, j = 0;
    int k = (NPMT_N > key_length) ? NPMT_N : key_length;
    for (; k; k--) {
        r->mt[i] = (uint32_t)((r->mt[i] ^ ((r->mt[i - 1] ^ (r->mt[i - 1] >> 30)) * 1664525UL))
                              + key[j] + (uint32_t)j);
        i++; j++;
        if (i >= NPMT_N) { r->mt[0] = r->mt[NPMT_N - 1]; i = 1; }
        if (j >= key_length) j = 0;
    }
    for (k = NPMT_N - 1; k; k--) {
        r->mt[i] = (uint32_t)((r->mt[i] ^ ((r->mt[i - 1] ^ (r->mt[i - 1] >> 30)) * 1566083941UL))
                              - (uint32_t)i);
        i++;
        if (i >= NPMT_N) { r->mt[0] = r->mt[NPMT_N - 1]; i = 1; }
    }
    r->mt[0] = 0x80000000UL;  /* MSB is 1; assuring non-zero initial array */
}

void n4m_rng_numpy_mt_seed(n4m_rng_numpy_mt* rng, uint64_t seed) {
    /* numpy RandomState(int) builds the key from the seed's 32-bit little-
     * endian words. A seed < 2^32 → key length 1. */
    uint32_t key[2];
    int key_len;
    if ((seed >> 32) == 0) {
        key[0] = (uint32_t)seed;
        key_len = 1;
    } else {
        key[0] = (uint32_t)(seed & 0xffffffffULL);
        key[1] = (uint32_t)(seed >> 32);
        key_len = 2;
    }
    npmt_init_by_array(rng, key, key_len);
    rng->has_gauss = 0;
    rng->gauss = 0.0;
}

uint32_t n4m_rng_numpy_mt_next_uint32(n4m_rng_numpy_mt* rng) {
    uint32_t y;
    static const uint32_t mag01[2] = {0x0UL, NPMT_MATRIX_A};
    if (rng->mti >= NPMT_N) {
        int kk;
        for (kk = 0; kk < NPMT_N - NPMT_M; kk++) {
            y = (rng->mt[kk] & NPMT_UPPER_MASK) | (rng->mt[kk + 1] & NPMT_LOWER_MASK);
            rng->mt[kk] = rng->mt[kk + NPMT_M] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        for (; kk < NPMT_N - 1; kk++) {
            y = (rng->mt[kk] & NPMT_UPPER_MASK) | (rng->mt[kk + 1] & NPMT_LOWER_MASK);
            rng->mt[kk] = rng->mt[kk + (NPMT_M - NPMT_N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        y = (rng->mt[NPMT_N - 1] & NPMT_UPPER_MASK) | (rng->mt[0] & NPMT_LOWER_MASK);
        rng->mt[NPMT_N - 1] = rng->mt[NPMT_M - 1] ^ (y >> 1) ^ mag01[y & 0x1UL];
        rng->mti = 0;
    }
    y = rng->mt[rng->mti++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);
    return y;
}

double n4m_rng_numpy_mt_double(n4m_rng_numpy_mt* rng) {
    /* numpy legacy random_sample: 53-bit. */
    uint32_t a = n4m_rng_numpy_mt_next_uint32(rng) >> 5;  /* 27 bits */
    uint32_t b = n4m_rng_numpy_mt_next_uint32(rng) >> 6;  /* 26 bits */
    return (a * 67108864.0 + b) / 9007199254740992.0;
}

double n4m_rng_numpy_mt_normal(n4m_rng_numpy_mt* rng) {
    /* numpy legacy_gauss: polar Box-Muller; returns f*x2, caches f*x1. */
    if (rng->has_gauss) {
        rng->has_gauss = 0;
        return rng->gauss;
    }
    double f, x1, x2, r2;
    do {
        x1 = 2.0 * n4m_rng_numpy_mt_double(rng) - 1.0;
        x2 = 2.0 * n4m_rng_numpy_mt_double(rng) - 1.0;
        r2 = x1 * x1 + x2 * x2;
    } while (r2 >= 1.0 || r2 == 0.0);
    f = sqrt(-2.0 * log(r2) / r2);
    rng->gauss = f * x1;
    rng->has_gauss = 1;
    return f * x2;
}
