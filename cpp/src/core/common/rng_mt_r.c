/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * R-compatible Mersenne-Twister + Inversion-normal RNG.
 * See rng_mt_r.h. Re-implemented from the published algorithms in R's
 * src/main/RNG.c and src/nmath/qnorm.c (Wichura 1988, AS 241).
 */
#include "rng_mt_r.h"

#include <math.h>
#include <stdlib.h>

/* ---- MT19937 constants (Matsumoto & Nishimura 1998, R's RNG.c values) ---- */
#define MT_N 624
#define MT_M 397
#define MT_MATRIX_A   0x9908b0dfUL
#define MT_UPPER_MASK 0x80000000UL
#define MT_LOWER_MASK 0x7fffffffUL
#define MT_TEMPER_B   0x9d2c5680UL
#define MT_TEMPER_C   0xefc60000UL

/* R's MT scaling is 1/2^32 (the factor baked into R's MT_genrand), NOT the
 * 1/(2^32-1) `i2_32m1` constant used by R's other generators. Using 1/(2^32-1)
 * here matches R to ~10 digits but not bit-for-bit; 1/2^32 is exact. */
static const double N4M_I2_32M1 = 2.3283064365386963e-10;
/* 2^27, R's INVERSION precision constant (BIG in RNG.c). */
static const double N4M_BIG = 134217728.0;

/* dummy[0] = mti, mt = dummy + 1  (R layout: i_seed[0]=mti, i_seed[1..624]=mt) */

void n4m_rng_mt_r_set_seed(n4m_rng_mt_r* rng, uint32_t seed) {
    rng->bm_norm_keep = 0.0;
    /* Initial scrambling: 50 steps of the 69069 LCG (R RNG_Init). */
    for (int j = 0; j < 50; j++)
        seed = (69069U * seed + 1U);
    /* Fill all 625 state words via the same LCG. */
    for (int j = 0; j < 625; j++) {
        seed = (69069U * seed + 1U);
        rng->i_seed[j] = (int32_t)seed;
    }
    /* FixupSeeds(MERSENNE_TWISTER, initial=1): mti := 624. */
    rng->i_seed[0] = MT_N;
}

/* R's MT_genrand: regenerate the 624-word block when mti >= N, then temper. */
uint32_t n4m_rng_mt_r_next_uint32(n4m_rng_mt_r* rng) {
    uint32_t* mt = (uint32_t*)&rng->i_seed[1];   /* dummy+1 */
    int mti = rng->i_seed[0];
    uint32_t y;
    static const uint32_t mag01[2] = {0x0UL, MT_MATRIX_A};

    if (mti >= MT_N) {
        int kk;
        /* mti == N+1 would mean "never seeded"; R seeds via set.seed so the
         * state is always initialised — we never hit the MT_sgenrand(4357)
         * fallback. mti == N (624) after set.seed lands here and regenerates
         * from the LCG-filled state, matching R exactly. */
        for (kk = 0; kk < MT_N - MT_M; kk++) {
            y = (mt[kk] & MT_UPPER_MASK) | (mt[kk + 1] & MT_LOWER_MASK);
            mt[kk] = mt[kk + MT_M] ^ (y >> 1) ^ mag01[y & 0x1U];
        }
        for (; kk < MT_N - 1; kk++) {
            y = (mt[kk] & MT_UPPER_MASK) | (mt[kk + 1] & MT_LOWER_MASK);
            mt[kk] = mt[kk + (MT_M - MT_N)] ^ (y >> 1) ^ mag01[y & 0x1U];
        }
        y = (mt[MT_N - 1] & MT_UPPER_MASK) | (mt[0] & MT_LOWER_MASK);
        mt[MT_N - 1] = mt[MT_M - 1] ^ (y >> 1) ^ mag01[y & 0x1U];
        mti = 0;
    }

    y = mt[mti++];
    y ^= (y >> 11);
    y ^= (y << 7) & MT_TEMPER_B;
    y ^= (y << 15) & MT_TEMPER_C;
    y ^= (y >> 18);
    rng->i_seed[0] = mti;
    return y;
}

/* R's fixup(): never return exactly 0 or 1. */
static double n4m_mt_fixup(double x) {
    if (x <= 0.0) return 0.5 * N4M_I2_32M1;
    if ((1.0 - x) <= 0.0) return 1.0 - 0.5 * N4M_I2_32M1;
    return x;
}

double n4m_rng_mt_r_unif(n4m_rng_mt_r* rng) {
    return n4m_mt_fixup((double)n4m_rng_mt_r_next_uint32(rng) * N4M_I2_32M1);
}

/* Wichura AS 241 — inverse standard-normal CDF, the exact routine R's
 * qnorm5 uses (mu=0, sigma=1, lower_tail=1, log_p=0). */
double n4m_qnorm5_std(double p) {
    double q, r, val;
    q = p - 0.5;
    if (fabs(q) <= 0.425) {
        r = 0.180625 - q * q;
        val = q * (((((((2509.0809287301226727 * r +
                33430.575583588128105) * r + 67265.770927008700853) * r +
                45921.953931549871457) * r + 13731.693765509461125) * r +
                1971.5909503065514427) * r + 133.14166789178437745) * r +
                3.387132872796366608)
            / (((((((5226.495278852854561 * r +
                28729.085735721942674) * r + 39307.89580009271061) * r +
                21213.794301586595867) * r + 5394.1960214247511077) * r +
                687.1870074920579083) * r + 42.313330701600911252) * r + 1.0);
        return val;
    }
    if (q < 0.0) r = p;
    else r = 1.0 - p;
    r = sqrt(-log(r));
    if (r <= 5.0) {
        r += -1.6;
        val = (((((((7.7454501427834140764e-4 * r +
               0.0227238449892691845833) * r + 0.24178072517745061177) *
               r + 1.27045825245236838258) * r +
               3.64784832476320460504) * r + 5.7694972214606914055) *
               r + 4.6303378461565452959) * r +
               1.42343711074968357734)
            / (((((((1.05075007164441684324e-9 * r +
               5.475938084995344946e-4) * r +
               0.0151986665636164571966) * r +
               0.14810397642748007459) * r + 0.68976733498510000455) *
               r + 1.6763848301838038494) * r +
               2.05319162663775882187) * r + 1.0);
    } else {
        r += -5.0;
        val = (((((((2.01033439929228813265e-7 * r +
               2.71155556874348757815e-5) * r +
               0.0012426609473880784386) * r + 0.026532189526576123093) *
               r + 0.29656057182850489123) * r +
               1.7848265399172913358) * r + 5.4637849111641143699) *
               r + 6.6579046435011037772)
            / (((((((2.04426310338993978564e-15 * r +
               1.4215117583164458887e-7) * r +
               1.8463183175100546818e-5) * r +
               7.868691311456132591e-4) * r + 0.0148753612908506148525)
               * r + 0.13692988092273580531) * r +
               0.59983220655588793769) * r + 1.0);
    }
    if (q < 0.0) val = -val;
    return val;
}

double n4m_rng_mt_r_norm(n4m_rng_mt_r* rng) {
    /* R INVERSION: u1 = unif; u1 = floor(BIG*u1) + unif; qnorm5(u1/BIG). */
    double u1 = n4m_rng_mt_r_unif(rng);
    u1 = (double)(int)(N4M_BIG * u1) + n4m_rng_mt_r_unif(rng);
    return n4m_qnorm5_std(u1 / N4M_BIG);
}

/* R's rbits(bits): assemble `bits` random bits from 16-bit unif chunks
 * (R src/main/RNG.c). */
static double n4m_mt_rbits(n4m_rng_mt_r* rng, int bits) {
    int64_t v = 0;
    for (int n = 0; n <= bits; n += 16) {
        int v1 = (int)floor(n4m_rng_mt_r_unif(rng) * 65536.0);
        v = 65536 * v + v1;
    }
    return (double)(v & (((int64_t)1 << bits) - 1));
}

double n4m_rng_mt_r_unif_index(n4m_rng_mt_r* rng, double dn) {
    /* R 3.6+ R_unif_index, sample.kind="Rejection". */
    if (dn <= 0.0) return 0.0;
    int bits = (int)ceil(log2(dn));
    double dv;
    do { dv = n4m_mt_rbits(rng, bits); } while (dn <= dv);
    return dv;
}

void n4m_rng_mt_r_sample(n4m_rng_mt_r* rng, int n, int* out) {
    /* R do_sample, equal-probability no-replacement Fisher-Yates
     * (R src/main/random.c): pool x[i]=i; for i: j=R_unif_index(n_);
     * out[i]=x[j]; x[j]=x[--n_]. Emits R's `sample(n) - 1` (0-based). */
    if (n <= 0) return;
    int* pool = (int*)malloc((size_t)n * sizeof(int));
    if (pool == NULL) {                  /* OOM: degrade to identity */
        for (int i = 0; i < n; i++) out[i] = i;
        return;
    }
    for (int i = 0; i < n; i++) pool[i] = i;
    int nn = n;
    for (int i = 0; i < n; i++) {
        int j = (int)n4m_rng_mt_r_unif_index(rng, (double)nn);
        out[i] = pool[j];
        pool[j] = pool[--nn];
    }
    free(pool);
}
