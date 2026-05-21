/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * PCG64 + NumPy SeedSequence implementation.
 *
 * Vendored/adapted from:
 *   numpy/random/src/pcg64/pcg64.{h,c}  (MIT licence — Melissa O'Neill,
 *                                        Robert Kern; relicensed 2019)
 *   numpy/random/bit_generator.pyx       (BSD-3-Clause — NumPy developers)
 *
 * The SeedSequence mixer reproduces NumPy 1.26.4's `SeedSequence(pool_size=4)`
 * exactly: a uint64 seed splits into uint32 words little-endian first, is
 * mixed into a 4-uint32 pool via `hashmix` (32-bit MURMUR-style multiplier
 * ratchet) plus a cross-mix pass, then `generate_state(2, dtype=uint64)`
 * extracts 4 uint32 words (= two uint64 little-endian-packed values) for
 * the PCG64 (state, inc) pair.
 */

#include "rng_pcg64.h"

#include <stddef.h>
#include <stdint.h>

/* ---------------------------------------------------------------------------
 * 128-bit primitives — emulated for MSVC, native otherwise.
 * ------------------------------------------------------------------------- */

#if N4M_PCG64_HAS_NATIVE_128

static n4m_pcg128_t n4m_pcg128_make(uint64_t high, uint64_t low) {
    return (((n4m_pcg128_t)high) << 64) | (n4m_pcg128_t)low;
}

static n4m_pcg128_t n4m_pcg128_zero(void) { return (n4m_pcg128_t)0; }
static n4m_pcg128_t n4m_pcg128_one(void)  { return (n4m_pcg128_t)1; }

static n4m_pcg128_t n4m_pcg128_add(n4m_pcg128_t a, n4m_pcg128_t b) {
    return a + b;
}

static n4m_pcg128_t n4m_pcg128_mul(n4m_pcg128_t a, n4m_pcg128_t b) {
    return a * b;
}

static int n4m_pcg128_nonzero(n4m_pcg128_t v) { return v != 0; }
static int n4m_pcg128_low_bit_set(n4m_pcg128_t v) { return (int)(v & 1u); }
static n4m_pcg128_t n4m_pcg128_shr1(n4m_pcg128_t v) { return v >> 1; }

static uint64_t n4m_pcg128_high(n4m_pcg128_t v) {
    return (uint64_t)(v >> 64);
}

static uint64_t n4m_pcg128_low(n4m_pcg128_t v) {
    return (uint64_t)(v & 0xFFFFFFFFFFFFFFFFULL);
}

#else  /* !N4M_PCG64_HAS_NATIVE_128 */

static n4m_pcg128_t n4m_pcg128_make(uint64_t high, uint64_t low) {
    n4m_pcg128_t r;
    r.high = high;
    r.low  = low;
    return r;
}

static n4m_pcg128_t n4m_pcg128_zero(void) { return n4m_pcg128_make(0, 0); }
static n4m_pcg128_t n4m_pcg128_one(void)  { return n4m_pcg128_make(0, 1); }

static n4m_pcg128_t n4m_pcg128_add(n4m_pcg128_t a, n4m_pcg128_t b) {
    n4m_pcg128_t r;
    r.low  = a.low + b.low;
    r.high = a.high + b.high + (r.low < b.low ? 1ULL : 0ULL);
    return r;
}

/* Schoolbook 64x64 -> 128-bit multiply: returns (z1=high, z0=low). */
static void n4m_pcg_mul64(uint64_t x, uint64_t y,
                           uint64_t* z1, uint64_t* z0) {
    uint64_t x0 = x & 0xFFFFFFFFULL;
    uint64_t x1 = x >> 32;
    uint64_t y0 = y & 0xFFFFFFFFULL;
    uint64_t y1 = y >> 32;
    uint64_t w0 = x0 * y0;
    uint64_t t  = x1 * y0 + (w0 >> 32);
    uint64_t w1 = t & 0xFFFFFFFFULL;
    uint64_t w2 = t >> 32;
    w1 += x0 * y1;
    *z0 = x * y;                       /* low half = wrapping product */
    *z1 = x1 * y1 + w2 + (w1 >> 32);
}

static n4m_pcg128_t n4m_pcg128_mul(n4m_pcg128_t a, n4m_pcg128_t b) {
    /* (a.high*b.low + a.low*b.high) << 64 + a.low * b.low */
    uint64_t h1 = a.high * b.low + a.low * b.high;
    n4m_pcg128_t r;
    n4m_pcg_mul64(a.low, b.low, &r.high, &r.low);
    r.high += h1;
    return r;
}

static int n4m_pcg128_nonzero(n4m_pcg128_t v) {
    return (v.high != 0) || (v.low != 0);
}
static int n4m_pcg128_low_bit_set(n4m_pcg128_t v) { return (int)(v.low & 1u); }
static n4m_pcg128_t n4m_pcg128_shr1(n4m_pcg128_t v) {
    n4m_pcg128_t r;
    r.low  = (v.low >> 1) | (v.high << 63);
    r.high = v.high >> 1;
    return r;
}
static uint64_t n4m_pcg128_high(n4m_pcg128_t v) { return v.high; }
static uint64_t n4m_pcg128_low(n4m_pcg128_t v)  { return v.low; }

#endif  /* N4M_PCG64_HAS_NATIVE_128 */

/* ---------------------------------------------------------------------------
 * PCG64 constants — bit-identical to NumPy's `pcg64.h`.
 * ------------------------------------------------------------------------- */

/* PCG default 128-bit multiplier: 0x2360_ED05_1FC6_5DA4_4385_DF64_9FCC_F645 */
#define N4M_PCG_MULT_HIGH 2549297995355413924ULL
#define N4M_PCG_MULT_LOW  4865540595714422341ULL

static n4m_pcg128_t n4m_pcg_default_multiplier(void) {
    return n4m_pcg128_make(N4M_PCG_MULT_HIGH, N4M_PCG_MULT_LOW);
}

/* ---------------------------------------------------------------------------
 * Bit-rotation: portable; compilers fold to ROR on x86/ARM.
 * ------------------------------------------------------------------------- */

static uint64_t n4m_rotr64(uint64_t value, unsigned int rot) {
    return (value >> rot) | (value << (((unsigned int)0 - rot) & 63u));
}

/* ---------------------------------------------------------------------------
 * NumPy SeedSequence (pool_size=4) replication.
 *
 * For a single uint64 seed `s`, NumPy emits a uint32 pool of 4 words derived
 * from `s` split little-endian first, then mixes (and cross-mixes) them via
 * the `hashmix` + `mix` pair, then `generate_state(2, dtype=uint64)` pulls
 * 4 uint32 words from the pool (modulating each with the INIT_B/MULT_B
 * ratchet) and reinterprets them as two uint64 values little-endian-packed.
 *
 * That packing matches what `_pcg64.pyx` passes to `pcg64_set_seed` as
 *   seed[0] = high(state), seed[1] = low(state),
 *   inc[0]  = high(inc),   inc[1]  = low(inc)
 * — see NumPy's `pcg64_set_seed` in `pcg64.c`, which forms
 *   state = (seed[0] << 64) | seed[1],
 *   inc   = (inc[0]  << 64) | inc[1].
 * ------------------------------------------------------------------------- */

#define N4M_SS_POOL_SIZE   4u
#define N4M_SS_INIT_A      0x43b0d7e5u
#define N4M_SS_MULT_A      0x931e8875u
#define N4M_SS_INIT_B      0x8b51f9ddu
#define N4M_SS_MULT_B      0x58f38dedu
#define N4M_SS_MIX_MULT_L  0xca01f9ddu
#define N4M_SS_MIX_MULT_R  0x4973f715u
#define N4M_SS_XSHIFT      16u

static uint32_t n4m_ss_hashmix(uint32_t value, uint32_t* hash_const) {
    /* mirrors `bit_generator.pyx` :: hashmix */
    value         ^= hash_const[0];
    hash_const[0] *= N4M_SS_MULT_A;
    value         *= hash_const[0];
    value         ^= value >> N4M_SS_XSHIFT;
    return value;
}

static uint32_t n4m_ss_mix(uint32_t x, uint32_t y) {
    /* mirrors `bit_generator.pyx` :: mix — unsigned 32-bit wrap is intended. */
    uint32_t result = N4M_SS_MIX_MULT_L * x - N4M_SS_MIX_MULT_R * y;
    result ^= result >> N4M_SS_XSHIFT;
    return result;
}

/* Mix the 1- or 2-word run entropy (uint32, little-endian split of seed)
 * into the 4-word pool. For a uint64 seed, the entropy array has length
 * 1 when seed < 2^32 and 2 otherwise (matching `_int_to_uint32_array`). */
static void n4m_ss_mix_entropy(uint32_t pool[N4M_SS_POOL_SIZE],
                                const uint32_t* entropy,
                                size_t entropy_len) {
    uint32_t hash_const = N4M_SS_INIT_A;
    size_t i;

    /* Stage 1 — seed the pool with the run entropy, then keep ratcheting on
     * zero once the entropy is exhausted (NumPy does exactly this). */
    for (i = 0; i < N4M_SS_POOL_SIZE; ++i) {
        uint32_t v = (i < entropy_len) ? entropy[i] : 0u;
        pool[i] = n4m_ss_hashmix(v, &hash_const);
    }

    /* Stage 2 — cross-mix all pool words so later words can affect earlier
     * ones. O(pool_size^2) = 16 iterations for the default pool. */
    for (size_t src = 0; src < N4M_SS_POOL_SIZE; ++src) {
        for (size_t dst = 0; dst < N4M_SS_POOL_SIZE; ++dst) {
            if (src == dst) continue;
            pool[dst] = n4m_ss_mix(pool[dst],
                                    n4m_ss_hashmix(pool[src], &hash_const));
        }
    }

    /* Stage 3 — fold any remaining entropy words (only fires for seeds
     * needing > pool_size uint32 words; uint64 seed never reaches here). */
    for (size_t src = N4M_SS_POOL_SIZE; src < entropy_len; ++src) {
        for (size_t dst = 0; dst < N4M_SS_POOL_SIZE; ++dst) {
            pool[dst] = n4m_ss_mix(pool[dst],
                                    n4m_ss_hashmix(entropy[src], &hash_const));
        }
    }
}

/* `generate_state(2, dtype=np.uint64)` — uint64 doubles n_words, so 4 uint32
 * words are emitted. Then they're viewed as `<u4` -> `<u8` (little-endian
 * pairs). We materialise that directly into 2 uint64 outputs. */
static void n4m_ss_generate_state_u64(const uint32_t pool[N4M_SS_POOL_SIZE],
                                       size_t n_uint64,
                                       uint64_t* out_u64) {
    uint32_t hash_const = N4M_SS_INIT_B;
    size_t pool_idx = 0;
    size_t n_words = n_uint64 * 2u;
    uint32_t words[16];  /* up to 8 uint64 -> 16 uint32; sufficient for our use */

    if (n_words > sizeof(words) / sizeof(words[0])) {
        /* Caller bug — but we never request more than 4 here. Bail without
         * touching `out_u64`. */
        return;
    }

    for (size_t i = 0; i < n_words; ++i) {
        uint32_t v = pool[pool_idx];
        v          ^= hash_const;
        hash_const *= N4M_SS_MULT_B;
        v          *= hash_const;
        v          ^= v >> N4M_SS_XSHIFT;
        words[i]   = v;
        pool_idx = (pool_idx + 1u) % N4M_SS_POOL_SIZE;
    }

    /* Little-endian pack of consecutive uint32 pairs into uint64. */
    for (size_t i = 0; i < n_uint64; ++i) {
        out_u64[i] = (uint64_t)words[2 * i] |
                     ((uint64_t)words[2 * i + 1] << 32);
    }
}

/* Expand a uint64 seed into (state_high, state_low, inc_high, inc_low) as
 * NumPy's `SeedSequence(seed).generate_state(4, np.uint64)` would. */
static void n4m_pcg_expand_seed(uint64_t seed, uint64_t out_u64_4[4]) {
    /* `_int_to_uint32_array(seed)` little-endian split. Special-case seed=0
     * (NumPy emits a 1-element [0] array, not an empty array). */
    uint32_t entropy[2];
    size_t entropy_len;
    if (seed == 0) {
        entropy[0]  = 0u;
        entropy_len = 1u;
    } else {
        uint64_t s = seed;
        entropy_len = 0u;
        while (s != 0u && entropy_len < 2u) {
            entropy[entropy_len] = (uint32_t)(s & 0xFFFFFFFFULL);
            s >>= 32;
            entropy_len++;
        }
    }

    uint32_t pool[N4M_SS_POOL_SIZE] = {0u, 0u, 0u, 0u};
    n4m_ss_mix_entropy(pool, entropy, entropy_len);
    n4m_ss_generate_state_u64(pool, 4u, out_u64_4);
}

/* ---------------------------------------------------------------------------
 * PCG64 core — step, output, advance.
 *
 * NumPy's PCG64 (XSL-RR variant) is "advance then output". `pcg64_set_seed`
 * runs the initialiser sequence:
 *     state = 0
 *     inc = (initseq << 1) | 1
 *     step()
 *     state += initstate
 *     step()
 * then each draw runs step() and returns XSL-RR(state).
 * ------------------------------------------------------------------------- */

static void n4m_pcg_step(n4m_rng_pcg64* rng) {
    rng->state = n4m_pcg128_add(n4m_pcg128_mul(rng->state,
                                                n4m_pcg_default_multiplier()),
                                  rng->inc);
}

static uint64_t n4m_pcg_output_xsl_rr(n4m_pcg128_t state) {
    /* high ^ low, rotate right by (state >> 122) = high >> 58. */
    uint64_t hi = n4m_pcg128_high(state);
    uint64_t lo = n4m_pcg128_low(state);
    return n4m_rotr64(hi ^ lo, (unsigned int)(hi >> 58));
}

static n4m_pcg128_t n4m_pcg_advance_lcg(n4m_pcg128_t state,
                                          n4m_pcg128_t delta,
                                          n4m_pcg128_t cur_mult,
                                          n4m_pcg128_t cur_plus) {
    /* Brown 1994 "Random Number Generation with Arbitrary Stride" —
     * O(log delta) fast-exponentiation walk. */
    n4m_pcg128_t acc_mult = n4m_pcg128_one();
    n4m_pcg128_t acc_plus = n4m_pcg128_zero();
    while (n4m_pcg128_nonzero(delta)) {
        if (n4m_pcg128_low_bit_set(delta)) {
            acc_mult = n4m_pcg128_mul(acc_mult, cur_mult);
            acc_plus = n4m_pcg128_add(n4m_pcg128_mul(acc_plus, cur_mult),
                                       cur_plus);
        }
        cur_plus = n4m_pcg128_mul(n4m_pcg128_add(cur_mult, n4m_pcg128_one()),
                                    cur_plus);
        cur_mult = n4m_pcg128_mul(cur_mult, cur_mult);
        delta    = n4m_pcg128_shr1(delta);
    }
    return n4m_pcg128_add(n4m_pcg128_mul(acc_mult, state), acc_plus);
}

/* ---------------------------------------------------------------------------
 * Public API (file-internal — see rng_pcg64.h).
 * ------------------------------------------------------------------------- */

void n4m_pcg64_engine_seed(n4m_rng_pcg64* rng, uint64_t seed) {
    uint64_t expanded[4];
    n4m_pcg_expand_seed(seed, expanded);
    /* NumPy passes seed[0..2] = state (high, low) and inc[0..2] = inc
     * (high, low). The packing order in `generate_state(4, uint64)` is
     * the natural array order [0..3] = state_hi, state_lo, inc_hi, inc_lo. */
    n4m_pcg128_t initstate = n4m_pcg128_make(expanded[0], expanded[1]);
    n4m_pcg128_t initseq   = n4m_pcg128_make(expanded[2], expanded[3]);

    /* pcg_setseq_128_srandom_r: state=0; inc=(initseq<<1)|1; step;
     *                          state+=initstate; step. */
    rng->state = n4m_pcg128_zero();
    /* inc = (initseq << 1) | 1: shift high/low by 1 with carry, then OR 1. */
    {
        uint64_t seq_hi = n4m_pcg128_high(initseq);
        uint64_t seq_lo = n4m_pcg128_low(initseq);
        uint64_t inc_hi = (seq_hi << 1) | (seq_lo >> 63);
        uint64_t inc_lo = (seq_lo << 1) | 1u;
        rng->inc = n4m_pcg128_make(inc_hi, inc_lo);
    }
    n4m_pcg_step(rng);
    rng->state = n4m_pcg128_add(rng->state, initstate);
    n4m_pcg_step(rng);
}

uint64_t n4m_pcg64_engine_next_uint64(n4m_rng_pcg64* rng) {
    /* "advance then output" — NumPy's `pcg_setseq_128_xsl_rr_64_random_r`. */
    n4m_pcg_step(rng);
    return n4m_pcg_output_xsl_rr(rng->state);
}

double n4m_pcg64_engine_next_double(n4m_rng_pcg64* rng) {
    /* (rnd >> 11) * 2^-53 — bit-exact match to NumPy's `uint64_to_double`. */
    uint64_t r = n4m_pcg64_engine_next_uint64(rng);
    return (double)(r >> 11) * (1.0 / 9007199254740992.0);
}

void n4m_pcg64_engine_advance(n4m_rng_pcg64* rng,
                            uint64_t delta_high, uint64_t delta_low) {
    n4m_pcg128_t delta = n4m_pcg128_make(delta_high, delta_low);
    rng->state = n4m_pcg_advance_lcg(rng->state, delta,
                                       n4m_pcg_default_multiplier(),
                                       rng->inc);
}
