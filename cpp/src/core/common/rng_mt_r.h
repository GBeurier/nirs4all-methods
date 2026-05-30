/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal R-compatible Mersenne-Twister RNG for n4m.
 *
 * This header is internal-only — never installed and never included by
 * `n4m/n4m.h`. It exists so the stochastic methods can OPTIONALLY draw from
 * an RNG stream that is bit-identical to base R's default generator
 * (`RNGkind("Mersenne-Twister", "Inversion")` + `set.seed(seed)`), which is
 * what the R reference libraries (plsVarSel, mdatools, …) use. The default
 * n4m RNG remains splitmix64 / PCG64; this is opt-in via the RNG-kind config.
 *
 * Reproduces, bit-for-bit:
 *   - set.seed(s)            ≡ n4m_rng_mt_r_set_seed(rng, s)
 *   - runif(n)               ≡ n4m_rng_mt_r_unif()      (fixup'd, never 0/1)
 *   - rnorm(n) [Inversion]   ≡ n4m_rng_mt_r_norm()      (2 uniforms + qnorm5)
 *
 * Algorithm vendored / re-derived from R's `src/main/RNG.c` and `qnorm.c`
 * (Wichura AS 241), GPL-2 — re-implemented from the public algorithm, not
 * copied verbatim, so it is license-compatible with CeCILL-2.1.
 */
#ifndef N4M_CORE_COMMON_RNG_MT_R_H
#define N4M_CORE_COMMON_RNG_MT_R_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* R keeps 625 Int32 state words: i_seed[0] is the MT position `mti`,
 * i_seed[1..624] is the 624-word MT state vector. */
typedef struct n4m_rng_mt_r {
    int32_t  i_seed[625];   /* [0]=mti, [1..624]=mt state (R layout) */
    double   bm_norm_keep;  /* reserved (Box-Muller cache; unused for Inversion) */
} n4m_rng_mt_r;

/* set.seed(seed): 50-step LCG scramble, fill 625 words via the 69069 LCG,
 * then FixupSeeds (mti := 624). Bit-identical to R's RNG_Init for MT. */
void   n4m_rng_mt_r_set_seed(n4m_rng_mt_r* rng, uint32_t seed);

/* One uint32 from the tempered MT stream (R's MT_genrand). */
uint32_t n4m_rng_mt_r_next_uint32(n4m_rng_mt_r* rng);

/* One uniform double in (0,1) — `fixup(MT_genrand() / (2^32 - 1))`,
 * bit-identical to R's unif_rand() under Mersenne-Twister. */
double n4m_rng_mt_r_unif(n4m_rng_mt_r* rng);

/* One standard-normal via R's INVERSION: u = (int)(2^27 * u1) + u2, then
 * qnorm5(u / 2^27). Bit-identical to R's norm_rand() under Inversion. */
double n4m_rng_mt_r_norm(n4m_rng_mt_r* rng);

/* Inverse standard-normal CDF (Wichura AS 241), exposed for testing. */
double n4m_qnorm5_std(double p);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_COMMON_RNG_MT_R_H */
