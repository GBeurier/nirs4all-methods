// SPDX-License-Identifier: CECILL-2.1
//
// Bit-exact parity tests for the R-compatible Mersenne-Twister RNG against
// base R (RNGkind("Mersenne-Twister","Inversion"), the R default).
//
// Ground truth captured from R 4.3.3:
//   set.seed(11); runif(6)   and   set.seed(11); rnorm(6)
// printed with sprintf("%.17g"). The C engine must match bit-for-bit so the
// stochastic selectors can OPTIONALLY draw an R-identical stream (the
// reference libraries plsVarSel / mdatools use base R's RNG). This locks that
// guarantee permanently.
//
// The RNG is an internal core module (cpp/src/core/common/rng_mt_r.{c,h}); we
// declare its prototypes here rather than include the internal header so the
// test stays a pure consumer.

#include <cstdint>
#include <cmath>
#include <string>

#include "harness.hpp"

extern "C" {
struct n4m_rng_mt_r;  // opaque to the test; we only need a byte buffer.
void   n4m_rng_mt_r_set_seed(void* rng, std::uint32_t seed);
double n4m_rng_mt_r_unif(void* rng);
double n4m_rng_mt_r_norm(void* rng);
double n4m_qnorm5_std(double p);
}

namespace {

// The state struct is { int32_t i_seed[625]; double bm_norm_keep; } — 625*4 + 8
// = 2508 bytes. Allocate a generous aligned buffer so we don't depend on the
// internal layout beyond "it fits".
struct alignas(8) RMtBuf { unsigned char bytes[4096]; };

// R 4.3.3 set.seed(11); runif(6)
constexpr double kRUnif[6] = {
    0.27724979422055185, 0.00051831291057169437, 0.51060837297700346,
    0.014047908363863826, 0.064689776627346873, 0.95484922546893358,
};
// R 4.3.3 set.seed(11); rnorm(6)   [Inversion]
constexpr double kRNorm[6] = {
    -0.59103110258436842, 0.026594369016167005, -1.516553097081865,
    -1.3626533492958086, 1.1784891560316197, -0.93415131967336518,
};

void test_runif_bit_exact_vs_R() {
    RMtBuf buf;
    n4m_rng_mt_r_set_seed(&buf, 11u);
    for (int i = 0; i < 6; ++i) {
        const double got = n4m_rng_mt_r_unif(&buf);
        N4M_TEST_REQUIRE(got == kRUnif[i]);  // bit-exact, not approximate
    }
}

void test_rnorm_bit_exact_vs_R() {
    RMtBuf buf;
    n4m_rng_mt_r_set_seed(&buf, 11u);
    for (int i = 0; i < 6; ++i) {
        const double got = n4m_rng_mt_r_norm(&buf);
        N4M_TEST_REQUIRE(got == kRNorm[i]);  // bit-exact, not approximate
    }
}

void test_set_seed_resets_stream() {
    RMtBuf a, b;
    n4m_rng_mt_r_set_seed(&a, 11u);
    (void)n4m_rng_mt_r_unif(&a);
    (void)n4m_rng_mt_r_unif(&a);
    n4m_rng_mt_r_set_seed(&b, 11u);  // re-seed must rewind to the same start
    n4m_rng_mt_r_set_seed(&a, 11u);
    N4M_TEST_REQUIRE(n4m_rng_mt_r_unif(&a) == n4m_rng_mt_r_unif(&b));
}

void test_qnorm5_median_and_symmetry() {
    // qnorm(0.5) == 0 exactly; qnorm is odd about 0.5.
    N4M_TEST_REQUIRE(n4m_qnorm5_std(0.5) == 0.0);
    const double hi = n4m_qnorm5_std(0.975);
    const double lo = n4m_qnorm5_std(0.025);
    N4M_TEST_REQUIRE(std::fabs(hi + lo) < 1e-12);
    // qnorm(0.975) ≈ 1.959963984540054 (the canonical 95% z).
    N4M_TEST_REQUIRE(std::fabs(hi - 1.959963984540054) < 1e-12);
}

}  // namespace

void register_rng_mt_r_tests(n4m_testing::Runner& r);
void register_rng_mt_r_tests(n4m_testing::Runner& r) {
    r.run("rng_mt_r_runif_bit_exact_vs_R",   test_runif_bit_exact_vs_R);
    r.run("rng_mt_r_rnorm_bit_exact_vs_R",   test_rnorm_bit_exact_vs_R);
    r.run("rng_mt_r_set_seed_resets_stream", test_set_seed_resets_stream);
    r.run("rng_mt_r_qnorm5_median_symmetry", test_qnorm5_median_and_symmetry);
}
