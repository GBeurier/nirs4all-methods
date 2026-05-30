// SPDX-License-Identifier: CECILL-2.1
//
// Tier B golden + dispatch tests for the unified RNG engine
// (cpp/src/core/common/rng_engine.{c,h}) and the numpy-RandomState MT
// (rng_numpy_mt.{c,h}, Tier A).
//
// The SPLITMIX64 path FREEZES n4m's historical stream: every stochastic
// selector used a file-local splitmix64 (golden 0x9E3779B97F4A7C15). These
// golden values lock that exact sequence so migrating a kernel from its
// file-local helper to the engine dispatch is provably byte-identical (the
// regression guard Codex required before the abstraction).
//
// The MT_R / NUMPY_MT routes assert the engine dispatches to the
// bit-exact-vs-R and bit-exact-vs-numpy engines respectively.

#include <cstdint>
#include <cmath>

#include "harness.hpp"

extern "C" {
// Mirror of the internal enum/struct (kept in sync with rng_engine.h). The
// struct is opaque to the test; allocate a byte buffer big enough for it.
typedef enum { ENG_SPLITMIX64 = 0, ENG_PCG64 = 1, ENG_MT_R = 2, ENG_NUMPY_MT = 3 } eng_kind;
void     n4m_rng_seed(void* rng, int kind, std::uint64_t seed);
std::uint64_t n4m_rng_next_u64(void* rng);
double   n4m_rng_next_double(void* rng);
double   n4m_rng_next_normal(void* rng);
// numpy-MT direct (Tier A), for the bit-exact-vs-numpy assertion.
void     n4m_rng_numpy_mt_seed(void* rng, std::uint64_t seed);
double   n4m_rng_numpy_mt_double(void* rng);
double   n4m_rng_numpy_mt_normal(void* rng);
}

namespace {

struct alignas(16) EngBuf  { unsigned char b[8192]; };  // >= sizeof(n4m_rng)
struct alignas(16) NpMtBuf { unsigned char b[4096]; };  // >= sizeof(n4m_rng_numpy_mt)

// Canonical splitmix64, seed=0 then seed=12345 (8 each). These are the
// reference SplitMix64 outputs; seed=0 is the well-known canonical sequence.
constexpr std::uint64_t kGolden0[8] = {
    0xe220a8397b1dcdafULL, 0x6e789e6aa1b965f4ULL, 0x06c45d188009454fULL, 0xf88bb8a8724c81ecULL,
    0x1b39896a51a8749bULL, 0x53cb9f0c747ea2eaULL, 0x2c829abe1f4532e1ULL, 0xc584133ac916ab3cULL,
};
constexpr std::uint64_t kGolden12345[8] = {
    0x6a81e5095c8b1bd3ULL, 0x2a5f9eeec0e9e3b3ULL, 0xa4cb96962d3b9ca8ULL, 0xda6c9d11c1f2a3b2ULL,
    0xa340df1f2c80a829ULL, 0xf41a6e0a3a3f1d5cULL, 0x639bf24b665eb0a6ULL, 0xb96ea0d703f27f3aULL,
};

void test_splitmix64_stream_frozen() {
    EngBuf e;
    n4m_rng_seed(&e, ENG_SPLITMIX64, 0);
    for (int i = 0; i < 8; ++i)
        N4M_TEST_REQUIRE(n4m_rng_next_u64(&e) == kGolden0[i]);
    n4m_rng_seed(&e, ENG_SPLITMIX64, 12345);
    for (int i = 0; i < 8; ++i)
        N4M_TEST_REQUIRE(n4m_rng_next_u64(&e) == kGolden12345[i]);
}

void test_engine_routes_to_R() {
    EngBuf e;
    n4m_rng_seed(&e, ENG_MT_R, 11);
    // R set.seed(11); runif(1) == 0.27724979422055185 (bit-exact)
    N4M_TEST_REQUIRE(n4m_rng_next_double(&e) == 0.27724979422055185);
}

void test_engine_routes_to_numpy() {
    EngBuf e;
    n4m_rng_seed(&e, ENG_NUMPY_MT, 11);
    // numpy RandomState(11).random_sample() == 0.18430501533259898 (bit-exact)
    N4M_TEST_REQUIRE(n4m_rng_next_double(&e) == 0.18430501533259898);
}

void test_numpy_mt_unif_bit_exact() {
    NpMtBuf r;
    n4m_rng_numpy_mt_seed(&r, 11);
    constexpr double want[6] = {
        0.18430501533259898, 0.41331584611395556, 0.5076895868479615,
        0.74880388253861187, 0.47241118570694411, 0.41172754247206768,
    };
    for (int i = 0; i < 6; ++i)
        N4M_TEST_REQUIRE(n4m_rng_numpy_mt_double(&r) == want[i]);
}

void test_numpy_mt_normal_bit_exact() {
    NpMtBuf r;
    n4m_rng_numpy_mt_seed(&r, 11);
    constexpr double want[3] = {
        0.13074365906142274, 0.046009735895881845, -0.46930528063856195,
    };
    for (int i = 0; i < 3; ++i)
        N4M_TEST_REQUIRE(n4m_rng_numpy_mt_normal(&r) == want[i]);
}

}  // namespace

void register_rng_engine_tests(n4m_testing::Runner& r);
void register_rng_engine_tests(n4m_testing::Runner& r) {
    r.run("rng_engine_splitmix64_stream_frozen", test_splitmix64_stream_frozen);
    r.run("rng_engine_routes_to_R",              test_engine_routes_to_R);
    r.run("rng_engine_routes_to_numpy",          test_engine_routes_to_numpy);
    r.run("rng_numpy_mt_unif_bit_exact",         test_numpy_mt_unif_bit_exact);
    r.run("rng_numpy_mt_normal_bit_exact",       test_numpy_mt_normal_bit_exact);
}
