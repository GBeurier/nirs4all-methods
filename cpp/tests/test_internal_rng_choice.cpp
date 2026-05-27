// SPDX-License-Identifier: CECILL-2.1
//
// Direct unit tests for the INTERNAL PCG64 bounded-integer + choice-without-
// replacement primitives. These symbols are hidden in libn4m (visibility +
// version script), so they cannot be reached from n4m_tests (which links the
// shared lib). This binary links the n4m_c_static archive instead — a static
// link resolves hidden symbols and carries the optional backend deps.
//
// Reference values are numpy.random.Generator outputs (generated with numpy
// 2.3.5 and cross-checked identical on 2.2.6). NumPy's PCG64 raw stream is a
// documented fixed-seed guarantee; the Generator-level choice/bounded
// algorithms are observed stable across the tested versions but are not a
// forever-contract. Each case is the fixed-seed stream a caller reproduces
// with numpy.random.default_rng(seed).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "core/augmentation/aug_rng_utils.h"
#include "core/common/rng_pcg64.h"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);   \
            ++g_failures;                                                   \
        }                                                                   \
    } while (0)

// --- n4m_aug_rng_bounded == numpy Generator.integers(0, m-1, endpoint=True) --
void test_bounded() {
    n4m_rng_pcg64 rng;
    n4m_pcg64_engine_seed(&rng, 123);
    const uint64_t m[]   = {200, 5, 1000, 7, 255, 16, 99999, 3};
    const uint64_t want[] = {3, 3, 592, 0, 231, 3, 25508, 0};
    for (int i = 0; i < 8; ++i) {
        CHECK(n4m_aug_rng_bounded(&rng, m[i] - 1) == want[i]);
    }
}

// --- n4m_aug_rng_integers(n) delegates to bounded(n-1) == numpy integers(0,n) -
void test_integers() {
    n4m_rng_pcg64 rng;
    n4m_pcg64_engine_seed(&rng, 123);
    const uint64_t m[]    = {200, 5, 1000, 7, 255, 16, 99999, 3};
    const uint64_t want[] = {3, 3, 592, 0, 231, 3, 25508, 0};  // same stream as bounded
    for (int i = 0; i < 8; ++i) {
        CHECK(n4m_aug_rng_integers(&rng, m[i]) == want[i]);  // integers(m) == bounded(m-1)
    }
    CHECK(n4m_aug_rng_integers(&rng, 1) == 0u);  // degenerate range -> 0, no draw
    CHECK(n4m_aug_rng_integers(&rng, 0) == 0u);
}

// --- advance() must reset the uint32 half-word cache (numpy PCG64.advance) ---
void test_advance_resets_cache() {
    n4m_rng_pcg64 rng;
    n4m_pcg64_engine_seed(&rng, 55);
    (void)n4m_aug_rng_bounded(&rng, 300);     // consumes a uint32, buffers high half
    n4m_pcg64_engine_advance(&rng, 0, 1000);  // must drop the stale buffered half
    CHECK(n4m_aug_rng_bounded(&rng, 300) == 145);
}

bool choice_matches(uint64_t seed, int64_t N, int64_t k,
                    const int64_t* want, int64_t want_len) {
    std::vector<int64_t> out(static_cast<std::size_t>(k));
    n4m_rng_pcg64 rng;
    n4m_pcg64_engine_seed(&rng, seed);
    if (n4m_aug_rng_choice_no_replace(&rng, N, k, out.data()) != 0) return false;
    for (int64_t i = 0; i < want_len; ++i) {
        if (out[static_cast<std::size_t>(i)] != want[i]) return false;
    }
    return true;
}

// Full bit-exact arrays from numpy choice(N, k, replace=False) (Floyd path).
const int64_t CH_A[] = {5, 6, 7, 3};  // seed=3298921130 N=16 k=4
const int64_t CH_B[] = {71,85,76,13,199,195,113,135,52,87,68,134,154,194,130,84,
    170,145,32,141,102,146,190,110,164,176,67,16,133,161,120,100,119,81,158,118,
    11,124,79,160,97,41,156,136,117,31,63,21,15,155};  // seed=42 N=200 k=50
const int64_t CH_C[] = {4346,7579,4933,10675,1591,3067,4729,5829,10555,3306,5263,
    3185,4338,8946,1699,1399,8079,11173,2478,6063,3704,10926,713,7371,6607,6733,
    2319,10923,410,8442,5578,1798,9502,5056,3163,10233,1562,6694,1825,7514,8968,
    8394,2774,6287,7620,10426,6503,7600,4334,1803,9654,5399,7155,327,9981,2613,
    5683,10271,5890,3597,11204,9365,1471,7994,11494,8352,5772,6189,8812,8952,
    11702,6739,8423,8994,6653,6660,6699,983,5894,1254,4622,7589,4018,66,6093,9185,
    6589,5924,11879,9113,2562,10008,2307,10095,177,10548,6089,6748,9045,5756,2579,
    1780,3677,11630,9900,7702,8636,7644,1524,8972,5891,3739,9985,442,8659,11480,
    3911,3525,10597,9233,9245,3866,466,8257,8195,9395,8851,7478,5734,9343,7743,798,
    4814,3297,7902,10114,4048,463,6735,4901,6590,10512,6043,533,7381,6507,9417,849,
    6776,11316,7178,6617,9374,152,7143,3744,10725,3105,1492,1348,648,10585,4474,
    1119,174,1660,8551,11345,3832,3387,7548,2738,7035,3085,7323,7619,6414,3145,
    9289,6856,2131,7274,87,2628,886,639,8596,7101,2610,8805,9515,495,8000,7882,
    9117,3950,2015,3027,3182,10611,7077,2667,7093,24,2253,6308,3821,9709,9084,8783,
    11091,710,936,7215,9318,6587,7429,2113,4494,64,2695,8406,1583,1692,11481,4892,
    1508,5218,742,5561,2327,10584,9461,1330,5701,7008,7233,8894,4419,7095,1071,
    10972,11026,946,6623,6939,8460,10445,10907,10420,10551,1354,11788,2547,11808,
    10028,9357,5752,3345,11551,8860,6135,3579,11588,8439,11705,3776,11491,9390,
    4171,11771,3502,9945,1773,67,8067,780,10801,5809,8963,5238,9557,318,156,11005,
    8642,10313,11107,11989,10974,8586,7706,9324,8006,7234,1371,3440,11530,10442,
    5058};  // seed=9 N=12000 k=300 — TAIL-SHUFFLE (k=300 > N/50=240)
// Large-population FLOYD: N>10000 but k<=N/50 (=240) stays on the Floyd branch.
const int64_t CH_D[] = {7900,8460,3337,2015,2730,11203,180,913,8945,3773,352,
    10752,10271,11441,8072,7554,3057,529,10673,7508,3128,8269,5970,4620,4240,2020,
    9993,2741,306,10213,6554,7332,3280,4045,2573,5002,8879,7532,4030,10957,11528,
    124,3096,9745,1328,4699,4879,1617,1515,8581,6044,7834,11740,2289,9479,3541,
    4558,2143,4963,2186,1299,7493,3556,4043,1044,2737,11435,727,4008,7619,9967,168,
    2106,9267,7990,4414,7684,11585,5989,8971,3452,215,11324,9728,8078,2477,7746,
    5742,10842,8240,10642,3702,6594,7361,1446,10927,5226,2410,8225,5489,11598,7191,
    11295,10586,9727,2863,9057,1380,10547,6899,9898,2265,3856,5215,5680,7946,5101,
    10648,3130,11178,7151,4543,5738,4062,9559,4556,8384,11799,141,1052,9413,5168,
    3789,1146,6733,8379,6861,9612,3318,1407,7869,5036,5557,142,5596,2676,7442,7665,
    3507,8614,6909,2220,8588,8433,10101,7044,9081,4776,261,8816,10934,7770,7448,3,
    6934,4800,1538,9491,9218,7263,2085,7692,2683,9024,7134,11018,9307,10422,10521,
    6831,10949,2623,11577,3084,9255,8277,11181,5501,5903,8633,7825,3473,2331,4822,
    2764,3714,7189,9427,8803,641};  // seed=21 N=12000 k=200

void test_choice_floyd() {
    CHECK(choice_matches(3298921130ULL, 16, 4, CH_A, 4));    // small pop
    CHECK(choice_matches(42ULL, 200, 50, CH_B, 50));         // small pop
    CHECK(choice_matches(21ULL, 12000, 200, CH_D, 200));     // large pop, Floyd branch
}

// Large-population tail-shuffle path (N>10000, k>N/50). CH_C is a full-array
// check; the second case fingerprints a 4000-element draw.
void test_choice_tail_shuffle() {
    CHECK(choice_matches(9ULL, 12000, 300, CH_C, 300));
    std::vector<int64_t> out(4000);
    n4m_rng_pcg64 rng;
    n4m_pcg64_engine_seed(&rng, 0);
    CHECK(n4m_aug_rng_choice_no_replace(&rng, 15000, 4000, out.data()) == 0);
    const int64_t first8[] = {3537, 2636, 897, 13659, 13712, 4255, 10666, 8649};
    for (int i = 0; i < 8; ++i) CHECK(out[i] == first8[i]);
    const int64_t last4[] = {4045, 7666, 9553, 12759};
    for (int i = 0; i < 4; ++i) CHECK(out[3996 + i] == last4[i]);
    int64_t sum = 0;
    for (int64_t v : out) sum += v;
    CHECK(sum == 30101813);
}

}  // namespace

int main() {
    test_bounded();
    test_integers();
    test_advance_resets_cache();
    test_choice_floyd();
    test_choice_tail_shuffle();
    if (g_failures == 0) {
        std::printf("=== n4m_internal_tests: all RNG choice/bounded parity OK ===\n");
        return 0;
    }
    std::printf("=== n4m_internal_tests: %d failure(s) ===\n", g_failures);
    return 1;
}
