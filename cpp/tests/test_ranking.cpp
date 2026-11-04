// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "harness.hpp"
#include "n4m/ranking.h"
namespace {
n4m_ranking_options_v1_t defaults(int method = 0) {
    n4m_ranking_options_v1_t o;
    N4M_TEST_REQUIRE(n4m_ranking_options_init_v1(&o) == N4M_OK);
    o.method = method;
    return o;
}
n4m_matrix_view_t view(const double* x, int64_t rows, int64_t cols) {
    return {const_cast<double*>(x), rows, cols, cols, 1, N4M_DTYPE_F64, 0};
}
void complete_ranks() {
    const double x[] = {1, 2, 3, 1, 2, 3, 1, 2, 3};
    const uint8_t m[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    auto v = view(x, 3, 3);
    auto o = defaults();
    std::array<double, 30> c;
    std::array<double, 9> s;
    N4M_TEST_REQUIRE(n4m_ranking_compute_v1(&v, m, &o, c.data(), 30, s.data(), 9) == N4M_OK);
    N4M_TEST_REQUIRE(c[0] == 1 && c[10] == 2 && c[20] == 3);
    N4M_TEST_REQUIRE(c[9] == 1 && c[19] == 2 && c[29] == 3);
    N4M_TEST_REQUIRE(s[2] == 9 && s[3] == 1 && std::abs(s[4] - 6) < 1e-12);
    N4M_TEST_REQUIRE(std::abs(s[8] - std::exp(-3)) < 1e-12);
    N4M_TEST_REQUIRE(c[6] == 6 && c[28] == 6 && std::isnan(c[1]));
}
void ties_and_complete_subset() {
    const double x[] = {1, 1, 3, 1, 2, 2, 2, 1, 3};
    const uint8_t m[] = {1, 1, 1, 1, 1, 1, 1, 1, 0};
    auto v = view(x, 3, 3);
    auto o = defaults();
    std::array<double, 30> c;
    std::array<double, 9> s;
    N4M_TEST_REQUIRE(n4m_ranking_compute_v1(&v, m, &o, c.data(), 30, s.data(), 9) == N4M_OK);
    N4M_TEST_REQUIRE(s[1] == 2 && std::abs(s[4] - 3) < 1e-12);
    N4M_TEST_REQUIRE(std::abs(c[2] - 1.5) < 1e-12 && std::abs(c[12] - 5.0 / 3) < 1e-12);
    o.tie_tolerance = 1.1;
    N4M_TEST_REQUIRE(n4m_ranking_compute_v1(&v, m, &o, c.data(), 30, s.data(), 9) == N4M_OK);
    // First complete row: bounded range groups [1,1] [3], not chained ties.
    N4M_TEST_REQUIRE(c[7] > 0);
}
void transitive_incomplete_bridge() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double x[] = {1, 2, nan, nan, 1, 2};
    const uint8_t m[] = {1, 1, 0, 0, 1, 1};
    auto v = view(x, 2, 3);
    std::array<double, 30> c;
    std::array<double, 9> s;
    for (int method = 1; method <= 4; method++) {
        auto o = defaults(method);
        N4M_TEST_REQUIRE(n4m_ranking_compute_v1(&v, m, &o, c.data(), 30, s.data(), 9) == N4M_OK);
        N4M_TEST_REQUIRE(s[1] == 0 && s[2] == 2 && s[3] == 1);
        N4M_TEST_REQUIRE(c[0] > c[10] && c[10] > c[20]);
        N4M_TEST_REQUIRE(c[9] == 1 && c[29] == 3);
        N4M_TEST_REQUIRE(c[3] == 1 && c[13] == 2 && c[23] == 1);
    }
}
void disconnected_and_missing() {
    const double x[] = {1, 2, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0};
    const uint8_t m[] = {1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0};
    auto v = view(x, 2, 6);
    auto o = defaults(3);
    std::array<double, 60> c;
    std::array<double, 9> s;
    N4M_TEST_REQUIRE(n4m_ranking_compute_v1(&v, m, &o, c.data(), 60, s.data(), 9) == N4M_OK);
    // Candidate 4 co-occurs with 0/1, so graph has three components including absent 5.
    N4M_TEST_REQUIRE(s[3] == 3 && c[5] == c[15] && c[25] != c[5]);
    for (int j = 0; j < 6; ++j)
        N4M_TEST_REQUIRE(std::isnan(c[j * 10 + 9]));
    N4M_TEST_REQUIRE(std::isnan(c[50]) && std::isnan(c[51]) && std::isnan(c[52]));
}
void elo_golden_and_direction() {
    const double x[] = {1, 2};
    const uint8_t m[] = {1, 1};
    auto v = view(x, 1, 2);
    auto o = defaults(1);
    std::array<double, 20> c;
    std::array<double, 9> s;
    N4M_TEST_REQUIRE(n4m_ranking_compute_v1(&v, m, &o, c.data(), 20, s.data(), 9) == N4M_OK);
    N4M_TEST_REQUIRE(c[0] == 41 && c[10] == 9);
    o.higher_is_better = 1;
    N4M_TEST_REQUIRE(n4m_ranking_compute_v1(&v, m, &o, c.data(), 20, s.data(), 9) == N4M_OK);
    N4M_TEST_REQUIRE(c[0] == 9 && c[10] == 41);
}
void arbitrary_stride() {
    const double x[] = {1, 99, 1, 99, 2, 99, 2, 99, 3, 99, 3, 99};
    const uint8_t m[] = {1, 1, 1, 1, 1, 1};
    auto v = view(x, 2, 3);
    v.row_stride = 2;
    v.col_stride = 4;
    auto o = defaults();
    std::array<double, 30> c;
    std::array<double, 9> s;
    N4M_TEST_REQUIRE(n4m_ranking_compute_v1(&v, m, &o, c.data(), 30, s.data(), 9) == N4M_OK);
    N4M_TEST_REQUIRE(c[0] == 1 && c[10] == 2 && c[20] == 3);
    v.row_stride = std::numeric_limits<int64_t>::max();
    N4M_TEST_REQUIRE(n4m_ranking_compute_v1(&v, m, &o, c.data(), 30, s.data(), 9)
                     == N4M_ERR_STRIDE_INVALID);
}
void failures_preserve_output() {
    const double x[] = {1, std::numeric_limits<double>::infinity()};
    uint8_t m[] = {1, 1};
    auto v = view(x, 1, 2);
    auto o = defaults();
    std::array<double, 20> c;
    c.fill(-77);
    std::array<double, 9> s;
    s.fill(-77);
    N4M_TEST_REQUIRE(n4m_ranking_compute_v1(&v, m, &o, c.data(), 20, s.data(), 9)
                     == N4M_ERR_INVALID_ARGUMENT);
    for (double z : c)
        N4M_TEST_REQUIRE(z == -77);
    for (double z : s)
        N4M_TEST_REQUIRE(z == -77);
    m[1] = 0;
    N4M_TEST_REQUIRE(n4m_ranking_compute_v1(&v, m, &o, c.data(), 20, s.data(), 9) == N4M_OK);
    m[1] = 2;
    N4M_TEST_REQUIRE(n4m_ranking_compute_v1(&v, m, &o, c.data(), 20, s.data(), 9)
                     == N4M_ERR_INVALID_ARGUMENT);
    m[1] = 0;
    o.method = 55;
    N4M_TEST_REQUIRE(n4m_ranking_compute_v1(&v, m, &o, c.data(), 20, s.data(), 9)
                     == N4M_ERR_UNSUPPORTED);
    o = defaults();
    o.struct_size = 0;
    N4M_TEST_REQUIRE(n4m_ranking_compute_v1(&v, m, &o, c.data(), 20, s.data(), 9)
                     == N4M_ERR_ABI_MISMATCH);
    o = defaults();
    o.beta = 1e300;
    N4M_TEST_REQUIRE(n4m_ranking_compute_v1(&v, m, &o, c.data(), 20, s.data(), 9)
                     == N4M_ERR_UNSUPPORTED);
}
void relative_gains() {
    const double a[] = {2, 4, 10}, b[] = {1, 3, 10}, vars[] = {.01, .04, .09};
    const uint8_t m[] = {1, 1, 1};
    std::array<double, 8> out;
    N4M_TEST_REQUIRE(n4m_ranking_relative_gain_v1(a, b, m, nullptr, 3, out.data(), 8) == N4M_OK);
    N4M_TEST_REQUIRE(out[0] == 3 && out[1] == .25 && std::abs(out[2] - .25) < 1e-12
                     && std::isnan(out[4]));
    N4M_TEST_REQUIRE(n4m_ranking_relative_gain_v1(a, b, m, vars, 3, out.data(), 8) == N4M_OK);
    N4M_TEST_REQUIRE(out[5] >= 0 && out[4] > 0);
    const double zero[] = {0, 1, 1};
    N4M_TEST_REQUIRE(n4m_ranking_relative_gain_v1(zero, b, m, nullptr, 3, out.data(), 8)
                     == N4M_ERR_INVALID_ARGUMENT);
}
}  // namespace
int main() {
    n4m_testing::Runner r("ranking");
    r.run("complete ranks and Friedman", complete_ranks);
    r.run("tied ranks and complete subset", ties_and_complete_subset);
    r.run("cross-dataset transitive incomplete bridge", transitive_incomplete_bridge);
    r.run("disconnected and missing", disconnected_and_missing);
    r.run("Elo closed form and direction", elo_golden_and_direction);
    r.run("arbitrary stride and overflow", arbitrary_stride);
    r.run("failure output atomicity", failures_preserve_output);
    r.run("positive loss gains", relative_gains);
    return r.finalize();
}
