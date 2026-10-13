// SPDX-License-Identifier: CECILL-2.1
// n4m_stats_group_means: hand-computed means, strided column-major input,
// extreme finite magnitudes and argument refusals.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "n4m/n4m.h"

#define REQUIRE(x) do { if (!(x)) throw std::runtime_error(std::string(#x) + " @" + std::to_string(__LINE__)); } while (false)

namespace {

void hand_values(n4m_context_t* ctx) {
    // Rows: group 1 {1,2},{3,6}; group 0 {10,-4}; group 2 {0,0},{1,1},{2,8}.
    std::vector<double> x = {1, 2, 3, 6, 10, -4, 0, 0, 1, 1, 2, 8};
    const int64_t codes[] = {1, 1, 0, 2, 2, 2};
    n4m_matrix_view_t v{};
    REQUIRE(n4m_matrix_view_init_rowmajor(&v, x.data(), 6, 2, N4M_DTYPE_F64) == N4M_OK);
    double means[6];
    int64_t counts[3];
    REQUIRE(n4m_stats_group_means(ctx, &v, codes, 3, means, counts) == N4M_OK);
    const double want[6] = {10, -4, 2, 4, 1, 3};
    for (int i = 0; i < 6; ++i) REQUIRE(std::abs(means[i] - want[i]) <= 1e-15 * (1 + std::abs(want[i])));
    REQUIRE(counts[0] == 1 && counts[1] == 2 && counts[2] == 3);
    // The same matrix stored column-major through strides.
    std::vector<double> c(12);
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 2; ++j) c[static_cast<size_t>(j * 6 + i)] = x[static_cast<size_t>(i * 2 + j)];
    n4m_matrix_view_t cv{c.data(), 6, 2, 1, 6, N4M_DTYPE_F64, 0};
    double cmeans[6];
    REQUIRE(n4m_stats_group_means(ctx, &cv, codes, 3, cmeans, counts) == N4M_OK);
    for (int i = 0; i < 6; ++i) REQUIRE(cmeans[i] == means[i]);
}

void extreme_magnitudes(n4m_context_t* ctx) {
    // A naive sum overflows; the running scaled mean is exact here.
    const double big = std::numeric_limits<double>::max();
    std::vector<double> x = {big, big, big, -big, 0.5 * big, 0.25 * big};
    const int64_t codes[] = {0, 0, 0, 1, 1, 1};
    n4m_matrix_view_t v{};
    REQUIRE(n4m_matrix_view_init_rowmajor(&v, x.data(), 6, 1, N4M_DTYPE_F64) == N4M_OK);
    double means[2];
    int64_t counts[2];
    REQUIRE(n4m_stats_group_means(ctx, &v, codes, 2, means, counts) == N4M_OK);
    REQUIRE(means[0] == big);
    REQUIRE(std::abs(means[1] - (-0.25 / 3.0) * big) <= 1e-15 * big);
}

void refusals(n4m_context_t* ctx) {
    std::vector<double> x = {1, 2, 3};
    n4m_matrix_view_t v{};
    REQUIRE(n4m_matrix_view_init_rowmajor(&v, x.data(), 3, 1, N4M_DTYPE_F64) == N4M_OK);
    double means[3];
    int64_t counts[3];
    const int64_t gap[] = {0, 0, 2};       // group 1 empty
    const int64_t outside[] = {0, 1, 3};   // code >= n_groups
    REQUIRE(n4m_stats_group_means(ctx, &v, gap, 3, means, counts) == N4M_ERR_INVALID_ARGUMENT);
    REQUIRE(n4m_stats_group_means(ctx, &v, outside, 3, means, counts) == N4M_ERR_INVALID_ARGUMENT);
    REQUIRE(n4m_stats_group_means(ctx, &v, gap, 4, means, counts) == N4M_ERR_INVALID_ARGUMENT);
    x[1] = std::numeric_limits<double>::quiet_NaN();
    const int64_t ok[] = {0, 1, 1};
    REQUIRE(n4m_stats_group_means(ctx, &v, ok, 2, means, counts) == N4M_ERR_INVALID_ARGUMENT);
    REQUIRE(n4m_stats_group_means(ctx, &v, nullptr, 2, means, counts) == N4M_ERR_NULL_POINTER);
}

}  // namespace

int main() {
    n4m_context_t* ctx = nullptr;
    if (n4m_context_create(&ctx) != N4M_OK) return 1;
    int failures = 0;
    for (auto [name, fn] : {std::pair{"hand_values", &hand_values},
                            std::pair{"extreme_magnitudes", &extreme_magnitudes},
                            std::pair{"refusals", &refusals}}) {
        try {
            fn(ctx);
            std::printf("ok %s\n", name);
        } catch (const std::exception& e) {
            std::printf("FAILED %s: %s\n", name, e.what());
            ++failures;
        }
    }
    n4m_context_destroy(ctx);
    return failures == 0 ? 0 : 1;
}
