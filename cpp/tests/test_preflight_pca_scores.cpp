// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
// Domains F4.3: resolved weak scores must describe the final loading vectors.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "fixture_parser.hpp"
#include "n4m/metrics.h"

#define REQUIRE(x) do { if (!(x)) throw std::runtime_error(std::string(#x) + " @" + std::to_string(__LINE__)); } while (false)

static n4m_matrix_view_t view(double* data, int64_t rows, int64_t cols) {
    n4m_matrix_view_t result{};
    REQUIRE(n4m_matrix_view_init_rowmajor(&result, data, rows, cols, N4M_DTYPE_F64) == N4M_OK);
    return result;
}

int main() {
    try {
        using namespace n4m_testing;
        const auto fixture = slurp(std::string(N4M_PARITY_FIXTURE_DIR) +
                                   "/preflight_pca_weak_scores_v1.json");
        const auto hex = [&](const std::string& key) {
            auto pos = find_value_for_key(fixture, key, 0, fixture.size());
            std::vector<double> values;
            parse_hex_double_array(fixture, pos, values);
            return values;
        };
        const auto input = hex("input_hex"), factor = hex("factor_hex");
        const auto reference = hex("reference_pc5_scores_hex");
        const auto reference_eigenvalues = hex("reference_eigenvalues_hex");
        const double reference_pearson = hex("reference_pearson_hex")[0];
        const double reference_eta = hex("reference_eta_squared_hex")[0];
        constexpr int64_t n = 24, p = 25;
        constexpr int component = 4;
        // Score L2 accuracy accounts for binary64 centring/projection on this
        // conditioned input. Factor estimates retain Domains' original 1e-10.
        constexpr double score_relative_l2_tolerance = 2e-9;
        constexpr double statistic_atol = 1e-10, statistic_rtol = 1e-10;
        std::vector<int64_t> codes;
        auto pos = find_value_for_key(fixture, "group_codes", 0, fixture.size()) + 1;
        while (codes.size() < static_cast<size_t>(n)) {
            skip_ws(fixture, pos);
            codes.push_back(parse_int64(fixture, pos));
            pos = fixture.find_first_of(",]", pos) + 1;
        }
        REQUIRE(input.size() == static_cast<size_t>(n * p));
        REQUIRE(factor.size() == static_cast<size_t>(n));
        n4m_context_t* ctx = nullptr;
        REQUIRE(n4m_context_create(&ctx) == N4M_OK);
        for (const int perm : {0, 1, 2, 3}) {
            for (const int exponent : {-30, 0, 30}) {
                std::vector<double> prefix;
                for (const int k : {5, 20, 24}) {
                    std::vector<double> x(input.size()), scores(static_cast<size_t>(n * k));
                    std::vector<double> loadings(static_cast<size_t>(k * p)), eigen(k), legacy(scores.size());
                    std::vector<uint8_t> unresolved(k);
                    std::vector<double> values(n), expected(n);
                    std::vector<int64_t> groups(n);
                    for (int64_t i = 0; i < n; ++i) {
                        const auto source_row = (perm & 1) ? n - 1 - i : i;
                        values[i] = factor[source_row];
                        groups[i] = codes[source_row];
                        expected[i] = std::ldexp(reference[source_row], exponent);
                        for (int64_t j = 0; j < p; ++j) {
                            const auto source_col = (perm & 2) ? p - 1 - j : j;
                            x[i * p + j] = std::ldexp(input[source_row * p + source_col], exponent);
                        }
                    }
                    auto X = view(x.data(), n, p), S = view(scores.data(), n, k);
                    auto L = view(loadings.data(), k, p);
                    double moments[2]{}, resolution = 0;
                    REQUIRE(n4m_preflight_pca_compute_resolved(ctx, &X, k, &S, &L, eigen.data(),
                                moments, &resolution, unresolved.data()) == N4M_OK);
                    REQUIRE(unresolved[component] == 0 && eigen[component] > resolution);
                    REQUIRE(std::abs(eigen[component] - std::ldexp(reference_eigenvalues[component],
                                2 * exponent)) <= resolution);
                    for (int a = 0; a < 5; ++a) {
                        double variance = 0;
                        for (int64_t i = 0; i < n; ++i) {
                            variance += scores[i * k + a] * scores[i * k + a] / (n - 1);
                            if (k != 5) REQUIRE(scores[i * k + a] == prefix[i * 5 + a]);
                        }
                        REQUIRE(std::abs(variance - eigen[a]) <= resolution);
                    }
                    if (k == 5) prefix = scores;
                    for (int a = 5; a < k; ++a) {
                        REQUIRE(unresolved[a] == 1 && eigen[a] == 0);
                        for (int64_t i = 0; i < n; ++i) REQUIRE(scores[i * k + a] == 0);
                    }
                    auto LS = view(legacy.data(), n, k);
                    REQUIRE(n4m_preflight_pca_compute(ctx, &X, k, &LS, &L, eigen.data(), moments) == N4M_OK);
                    REQUIRE(legacy == scores);  // Both C ABI entrypoints share the correction.
                    std::vector<double> weak(n);
                    double dot = 0, error2 = 0, norm2 = 0;
                    for (int64_t i = 0; i < n; ++i) dot += scores[i * k + component] * expected[i];
                    const double sign = dot < 0 ? -1.0 : 1.0;
                    for (int64_t i = 0; i < n; ++i) {
                        weak[i] = sign * scores[i * k + component];
                        error2 += (weak[i] - expected[i]) * (weak[i] - expected[i]);
                        norm2 += expected[i] * expected[i];
                    }
                    REQUIRE(std::sqrt(error2 / norm2) <= score_relative_l2_tolerance);
                    double correlation[5]{}, association[8]{};
                    int64_t counts[2]{};
                    REQUIRE(n4m_stats_correlation_test(ctx, values.data(), weak.data(), n, 0, 0, 0,
                                                       correlation, counts) == N4M_OK);
                    REQUIRE(n4m_preflight_factor_association(ctx, groups.data(), weak.data(), n,
                                0, 0, 0, association, counts) == N4M_OK);
                    REQUIRE(std::abs(correlation[0] - reference_pearson) <=
                            statistic_atol + statistic_rtol * std::abs(reference_pearson));
                    REQUIRE(std::abs(association[0] - reference_eta) <=
                            statistic_atol + statistic_rtol * std::abs(reference_eta));
                }
            }
        }
        n4m_context_destroy(ctx);
        std::puts("PCA weak-score projection: 36 cases passed, both public entrypoints");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
