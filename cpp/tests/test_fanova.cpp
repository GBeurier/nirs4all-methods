// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
#include "harness.hpp"
#include "n4m/fanova.h"
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace {
n4m_matrix_view_t view(const double* data, int64_t rows, int64_t cols) {
    return {const_cast<double*>(data), rows, cols, cols, 1, N4M_DTYPE_F64, 0};
}
struct Fixture {
    std::vector<double> X, y;
    std::array<double, 4> bounds{0, 1, 0, 1};
    std::array<uint32_t, 2> types{0, 0};
    std::array<double, 4> mains;
    std::array<double, 8> pairs;
    n4m_fanova_options_v1_t options;
    n4m_fanova_summary_v1_t summary{};
    Fixture() {
        N4M_TEST_REQUIRE(n4m_fanova_options_init_v1(&options) == N4M_OK);
        options.num_trees = 12;
        options.max_depth = 8;
        summary.struct_size = sizeof(summary);
        mains.fill(-123);
        pairs.fill(-456);
    }
    n4m_status_t fit(bool validate = false) {
        auto vx = view(X.data(), static_cast<int64_t>(y.size()), 2);
        auto vy = view(y.data(), static_cast<int64_t>(y.size()), 1);
        auto vb = view(bounds.data(), 2, 2);
        return n4m_fanova_fit_v1(&vx, &vy, &vb, types.data(), validate ? &vx : nullptr,
            validate ? &vy : nullptr, &options, mains.data(), static_cast<int64_t>(mains.size()),
            pairs.data(), static_cast<int64_t>(pairs.size()), &summary);
    }
    void xor_data() {
        types = {2, 2};
        for (int i = 0; i < 128; ++i)
            for (int a = 0; a < 2; ++a)
                for (int b = 0; b < 2; ++b) {
                    X.insert(X.end(), {static_cast<double>(a), static_cast<double>(b)});
                    y.push_back(a == b ? -1 : 1);
                }
    }
};
void defaults_and_backend() {
    N4M_TEST_REQUIRE(n4m_fanova_options_init_v1(nullptr) == N4M_ERR_NULL_POINTER);
    Fixture f;
    f.xor_data();
    const auto status = f.fit();
    if (n4m_fanova_backend_available_v1())
        N4M_TEST_REQUIRE(status == N4M_OK);
    else {
        N4M_TEST_REQUIRE(status == N4M_ERR_BACKEND_UNAVAILABLE);
        N4M_TEST_REQUIRE(f.mains[0] == -123 && f.pairs[0] == -456);
        N4M_TEST_REQUIRE(f.summary.trees == 0);
    }
}
void interaction_and_repeated_calls() {
    if (!n4m_fanova_backend_available_v1()) return;
    Fixture f;
    f.xor_data();
    N4M_TEST_REQUIRE(f.fit(true) == N4M_OK);
    N4M_TEST_REQUIRE(f.summary.fit_state == N4M_FANOVA_FITTED);
    N4M_TEST_REQUIRE(f.summary.informative_trees == f.options.num_trees);
    N4M_TEST_REQUIRE(std::abs(f.mains[0]) < 1e-12 && std::abs(f.mains[2]) < 1e-12);
    N4M_TEST_REQUIRE(std::abs(f.pairs[2] - 1) < 1e-12);
    N4M_TEST_REQUIRE(f.pairs[2] == f.pairs[4] && std::isnan(f.pairs[0]));
    N4M_TEST_REQUIRE(f.summary.train_rmse < 1e-12 && f.summary.validation_rmse < 1e-12);
    N4M_TEST_REQUIRE(std::abs(f.summary.validation_r2 - 1) < 1e-12);
    N4M_TEST_REQUIRE(f.summary.remainder_fraction < 1e-12);
    const auto mains = f.mains;
    const auto pairs = f.pairs;
    const auto evaluations = f.summary.marginal_evaluations;
    N4M_TEST_REQUIRE(f.fit() == N4M_OK);
    N4M_TEST_REQUIRE(std::memcmp(mains.data(), f.mains.data(), sizeof(mains)) == 0);
    N4M_TEST_REQUIRE(std::memcmp(pairs.data(), f.pairs.data(), sizeof(pairs)) == 0);
    N4M_TEST_REQUIRE(evaluations == f.summary.marginal_evaluations);
    N4M_TEST_REQUIRE(std::isnan(f.summary.validation_rmse));
    f.options.max_order = 1;
    N4M_TEST_REQUIRE(f.fit() == N4M_OK);
    for (double value : f.pairs) N4M_TEST_REQUIRE(std::isnan(value));
    N4M_TEST_REQUIRE(std::abs(f.summary.remainder_fraction - 1) < 1e-12);
}
void continuous_additive_and_strides() {
    if (!n4m_fanova_backend_available_v1()) return;
    Fixture f;
    for (int a = 0; a <= 20; ++a)
        for (int b = 0; b <= 20; ++b) {
            f.X.insert(f.X.end(), {a / 20.0, b / 20.0});
            f.y.push_back(a / 20.0 + 2 * b / 20.0);
        }
    N4M_TEST_REQUIRE(f.fit() == N4M_OK);
    N4M_TEST_REQUIRE(f.summary.fit_state == N4M_FANOVA_FITTED);
    N4M_TEST_REQUIRE(std::abs(f.mains[0] - .2) < .08);
    N4M_TEST_REQUIRE(std::abs(f.mains[2] - .8) < .08);
    N4M_TEST_REQUIRE(f.pairs[2] < .08);
    const auto expected = f.mains;
    std::vector<double> columns(f.X.size() * 2, 99), ys(f.y.size() * 3, 99);
    for (size_t i = 0; i < f.y.size(); ++i) {
        columns[2 * i] = f.X[2 * i];
        columns[2 * (i + f.y.size())] = f.X[2 * i + 1];
        ys[i * 3] = f.y[i];
    }
    auto vx = view(columns.data(), static_cast<int64_t>(f.y.size()), 2);
    vx.row_stride = 2;
    vx.col_stride = 2 * static_cast<int64_t>(f.y.size());
    auto vy = view(ys.data(), static_cast<int64_t>(f.y.size()), 1);
    vy.row_stride = 3;
    auto vb = view(f.bounds.data(), 2, 2);
    N4M_TEST_REQUIRE(n4m_fanova_fit_v1(&vx, &vy, &vb, f.types.data(), nullptr, nullptr,
        &f.options, f.mains.data(), 4, f.pairs.data(), 8, &f.summary) == N4M_OK);
    N4M_TEST_REQUIRE(std::memcmp(expected.data(), f.mains.data(), sizeof(expected)) == 0);
    // A matrix view owns a byte pointer; contiguous NumPy buffers can be
    // unaligned even when their strides are exact element multiples.
    std::vector<unsigned char> unaligned_x(f.X.size() * sizeof(double) + 1);
    std::vector<unsigned char> unaligned_y(f.y.size() * sizeof(double) + 1);
    std::vector<unsigned char> unaligned_bounds(sizeof(f.bounds) + 1);
    std::memcpy(unaligned_x.data() + 1, f.X.data(), f.X.size() * sizeof(double));
    std::memcpy(unaligned_y.data() + 1, f.y.data(), f.y.size() * sizeof(double));
    std::memcpy(unaligned_bounds.data() + 1, f.bounds.data(), sizeof(f.bounds));
    vx = view(f.X.data(), static_cast<int64_t>(f.y.size()), 2);
    vy = view(f.y.data(), static_cast<int64_t>(f.y.size()), 1);
    vx.data = unaligned_x.data() + 1;
    vy.data = unaligned_y.data() + 1;
    vb.data = unaligned_bounds.data() + 1;
    N4M_TEST_REQUIRE(n4m_fanova_fit_v1(&vx, &vy, &vb, f.types.data(), nullptr, nullptr,
        &f.options, f.mains.data(), 4, f.pairs.data(), 8, &f.summary) == N4M_OK);
    N4M_TEST_REQUIRE(std::memcmp(expected.data(), f.mains.data(), sizeof(expected)) == 0);
    // Both a different seed and independent repeated fits must remain valid.
    f.options.seed = 123456789;
    N4M_TEST_REQUIRE(f.fit() == N4M_OK);
    N4M_TEST_REQUIRE(f.summary.fit_state == N4M_FANOVA_FITTED);
    N4M_TEST_REQUIRE(std::memcmp(expected.data(), f.mains.data(), sizeof(expected)) != 0);
}
void categorical_full_domain_and_constant() {
    if (!n4m_fanova_backend_available_v1()) return;
    Fixture f;
    f.types = {3, 1}; f.bounds = {0, 2, 0, 0};
    for (int i = 0; i < 128; ++i)
        for (int k = 0; k < 3; ++k) {
            f.X.insert(f.X.end(), {static_cast<double>(k), 0});
            f.y.push_back(k == 2 ? 2 : -1);
        }
    N4M_TEST_REQUIRE(f.fit() == N4M_OK);
    N4M_TEST_REQUIRE(f.summary.fit_state == N4M_FANOVA_FITTED);
    N4M_TEST_REQUIRE(std::abs(f.mains[0] - 1) < 1e-12 && f.mains[2] == 0);
    // Uniform over all three categories: variance of {-1,-1,2} is 2.
    N4M_TEST_REQUIRE(std::abs(f.summary.mean_tree_variance - 2) < 1e-12);
    // A declared but unobserved category is still integrated under its uniform
    // mass. Its predicted response is surrogate extrapolation, not new evidence.
    for (size_t i = 0; i < f.y.size(); ++i) {
        f.X[2 * i] = (i % 2) ? 0 : 2;
        f.y[i] = (i % 2) ? -1 : 2;
    }
    N4M_TEST_REQUIRE(f.fit() == N4M_OK && f.summary.fit_state == N4M_FANOVA_FITTED);
    N4M_TEST_REQUIRE(std::abs(f.mains[0] - 1) < 1e-12 && f.mains[2] == 0);
    f.y.assign(f.y.size(), 7);
    N4M_TEST_REQUIRE(f.fit(true) == N4M_OK);
    N4M_TEST_REQUIRE(f.summary.fit_state == N4M_FANOVA_CONSTANT_RESPONSE);
    for (double value : f.mains) N4M_TEST_REQUIRE(std::isnan(value));
    for (double value : f.pairs) N4M_TEST_REQUIRE(std::isnan(value));
    N4M_TEST_REQUIRE(f.summary.trees == 0);
}
void invalid_and_budget_atomicity() {
    if (!n4m_fanova_backend_available_v1()) return;
    Fixture f; f.xor_data();
    const auto summary = f.summary;
    auto unchanged = [&]() {
        N4M_TEST_REQUIRE(f.mains[0] == -123 && f.pairs[0] == -456);
        N4M_TEST_REQUIRE(std::memcmp(&f.summary, &summary, sizeof(summary)) == 0);
    };
    f.X[0] = .5;
    N4M_TEST_REQUIRE(f.fit() == N4M_ERR_INVALID_ARGUMENT); unchanged();
    f.X[0] = 0; f.options.max_marginal_evaluations = 1;
    N4M_TEST_REQUIRE(f.fit() == N4M_ERR_UNSUPPORTED); unchanged();
    f.options.max_marginal_evaluations = 1000000; f.options.reserved = 1;
    N4M_TEST_REQUIRE(f.fit() == N4M_ERR_INVALID_ARGUMENT); unchanged();
    f.options.reserved = 0; f.y[0] = std::numeric_limits<double>::infinity();
    N4M_TEST_REQUIRE(f.fit() == N4M_ERR_INVALID_ARGUMENT); unchanged();
    f.y[0] = -1; f.bounds[1] = 2;
    N4M_TEST_REQUIRE(f.fit() == N4M_ERR_INVALID_ARGUMENT); unchanged();
    f.bounds[1] = 1;
    auto vx = view(f.X.data(), static_cast<int64_t>(f.y.size()), 2), vy = view(f.y.data(), static_cast<int64_t>(f.y.size()), 1);
    auto vb = view(f.bounds.data(), 2, 2);
    vx.row_stride = std::numeric_limits<int64_t>::max();
    N4M_TEST_REQUIRE(n4m_fanova_fit_v1(&vx, &vy, &vb, f.types.data(), nullptr, nullptr,
        &f.options, f.mains.data(), 4, f.pairs.data(), 8, &f.summary) == N4M_ERR_STRIDE_INVALID);
    unchanged();
}
void higher_order_oracle_and_constant_trees() {
    if (!n4m_fanova_backend_available_v1()) return;
    std::vector<double> X, y;
    // Orthogonal +/-1 factorial terms: V={1,4,0,0,0,0,9}, total14.
    // The three-way interaction belongs in the remainder at max_order=2.
    for (int repeat = 0; repeat < 128; ++repeat)
        for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 2; ++b)
                for (int c = 0; c < 2; ++c) {
                    X.insert(X.end(), {static_cast<double>(a), static_cast<double>(b),
                                       static_cast<double>(c)});
                    const double aa = 2 * a - 1, bb = 2 * b - 1, cc = 2 * c - 1;
                    y.push_back(aa + 2 * bb + 3 * aa * bb * cc);
                }
    auto vx = view(X.data(), static_cast<int64_t>(y.size()), 3);
    auto vy = view(y.data(), static_cast<int64_t>(y.size()), 1);
    const double bounds[] = {0, 1, 0, 1, 0, 1};
    const uint32_t types[] = {2, 2, 2};
    auto vb = view(bounds, 3, 2);
    std::array<double, 6> mains;
    std::array<double, 18> pairs;
    n4m_fanova_options_v1_t options;
    n4m_fanova_options_init_v1(&options);
    options.num_trees = 1;
    n4m_fanova_summary_v1_t summary{}; summary.struct_size = sizeof(summary);
    auto fit = [&]() {
        return n4m_fanova_fit_v1(&vx, &vy, &vb, types, nullptr, nullptr, &options,
            mains.data(), 6, pairs.data(), 18, &summary);
    };
    N4M_TEST_REQUIRE(fit() == N4M_OK && summary.fit_state == N4M_FANOVA_FITTED);
    N4M_TEST_REQUIRE(std::abs(mains[0] - 1.0 / 14) < 1e-12);
    N4M_TEST_REQUIRE(std::abs(mains[2] - 4.0 / 14) < 1e-12 && mains[4] < 1e-12);
    N4M_TEST_REQUIRE(std::isnan(mains[1]));
    N4M_TEST_REQUIRE(std::abs(summary.remainder_fraction - 9.0 / 14) < 1e-12);
    N4M_TEST_REQUIRE(std::abs(summary.mean_tree_variance - 14) < 1e-12);
    options.min_samples_leaf = 8192;
    N4M_TEST_REQUIRE(fit() == N4M_OK);
    N4M_TEST_REQUIRE(summary.fit_state == N4M_FANOVA_NO_INFORMATIVE_TREES);
    N4M_TEST_REQUIRE(summary.trees == 1 && summary.informative_trees == 0);
    for (double value : mains) N4M_TEST_REQUIRE(std::isnan(value));
    for (double value : pairs) N4M_TEST_REQUIRE(std::isnan(value));
}
void declared_measure_and_extreme_ranges() {
    if (!n4m_fanova_backend_available_v1()) return;
    Fixture f;
    f.types = {0, 2}; f.bounds = {0, 1000, 0, 1};
    for (int repeat = 0; repeat < 128; ++repeat)
        for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 2; ++b) {
                f.X.insert(f.X.end(), {static_cast<double>(a), static_cast<double>(b)});
                f.y.push_back(a + b);
            }
    N4M_TEST_REQUIRE(f.fit() == N4M_OK && f.summary.fit_state == N4M_FANOVA_FITTED);
    // Upstream randomizes split locations between observed values. Every x
    // split is inside [0,1], hence occupies at most 1/1000 of this domain.
    // Its total contribution is tiny compared with the categorical variance 1/4.
    // This rejects an accidental empirical-domain integration (roughly 1/2 each).
    N4M_TEST_REQUIRE(f.mains[0] < .004 && f.mains[2] > .99);
    const double wide_fraction = f.mains[0];
    f.bounds[1] = 1;
    N4M_TEST_REQUIRE(f.fit() == N4M_OK && f.summary.fit_state == N4M_FANOVA_FITTED);
    N4M_TEST_REQUIRE(f.mains[0] > wide_fraction * 10);
    f.types = {0, 1}; f.bounds = {-1e308, 1e308, 0, 0};
    for (size_t i = 0; i < f.y.size(); ++i) {
        f.X[2 * i] = (i % 2) ? -1e308 : 1e308;
        f.X[2 * i + 1] = 0;
        f.y[i] = (i % 2) ? -1 : 1;
    }
    N4M_TEST_REQUIRE(f.fit() == N4M_OK);
    if (std::numeric_limits<long double>::max_exponent > 1024) {
        N4M_TEST_REQUIRE(f.summary.fit_state == N4M_FANOVA_FITTED);
        N4M_TEST_REQUIRE(std::abs(f.mains[0] - 1) < 1e-12 && f.mains[2] == 0);
    } else {
        N4M_TEST_REQUIRE(f.summary.fit_state == N4M_FANOVA_NUMERICAL_REFUSAL);
    }
    // Output variance cannot represent the square of this response range.
    const double maximum = std::numeric_limits<double>::max();
    for (size_t i = 0; i < f.y.size(); ++i) f.y[i] = (i % 2) ? -maximum : maximum;
    N4M_TEST_REQUIRE(f.fit() == N4M_OK);
    N4M_TEST_REQUIRE(f.summary.fit_state == N4M_FANOVA_NUMERICAL_REFUSAL);
    for (double value : f.mains) N4M_TEST_REQUIRE(std::isnan(value));
    // Subnormal responses are genuinely nonconstant, but their squared variance
    // underflows binary64: explicit numerical refusal preserves that distinction.
    f.bounds = {-1, 1, 0, 0};
    const double tiny = std::numeric_limits<double>::denorm_min();
    for (size_t i = 0; i < f.y.size(); ++i) {
        f.X[2 * i] = (i % 2) ? -1 : 1;
        f.y[i] = (i % 2) ? -tiny : tiny;
    }
    N4M_TEST_REQUIRE(f.fit() == N4M_OK);
    N4M_TEST_REQUIRE(f.summary.fit_state == N4M_FANOVA_NUMERICAL_REFUSAL);
    for (double value : f.mains) N4M_TEST_REQUIRE(std::isnan(value));
    // Incomplete correlated observations do not identify real component effects;
    // the returned quantity remains a fitted-surface product-measure attribution.
    f.types = {2, 2}; f.bounds = {0, 1, 0, 1};
    for (size_t i = 0; i < f.y.size(); ++i) {
        f.X[2 * i] = f.X[2 * i + 1] = static_cast<double>(i % 2);
        f.y[i] = static_cast<double>(i % 2);
    }
    N4M_TEST_REQUIRE(f.fit() == N4M_OK && f.summary.fit_state == N4M_FANOVA_FITTED);
    N4M_TEST_REQUIRE(std::abs(f.mains[0] + f.mains[2] - 1) < 1e-12);
}
}  // namespace

int main() {
    n4m_testing::Runner runner("fanova");
    runner.run("defaults and optional backend", defaults_and_backend);
    runner.run("pure interaction, holdout and repeated fit", interaction_and_repeated_calls);
    runner.run("continuous additive response, strides and seed", continuous_additive_and_strides);
    runner.run("categorical uniform domain and constant refusal", categorical_full_domain_and_constant);
    runner.run("invalid inputs and budget preserve outputs", invalid_and_budget_atomicity);
    runner.run("three-way oracle and noninformative trees", higher_order_oracle_and_constant_trees);
    runner.run("declared measure and finite extreme ranges", declared_measure_and_extreme_ranges);
    return runner.finalize();
}
