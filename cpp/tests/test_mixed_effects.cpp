// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "harness.hpp"
#include "n4m/mixed_effects.h"
namespace {
struct Fixture {
    std::vector<double> x, y;
    std::vector<int64_t> groups;
    Fixture() {
        const double noise[] = {-0.1, 0.3, -0.3, 0.1};
        for (int64_t g = 0; g < 5; ++g)
            for (int j = 0; j < 4; ++j) {
                const double slope = j - 1.5;
                x.push_back(slope);
                x.push_back(1.0);  // Intercept deliberately second.
                y.push_back(2.0 + 0.75 * slope + (static_cast<double>(g) - 2.0) + noise[j]);
                groups.push_back(g);
            }
    }
};
struct Outputs {
    std::array<double, 2> beta{{77, 77}}, se{{77, 77}}, variances{{77, 77}};
    std::array<double, 4> covariance{{77, 77, 77, 77}};
    n4m_mixed_effects_summary_v1_t summary{};
    Outputs() {
        summary.struct_size = sizeof(summary);
        summary.reason = 77;
    }
};
n4m_mixed_effects_options_v1_t options() {
    n4m_mixed_effects_options_v1_t o{};
    N4M_TEST_REQUIRE(n4m_mixed_effects_options_init_v1(&o) == N4M_OK);
    return o;
}
n4m_matrix_view_t view(void* ptr, int64_t rows, int64_t cols, n4m_dtype_t dtype = N4M_DTYPE_F64) {
    return {ptr, rows, cols, cols, 1, dtype, 0};
}
n4m_status_t call(Fixture& f, Outputs& out, n4m_mixed_effects_options_v1_t o) {
    auto x = view(f.x.data(), 20, 2), y = view(f.y.data(), 20, 1),
         g = view(f.groups.data(), 20, 1, N4M_DTYPE_I64);
    return n4m_mixed_effects_fit_v1(&x,
                                    &y,
                                    &g,
                                    &o,
                                    out.beta.data(),
                                    2,
                                    out.covariance.data(),
                                    4,
                                    out.se.data(),
                                    2,
                                    out.variances.data(),
                                    2,
                                    &out.summary);
}
void backend_off_or_real_fit() {
    Fixture f;
    Outputs out;
    auto o = options();
    const auto status = call(f, out, o);
    if (!n4m_mixed_effects_backend_available_v1()) {
        N4M_TEST_REQUIRE(status == N4M_ERR_BACKEND_UNAVAILABLE);
        N4M_TEST_REQUIRE(out.beta[0] == 77 && out.summary.reason == 77);
        return;
    }
    N4M_TEST_REQUIRE(status == N4M_OK);
    N4M_TEST_REQUIRE(out.summary.fit_state == N4M_MIXED_EFFECTS_FITTED);
    N4M_TEST_REQUIRE(out.summary.design_rank == 2 && out.summary.converged == 1);
    N4M_TEST_REQUIRE(std::abs(out.beta[0] - 0.75) < 1e-8 && std::abs(out.beta[1] - 2.0) < 1e-8);
    N4M_TEST_REQUIRE(std::abs(out.variances[0] - 1.0 / 14.0) < 1e-5);
    N4M_TEST_REQUIRE(std::abs(out.covariance[3] - 0.5) < 1e-4);
    N4M_TEST_REQUIRE(std::abs(out.se[1] * out.se[1] - out.covariance[3]) < 1e-12);
    N4M_TEST_REQUIRE(std::isfinite(out.summary.log_likelihood));
}
void strides_and_refits_preserve_order() {
    if (!n4m_mixed_effects_backend_available_v1())
        return;
    Fixture f;
    Outputs expected;
    auto o = options();
    N4M_TEST_REQUIRE(call(f, expected, o) == N4M_OK);
    std::vector<double> xx(80, -99), yy(60, -99);
    std::vector<int64_t> gg(60, -99);
    for (size_t i = 0; i < 20; ++i) {
        xx[2 * i] = f.x[2 * i];
        xx[40 + 2 * i] = f.x[2 * i + 1];
        yy[3 * i] = f.y[i];
        gg[3 * i] = f.groups[i];
    }
    auto x = view(xx.data(), 20, 2);
    x.row_stride = 2;
    x.col_stride = 40;
    auto y = view(yy.data(), 20, 1);
    y.row_stride = 3;
    auto g = view(gg.data(), 20, 1, N4M_DTYPE_I64);
    g.row_stride = 3;
    for (int repeat = 0; repeat < 2; ++repeat) {
        Outputs out;
        N4M_TEST_REQUIRE(n4m_mixed_effects_fit_v1(&x,
                                                  &y,
                                                  &g,
                                                  &o,
                                                  out.beta.data(),
                                                  2,
                                                  out.covariance.data(),
                                                  4,
                                                  out.se.data(),
                                                  2,
                                                  out.variances.data(),
                                                  2,
                                                  &out.summary)
                         == N4M_OK);
        for (size_t j = 0; j < 2; ++j)
            N4M_TEST_REQUIRE(std::abs(out.beta[j] - expected.beta[j]) < 1e-12);
        for (size_t j = 0; j < 4; ++j)
            N4M_TEST_REQUIRE(std::abs(out.covariance[j] - expected.covariance[j]) < 1e-12);
    }
}
void technical_failures_are_atomic() {
    if (!n4m_mixed_effects_backend_available_v1())
        return;
    Fixture f;
    Outputs out;
    auto o = options();
    auto x = view(f.x.data(), 20, 2), y = view(f.y.data(), 20, 1),
         g = view(f.groups.data(), 20, 1, N4M_DTYPE_I64);
    auto run = [&](double* se, int64_t bs = 2) {
        return n4m_mixed_effects_fit_v1(&x,
                                        &y,
                                        &g,
                                        &o,
                                        out.beta.data(),
                                        bs,
                                        out.covariance.data(),
                                        4,
                                        se,
                                        2,
                                        out.variances.data(),
                                        2,
                                        &out.summary);
    };
    N4M_TEST_REQUIRE(run(out.se.data(), 1) == N4M_ERR_SHAPE_MISMATCH);
    g.dtype = N4M_DTYPE_F64;
    N4M_TEST_REQUIRE(run(out.se.data()) == N4M_ERR_DTYPE_MISMATCH);
    g.dtype = N4M_DTYPE_I64;
    g.rows = 19;
    N4M_TEST_REQUIRE(run(out.se.data()) == N4M_ERR_SHAPE_MISMATCH);
    g.rows = 20;
    x.row_stride = -1;
    N4M_TEST_REQUIRE(run(out.se.data()) == N4M_ERR_STRIDE_INVALID);
    x.row_stride = std::numeric_limits<int64_t>::max();
    N4M_TEST_REQUIRE(run(out.se.data()) == N4M_ERR_STRIDE_INVALID);
    x.row_stride = 2;
    f.y[3] = std::numeric_limits<double>::infinity();
    N4M_TEST_REQUIRE(run(out.se.data()) == N4M_ERR_INVALID_ARGUMENT);
    f.y[3] = 0;
    f.groups[2] = -1;
    N4M_TEST_REQUIRE(run(out.se.data()) == N4M_ERR_INVALID_ARGUMENT);
    f.groups[2] = 0;
    N4M_TEST_REQUIRE(run(out.beta.data()) == N4M_ERR_INVALID_ARGUMENT);
    o.reml = 2;
    N4M_TEST_REQUIRE(run(out.se.data()) == N4M_ERR_INVALID_ARGUMENT);
    o.reml = 1;
    o.struct_size = 0;
    N4M_TEST_REQUIRE(run(out.se.data()) == N4M_ERR_ABI_MISMATCH);
    N4M_TEST_REQUIRE(out.beta[0] == 77 && out.se[0] == 77 && out.variances[0] == 77
                     && out.covariance[0] == 77 && out.summary.reason == 77);
}
void statistical_refusals_withhold_estimates() {
    if (!n4m_mixed_effects_backend_available_v1())
        return;
    Fixture f;
    Outputs out;
    auto o = options();
    for (size_t i = 0; i < 20; ++i)
        f.x[2 * i] = 1;
    N4M_TEST_REQUIRE(call(f, out, o) == N4M_OK);
    N4M_TEST_REQUIRE(out.summary.fit_state == N4M_MIXED_EFFECTS_RANK_DEFICIENT
                     && out.summary.design_rank == 1);
    N4M_TEST_REQUIRE(std::isnan(out.beta[0]) && std::isnan(out.covariance[0])
                     && std::isnan(out.se[0]) && std::isnan(out.variances[0]));
    Fixture full;
    Outputs limited;
    o.max_evaluations = 1;
    N4M_TEST_REQUIRE(call(full, limited, o) == N4M_OK);
    N4M_TEST_REQUIRE(limited.summary.fit_state == N4M_MIXED_EFFECTS_NONCONVERGED
                     && !limited.summary.converged);
    N4M_TEST_REQUIRE(std::isnan(limited.beta[0]));
}
}  // namespace
int main() {
    n4m_testing::Runner runner("mixed_effects");
    runner.run("backend availability or real analytic fit", backend_off_or_real_fit);
    runner.run("strides and repeated fits preserve design order",
               strides_and_refits_preserve_order);
    runner.run("technical failures never modify outputs", technical_failures_are_atomic);
    runner.run("statistical refusals withhold all estimates",
               statistical_refusals_withhold_estimates);
    return runner.finalize();
}
