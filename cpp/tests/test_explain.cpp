// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
//
// Tests of the n4m/explain.h public ABI: ask/tell protocol, exact /
// permutation / Kernel SHAP against independent brute-force Shapley values,
// LIME and occlusion properties, the INTERPOLATE bridge, linear SHAP,
// path-dependent TreeSHAP against a brute-force conditional-expectation
// oracle, grouping helpers and summaries. Models are C++ lambdas.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

#include "harness.hpp"
#include "n4m/explain.h"

namespace {

using Model = std::function<void(const double* rows, int64_t n, int64_t f, double* out)>;
using Values = std::vector<double>;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

std::size_t sz(int64_t v) {
    return static_cast<std::size_t>(v);
}

bool close(double a, double b, double tol) {
    return std::fabs(a - b) <= tol * (1.0 + std::fabs(b));
}

struct Ctx {
    n4m_context_t* p = nullptr;
    Ctx() { N4M_TEST_REQUIRE(n4m_context_create(&p) == N4M_OK); }
    ~Ctx() { n4m_context_destroy(p); }
    Ctx(const Ctx&) = delete;
    Ctx& operator=(const Ctx&) = delete;
};

n4m_matrix_view_t view(const double* x, int64_t rows, int64_t cols) {
    return {const_cast<double*>(x), rows, cols, cols, 1, N4M_DTYPE_F64, 0};
}

n4m_explain_options_v1_t options(int32_t method, int32_t baseline, int64_t n_outputs = 1) {
    n4m_explain_options_v1_t o;
    N4M_TEST_REQUIRE(n4m_explain_options_init_v1(&o) == N4M_OK);
    o.method = method;
    o.baseline = baseline;
    o.n_outputs = n_outputs;
    return o;
}

struct Run {
    Values attr, expected, pred, se, rows;
    int64_t total = 0;
    int64_t evaluated = 0;
};

Run run_session(const Values& X, int64_t R, int64_t F, const Values& bg, int64_t B,
                const std::vector<int64_t>& groups, int64_t G,
                const n4m_explain_options_v1_t& opt, const Model& model,
                int64_t capacity = 7, bool keep_rows = false) {
    Ctx ctx;
    const auto xv = view(X.data(), R, F);
    const auto bv = view(bg.data(), B, F);
    n4m_explain_session_t* s = nullptr;
    N4M_TEST_REQUIRE(n4m_explain_session_create_v1(ctx.p, &xv, B > 0 ? &bv : nullptr,
                                                   groups.empty() ? nullptr : groups.data(), G,
                                                   &opt, &s)
                     == N4M_OK);
    Run run;
    int64_t nr = 0, nf = 0, ng = 0, no = 0;
    N4M_TEST_REQUIRE(n4m_explain_session_info_v1(s, &nr, &nf, &ng, &no, &run.total) == N4M_OK);
    N4M_TEST_REQUIRE(nr == R && nf == F && ng == G && no == opt.n_outputs);
    const std::size_t O = sz(opt.n_outputs);
    Values rows(sz(capacity * F)), preds(sz(capacity) * O);
    for (;;) {
        int64_t written = -1;
        N4M_TEST_REQUIRE(n4m_explain_session_next_batch_v1(ctx.p, s, rows.data(), capacity,
                                                           &written)
                         == N4M_OK);
        if (written == 0) {
            break;
        }
        model(rows.data(), written, F, preds.data());
        run.evaluated += written;
        if (keep_rows) {
            run.rows.insert(run.rows.end(), rows.begin(), rows.begin() + written * F);
        }
        N4M_TEST_REQUIRE(n4m_explain_session_submit_v1(ctx.p, s, preds.data(), written)
                         == N4M_OK);
    }
    N4M_TEST_REQUIRE(run.evaluated == run.total);
    run.attr.assign(sz(R * G) * O, 0.0);
    run.expected.assign(sz(R) * O, 0.0);
    run.pred.assign(sz(R) * O, 0.0);
    run.se.assign(run.attr.size(), 0.0);
    const auto na = static_cast<int64_t>(run.attr.size());
    const auto ne = static_cast<int64_t>(run.expected.size());
    N4M_TEST_REQUIRE(n4m_explain_session_result_v1(ctx.p, s, run.attr.data(), na,
                                                   run.expected.data(), ne, run.pred.data(), ne,
                                                   run.se.data(), na)
                     == N4M_OK);
    n4m_explain_session_destroy(s);
    return run;
}

// Brute-force Shapley values over all G! orderings (independent of the library).
Values shapley(std::size_t G, std::size_t O,
               const std::function<Values(const std::vector<int>&)>& v) {
    std::vector<std::size_t> perm(G);
    std::iota(perm.begin(), perm.end(), std::size_t{0});
    Values phi(G * O, 0.0);
    double count = 0.0;
    do {
        std::vector<int> mask(G, 0);
        Values prev = v(mask);
        for (const std::size_t g : perm) {
            mask[g] = 1;
            const Values cur = v(mask);
            for (std::size_t o = 0; o < O; ++o) {
                phi[g * O + o] += cur[o] - prev[o];
            }
            prev = cur;
        }
        count += 1.0;
    } while (std::next_permutation(perm.begin(), perm.end()));
    for (double& p : phi) {
        p /= count;
    }
    return phi;
}

// Interventional group game v(S) = mean_b f(x_S, b_notS).
Values background_game(const Values& x, const Values& bg, std::size_t B, std::size_t F,
                       const std::vector<int64_t>& groups, const std::vector<int>& mask,
                       std::size_t O, const Model& model) {
    Values row(F), out(O), acc(O, 0.0);
    for (std::size_t b = 0; b < B; ++b) {
        for (std::size_t j = 0; j < F; ++j) {
            const std::size_t g = groups.empty() ? j : sz(groups[j]);
            row[j] = mask[g] != 0 ? x[j] : bg[b * F + j];
        }
        model(row.data(), 1, static_cast<int64_t>(F), out.data());
        for (std::size_t o = 0; o < O; ++o) {
            acc[o] += out[o] / static_cast<double>(B);
        }
    }
    return acc;
}

const Model kInteraction = [](const double* r, int64_t n, int64_t f, double* out) {
    for (int64_t i = 0; i < n; ++i) {
        const double* z = r + i * f;
        out[i] = z[0] * z[1] + 2.0 * z[2];
    }
};

void check_additive(const Run& run, std::size_t R, std::size_t G, std::size_t O, double tol) {
    for (std::size_t r = 0; r < R; ++r) {
        for (std::size_t o = 0; o < O; ++o) {
            double sum = run.expected[r * O + o];
            for (std::size_t g = 0; g < G; ++g) {
                sum += run.attr[(r * G + g) * O + o];
            }
            N4M_TEST_REQUIRE(std::fabs(sum - run.pred[r * O + o]) <= tol);
        }
    }
}

// ---------------------------------------------------------------------------

void options_init_defaults() {
    n4m_explain_options_v1_t o;
    N4M_TEST_REQUIRE(n4m_explain_options_init_v1(nullptr) == N4M_ERR_NULL_POINTER);
    N4M_TEST_REQUIRE(n4m_explain_options_init_v1(&o) == N4M_OK);
    N4M_TEST_REQUIRE(o.struct_size == sizeof(o) && sizeof(o) == 72);
    N4M_TEST_REQUIRE(o.method == N4M_EXPLAIN_PERMUTATION_SHAP);
    N4M_TEST_REQUIRE(o.baseline == N4M_EXPLAIN_BASELINE_BACKGROUND);
    N4M_TEST_REQUIRE(o.n_outputs == 1 && o.n_permutations == 10 && o.n_coalitions == 0);
    N4M_TEST_REQUIRE(o.lime_kernel_width == 0.0 && o.lime_ridge == 1.0);
    N4M_TEST_REQUIRE(o.max_batch_rows == 4096 && o.seed == 0);
}

void create_rejections() {
    Ctx ctx;
    const Values x = {1, 2, 3, 4, 5, 6};
    const Values bg = {0, 0, 0, 1, 1, 1};
    const auto xv = view(x.data(), 2, 3);
    const auto bv = view(bg.data(), 2, 3);
    auto o = options(N4M_EXPLAIN_EXACT_SHAP, N4M_EXPLAIN_BASELINE_BACKGROUND);
    n4m_explain_session_t* s = nullptr;
    N4M_TEST_REQUIRE(n4m_explain_session_create_v1(nullptr, &xv, &bv, nullptr, 3, &o, &s)
                     == N4M_ERR_NULL_POINTER);
    N4M_TEST_REQUIRE(n4m_explain_session_create_v1(ctx.p, &xv, &bv, nullptr, 3, &o, nullptr)
                     == N4M_ERR_NULL_POINTER);
    auto bad = o;
    bad.struct_size = 8;
    N4M_TEST_REQUIRE(n4m_explain_session_create_v1(ctx.p, &xv, &bv, nullptr, 3, &bad, &s)
                     == N4M_ERR_ABI_MISMATCH);
    N4M_TEST_REQUIRE(s == nullptr);
    bad = o;
    bad.method = 9;
    N4M_TEST_REQUIRE(n4m_explain_session_create_v1(ctx.p, &xv, &bv, nullptr, 3, &bad, &s)
                     == N4M_ERR_INVALID_ARGUMENT);
    // NULL feature_group requires n_groups == n_features.
    N4M_TEST_REQUIRE(n4m_explain_session_create_v1(ctx.p, &xv, &bv, nullptr, 2, &o, &s)
                     == N4M_ERR_INVALID_ARGUMENT);
    const int64_t empty_group[] = {0, 0, 1};
    N4M_TEST_REQUIRE(n4m_explain_session_create_v1(ctx.p, &xv, &bv, empty_group, 3, &o, &s)
                     == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(n4m_explain_session_create_v1(ctx.p, &xv, nullptr, nullptr, 3, &o, &s)
                     == N4M_ERR_NULL_POINTER);
    const auto narrow = view(bg.data(), 3, 2);
    N4M_TEST_REQUIRE(n4m_explain_session_create_v1(ctx.p, &xv, &narrow, nullptr, 3, &o, &s)
                     == N4M_ERR_SHAPE_MISMATCH);
    const Values xnan = {1, kNaN, 3, 4, 5, 6};
    const auto xnv = view(xnan.data(), 2, 3);
    N4M_TEST_REQUIRE(n4m_explain_session_create_v1(ctx.p, &xnv, &bv, nullptr, 3, &o, &s)
                     == N4M_ERR_INVALID_ARGUMENT);
    // EXACT_SHAP refuses more than 16 groups.
    const Values wide(17, 1.0);
    const auto wv = view(wide.data(), 1, 17);
    N4M_TEST_REQUIRE(n4m_explain_session_create_v1(ctx.p, &wv, &wv, nullptr, 17, &o, &s)
                     == N4M_ERR_UNSUPPORTED);
    // INTERPOLATE refuses non-contiguous groups; the background may be absent.
    auto interp = options(N4M_EXPLAIN_EXACT_SHAP, N4M_EXPLAIN_BASELINE_INTERPOLATE);
    const int64_t split[] = {0, 1, 0};
    N4M_TEST_REQUIRE(n4m_explain_session_create_v1(ctx.p, &xv, nullptr, split, 2, &interp, &s)
                     == N4M_ERR_INVALID_ARGUMENT);
    const int64_t contiguous[] = {0, 0, 1};
    N4M_TEST_REQUIRE(n4m_explain_session_create_v1(ctx.p, &xv, nullptr, contiguous, 2, &interp,
                                                   &s)
                     == N4M_OK);
    n4m_explain_session_destroy(s);
    auto perm = options(N4M_EXPLAIN_PERMUTATION_SHAP, N4M_EXPLAIN_BASELINE_BACKGROUND);
    perm.n_permutations = 0;
    s = nullptr;
    N4M_TEST_REQUIRE(n4m_explain_session_create_v1(ctx.p, &xv, &bv, nullptr, 3, &perm, &s)
                     == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(s == nullptr);
}

void protocol_misuse() {
    Ctx ctx;
    const Values x = {1, 2, 3, 4, 5, 6};
    const Values bg = {0, 0, 0, 1, 1, 1};
    const auto xv = view(x.data(), 2, 3);
    const auto bv = view(bg.data(), 2, 3);
    auto o = options(N4M_EXPLAIN_EXACT_SHAP, N4M_EXPLAIN_BASELINE_BACKGROUND);
    n4m_explain_session_t* s = nullptr;
    N4M_TEST_REQUIRE(n4m_explain_session_create_v1(ctx.p, &xv, &bv, nullptr, 3, &o, &s)
                     == N4M_OK);
    Values rows(4 * 3, 0.0), preds(4, 0.0), attr(18), e(2), p(2);
    int64_t written = 0;
    N4M_TEST_REQUIRE(n4m_explain_session_result_v1(ctx.p, s, attr.data(), 18, e.data(), 2,
                                                   p.data(), 2, nullptr, 0)
                     == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(n4m_explain_session_submit_v1(ctx.p, s, preds.data(), 4)
                     == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(n4m_explain_session_next_batch_v1(ctx.p, s, rows.data(), 0, &written)
                     == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(n4m_explain_session_next_batch_v1(ctx.p, s, rows.data(), 4, &written)
                     == N4M_OK);
    // Global empty phase (two background rows), then x itself, then {g0} on background 0.
    N4M_TEST_REQUIRE(written == 4);
    N4M_TEST_REQUIRE(rows[0] == 0 && rows[3] == 1 && rows[6] == 1 && rows[8] == 3);
    N4M_TEST_REQUIRE(rows[9] == 1 && rows[10] == 0 && rows[11] == 0);
    N4M_TEST_REQUIRE(n4m_explain_session_next_batch_v1(ctx.p, s, rows.data(), 4, &written)
                     == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(n4m_explain_session_submit_v1(ctx.p, s, preds.data(), 1)
                     == N4M_ERR_SHAPE_MISMATCH);
    N4M_TEST_REQUIRE(n4m_explain_session_submit_v1(ctx.p, s, preds.data(), 4) == N4M_OK);
    N4M_TEST_REQUIRE(n4m_explain_session_next_batch_v1(ctx.p, s, rows.data(), 4, &written)
                     == N4M_OK);
    N4M_TEST_REQUIRE(written == 4);
    preds[1] = kNaN;
    N4M_TEST_REQUIRE(n4m_explain_session_submit_v1(ctx.p, s, preds.data(), 4)
                     == N4M_ERR_NUMERICAL_FAILURE);
    N4M_TEST_REQUIRE(n4m_explain_session_next_batch_v1(ctx.p, s, rows.data(), 4, &written)
                     == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(n4m_explain_session_result_v1(ctx.p, s, attr.data(), 18, e.data(), 2,
                                                   p.data(), 2, nullptr, 0)
                     == N4M_ERR_INVALID_ARGUMENT);
    n4m_explain_session_destroy(s);
    n4m_explain_session_destroy(nullptr);
}

void exact_matches_bruteforce() {
    const Values x = {1.0, 2.0, -1.0, 0.5, -3.0, 2.5};
    const Values bg = {0.0, 0.0, 0.0, 1.0, -1.0, 2.0, -0.5, 3.0, 1.0};
    for (int grouping = 0; grouping < 2; ++grouping) {
        const std::vector<int64_t> groups = grouping == 0 ? std::vector<int64_t>{}
                                                          : std::vector<int64_t>{0, 0, 1};
        const int64_t G = grouping == 0 ? 3 : 2;
        const auto o = options(N4M_EXPLAIN_EXACT_SHAP, N4M_EXPLAIN_BASELINE_BACKGROUND);
        const Run run = run_session(x, 2, 3, bg, 3, groups, G, o, kInteraction, 5);
        // Plan size: 3 (global empty) + 2 * (1 + (2^G - 2) * 3).
        N4M_TEST_REQUIRE(run.total == 3 + 2 * (1 + ((int64_t{1} << G) - 2) * 3));
        for (std::size_t r = 0; r < 2; ++r) {
            const Values xr(x.begin() + static_cast<std::ptrdiff_t>(3 * r),
                            x.begin() + static_cast<std::ptrdiff_t>(3 * r + 3));
            const Values phi = shapley(sz(G), 1, [&](const std::vector<int>& mask) {
                return background_game(xr, bg, 3, 3, groups, mask, 1, kInteraction);
            });
            for (std::size_t g = 0; g < sz(G); ++g) {
                N4M_TEST_REQUIRE(close(run.attr[r * sz(G) + g], phi[g], 1e-12));
            }
            const double fx = xr[0] * xr[1] + 2.0 * xr[2];
            N4M_TEST_REQUIRE(close(run.pred[r], fx, 1e-15));
            const double e = (0.0 + (-1.0 + 4.0) + (-1.5 + 2.0)) / 3.0;
            N4M_TEST_REQUIRE(close(run.expected[r], e, 1e-14));
            N4M_TEST_REQUIRE(std::isnan(run.se[r * sz(G)]));
        }
        check_additive(run, 2, sz(G), 1, 1e-12);
    }
}

const Model kRich = [](const double* r, int64_t n, int64_t f, double* out) {
    for (int64_t i = 0; i < n; ++i) {
        const double* z = r + i * f;
        out[i] = z[0] * z[1] * z[2] + std::sin(z[3]) * z[0] + z[1] - 0.5 * z[2] * z[3];
    }
};

void permutation_properties() {
    const Values x = {1.0, 2.0, -1.0, 0.5, -3.0, 2.5};
    const Values bg = {0.0, 0.0, 0.0, 1.0, -1.0, 2.0, -0.5, 3.0, 1.0};
    // G = 2 with one antithetic pair is exact.
    const std::vector<int64_t> two = {0, 0, 1};
    auto o = options(N4M_EXPLAIN_PERMUTATION_SHAP, N4M_EXPLAIN_BASELINE_BACKGROUND);
    o.n_permutations = 1;
    o.seed = 7;
    Run run = run_session(x, 2, 3, bg, 3, two, 2, o, kInteraction);
    N4M_TEST_REQUIRE(run.total == 3 + 2 * (1 + 1 * 2 * 1 * 3));
    for (std::size_t r = 0; r < 2; ++r) {
        const Values xr(x.begin() + static_cast<std::ptrdiff_t>(3 * r),
                        x.begin() + static_cast<std::ptrdiff_t>(3 * r + 3));
        const Values phi = shapley(2, 1, [&](const std::vector<int>& mask) {
            return background_game(xr, bg, 3, 3, two, mask, 1, kInteraction);
        });
        N4M_TEST_REQUIRE(close(run.attr[2 * r], phi[0], 1e-12));
        N4M_TEST_REQUIRE(close(run.attr[2 * r + 1], phi[1], 1e-12));
        N4M_TEST_REQUIRE(std::isnan(run.se[2 * r]));
    }
    // Additivity holds for every draw count and the plan is batch-size independent.
    o.n_permutations = 3;
    run = run_session(x, 2, 3, bg, 3, {}, 3, o, kInteraction, 4);
    N4M_TEST_REQUIRE(run.total == 3 + 2 * (1 + 3 * 2 * 2 * 3));
    check_additive(run, 2, 3, 1, 1e-12);
    for (const double se : run.se) {
        N4M_TEST_REQUIRE(std::isfinite(se) && se >= 0.0);
    }
    const Run again = run_session(x, 2, 3, bg, 3, {}, 3, o, kInteraction, 1000);
    N4M_TEST_REQUIRE(again.attr == run.attr && again.se == run.se);
    // Unbiased: many draws converge to the exact values (MEAN baseline, G = 4).
    const Values x4 = {0.8, -1.2, 1.5, 2.0};
    const Values bg4 = {0.0, 0.5, -0.5, 1.0, 1.0, -1.0, 0.5, -2.0};
    auto exact = options(N4M_EXPLAIN_EXACT_SHAP, N4M_EXPLAIN_BASELINE_MEAN);
    const Run truth = run_session(x4, 1, 4, bg4, 2, {}, 4, exact, kRich);
    auto many = options(N4M_EXPLAIN_PERMUTATION_SHAP, N4M_EXPLAIN_BASELINE_MEAN);
    many.n_permutations = 3000;
    many.seed = 11;
    const Run approx = run_session(x4, 1, 4, bg4, 2, {}, 4, many, kRich, 4096);
    N4M_TEST_REQUIRE(approx.total == 1 + 1 + 3000 * 2 * 3);
    for (std::size_t g = 0; g < 4; ++g) {
        const double err = std::fabs(approx.attr[g] - truth.attr[g]);
        N4M_TEST_REQUIRE(err <= 5.0 * approx.se[g] + 1e-9);
        N4M_TEST_REQUIRE(approx.se[g] > 0.0 && approx.se[g] < 0.1);
    }
    check_additive(approx, 1, 4, 1, 1e-10);
    N4M_TEST_REQUIRE(close(approx.expected[0], truth.expected[0], 1e-15));
}

void kernel_properties() {
    const Values x4 = {0.8, -1.2, 1.5, 2.0, -0.3, 0.7, 0.1, 1.1};
    const Values bg4 = {0.0, 0.5, -0.5, 1.0, 1.0, -1.0, 0.5, -2.0, 0.2, 0.2, 0.2, 0.2};
    const auto exact = options(N4M_EXPLAIN_EXACT_SHAP, N4M_EXPLAIN_BASELINE_BACKGROUND);
    const Run truth = run_session(x4, 2, 4, bg4, 3, {}, 4, exact, kRich);
    // Auto budget min(2^4 - 2, 2*4 + 2048) enumerates every proper coalition: exact.
    const auto kernel = options(N4M_EXPLAIN_KERNEL_SHAP, N4M_EXPLAIN_BASELINE_BACKGROUND);
    const Run full = run_session(x4, 2, 4, bg4, 3, {}, 4, kernel, kRich);
    N4M_TEST_REQUIRE(full.total == truth.total);
    for (std::size_t i = 0; i < truth.attr.size(); ++i) {
        N4M_TEST_REQUIRE(close(full.attr[i], truth.attr[i], 1e-10));
    }
    check_additive(full, 2, 4, 1, 1e-12);
    // Sampled coalitions (budget rounded up to complement pairs): efficiency exact.
    const Model quad = [](const double* r, int64_t n, int64_t f, double* out) {
        for (int64_t i = 0; i < n; ++i) {
            const double* z = r + i * f;
            double s = z[0] * z[9];
            for (int64_t j = 0; j < f; ++j) {
                s += z[j] * z[j] * static_cast<double>(j + 1);
            }
            out[i] = s;
        }
    };
    Values x10(10), bg10(20);
    for (std::size_t j = 0; j < 10; ++j) {
        x10[j] = std::cos(static_cast<double>(j));
        bg10[j] = 0.1 * static_cast<double>(j);
        bg10[10 + j] = -0.2 * static_cast<double>(j);
    }
    auto sampled = options(N4M_EXPLAIN_KERNEL_SHAP, N4M_EXPLAIN_BASELINE_BACKGROUND);
    sampled.n_coalitions = 63;
    sampled.seed = 5;
    const Run approx = run_session(x10, 1, 10, bg10, 2, {}, 10, sampled, quad, 50);
    N4M_TEST_REQUIRE(approx.total == 2 + 1 + 64 * 2);
    check_additive(approx, 1, 10, 1, 1e-10);
    for (std::size_t i = 0; i < approx.attr.size(); ++i) {
        N4M_TEST_REQUIRE(std::isfinite(approx.attr[i]) && std::isnan(approx.se[i]));
    }
}

void linear_matches_exact_session() {
    // f(z) = z . B + c with two outputs; groups {0,1} {2} {3}.
    const Values B = {1.0, -2.0, 0.5, 0.0, -1.5, 3.0, 2.0, 1.0};  // 4 x 2
    const Values c = {0.25, -1.0};
    const Model affine = [&](const double* r, int64_t n, int64_t f, double* out) {
        for (int64_t i = 0; i < n; ++i) {
            for (std::size_t o = 0; o < 2; ++o) {
                double s = c[o];
                for (int64_t j = 0; j < f; ++j) {
                    s += r[i * f + j] * B[sz(j) * 2 + o];
                }
                out[sz(i) * 2 + o] = s;
            }
        }
    };
    const Values x = {1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 0.0, 2.0};
    const Values bg = {0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, 1.0};
    const std::vector<int64_t> groups = {0, 0, 1, 2};
    const auto o = options(N4M_EXPLAIN_EXACT_SHAP, N4M_EXPLAIN_BASELINE_BACKGROUND, 2);
    const Run run = run_session(x, 2, 4, bg, 3, groups, 3, o, affine);
    Values reference(4, 0.0);
    for (std::size_t b = 0; b < 3; ++b) {
        for (std::size_t j = 0; j < 4; ++j) {
            reference[j] += bg[b * 4 + j] / 3.0;
        }
    }
    // Column-major X view exercises the strided path.
    const Values xcol = {1.0, -1.0, 2.0, 0.5, 3.0, 0.0, 4.0, 2.0};
    const n4m_matrix_view_t xv = {const_cast<double*>(xcol.data()), 2, 4, 1, 2, N4M_DTYPE_F64, 0};
    const auto bv = view(B.data(), 4, 2);
    Values attr(12, 0.0), e(4, 0.0), p(4, 0.0);
    Ctx ctx;
    N4M_TEST_REQUIRE(n4m_explain_linear_v1(ctx.p, &xv, &bv, c.data(), reference.data(),
                                           groups.data(), 3, attr.data(), 12, e.data(), 4,
                                           p.data(), 4)
                     == N4M_OK);
    for (std::size_t i = 0; i < 12; ++i) {
        N4M_TEST_REQUIRE(close(attr[i], run.attr[i], 1e-12));
    }
    for (std::size_t i = 0; i < 4; ++i) {
        N4M_TEST_REQUIRE(close(e[i], run.expected[i], 1e-12));
        N4M_TEST_REQUIRE(close(p[i], run.pred[i], 1e-12));
    }
    // Validation: wrong coefficient rows and short buffers leave outputs untouched.
    const auto short_b = view(B.data(), 3, 2);
    Values untouched(12, 42.0);
    N4M_TEST_REQUIRE(n4m_explain_linear_v1(ctx.p, &xv, &short_b, nullptr, reference.data(),
                                           groups.data(), 3, untouched.data(), 12, e.data(), 4,
                                           nullptr, 0)
                     == N4M_ERR_SHAPE_MISMATCH);
    N4M_TEST_REQUIRE(n4m_explain_linear_v1(ctx.p, &xv, &bv, nullptr, reference.data(),
                                           groups.data(), 3, untouched.data(), 11, e.data(), 4,
                                           nullptr, 0)
                     == N4M_ERR_SHAPE_MISMATCH);
    N4M_TEST_REQUIRE(std::all_of(untouched.begin(), untouched.end(),
                                 [](double v) { return v == 42.0; }));
}

void occlusion_and_lime() {
    // Additive model: f(z) = sum_j b_j z_j + c.
    const Values b = {2.0, -1.0, 0.5, 3.0, -0.25};
    const Model additive = [&](const double* r, int64_t n, int64_t f, double* out) {
        for (int64_t i = 0; i < n; ++i) {
            double s = 1.5;
            for (int64_t j = 0; j < f; ++j) {
                s += b[sz(j)] * r[i * f + j];
            }
            out[i] = s;
        }
    };
    const Values x = {1.0, 2.0, 3.0, 4.0, 5.0};
    const Values bg = {0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0};  // mean = 1
    const std::vector<int64_t> groups = {0, 0, 1, 2, 2};
    Values truth(3, 0.0);
    for (std::size_t j = 0; j < 5; ++j) {
        truth[sz(groups[j])] += b[j] * (x[j] - 1.0);
    }
    double f_ref = 1.5;
    for (const double bj : b) {
        f_ref += bj;
    }
    const auto occ = options(N4M_EXPLAIN_OCCLUSION, N4M_EXPLAIN_BASELINE_MEAN);
    const Run run = run_session(x, 1, 5, bg, 2, groups, 3, occ, additive);
    N4M_TEST_REQUIRE(run.total == 1 + 1 + 3);
    for (std::size_t g = 0; g < 3; ++g) {
        N4M_TEST_REQUIRE(close(run.attr[g], truth[g], 1e-12));
        N4M_TEST_REQUIRE(std::isnan(run.se[g]));
    }
    N4M_TEST_REQUIRE(close(run.expected[0], f_ref, 1e-12));
    // Occlusion of an interaction model is f(x) - f(x without g); not additive.
    const Values x3 = {1.0, 2.0, -1.0};
    const Values bg3 = {0.0, 0.0, 0.0};
    const Run inter = run_session(x3, 1, 3, bg3, 1, {}, 3,
                                  options(N4M_EXPLAIN_OCCLUSION, N4M_EXPLAIN_BASELINE_MEAN),
                                  kInteraction);
    N4M_TEST_REQUIRE(close(inter.attr[0], 2.0, 1e-15) && close(inter.attr[1], 2.0, 1e-15));
    N4M_TEST_REQUIRE(close(inter.attr[2], -2.0, 1e-15));
    // LIME with no ridge recovers the mask-space coefficients of a linear game.
    auto lime = options(N4M_EXPLAIN_LIME, N4M_EXPLAIN_BASELINE_MEAN);
    lime.lime_ridge = 0.0;
    lime.n_coalitions = 200;
    lime.seed = 3;
    const Run fit = run_session(x, 1, 5, bg, 2, groups, 3, lime, additive);
    N4M_TEST_REQUIRE(fit.total == 200);
    for (std::size_t g = 0; g < 3; ++g) {
        N4M_TEST_REQUIRE(close(fit.attr[g], truth[g], 1e-9));
    }
    N4M_TEST_REQUIRE(close(fit.expected[0], f_ref, 1e-9));
    // The ridge penalty shrinks the coefficients (intercept unpenalized).
    lime.lime_ridge = 5.0;
    const Run shrunk = run_session(x, 1, 5, bg, 2, groups, 3, lime, additive);
    double norm_fit = 0.0, norm_shrunk = 0.0;
    for (std::size_t g = 0; g < 3; ++g) {
        norm_fit += fit.attr[g] * fit.attr[g];
        norm_shrunk += shrunk.attr[g] * shrunk.attr[g];
    }
    N4M_TEST_REQUIRE(norm_shrunk < norm_fit);
}

void interpolate_bridges() {
    const Values x = {1.0, 2.0, 4.0, 8.0, 16.0, 32.0};
    const std::vector<int64_t> groups = {0, 0, 1, 1, 2, 2};
    const Model sum = [](const double* r, int64_t n, int64_t f, double* out) {
        for (int64_t i = 0; i < n; ++i) {
            out[i] = std::accumulate(r + i * f, r + (i + 1) * f, 0.0);
        }
    };
    const auto o = options(N4M_EXPLAIN_EXACT_SHAP, N4M_EXPLAIN_BASELINE_INTERPOLATE);
    const Run run = run_session(x, 1, 6, {}, 0, groups, 3, o, sum, 3, true);
    N4M_TEST_REQUIRE(run.total == 8);
    const double t = 14.0 / 3.0;
    const Values expected_rows = {
        1, 2, 4, 8, 16, 32,                    // full
        1, 7.2, 13.4, 19.6, 25.8, 32,          // empty: line first -> last
        1, 2, 2, 2, 2, 2,                      // {g0}: right edge flat
        4, 4, 4, 8, 8, 8,                      // {g1}: both edges flat
        1, 2, 4, 8, 8, 8,                      // {g0, g1}
        16, 16, 16, 16, 16, 32,                // {g2}: merged absent run
        1, 2, 2 + t, 2 + 2 * t, 16, 32,        // {g0, g2}: bridge
        4, 4, 4, 8, 16, 32};                   // {g1, g2}
    N4M_TEST_REQUIRE(run.rows.size() == expected_rows.size());
    for (std::size_t i = 0; i < expected_rows.size(); ++i) {
        N4M_TEST_REQUIRE(close(run.rows[i], expected_rows[i], 1e-14));
    }
    N4M_TEST_REQUIRE(close(run.expected[0], 99.0, 1e-13));
    check_additive(run, 1, 3, 1, 1e-12);
}

// ---------------------------------------------------------------------------
// TreeSHAP

struct Ensemble {
    std::vector<int64_t> offsets, feature, left, right;
    Values threshold, cover, value;
    std::vector<uint8_t> default_left;
    int64_t n_outputs = 1;
    int32_t decision_type = 0;
    Values base_offset;
    double scale = 1.0;

    void add(const std::vector<int64_t>& f, const Values& thr, const std::vector<int64_t>& l,
             const std::vector<int64_t>& r, const Values& cov, const Values& val,
             const std::vector<uint8_t>& dl = {}) {
        if (offsets.empty()) {
            offsets.push_back(0);
        }
        feature.insert(feature.end(), f.begin(), f.end());
        threshold.insert(threshold.end(), thr.begin(), thr.end());
        left.insert(left.end(), l.begin(), l.end());
        right.insert(right.end(), r.begin(), r.end());
        cover.insert(cover.end(), cov.begin(), cov.end());
        value.insert(value.end(), val.begin(), val.end());
        default_left.insert(default_left.end(), dl.begin(), dl.end());
        offsets.push_back(offsets.back() + static_cast<int64_t>(f.size()));
    }

    bool goes_left(std::size_t k, double v) const {
        if (std::isnan(v)) {
            return default_left[k] != 0;
        }
        return decision_type == 0 ? v <= threshold[k] : v < threshold[k];
    }

    // Path-dependent conditional expectation (Lundberg Algorithm 1).
    double cond(std::size_t base, std::size_t node, const double* x, const std::vector<int>& in,
                std::size_t o) const {
        const std::size_t k = base + node;
        if (left[k] < 0) {
            return value[k * sz(n_outputs) + o];
        }
        const auto f = sz(feature[k]);
        const auto l = sz(left[k]);
        const auto r = sz(right[k]);
        if (in[f] != 0) {
            return cond(base, goes_left(k, x[f]) ? l : r, x, in, o);
        }
        const double cl = cover[base + l], cr = cover[base + r];
        return (cl * cond(base, l, x, in, o) + cr * cond(base, r, x, in, o)) / (cl + cr);
    }

    Values game(const double* x, const std::vector<int>& in) const {
        const std::size_t O = sz(n_outputs);
        Values v(O, 0.0);
        for (std::size_t o = 0; o < O; ++o) {
            double s = 0.0;
            for (std::size_t t = 0; t + 1 < offsets.size(); ++t) {
                s += cond(sz(offsets[t]), 0, x, in, o);
            }
            v[o] = (base_offset.empty() ? 0.0 : base_offset[o]) + scale * s;
        }
        return v;
    }

    n4m_status_t explain(n4m_context_t* ctx, const Values& X, int64_t R, int64_t F,
                         const std::vector<int64_t>& groups, int64_t G, Values& attr, Values& e,
                         Values& p) const {
        const auto xv = view(X.data(), R, F);
        attr.assign(sz(R * G * n_outputs), 0.0);
        e.assign(sz(R * n_outputs), 0.0);
        p.assign(sz(R * n_outputs), 0.0);
        return n4m_explain_tree_path_dependent_v1(
            ctx, &xv, static_cast<int64_t>(offsets.size()) - 1, offsets.data(), feature.data(),
            threshold.data(), left.data(), right.data(),
            default_left.empty() ? nullptr : default_left.data(), cover.data(), value.data(),
            n_outputs, decision_type, base_offset.empty() ? nullptr : base_offset.data(), scale,
            groups.empty() ? nullptr : groups.data(), G, attr.data(),
            static_cast<int64_t>(attr.size()), e.data(), static_cast<int64_t>(e.size()),
            p.data(), static_cast<int64_t>(p.size()));
    }

    // Compares per-feature TreeSHAP with brute-force Shapley values of the oracle game.
    void check(const Values& X, int64_t R, int64_t F) const {
        Ctx ctx;
        Values attr, e, p;
        N4M_TEST_REQUIRE(explain(ctx.p, X, R, F, {}, F, attr, e, p) == N4M_OK);
        const std::size_t O = sz(n_outputs);
        for (std::size_t r = 0; r < sz(R); ++r) {
            const double* xr = &X[r * sz(F)];
            const Values phi = shapley(sz(F), O, [&](const std::vector<int>& in) {
                return game(xr, in);
            });
            const Values full = game(xr, std::vector<int>(sz(F), 1));
            const Values none = game(xr, std::vector<int>(sz(F), 0));
            for (std::size_t o = 0; o < O; ++o) {
                double sum = e[r * O + o];
                for (std::size_t j = 0; j < sz(F); ++j) {
                    N4M_TEST_REQUIRE(close(attr[(r * sz(F) + j) * O + o], phi[j * O + o], 1e-12));
                    sum += attr[(r * sz(F) + j) * O + o];
                }
                N4M_TEST_REQUIRE(close(p[r * O + o], full[o], 1e-14));
                N4M_TEST_REQUIRE(close(e[r * O + o], none[o], 1e-14));
                N4M_TEST_REQUIRE(std::fabs(sum - p[r * O + o]) <= 1e-12);
            }
        }
    }
};

// T1 (features 0, 1): root f0 <= 0.5, left f1 <= 0.3.
void add_t1(Ensemble& ens, double root_cover = 10.0) {
    ens.add({0, 1, -1, -1, -1}, {0.5, 0.3, 0, 0, 0}, {1, 2, -1, -1, -1}, {4, 3, -1, -1, -1},
            {root_cover, 6, 2, 4, 4}, {0, 0, 1, 3, 7}, {1, 0, 0, 0, 0});
}

// T2 (features 0, 1, 2) with the repeated feature path f0 -> f1 -> f0.
void add_t2(Ensemble& ens, bool two_outputs = false) {
    const Values v1 = {0, 0, 0, -2, 1, 4, 0, 10, -5};
    Values val = v1;
    if (two_outputs) {
        val.clear();
        for (const double v : v1) {
            val.push_back(v);
            val.push_back(2.0 * v * v - 1.0);
        }
    }
    ens.add({0, 1, 0, -1, -1, -1, 2, -1, -1}, {0.0, 0.0, -1.0, 0, 0, 0, 0.5, 0, 0},
            {1, 2, 3, -1, -1, -1, 7, -1, -1}, {6, 5, 4, -1, -1, -1, 8, -1, -1},
            {20, 12, 8, 3, 5, 4, 8, 5, 3}, val, {0, 1, 0, 0, 0, 0, 1, 0, 0});
}

const Values kTreeX = {-2.0, -1.0, 0.0, -0.5, 1.0, 1.0, 1.0, 0.0, 1.0,
                       -0.5, -1.0, 0.0, 0.5, 0.3, 0.6, 0.0, 0.0, 0.5};

void tree_single_and_repeated_feature() {
    Ensemble t1;
    add_t1(t1);
    t1.check(kTreeX, 6, 3);
    Ensemble t2;
    add_t2(t2);
    t2.check(kTreeX, 6, 3);
}

void tree_ensemble_multi_output_and_scale() {
    Ensemble ens;
    add_t1(ens);
    add_t2(ens);
    ens.scale = 0.5;
    ens.base_offset = {0.25};
    ens.check(kTreeX, 6, 3);
    Ensemble multi;
    multi.n_outputs = 2;
    add_t2(multi, true);
    add_t2(multi, true);
    multi.base_offset = {1.0, -1.0};
    multi.check(kTreeX, 6, 3);
    // Grouped tree values are sums of per-feature values.
    Ctx ctx;
    Values attr, e, p, grouped, ge, gp;
    N4M_TEST_REQUIRE(ens.explain(ctx.p, kTreeX, 6, 3, {}, 3, attr, e, p) == N4M_OK);
    N4M_TEST_REQUIRE(ens.explain(ctx.p, kTreeX, 6, 3, {0, 0, 1}, 2, grouped, ge, gp) == N4M_OK);
    for (std::size_t r = 0; r < 6; ++r) {
        N4M_TEST_REQUIRE(close(grouped[r * 2], attr[r * 3] + attr[r * 3 + 1], 1e-14));
        N4M_TEST_REQUIRE(close(grouped[r * 2 + 1], attr[r * 3 + 2], 1e-14));
    }
}

void tree_decision_type_nan_and_covers() {
    const Values at_threshold = {0.5, 0.3, 0.0};
    Ensemble le;
    add_t1(le);
    le.check(at_threshold, 1, 3);
    Ensemble lt = le;
    lt.decision_type = 1;
    lt.check(at_threshold, 1, 3);
    Ctx ctx;
    Values a, e, p_le, p_lt;
    N4M_TEST_REQUIRE(le.explain(ctx.p, at_threshold, 1, 3, {}, 3, a, e, p_le) == N4M_OK);
    N4M_TEST_REQUIRE(lt.explain(ctx.p, at_threshold, 1, 3, {}, 3, a, e, p_lt) == N4M_OK);
    N4M_TEST_REQUIRE(p_le[0] == 1.0 && p_lt[0] == 7.0);
    // NaN routing follows default_left; without it NaN is rejected.
    const Values nan_x = {kNaN, -1.0, kNaN, kNaN, 1.0, 0.0};
    Ensemble nan_ens;
    add_t2(nan_ens);
    nan_ens.check(nan_x, 2, 3);
    Ensemble no_default = nan_ens;
    no_default.default_left.clear();
    N4M_TEST_REQUIRE(no_default.explain(ctx.p, nan_x, 2, 3, {}, 3, a, e, p_le)
                     == N4M_ERR_INVALID_ARGUMENT);
    // A parent cover differing from its children sum keeps exact additivity.
    Ensemble odd;
    add_t1(odd, 100.0);
    odd.check(kTreeX, 6, 3);
}

void tree_rejections() {
    Ctx ctx;
    Values a, e, p;
    const Values x = {0.0, 0.0, 0.0};
    Ensemble cycle;
    add_t1(cycle);
    cycle.left[1] = 0;
    N4M_TEST_REQUIRE(cycle.explain(ctx.p, x, 1, 3, {}, 3, a, e, p) == N4M_ERR_INVALID_ARGUMENT);
    Ensemble half_leaf;
    add_t1(half_leaf);
    half_leaf.left[4] = 2;
    N4M_TEST_REQUIRE(half_leaf.explain(ctx.p, x, 1, 3, {}, 3, a, e, p)
                     == N4M_ERR_INVALID_ARGUMENT);
    Ensemble zero_cover;
    add_t1(zero_cover);
    zero_cover.cover[3] = 0.0;
    N4M_TEST_REQUIRE(zero_cover.explain(ctx.p, x, 1, 3, {}, 3, a, e, p)
                     == N4M_ERR_INVALID_ARGUMENT);
    Ensemble bad_feature;
    add_t1(bad_feature);
    bad_feature.feature[1] = 3;
    N4M_TEST_REQUIRE(bad_feature.explain(ctx.p, x, 1, 3, {}, 3, a, e, p)
                     == N4M_ERR_INVALID_ARGUMENT);
    Ensemble nan_threshold;
    add_t1(nan_threshold);
    nan_threshold.threshold[0] = kNaN;
    N4M_TEST_REQUIRE(nan_threshold.explain(ctx.p, x, 1, 3, {}, 3, a, e, p)
                     == N4M_ERR_INVALID_ARGUMENT);
    Ensemble bad_offsets;
    add_t1(bad_offsets);
    bad_offsets.offsets = {0, 5, 5};
    N4M_TEST_REQUIRE(bad_offsets.explain(ctx.p, x, 1, 3, {}, 3, a, e, p)
                     == N4M_ERR_INVALID_ARGUMENT);
    Ensemble bad_type;
    add_t1(bad_type);
    bad_type.decision_type = 2;
    N4M_TEST_REQUIRE(bad_type.explain(ctx.p, x, 1, 3, {}, 3, a, e, p)
                     == N4M_ERR_INVALID_ARGUMENT);
}

// ---------------------------------------------------------------------------

void grouping_helpers() {
    std::vector<int64_t> fg(10, -7);
    int64_t n = 0;
    N4M_TEST_REQUIRE(n4m_explain_groups_fixed_width_v1(10, 4, fg.data(), 10, &n) == N4M_OK);
    N4M_TEST_REQUIRE(n == 3);
    N4M_TEST_REQUIRE((fg == std::vector<int64_t>{0, 0, 0, 0, 1, 1, 1, 1, 2, 2}));
    N4M_TEST_REQUIRE(n4m_explain_groups_fixed_width_v1(10, 0, fg.data(), 10, &n)
                     == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(n4m_explain_groups_fixed_width_v1(10, 4, fg.data(), 9, &n)
                     == N4M_ERR_SHAPE_MISMATCH);
    N4M_TEST_REQUIRE(n4m_explain_groups_fixed_width_v1(3, 5, fg.data(), 10, &n) == N4M_OK);
    N4M_TEST_REQUIRE(n == 1 && fg[0] == 0 && fg[2] == 0);

    const int64_t ranges[] = {5, 7, 1, 3};
    std::vector<int64_t> rog(12, -9);
    fg.assign(10, -7);
    N4M_TEST_REQUIRE(n4m_explain_groups_from_ranges_v1(10, ranges, 2, fg.data(), 10, rog.data(),
                                                       12, &n)
                     == N4M_OK);
    N4M_TEST_REQUIRE(n == 5);
    N4M_TEST_REQUIRE((fg == std::vector<int64_t>{2, 1, 1, 3, 3, 0, 0, 4, 4, 4}));
    N4M_TEST_REQUIRE(rog[0] == 0 && rog[1] == 1 && rog[2] == -1 && rog[3] == -1 && rog[4] == -1);
    N4M_TEST_REQUIRE(rog[5] == -9);
    // Full coverage, no ranges, and rejections leaving outputs untouched.
    const int64_t all[] = {0, 10};
    N4M_TEST_REQUIRE(n4m_explain_groups_from_ranges_v1(10, all, 1, fg.data(), 10, nullptr, 0, &n)
                     == N4M_OK);
    N4M_TEST_REQUIRE(n == 1 && fg[9] == 0);
    N4M_TEST_REQUIRE(n4m_explain_groups_from_ranges_v1(10, nullptr, 0, fg.data(), 10, nullptr, 0,
                                                       &n)
                     == N4M_OK);
    N4M_TEST_REQUIRE(n == 1 && fg[0] == 0 && fg[9] == 0);
    fg.assign(10, -7);
    const int64_t overlap[] = {0, 4, 3, 6};
    const int64_t outside[] = {8, 11};
    const int64_t empty_range[] = {2, 2};
    N4M_TEST_REQUIRE(n4m_explain_groups_from_ranges_v1(10, overlap, 2, fg.data(), 10, nullptr, 0,
                                                       &n)
                     == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(n4m_explain_groups_from_ranges_v1(10, outside, 1, fg.data(), 10, nullptr, 0,
                                                       &n)
                     == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(n4m_explain_groups_from_ranges_v1(10, empty_range, 1, fg.data(), 10,
                                                       nullptr, 0, &n)
                     == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(n4m_explain_groups_from_ranges_v1(10, ranges, 2, fg.data(), 10, rog.data(),
                                                       4, &n)
                     == N4M_ERR_SHAPE_MISMATCH);
    N4M_TEST_REQUIRE(std::all_of(fg.begin(), fg.end(), [](int64_t v) { return v == -7; }));
}

void summaries() {
    // [row][group][output], 2 x 2 x 2.
    const Values attr = {1.0, -1.0, 2.0, 0.5, -3.0, 4.0, 0.0, -0.5};
    Values mean_abs(2), mean_signed(2);
    N4M_TEST_REQUIRE(n4m_explain_summarize_v1(attr.data(), 2, 2, 2, 1, mean_abs.data(),
                                              mean_signed.data(), 2)
                     == N4M_OK);
    N4M_TEST_REQUIRE(close(mean_abs[0], 2.5, 1e-15) && close(mean_signed[0], 1.5, 1e-15));
    N4M_TEST_REQUIRE(close(mean_abs[1], 0.5, 1e-15) && close(mean_signed[1], 0.0, 1e-15));
    N4M_TEST_REQUIRE(n4m_explain_summarize_v1(attr.data(), 2, 2, 2, 0, mean_abs.data(),
                                              mean_signed.data(), 2)
                     == N4M_OK);
    N4M_TEST_REQUIRE(close(mean_abs[0], 2.0, 1e-15) && close(mean_signed[0], -1.0, 1e-15));
    N4M_TEST_REQUIRE(n4m_explain_summarize_v1(attr.data(), 2, 2, 2, 2, mean_abs.data(),
                                              mean_signed.data(), 2)
                     == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(n4m_explain_summarize_v1(attr.data(), 2, 2, 2, 0, mean_abs.data(),
                                              mean_signed.data(), 1)
                     == N4M_ERR_SHAPE_MISMATCH);
}

}  // namespace

int main() {
    n4m_testing::Runner r("explain");
    r.run("options init defaults", options_init_defaults);
    r.run("session create rejections", create_rejections);
    r.run("ask/tell protocol misuse and poisoning", protocol_misuse);
    r.run("exact SHAP equals brute-force Shapley", exact_matches_bruteforce);
    r.run("permutation SHAP additive, exact at G=2, unbiased", permutation_properties);
    r.run("kernel SHAP enumeration exact, sampled efficient", kernel_properties);
    r.run("linear SHAP equals exact session", linear_matches_exact_session);
    r.run("occlusion and LIME properties", occlusion_and_lime);
    r.run("INTERPOLATE baseline bridges", interpolate_bridges);
    r.run("TreeSHAP single tree and repeated feature", tree_single_and_repeated_feature);
    r.run("TreeSHAP ensemble, multi-output, scale, grouping",
          tree_ensemble_multi_output_and_scale);
    r.run("TreeSHAP decision type, NaN routing, covers", tree_decision_type_nan_and_covers);
    r.run("TreeSHAP structural rejections", tree_rejections);
    r.run("grouping helpers", grouping_helpers);
    r.run("summaries", summaries);
    return r.finalize();
}
