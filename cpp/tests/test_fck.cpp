// SPDX-License-Identifier: CECILL-2.1
//
// Phase 21 — FCKStaticTransformer parity tests.
//
// The Phase 21 ABI surface is split off from c4a.h (it lives in c4a_fck.h),
// and its test entry point is registered from a self-contained main()
// rather than being appended to the Phase 0 harness's main.cpp. This keeps
// the Phase 21 worktree fully self-contained: the parent project picks up
// the new executable through cpp/tests/CMakeLists.txt and `ctest` runs both
// suites side by side.
//
// Four tests:
//   1. fck_smoke               — create / transform / destroy lifecycle on
//                                a small inline matrix, plus shape-helper
//                                and error-status verification.
//   2. fck_kernel_properties   — drives the static transformer with a unit-
//                                delta input so the resulting output column
//                                exposes the kernel (modulo scipy's
//                                convolve1d reversal) and checks the
//                                symmetry / zero-mean / L1 normalisation
//                                contracts for pure-Gaussian (alpha=0) and
//                                signed-derivative (alpha=1.0, 1.5)
//                                kernels via the public ABI only.
//   3. fck_output_cols         — covers the helper's overflow and
//                                NULL-pointer paths.
//   4. fck_parity              — load fck_static_v1.json fixture and verify
//                                bit-tight parity against the canonical
//                                Python reference (tolerance 1e-12 abs /
//                                1e-13 rel).

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "chemometrics4all/c4a.h"

#include "fixture_parser.hpp"
#include "harness.hpp"

#ifndef C4A_PARITY_FIXTURE_DIR
#  error "C4A_PARITY_FIXTURE_DIR must be defined"
#endif

// The kernel builder is private to libc4a (no `C4A_API` export, hidden
// visibility, intentionally absent from c4a.h). Phase 21 review feedback:
// the kernel must only be reachable via the static-transformer apply path,
// so we exercise its algebraic properties through that ABI rather than
// calling it directly.

namespace {

using ParityFixture = ::c4a_testing::Fixture;

ParityFixture load_fixture(const std::string& filename) {
    return ::c4a_testing::load_fixture(
        std::string(C4A_PARITY_FIXTURE_DIR) + "/" + filename,
        /*require_per_case_output_shape=*/true);
}

c4a_matrix_view_t make_rowmajor_view(double* data, std::int64_t rows,
                                      std::int64_t cols) {
    c4a_matrix_view_t v{};
    const c4a_status_t st =
        c4a_matrix_view_init_rowmajor(&v, data, rows, cols, C4A_DTYPE_F64);
    C4A_TEST_REQUIRE(st == C4A_OK);
    return v;
}

// ---------------------------------------------------------------------------
// Test 1 — smoke / lifecycle.
// ---------------------------------------------------------------------------

void test_fck_smoke() {
    // 1 row x 10 cols, two kernels (alpha=0.0 smoother + alpha=1.0
    // derivative-like) at scale=2.0.
    double X[10] = {1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0, 37.0, 46.0};
    double Y[20] = {0};

    const double alphas[2] = {0.0, 1.0};
    const double scales[1] = {2.0};
    c4a_pp_fck_static_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_fck_static_create(&h, /*K=*/5,
                                                alphas, 2,
                                                scales, 1) == C4A_OK);
    C4A_TEST_REQUIRE(h != nullptr);

    // The output should be (1, 2 * 10) = (1, 20).
    int32_t out_cols = 0;
    C4A_TEST_REQUIRE(c4a_pp_fck_static_output_cols(2, 10, &out_cols) == C4A_OK);
    C4A_TEST_REQUIRE(out_cols == 20);

    c4a_matrix_view_t Xv = make_rowmajor_view(X, 1, 10);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 1, 20);
    C4A_TEST_REQUIRE(c4a_pp_fck_static_transform(h, Xv, Yv) == C4A_OK);

    // Band 0 (alpha=0, pure Gaussian smoother) — interior values should be
    // close to the local moving average (input is monotonically increasing).
    C4A_TEST_REQUIRE(Y[3] > 5.0 && Y[3] < 9.0);
    // Band 1 (alpha=1, derivative-like) — scipy.ndimage.convolve1d reverses
    // the kernel before correlating, so the antisymmetric kernel with
    // positive weights on the right becomes negative-on-the-right after
    // reversal. For an increasing signal this yields a negative response.
    C4A_TEST_REQUIRE(Y[10 + 5] < 0.0);

    // Invalid parameter rejection.
    c4a_pp_fck_static_handle_t* bad = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_fck_static_create(&bad, /*K=*/0,
                                                alphas, 2, scales, 1) ==
                     C4A_ERR_INVALID_ARGUMENT);
    C4A_TEST_REQUIRE(c4a_pp_fck_static_create(&bad, /*K=*/5,
                                                alphas, 0, scales, 1) ==
                     C4A_ERR_INVALID_ARGUMENT);
    const double zero_scale[1] = {0.0};
    C4A_TEST_REQUIRE(c4a_pp_fck_static_create(&bad, /*K=*/5,
                                                alphas, 2, zero_scale, 1) ==
                     C4A_ERR_INVALID_ARGUMENT);

    // NULL out / pointers.
    C4A_TEST_REQUIRE(c4a_pp_fck_static_create(nullptr, 5, alphas, 2,
                                                scales, 1) ==
                     C4A_ERR_NULL_POINTER);
    C4A_TEST_REQUIRE(c4a_pp_fck_static_create(&bad, 5, nullptr, 2,
                                                scales, 1) ==
                     C4A_ERR_NULL_POINTER);

    // Shape-mismatch on transform.
    double Y_wrong[10] = {0};
    c4a_matrix_view_t Yv_bad = make_rowmajor_view(Y_wrong, 1, 10);
    C4A_TEST_REQUIRE(c4a_pp_fck_static_transform(h, Xv, Yv_bad) ==
                     C4A_ERR_SHAPE_MISMATCH);

    c4a_pp_fck_static_destroy(h);
    c4a_pp_fck_static_destroy(nullptr);
}

// ---------------------------------------------------------------------------
// Test 2 — kernel algebraic properties via the static transformer ABI.
//
// The kernel builder (`c4a_fck_kernel_1d`) is internal; we exercise its
// algebraic contracts (Gaussian symmetry, antisymmetry, zero-mean for
// alpha>0.1, L1 normalisation) through `c4a_pp_fck_static_transform`.
//
// Recovery trick: feed a unit-delta row (single 1.0 at the centre, zeros
// elsewhere). `scipy.ndimage.convolve1d` reverses the kernel before
// correlating, so:
//   y[i] = kernel[centre - i + (K-1)/2]   for i within K/2 of the centre
//        = 0                                outside that band.
// In words: the output column exposes the reversed kernel, indexed around
// the delta position. For alpha=0 (even kernel) reversal is a no-op; for
// alpha=1 / 1.5 (antisymmetric kernel) reversal negates each tap, so we
// check antisymmetry as `y[i] + y[2*centre - i] == 0`. In both branches
// the magnitudes (|y[i]|) match the kernel taps, so the L1-normalisation
// and zero-mean contracts transfer through.
// ---------------------------------------------------------------------------

void test_fck_kernel_properties() {
    // We use a single (alpha, scale) pair per case so the output has one
    // kernel band per row. The input length must be large enough to keep
    // the kernel response away from the implicit-zero boundary (mode
    // "constant" in scipy.ndimage.convolve1d).

    auto run_kernel_via_transform = [](double alpha, double scale,
                                        int32_t K, int64_t N,
                                        std::vector<double>& y_out) {
        const int64_t centre = N / 2;
        std::vector<double> x(static_cast<std::size_t>(N), 0.0);
        x[static_cast<std::size_t>(centre)] = 1.0;
        y_out.assign(static_cast<std::size_t>(N), 0.0);

        const double alphas[1] = {alpha};
        const double scales[1] = {scale};
        c4a_pp_fck_static_handle_t* h = nullptr;
        C4A_TEST_REQUIRE(c4a_pp_fck_static_create(&h, K,
                                                    alphas, 1,
                                                    scales, 1) == C4A_OK);
        c4a_matrix_view_t Xv = make_rowmajor_view(x.data(), 1, N);
        c4a_matrix_view_t Yv = make_rowmajor_view(y_out.data(), 1, N);
        C4A_TEST_REQUIRE(c4a_pp_fck_static_transform(h, Xv, Yv) == C4A_OK);
        c4a_pp_fck_static_destroy(h);
    };

    // Sign-clean helper: convert a (possibly negative) signed index into a
    // std::size_t for vector indexing without triggering -Wsign-conversion.
    auto at = [](const std::vector<double>& v, int64_t i) -> double {
        return v[static_cast<std::size_t>(i)];
    };

    // Case A — alpha = 0 (pure Gaussian, even-symmetric, positive everywhere).
    {
        constexpr int32_t K = 11;
        constexpr int64_t N = 41;
        std::vector<double> y;
        run_kernel_via_transform(0.0, 3.0, K, N, y);
        const int64_t c = N / 2;
        const int64_t half = (K - 1) / 2;
        // Even-symmetric kernel: convolve1d's reversal is a no-op. y[c-d] == y[c+d].
        for (int64_t d = 0; d <= half; ++d) {
            C4A_TEST_REQUIRE(std::fabs(at(y, c - d) - at(y, c + d)) < 1e-15);
        }
        // Positive in the [c - half, c + half] band.
        for (int64_t d = 0; d <= half; ++d) {
            C4A_TEST_REQUIRE(at(y, c + d) > 0.0);
            C4A_TEST_REQUIRE(at(y, c - d) > 0.0);
        }
        // Zero outside the kernel window.
        for (int64_t i = 0; i < N; ++i) {
            if (i < c - half || i > c + half) {
                C4A_TEST_REQUIRE(std::fabs(at(y, i)) < 1e-15);
            }
        }
        // L1 norm uses the nirs4all epsilon denominator:
        // raw_sum / (raw_sum + 1e-8), so it is within epsilon of 1.
        double sum_abs = 0.0;
        for (int64_t d = -half; d <= half; ++d) {
            sum_abs += std::fabs(at(y, c + d));
        }
        C4A_TEST_REQUIRE(std::fabs(sum_abs - 1.0) < 1e-8);
        // Peak at the centre.
        C4A_TEST_REQUIRE(at(y, c) >= at(y, c - half));
    }

    // Case B — alpha = 1.0 (zero-mean, antisymmetric kernel).
    {
        constexpr int32_t K = 11;
        constexpr int64_t N = 41;
        std::vector<double> y;
        run_kernel_via_transform(1.0, 3.0, K, N, y);
        const int64_t c = N / 2;
        const int64_t half = (K - 1) / 2;
        // Antisymmetric kernel; convolve1d reverses (still antisymmetric).
        for (int64_t d = 1; d <= half; ++d) {
            C4A_TEST_REQUIRE(std::fabs(at(y, c - d) + at(y, c + d)) < 1e-13);
        }
        // Centre is 0 (sign(0) == 0 -> kernel value 0 before normalisation).
        C4A_TEST_REQUIRE(std::fabs(at(y, c)) < 1e-13);
        // Zero mean across the kernel window.
        double mean = 0.0;
        for (int64_t d = -half; d <= half; ++d) mean += at(y, c + d);
        mean /= static_cast<double>(K);
        C4A_TEST_REQUIRE(std::fabs(mean) < 1e-13);
        // L1 norm of the kernel taps is within the nirs4all epsilon of 1.
        double sum_abs = 0.0;
        for (int64_t d = -half; d <= half; ++d) sum_abs += std::fabs(at(y, c + d));
        C4A_TEST_REQUIRE(std::fabs(sum_abs - 1.0) < 1e-8);
    }

    // Case C — alpha = 1.5 (zero-mean, still antisymmetric because sign(idx)
    // makes the kernel odd; the modulation strength is different from a=1).
    {
        constexpr int32_t K = 15;
        constexpr int64_t N = 51;
        std::vector<double> y;
        run_kernel_via_transform(1.5, 2.5, K, N, y);
        const int64_t c = N / 2;
        const int64_t half = (K - 1) / 2;
        for (int64_t d = 1; d <= half; ++d) {
            C4A_TEST_REQUIRE(std::fabs(at(y, c - d) + at(y, c + d)) < 1e-13);
        }
        double mean = 0.0;
        for (int64_t d = -half; d <= half; ++d) mean += at(y, c + d);
        mean /= static_cast<double>(K);
        C4A_TEST_REQUIRE(std::fabs(mean) < 1e-13);
    }
}

// ---------------------------------------------------------------------------
// Test 3 — output-cols helper.
// ---------------------------------------------------------------------------

void test_fck_output_cols() {
    int32_t out = -1;
    C4A_TEST_REQUIRE(c4a_pp_fck_static_output_cols(4, 200, &out) == C4A_OK);
    C4A_TEST_REQUIRE(out == 800);

    C4A_TEST_REQUIRE(c4a_pp_fck_static_output_cols(0, 200, &out) == C4A_OK);
    C4A_TEST_REQUIRE(out == 0);

    C4A_TEST_REQUIRE(c4a_pp_fck_static_output_cols(1, 0, &out) == C4A_OK);
    C4A_TEST_REQUIRE(out == 0);

    // Invalid arguments.
    C4A_TEST_REQUIRE(c4a_pp_fck_static_output_cols(-1, 10, &out) ==
                     C4A_ERR_INVALID_ARGUMENT);
    C4A_TEST_REQUIRE(c4a_pp_fck_static_output_cols(10, -1, &out) ==
                     C4A_ERR_INVALID_ARGUMENT);
    // Product would overflow int32_t.
    C4A_TEST_REQUIRE(c4a_pp_fck_static_output_cols(65536, 65536, &out) ==
                     C4A_ERR_INVALID_ARGUMENT);

    // NULL output.
    C4A_TEST_REQUIRE(c4a_pp_fck_static_output_cols(2, 10, nullptr) ==
                     C4A_ERR_NULL_POINTER);
}

// ---------------------------------------------------------------------------
// Test 4 — fixture parity against the canonical Python reference.
// ---------------------------------------------------------------------------

void test_fck_parity() {
    ParityFixture fx = load_fixture("fck_static_v1.json");
    for (const auto& c : fx.cases) {
        const int32_t K = static_cast<int32_t>(
            ::c4a_testing::params_get_int(c.params_json, "kernel_size", 11));
        // The fixture writes the alphas / sigmas vectors as JSON arrays. The
        // historical fixture key "sigmas" now carries FCK scales.
        // Pull their slices out of the params blob with the same `find +
        // read_value_span` plumbing used for nested objects.
        auto extract_vec = [&](const std::string& key) {
            std::vector<double> v;
            const std::string q = "\"" + key + "\"";
            const std::size_t at = c.params_json.find(q);
            if (at == std::string::npos) {
                throw std::runtime_error("fck_parity: missing " + key);
            }
            std::size_t p = at + q.size();
            while (p < c.params_json.size() &&
                   (c.params_json[p] == ' ' || c.params_json[p] == ':')) {
                ++p;
            }
            if (p >= c.params_json.size() || c.params_json[p] != '[') {
                throw std::runtime_error("fck_parity: expected '[' at " + key);
            }
            ++p;
            while (p < c.params_json.size()) {
                while (p < c.params_json.size() &&
                       (c.params_json[p] == ' ' || c.params_json[p] == ',')) {
                    ++p;
                }
                if (p >= c.params_json.size() || c.params_json[p] == ']') break;
                char* tail = nullptr;
                const double val = std::strtod(c.params_json.c_str() + p,
                                                &tail);
                if (tail == c.params_json.c_str() + p) break;
                v.push_back(val);
                p = static_cast<std::size_t>(tail - c.params_json.c_str());
            }
            return v;
        };
        const std::vector<double> alphas = extract_vec("alphas");
        const std::vector<double> sigmas = extract_vec("sigmas");
        const int32_t n_orders = static_cast<int32_t>(alphas.size());
        const int32_t n_scales = static_cast<int32_t>(sigmas.size());

        c4a_pp_fck_static_handle_t* h = nullptr;
        C4A_TEST_REQUIRE(c4a_pp_fck_static_create(&h, K,
                                                    alphas.data(), n_orders,
                                                    sigmas.data(), n_scales) ==
                         C4A_OK);

        std::vector<double> in = fx.input;
        const std::int64_t out_rows = c.output_rows;
        const std::int64_t out_cols = c.output_cols;
        std::vector<double> out(static_cast<std::size_t>(out_rows) *
                                 static_cast<std::size_t>(out_cols),
                                 0.0);
        c4a_matrix_view_t Xv = make_rowmajor_view(in.data(), fx.rows, fx.cols);
        c4a_matrix_view_t Yv = make_rowmajor_view(out.data(), out_rows,
                                                    out_cols);
        C4A_TEST_REQUIRE(c4a_pp_fck_static_transform(h, Xv, Yv) == C4A_OK);

        // Pure arithmetic (kernel build is one Gaussian + one pow + L1
        // normalisation; convolution is a small dot product), so the parity
        // is tight: 1e-12 abs / 1e-13 rel matches every other arithmetic
        // operator in the suite.
        ::c4a_testing::assert_close(out, c.expected_output,
                                     "fck_static/" + c.name,
                                     1e-12, 1e-13);
        c4a_pp_fck_static_destroy(h);
    }
}

}  // namespace

void register_fck_tests(c4a_testing::Runner& r);
void register_fck_tests(c4a_testing::Runner& r) {
    r.run("fck_smoke",             test_fck_smoke);
    r.run("fck_kernel_properties", test_fck_kernel_properties);
    r.run("fck_output_cols",       test_fck_output_cols);
    r.run("fck_parity",            test_fck_parity);
}
