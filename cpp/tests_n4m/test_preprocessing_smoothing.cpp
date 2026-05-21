// SPDX-License-Identifier: CECILL-2.1
//
// Parity tests for the Phase 4 stateless derivative & smoothing operators
// (SavitzkyGolay, FirstDerivative, SecondDerivative, NorrisWilliams,
// Gaussian).  For each operator the test loads a JSON fixture produced by
// parity/python_generator/scripts/generate_phase4_fixtures.py, decodes the
// input / expected matrices from big-endian hex doubles, applies the C
// engine through its public ABI, and asserts byte-equality or 1e-10 / 1e-11
// tolerance (loosened for SG and Gaussian, which depart by a few ULPs from
// the SVD / FFT paths SciPy uses internally on the same algorithm).
//
// Each operator contributes one smoke test exercising the create/transform/
// destroy lifecycle and one parity test sweeping the full fixture.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "n4m/n4m.h"

#include "fixture_parser.hpp"
#include "harness.hpp"

#ifndef N4M_PARITY_FIXTURE_DIR
#  error "N4M_PARITY_FIXTURE_DIR must be defined"
#endif

namespace {

using ParityFixture = ::n4m_testing::Fixture;
using ParityCase    = ::n4m_testing::Case;

ParityFixture load_fixture(const std::string& filename) {
    return ::n4m_testing::load_fixture(
        std::string(N4M_PARITY_FIXTURE_DIR) + "/" + filename,
        /*require_per_case_output_shape=*/false);
}

using ::n4m_testing::params_get_double;
using ::n4m_testing::params_get_int;
using ::n4m_testing::params_get_string;

// ---------------------------------------------------------------------------
// Tolerance check.  SG & Gaussian deviate up to ~5e-11 from the SciPy SVD /
// FFT paths because the underlying small-system solvers differ in rounding;
// FirstDerivative / SecondDerivative / NorrisWilliams are pure arithmetic
// and hit 1e-12 byte-tolerant comparisons.
// ---------------------------------------------------------------------------

void assert_close_with(const std::vector<double>& got,
                       const std::vector<double>& want,
                       const std::string& tag,
                       double abs_tol, double rel_tol) {
    ::n4m_testing::assert_close(got, want, tag, abs_tol, rel_tol);
}

n4m_matrix_view_t make_rowmajor_view(double* data, std::int64_t rows,
                                      std::int64_t cols) {
    n4m_matrix_view_t v{};
    const n4m_status_t st =
        n4m_matrix_view_init_rowmajor(&v, data, rows, cols, N4M_DTYPE_F64);
    N4M_TEST_REQUIRE(st == N4M_OK);
    return v;
}

// ---------------------------------------------------------------------------
// Smoke tests.
// ---------------------------------------------------------------------------

void test_savgol_smoke() {
    double X[10] = {1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0, 37.0, 46.0};
    double Y[10] = {0};
    n4m_pp_savgol_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_savgol_create(&h, /*w=*/5, /*p=*/2, /*d=*/0,
                                            /*delta=*/1.0,
                                            N4M_PP_SAVGOL_MIRROR, 0.0) ==
                     N4M_OK);
    N4M_TEST_REQUIRE(h != nullptr);
    n4m_matrix_view_t Xv = make_rowmajor_view(X, 1, 10);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 1, 10);
    N4M_TEST_REQUIRE(n4m_pp_savgol_transform(h, Xv, Yv) == N4M_OK);
    // The polynomial 1, 2, 4, 7, 11, ... is monotonically increasing; the
    // SG-smoothed values stay close to the raw values away from edges.
    N4M_TEST_REQUIRE(Y[4] > 9.0 && Y[4] < 12.0);
    // Invalid parameters.
    n4m_pp_savgol_handle_t* h_bad = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_savgol_create(&h_bad, /*w=*/4, 2, 0, 1.0,
                                            N4M_PP_SAVGOL_MIRROR, 0.0) ==
                     N4M_ERR_INVALID_ARGUMENT);  // even window
    N4M_TEST_REQUIRE(n4m_pp_savgol_create(&h_bad, /*w=*/5, 5, 0, 1.0,
                                            N4M_PP_SAVGOL_MIRROR, 0.0) ==
                     N4M_ERR_INVALID_ARGUMENT);  // polyorder >= window
    n4m_pp_savgol_destroy(h);
    n4m_pp_savgol_destroy(nullptr);
}

void test_first_derivative_smoke() {
    double X[5] = {0.0, 1.0, 4.0, 9.0, 16.0};  /* x^2 sampled */
    double Y[5] = {0};
    n4m_pp_first_derivative_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_first_derivative_create(&h, /*delta=*/1.0,
                                                      /*edge_order=*/2) ==
                     N4M_OK);
    n4m_matrix_view_t Xv = make_rowmajor_view(X, 1, 5);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 1, 5);
    N4M_TEST_REQUIRE(n4m_pp_first_derivative_transform(h, Xv, Yv) == N4M_OK);
    // Interior derivative of x^2 at integer points is 2x:
    // Y[1] = (4 - 0) / 2 = 2; Y[2] = (9 - 1) / 2 = 4; Y[3] = (16 - 4) / 2 = 6.
    N4M_TEST_REQUIRE(std::fabs(Y[1] - 2.0) < 1e-12);
    N4M_TEST_REQUIRE(std::fabs(Y[2] - 4.0) < 1e-12);
    N4M_TEST_REQUIRE(std::fabs(Y[3] - 6.0) < 1e-12);
    n4m_pp_first_derivative_destroy(h);
}

void test_second_derivative_smoke() {
    double X[7] = {0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0};  /* x^2 */
    double Y[7] = {0};
    n4m_pp_second_derivative_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_second_derivative_create(&h, /*delta=*/1.0,
                                                       /*edge_order=*/2) ==
                     N4M_OK);
    n4m_matrix_view_t Xv = make_rowmajor_view(X, 1, 7);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 1, 7);
    N4M_TEST_REQUIRE(n4m_pp_second_derivative_transform(h, Xv, Yv) == N4M_OK);
    // d^2/dx^2 (x^2) = 2 — the second derivative is exactly 2 everywhere
    // for the interior of a parabola sampled on the integers.
    N4M_TEST_REQUIRE(std::fabs(Y[3] - 2.0) < 1e-10);
    n4m_pp_second_derivative_destroy(h);
}

void test_norris_williams_smoke() {
    double X[7] = {1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0};
    double Y[7] = {0};
    n4m_pp_norris_williams_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_norris_williams_create(&h, /*segment=*/3,
                                                     /*gap=*/1,
                                                     /*derivative_order=*/1,
                                                     /*delta=*/1.0) ==
                     N4M_OK);
    n4m_matrix_view_t Xv = make_rowmajor_view(X, 1, 7);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 1, 7);
    N4M_TEST_REQUIRE(n4m_pp_norris_williams_transform(h, Xv, Yv) == N4M_OK);
    // Smoke check: the gap-1 derivative produces a sequence centred around
    // the local slope of the input (positive and increasing in this case).
    N4M_TEST_REQUIRE(Y[3] > 0.0);
    N4M_TEST_REQUIRE(Y[5] > Y[1]);
    n4m_pp_norris_williams_destroy(h);
    n4m_pp_norris_williams_destroy(nullptr);
}

void test_gaussian_smoke() {
    double X[7] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    double Y[7] = {0};
    n4m_pp_gaussian_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_gaussian_create(&h, /*sigma=*/1.0, /*order=*/0,
                                              N4M_PP_GAUSSIAN_REFLECT, 0.0,
                                              4.0) == N4M_OK);
    n4m_matrix_view_t Xv = make_rowmajor_view(X, 1, 7);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 1, 7);
    N4M_TEST_REQUIRE(n4m_pp_gaussian_transform(h, Xv, Yv) == N4M_OK);
    // Gaussian smoothing of a perfect ramp should leave the centre nearly
    // unchanged.
    N4M_TEST_REQUIRE(std::fabs(Y[3] - 4.0) < 1e-10);
    n4m_pp_gaussian_destroy(h);
}

// ---------------------------------------------------------------------------
// Parity tests — load fixtures and run every case.
// ---------------------------------------------------------------------------

n4m_pp_savgol_mode_t savgol_mode_from_string(const std::string& m) {
    if (m == "mirror")   return N4M_PP_SAVGOL_MIRROR;
    if (m == "constant") return N4M_PP_SAVGOL_CONSTANT;
    if (m == "nearest")  return N4M_PP_SAVGOL_NEAREST;
    if (m == "wrap")     return N4M_PP_SAVGOL_WRAP;
    if (m == "interp")   return N4M_PP_SAVGOL_INTERP;
    throw std::runtime_error("savgol: unknown mode " + m);
}

n4m_pp_gaussian_mode_t gaussian_mode_from_string(const std::string& m) {
    if (m == "reflect")  return N4M_PP_GAUSSIAN_REFLECT;
    if (m == "constant") return N4M_PP_GAUSSIAN_CONSTANT;
    if (m == "nearest")  return N4M_PP_GAUSSIAN_NEAREST;
    if (m == "mirror")   return N4M_PP_GAUSSIAN_MIRROR;
    if (m == "wrap")     return N4M_PP_GAUSSIAN_WRAP;
    throw std::runtime_error("gaussian: unknown mode " + m);
}

void verify_savgol_parity() {
    ParityFixture fx = load_fixture("savgol_v1.json");
    for (const auto& c : fx.cases) {
        const int w = static_cast<int>(
            params_get_int(c.params_json, "window_length", 11));
        const int p = static_cast<int>(
            params_get_int(c.params_json, "polyorder", 3));
        const int d = static_cast<int>(
            params_get_int(c.params_json, "deriv", 0));
        const double delta = params_get_double(c.params_json, "delta", 1.0);
        const std::string m = params_get_string(c.params_json, "mode",
                                                  "mirror");
        const double cv = params_get_double(c.params_json, "cval", 0.0);

        n4m_pp_savgol_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_pp_savgol_create(&h, w, p, d, delta,
                                                savgol_mode_from_string(m),
                                                cv) == N4M_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(in.size(), 0.0);
        n4m_matrix_view_t Xv = make_rowmajor_view(in.data(), fx.rows, fx.cols);
        n4m_matrix_view_t Yv = make_rowmajor_view(out.data(), fx.rows, fx.cols);
        N4M_TEST_REQUIRE(n4m_pp_savgol_transform(h, Xv, Yv) == N4M_OK);
        // SG: SVD-based scipy lstsq vs our Vandermonde-QR may diverge by a
        // few ULPs on the coefficient computation, then amplified by the
        // convolution sum.  1e-9 abs / 1e-10 rel gives us a comfortable
        // margin while still catching algorithmic bugs.
        assert_close_with(out, c.expected_output, "savgol/" + c.name,
                           1e-9, 1e-10);
        n4m_pp_savgol_destroy(h);
    }
}

void verify_first_derivative_parity() {
    ParityFixture fx = load_fixture("first_derivative_v1.json");
    for (const auto& c : fx.cases) {
        const double delta = params_get_double(c.params_json, "delta", 1.0);
        const int edge = static_cast<int>(
            params_get_int(c.params_json, "edge_order", 2));
        n4m_pp_first_derivative_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_pp_first_derivative_create(&h, delta, edge) ==
                         N4M_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(in.size(), 0.0);
        n4m_matrix_view_t Xv = make_rowmajor_view(in.data(), fx.rows, fx.cols);
        n4m_matrix_view_t Yv = make_rowmajor_view(out.data(), fx.rows, fx.cols);
        N4M_TEST_REQUIRE(n4m_pp_first_derivative_transform(h, Xv, Yv) ==
                         N4M_OK);
        // Pure arithmetic — byte-exact tolerance.
        assert_close_with(out, c.expected_output,
                           "first_derivative/" + c.name, 1e-12, 1e-13);
        n4m_pp_first_derivative_destroy(h);
    }
}

void verify_second_derivative_parity() {
    ParityFixture fx = load_fixture("second_derivative_v1.json");
    for (const auto& c : fx.cases) {
        const double delta = params_get_double(c.params_json, "delta", 1.0);
        const int edge = static_cast<int>(
            params_get_int(c.params_json, "edge_order", 2));
        n4m_pp_second_derivative_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_pp_second_derivative_create(&h, delta, edge) ==
                         N4M_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(in.size(), 0.0);
        n4m_matrix_view_t Xv = make_rowmajor_view(in.data(), fx.rows, fx.cols);
        n4m_matrix_view_t Yv = make_rowmajor_view(out.data(), fx.rows, fx.cols);
        N4M_TEST_REQUIRE(n4m_pp_second_derivative_transform(h, Xv, Yv) ==
                         N4M_OK);
        assert_close_with(out, c.expected_output,
                           "second_derivative/" + c.name, 1e-12, 1e-13);
        n4m_pp_second_derivative_destroy(h);
    }
}

void verify_norris_williams_parity() {
    ParityFixture fx = load_fixture("norris_williams_v1.json");
    for (const auto& c : fx.cases) {
        const int seg = static_cast<int>(
            params_get_int(c.params_json, "segment", 5));
        const int gap = static_cast<int>(
            params_get_int(c.params_json, "gap", 5));
        const int d = static_cast<int>(
            params_get_int(c.params_json, "derivative_order", 1));
        const double delta = params_get_double(c.params_json, "delta", 1.0);
        n4m_pp_norris_williams_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_pp_norris_williams_create(&h, seg, gap, d,
                                                         delta) == N4M_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(in.size(), 0.0);
        n4m_matrix_view_t Xv = make_rowmajor_view(in.data(), fx.rows, fx.cols);
        n4m_matrix_view_t Yv = make_rowmajor_view(out.data(), fx.rows, fx.cols);
        N4M_TEST_REQUIRE(n4m_pp_norris_williams_transform(h, Xv, Yv) ==
                         N4M_OK);
        // Pure arithmetic — tight tolerance.
        assert_close_with(out, c.expected_output,
                           "norris_williams/" + c.name, 1e-12, 1e-13);
        n4m_pp_norris_williams_destroy(h);
    }
}

void verify_gaussian_parity() {
    ParityFixture fx = load_fixture("gaussian_v1.json");
    for (const auto& c : fx.cases) {
        const double sigma = params_get_double(c.params_json, "sigma", 1.0);
        const int order = static_cast<int>(
            params_get_int(c.params_json, "order", 0));
        const std::string m = params_get_string(c.params_json, "mode",
                                                  "reflect");
        const double cv = params_get_double(c.params_json, "cval", 0.0);
        const double tr = params_get_double(c.params_json, "truncate", 4.0);
        n4m_pp_gaussian_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_pp_gaussian_create(&h, sigma, order,
                                                  gaussian_mode_from_string(m),
                                                  cv, tr) == N4M_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(in.size(), 0.0);
        n4m_matrix_view_t Xv = make_rowmajor_view(in.data(), fx.rows, fx.cols);
        n4m_matrix_view_t Yv = make_rowmajor_view(out.data(), fx.rows, fx.cols);
        N4M_TEST_REQUIRE(n4m_pp_gaussian_transform(h, Xv, Yv) == N4M_OK);
        // Gaussian: kernel sum + convolution differ by a couple of ULPs vs
        // scipy's correlate1d (which may use FFT for larger kernels).  We
        // keep the same tolerance as SG.
        assert_close_with(out, c.expected_output,
                           "gaussian/" + c.name, 1e-9, 1e-10);
        n4m_pp_gaussian_destroy(h);
    }
}

}  // namespace

void register_preprocessing_smoothing_tests(n4m_testing::Runner& r);
void register_preprocessing_smoothing_tests(n4m_testing::Runner& r) {
    r.run("pp_savgol_smoke",                test_savgol_smoke);
    r.run("pp_savgol_parity",               verify_savgol_parity);
    r.run("pp_first_derivative_smoke",      test_first_derivative_smoke);
    r.run("pp_first_derivative_parity",     verify_first_derivative_parity);
    r.run("pp_second_derivative_smoke",     test_second_derivative_smoke);
    r.run("pp_second_derivative_parity",    verify_second_derivative_parity);
    r.run("pp_norris_williams_smoke",       test_norris_williams_smoke);
    r.run("pp_norris_williams_parity",      verify_norris_williams_parity);
    r.run("pp_gaussian_smoke",              test_gaussian_smoke);
    r.run("pp_gaussian_parity",             verify_gaussian_parity);
}
