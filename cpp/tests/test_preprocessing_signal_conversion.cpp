// SPDX-License-Identifier: CECILL-2.1
//
// Parity tests for the Phase 7 stateless signal-conversion preprocessing
// operators (ToAbsorbance, FromAbsorbance, PercentToFraction,
// FractionToPercent, KubelkaMunk). For each operator the test loads a JSON
// fixture produced by parity/python_generator/scripts/generate_phase7_fixtures.py,
// decodes the input/output matrices from big-endian hex doubles, applies
// the C engine through its public ABI, and asserts byte-equality (or
// 1e-12 abs / 1e-13 rel tolerance) against the reference output.
//
// Each operator contributes two tests:
//
//   * a smoke test exercising create/transform/destroy on a small inline
//     matrix to validate the public ABI surface and lifecycle semantics;
//   * a parity test sweeping every case in the fixture and asserting the
//     output matches the reference within tolerance.
//
// Percent-mode cases (`is_percent: true`) load the fixture's fractional
// input matrix and multiply by 100.0 before feeding the engine — the
// Python reference generator did the same scale, so the comparison stays
// at the same 1e-12 / 1e-13 tolerance bar.

#include <cmath>
#include <cstddef>
#include <cstdint>
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

using ::n4m_testing::params_get_bool;
using ::n4m_testing::params_get_double;

void assert_close(const std::vector<double>& got,
                  const std::vector<double>& want,
                  const std::string& tag) {
    ::n4m_testing::assert_close(got, want, tag, /*abs_tol=*/1e-12,
                                 /*rel_tol=*/1e-13);
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
// Smoke tests — exercise create/transform/destroy on a small hand-crafted
// matrix for each operator.
// ---------------------------------------------------------------------------

void test_to_absorbance_smoke() {
    // R = [[0.5, 0.4, 0.3], [0.6, 0.5, 0.4]] → A = -log10(R).
    double X[6] = {0.5, 0.4, 0.3, 0.6, 0.5, 0.4};
    double Y[6] = {0};
    n4m_pp_to_absorbance_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_to_absorbance_create(
                         &h, /*is_percent=*/0, /*epsilon=*/1e-10,
                         /*clip_negative=*/1) == N4M_OK);
    N4M_TEST_REQUIRE(h != nullptr);
    n4m_matrix_view_t Xv = make_rowmajor_view(X, 2, 3);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 2, 3);
    N4M_TEST_REQUIRE(n4m_pp_to_absorbance_transform(h, Xv, Yv) == N4M_OK);
    // -log10(0.5) ≈ 0.30103, -log10(0.1) is not present; check first cell.
    N4M_TEST_REQUIRE(std::fabs(Y[0] - (-std::log10(0.5))) < 1e-15);
    N4M_TEST_REQUIRE(std::fabs(Y[5] - (-std::log10(0.4))) < 1e-15);
    n4m_pp_to_absorbance_destroy(h);
    n4m_pp_to_absorbance_destroy(nullptr);  // null-safe
}

void test_from_absorbance_smoke() {
    // A = [[0.3, 0.4], [0.2, 0.1]] → R = 10^(-A).
    double X[4] = {0.3, 0.4, 0.2, 0.1};
    double Y[4] = {0};
    n4m_pp_from_absorbance_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_from_absorbance_create(&h, /*is_percent=*/0)
                     == N4M_OK);
    n4m_matrix_view_t Xv = make_rowmajor_view(X, 2, 2);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 2, 2);
    N4M_TEST_REQUIRE(n4m_pp_from_absorbance_transform(h, Xv, Yv) == N4M_OK);
    N4M_TEST_REQUIRE(std::fabs(Y[0] - std::pow(10.0, -0.3)) < 1e-15);
    N4M_TEST_REQUIRE(std::fabs(Y[3] - std::pow(10.0, -0.1)) < 1e-15);
    n4m_pp_from_absorbance_destroy(h);
    n4m_pp_from_absorbance_destroy(nullptr);
}

void test_pct_to_frac_smoke() {
    double X[4] = {50.0, 60.0, 70.0, 80.0};
    double Y[4] = {0};
    n4m_pp_pct_to_frac_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_pct_to_frac_create(&h) == N4M_OK);
    n4m_matrix_view_t Xv = make_rowmajor_view(X, 2, 2);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 2, 2);
    N4M_TEST_REQUIRE(n4m_pp_pct_to_frac_transform(h, Xv, Yv) == N4M_OK);
    N4M_TEST_REQUIRE(std::fabs(Y[0] - 0.5) < 1e-15);
    N4M_TEST_REQUIRE(std::fabs(Y[3] - 0.8) < 1e-15);
    n4m_pp_pct_to_frac_destroy(h);
    n4m_pp_pct_to_frac_destroy(nullptr);
}

void test_frac_to_pct_smoke() {
    double X[4] = {0.5, 0.6, 0.7, 0.8};
    double Y[4] = {0};
    n4m_pp_frac_to_pct_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_frac_to_pct_create(&h) == N4M_OK);
    n4m_matrix_view_t Xv = make_rowmajor_view(X, 2, 2);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 2, 2);
    N4M_TEST_REQUIRE(n4m_pp_frac_to_pct_transform(h, Xv, Yv) == N4M_OK);
    N4M_TEST_REQUIRE(std::fabs(Y[0] - 50.0) < 1e-15);
    N4M_TEST_REQUIRE(std::fabs(Y[3] - 80.0) < 1e-15);
    n4m_pp_frac_to_pct_destroy(h);
    n4m_pp_frac_to_pct_destroy(nullptr);
}

void test_kubelka_munk_smoke() {
    // F(0.5) = (1 - 0.5)^2 / (2 * 0.5) = 0.25 / 1.0 = 0.25.
    double X[4] = {0.5, 0.4, 0.6, 0.5};
    double Y[4] = {0};
    n4m_pp_kubelka_munk_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_kubelka_munk_create(&h, /*is_percent=*/0,
                                                  /*epsilon=*/1e-10) == N4M_OK);
    n4m_matrix_view_t Xv = make_rowmajor_view(X, 2, 2);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 2, 2);
    N4M_TEST_REQUIRE(n4m_pp_kubelka_munk_transform(h, Xv, Yv) == N4M_OK);
    N4M_TEST_REQUIRE(std::fabs(Y[0] - 0.25) < 1e-15);
    // F(0.4) = 0.36 / 0.8 = 0.45.
    N4M_TEST_REQUIRE(std::fabs(Y[1] - 0.45) < 1e-15);
    n4m_pp_kubelka_munk_destroy(h);
    n4m_pp_kubelka_munk_destroy(nullptr);
}

// ---------------------------------------------------------------------------
// Parity tests — load fixture, dispatch each case to the matching engine.
// ---------------------------------------------------------------------------

void verify_to_absorbance_parity() {
    ParityFixture fx = load_fixture("to_absorbance_v1.json");
    for (const auto& c : fx.cases) {
        const int is_percent =
            params_get_bool(c.params_json, "is_percent", false) ? 1 : 0;
        const double epsilon =
            params_get_double(c.params_json, "epsilon", 1e-10);
        const int clip_negative =
            params_get_bool(c.params_json, "clip_negative", true) ? 1 : 0;

        n4m_pp_to_absorbance_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_pp_to_absorbance_create(
                             &h, is_percent, epsilon, clip_negative) == N4M_OK);
        std::vector<double> in = fx.input;
        // Percent cases: the Python reference was invoked on R * 100.0, so
        // we scale the loaded fractional input the same way before feeding
        // the engine. 100.0 is exactly representable, so the scale is
        // bit-exact.
        if (is_percent) {
            for (auto& v : in) v = v * 100.0;
        }
        std::vector<double> out(in.size(), 0.0);
        n4m_matrix_view_t Xv = make_rowmajor_view(in.data(),  fx.rows, fx.cols);
        n4m_matrix_view_t Yv = make_rowmajor_view(out.data(), fx.rows, fx.cols);
        N4M_TEST_REQUIRE(n4m_pp_to_absorbance_transform(h, Xv, Yv) == N4M_OK);
        assert_close(out, c.expected_output, "to_absorbance/" + c.name);
        n4m_pp_to_absorbance_destroy(h);
    }
}

void verify_from_absorbance_parity() {
    ParityFixture fx = load_fixture("from_absorbance_v1.json");
    for (const auto& c : fx.cases) {
        const int is_percent =
            params_get_bool(c.params_json, "is_percent", false) ? 1 : 0;

        n4m_pp_from_absorbance_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_pp_from_absorbance_create(&h, is_percent) == N4M_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(in.size(), 0.0);
        n4m_matrix_view_t Xv = make_rowmajor_view(in.data(),  fx.rows, fx.cols);
        n4m_matrix_view_t Yv = make_rowmajor_view(out.data(), fx.rows, fx.cols);
        N4M_TEST_REQUIRE(n4m_pp_from_absorbance_transform(h, Xv, Yv) == N4M_OK);
        assert_close(out, c.expected_output, "from_absorbance/" + c.name);
        n4m_pp_from_absorbance_destroy(h);
    }
}

void verify_pct_to_frac_parity() {
    ParityFixture fx = load_fixture("pct_to_frac_v1.json");
    for (const auto& c : fx.cases) {
        n4m_pp_pct_to_frac_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_pp_pct_to_frac_create(&h) == N4M_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(in.size(), 0.0);
        n4m_matrix_view_t Xv = make_rowmajor_view(in.data(),  fx.rows, fx.cols);
        n4m_matrix_view_t Yv = make_rowmajor_view(out.data(), fx.rows, fx.cols);
        N4M_TEST_REQUIRE(n4m_pp_pct_to_frac_transform(h, Xv, Yv) == N4M_OK);
        assert_close(out, c.expected_output, "pct_to_frac/" + c.name);
        n4m_pp_pct_to_frac_destroy(h);
    }
}

void verify_frac_to_pct_parity() {
    ParityFixture fx = load_fixture("frac_to_pct_v1.json");
    for (const auto& c : fx.cases) {
        n4m_pp_frac_to_pct_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_pp_frac_to_pct_create(&h) == N4M_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(in.size(), 0.0);
        n4m_matrix_view_t Xv = make_rowmajor_view(in.data(),  fx.rows, fx.cols);
        n4m_matrix_view_t Yv = make_rowmajor_view(out.data(), fx.rows, fx.cols);
        N4M_TEST_REQUIRE(n4m_pp_frac_to_pct_transform(h, Xv, Yv) == N4M_OK);
        assert_close(out, c.expected_output, "frac_to_pct/" + c.name);
        n4m_pp_frac_to_pct_destroy(h);
    }
}

void verify_kubelka_munk_parity() {
    ParityFixture fx = load_fixture("kubelka_munk_v1.json");
    for (const auto& c : fx.cases) {
        const int is_percent =
            params_get_bool(c.params_json, "is_percent", false) ? 1 : 0;
        const double epsilon =
            params_get_double(c.params_json, "epsilon", 1e-10);

        n4m_pp_kubelka_munk_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_pp_kubelka_munk_create(&h, is_percent, epsilon)
                         == N4M_OK);
        std::vector<double> in = fx.input;
        if (is_percent) {
            for (auto& v : in) v = v * 100.0;
        }
        std::vector<double> out(in.size(), 0.0);
        n4m_matrix_view_t Xv = make_rowmajor_view(in.data(),  fx.rows, fx.cols);
        n4m_matrix_view_t Yv = make_rowmajor_view(out.data(), fx.rows, fx.cols);
        N4M_TEST_REQUIRE(n4m_pp_kubelka_munk_transform(h, Xv, Yv) == N4M_OK);
        assert_close(out, c.expected_output, "kubelka_munk/" + c.name);
        n4m_pp_kubelka_munk_destroy(h);
    }
}

}  // namespace

void register_preprocessing_signal_conversion_tests(n4m_testing::Runner& r);
void register_preprocessing_signal_conversion_tests(n4m_testing::Runner& r) {
    r.run("pp_to_absorbance_smoke",     test_to_absorbance_smoke);
    r.run("pp_to_absorbance_parity",    verify_to_absorbance_parity);
    r.run("pp_from_absorbance_smoke",   test_from_absorbance_smoke);
    r.run("pp_from_absorbance_parity",  verify_from_absorbance_parity);
    r.run("pp_pct_to_frac_smoke",       test_pct_to_frac_smoke);
    r.run("pp_pct_to_frac_parity",      verify_pct_to_frac_parity);
    r.run("pp_frac_to_pct_smoke",       test_frac_to_pct_smoke);
    r.run("pp_frac_to_pct_parity",      verify_frac_to_pct_parity);
    r.run("pp_kubelka_munk_smoke",      test_kubelka_munk_smoke);
    r.run("pp_kubelka_munk_parity",     verify_kubelka_munk_parity);
}
