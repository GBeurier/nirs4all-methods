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

#include "chemometrics4all/c4a.h"

#include "fixture_parser.hpp"
#include "harness.hpp"

#ifndef C4A_PARITY_FIXTURE_DIR
#  error "C4A_PARITY_FIXTURE_DIR must be defined"
#endif

namespace {

using ParityFixture = ::c4a_testing::Fixture;
using ParityCase    = ::c4a_testing::Case;

ParityFixture load_fixture(const std::string& filename) {
    return ::c4a_testing::load_fixture(
        std::string(C4A_PARITY_FIXTURE_DIR) + "/" + filename,
        /*require_per_case_output_shape=*/false);
}

using ::c4a_testing::params_get_bool;
using ::c4a_testing::params_get_double;

void assert_close(const std::vector<double>& got,
                  const std::vector<double>& want,
                  const std::string& tag) {
    ::c4a_testing::assert_close(got, want, tag, /*abs_tol=*/1e-12,
                                 /*rel_tol=*/1e-13);
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
// Smoke tests — exercise create/transform/destroy on a small hand-crafted
// matrix for each operator.
// ---------------------------------------------------------------------------

void test_to_absorbance_smoke() {
    // R = [[0.5, 0.4, 0.3], [0.6, 0.5, 0.4]] → A = -log10(R).
    double X[6] = {0.5, 0.4, 0.3, 0.6, 0.5, 0.4};
    double Y[6] = {0};
    c4a_pp_to_absorbance_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_to_absorbance_create(
                         &h, /*is_percent=*/0, /*epsilon=*/1e-10,
                         /*clip_negative=*/1) == C4A_OK);
    C4A_TEST_REQUIRE(h != nullptr);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 2, 3);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 2, 3);
    C4A_TEST_REQUIRE(c4a_pp_to_absorbance_transform(h, Xv, Yv) == C4A_OK);
    // -log10(0.5) ≈ 0.30103, -log10(0.1) is not present; check first cell.
    C4A_TEST_REQUIRE(std::fabs(Y[0] - (-std::log10(0.5))) < 1e-15);
    C4A_TEST_REQUIRE(std::fabs(Y[5] - (-std::log10(0.4))) < 1e-15);
    c4a_pp_to_absorbance_destroy(h);
    c4a_pp_to_absorbance_destroy(nullptr);  // null-safe
}

void test_from_absorbance_smoke() {
    // A = [[0.3, 0.4], [0.2, 0.1]] → R = 10^(-A).
    double X[4] = {0.3, 0.4, 0.2, 0.1};
    double Y[4] = {0};
    c4a_pp_from_absorbance_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_from_absorbance_create(&h, /*is_percent=*/0)
                     == C4A_OK);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 2, 2);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 2, 2);
    C4A_TEST_REQUIRE(c4a_pp_from_absorbance_transform(h, Xv, Yv) == C4A_OK);
    C4A_TEST_REQUIRE(std::fabs(Y[0] - std::pow(10.0, -0.3)) < 1e-15);
    C4A_TEST_REQUIRE(std::fabs(Y[3] - std::pow(10.0, -0.1)) < 1e-15);
    c4a_pp_from_absorbance_destroy(h);
    c4a_pp_from_absorbance_destroy(nullptr);
}

void test_pct_to_frac_smoke() {
    double X[4] = {50.0, 60.0, 70.0, 80.0};
    double Y[4] = {0};
    c4a_pp_pct_to_frac_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_pct_to_frac_create(&h) == C4A_OK);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 2, 2);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 2, 2);
    C4A_TEST_REQUIRE(c4a_pp_pct_to_frac_transform(h, Xv, Yv) == C4A_OK);
    C4A_TEST_REQUIRE(std::fabs(Y[0] - 0.5) < 1e-15);
    C4A_TEST_REQUIRE(std::fabs(Y[3] - 0.8) < 1e-15);
    c4a_pp_pct_to_frac_destroy(h);
    c4a_pp_pct_to_frac_destroy(nullptr);
}

void test_frac_to_pct_smoke() {
    double X[4] = {0.5, 0.6, 0.7, 0.8};
    double Y[4] = {0};
    c4a_pp_frac_to_pct_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_frac_to_pct_create(&h) == C4A_OK);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 2, 2);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 2, 2);
    C4A_TEST_REQUIRE(c4a_pp_frac_to_pct_transform(h, Xv, Yv) == C4A_OK);
    C4A_TEST_REQUIRE(std::fabs(Y[0] - 50.0) < 1e-15);
    C4A_TEST_REQUIRE(std::fabs(Y[3] - 80.0) < 1e-15);
    c4a_pp_frac_to_pct_destroy(h);
    c4a_pp_frac_to_pct_destroy(nullptr);
}

void test_kubelka_munk_smoke() {
    // F(0.5) = (1 - 0.5)^2 / (2 * 0.5) = 0.25 / 1.0 = 0.25.
    double X[4] = {0.5, 0.4, 0.6, 0.5};
    double Y[4] = {0};
    c4a_pp_kubelka_munk_handle_t* h = nullptr;
    C4A_TEST_REQUIRE(c4a_pp_kubelka_munk_create(&h, /*is_percent=*/0,
                                                  /*epsilon=*/1e-10) == C4A_OK);
    c4a_matrix_view_t Xv = make_rowmajor_view(X, 2, 2);
    c4a_matrix_view_t Yv = make_rowmajor_view(Y, 2, 2);
    C4A_TEST_REQUIRE(c4a_pp_kubelka_munk_transform(h, Xv, Yv) == C4A_OK);
    C4A_TEST_REQUIRE(std::fabs(Y[0] - 0.25) < 1e-15);
    // F(0.4) = 0.36 / 0.8 = 0.45.
    C4A_TEST_REQUIRE(std::fabs(Y[1] - 0.45) < 1e-15);
    c4a_pp_kubelka_munk_destroy(h);
    c4a_pp_kubelka_munk_destroy(nullptr);
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

        c4a_pp_to_absorbance_handle_t* h = nullptr;
        C4A_TEST_REQUIRE(c4a_pp_to_absorbance_create(
                             &h, is_percent, epsilon, clip_negative) == C4A_OK);
        std::vector<double> in = fx.input;
        // Percent cases: the Python reference was invoked on R * 100.0, so
        // we scale the loaded fractional input the same way before feeding
        // the engine. 100.0 is exactly representable, so the scale is
        // bit-exact.
        if (is_percent) {
            for (auto& v : in) v = v * 100.0;
        }
        std::vector<double> out(in.size(), 0.0);
        c4a_matrix_view_t Xv = make_rowmajor_view(in.data(),  fx.rows, fx.cols);
        c4a_matrix_view_t Yv = make_rowmajor_view(out.data(), fx.rows, fx.cols);
        C4A_TEST_REQUIRE(c4a_pp_to_absorbance_transform(h, Xv, Yv) == C4A_OK);
        assert_close(out, c.expected_output, "to_absorbance/" + c.name);
        c4a_pp_to_absorbance_destroy(h);
    }
}

void verify_from_absorbance_parity() {
    ParityFixture fx = load_fixture("from_absorbance_v1.json");
    for (const auto& c : fx.cases) {
        const int is_percent =
            params_get_bool(c.params_json, "is_percent", false) ? 1 : 0;

        c4a_pp_from_absorbance_handle_t* h = nullptr;
        C4A_TEST_REQUIRE(c4a_pp_from_absorbance_create(&h, is_percent) == C4A_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(in.size(), 0.0);
        c4a_matrix_view_t Xv = make_rowmajor_view(in.data(),  fx.rows, fx.cols);
        c4a_matrix_view_t Yv = make_rowmajor_view(out.data(), fx.rows, fx.cols);
        C4A_TEST_REQUIRE(c4a_pp_from_absorbance_transform(h, Xv, Yv) == C4A_OK);
        assert_close(out, c.expected_output, "from_absorbance/" + c.name);
        c4a_pp_from_absorbance_destroy(h);
    }
}

void verify_pct_to_frac_parity() {
    ParityFixture fx = load_fixture("pct_to_frac_v1.json");
    for (const auto& c : fx.cases) {
        c4a_pp_pct_to_frac_handle_t* h = nullptr;
        C4A_TEST_REQUIRE(c4a_pp_pct_to_frac_create(&h) == C4A_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(in.size(), 0.0);
        c4a_matrix_view_t Xv = make_rowmajor_view(in.data(),  fx.rows, fx.cols);
        c4a_matrix_view_t Yv = make_rowmajor_view(out.data(), fx.rows, fx.cols);
        C4A_TEST_REQUIRE(c4a_pp_pct_to_frac_transform(h, Xv, Yv) == C4A_OK);
        assert_close(out, c.expected_output, "pct_to_frac/" + c.name);
        c4a_pp_pct_to_frac_destroy(h);
    }
}

void verify_frac_to_pct_parity() {
    ParityFixture fx = load_fixture("frac_to_pct_v1.json");
    for (const auto& c : fx.cases) {
        c4a_pp_frac_to_pct_handle_t* h = nullptr;
        C4A_TEST_REQUIRE(c4a_pp_frac_to_pct_create(&h) == C4A_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(in.size(), 0.0);
        c4a_matrix_view_t Xv = make_rowmajor_view(in.data(),  fx.rows, fx.cols);
        c4a_matrix_view_t Yv = make_rowmajor_view(out.data(), fx.rows, fx.cols);
        C4A_TEST_REQUIRE(c4a_pp_frac_to_pct_transform(h, Xv, Yv) == C4A_OK);
        assert_close(out, c.expected_output, "frac_to_pct/" + c.name);
        c4a_pp_frac_to_pct_destroy(h);
    }
}

void verify_kubelka_munk_parity() {
    ParityFixture fx = load_fixture("kubelka_munk_v1.json");
    for (const auto& c : fx.cases) {
        const int is_percent =
            params_get_bool(c.params_json, "is_percent", false) ? 1 : 0;
        const double epsilon =
            params_get_double(c.params_json, "epsilon", 1e-10);

        c4a_pp_kubelka_munk_handle_t* h = nullptr;
        C4A_TEST_REQUIRE(c4a_pp_kubelka_munk_create(&h, is_percent, epsilon)
                         == C4A_OK);
        std::vector<double> in = fx.input;
        if (is_percent) {
            for (auto& v : in) v = v * 100.0;
        }
        std::vector<double> out(in.size(), 0.0);
        c4a_matrix_view_t Xv = make_rowmajor_view(in.data(),  fx.rows, fx.cols);
        c4a_matrix_view_t Yv = make_rowmajor_view(out.data(), fx.rows, fx.cols);
        C4A_TEST_REQUIRE(c4a_pp_kubelka_munk_transform(h, Xv, Yv) == C4A_OK);
        assert_close(out, c.expected_output, "kubelka_munk/" + c.name);
        c4a_pp_kubelka_munk_destroy(h);
    }
}

}  // namespace

void register_preprocessing_signal_conversion_tests(c4a_testing::Runner& r);
void register_preprocessing_signal_conversion_tests(c4a_testing::Runner& r) {
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
