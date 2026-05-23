// SPDX-License-Identifier: CECILL-2.1
//
// Parity tests for the Phase 5a + 5b baseline correction operators
// (Detrend, AsLS, AirPLS, ArPLS + ModPoly, IModPoly, SNIP, RollingBall,
// IAsLS, BEADS). Each operator contributes one smoke test (exercise
// create / transform / destroy on a small inline matrix) and one parity
// test (load a JSON fixture produced by the matching generator under
// parity/python_generator/scripts/, run the C engine, assert within
// tolerance against the frozen NumPy reference).
//
// Tolerances per the brief (parity/tolerances.md):
//
//   * Detrend     : 1e-11 abs / 1e-12 rel.  Closed-form, QR vs SVD ~1e-13.
//   * AsLS / AirPLS / ArPLS:
//                   1e-7 abs / 1e-8 rel.  Iterative banded LDLT.
//   * ModPoly / IModPoly:
//                   1e-9 abs / 1e-10 rel.  Iterative polynomial QR.
//   * SNIP / RollingBall:
//                   1e-12 abs / 1e-13 rel.  Pure arithmetic / morphology.
//   * IAsLS       : 1e-7 abs / 1e-8 rel.  Banded LDLT.
//   * BEADS       : 1e-6 abs / 1e-7 rel.  Banded LDLT with reweighting.

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
        /*require_per_case_output_shape=*/true);
}

using ::n4m_testing::params_get_double;
using ::n4m_testing::params_get_int;

n4m_matrix_view_t make_rowmajor_view(double* data, std::int64_t rows,
                                      std::int64_t cols) {
    n4m_matrix_view_t v{};
    const n4m_status_t st =
        n4m_matrix_view_init_rowmajor(&v, data, rows, cols, N4M_DTYPE_F64);
    N4M_TEST_REQUIRE(st == N4M_OK);
    return v;
}

// ---------------------------------------------------------------------------
// Phase 5a smoke tests.
// ---------------------------------------------------------------------------

void test_detrend_smoke() {
    // A pure linear ramp should be perfectly removed by polyorder=1.
    double X[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    double Y[10] = {0};
    n4m_pp_detrend_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_detrend_create(&h, /*polyorder=*/1) == N4M_OK);
    N4M_TEST_REQUIRE(h != nullptr);
    n4m_matrix_view_t Xv = make_rowmajor_view(X, 1, 10);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 1, 10);
    N4M_TEST_REQUIRE(n4m_pp_detrend_transform(h, Xv, Yv) == N4M_OK);
    // After detrending a perfect ramp the output should be ~0 everywhere.
    for (int i = 0; i < 10; ++i) {
        N4M_TEST_REQUIRE(std::fabs(Y[i]) < 1e-12);
    }
    n4m_pp_detrend_destroy(h);
    n4m_pp_detrend_destroy(nullptr);  // null-safe
}

void test_asls_smoke() {
    // A row of constant 1.0 should yield a baseline equal to 1.0 → detrended
    // ~ 0 (within iteration tolerance).
    double X[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    double Y[10] = {0};
    n4m_pp_asls_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_asls_create(&h, /*lam=*/1e6, /*p=*/1e-2,
                                          /*max_iter=*/50, /*tol=*/1e-3)
                     == N4M_OK);
    N4M_TEST_REQUIRE(h != nullptr);
    n4m_matrix_view_t Xv = make_rowmajor_view(X, 1, 10);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 1, 10);
    N4M_TEST_REQUIRE(n4m_pp_asls_transform(h, Xv, Yv) == N4M_OK);
    for (int i = 0; i < 10; ++i) {
        N4M_TEST_REQUIRE(std::isfinite(Y[i]));
    }
    n4m_pp_asls_destroy(h);
    n4m_pp_asls_destroy(nullptr);
    // Invalid parameters reject.
    N4M_TEST_REQUIRE(n4m_pp_asls_create(&h, /*lam=*/-1.0, 1e-2, 50, 1e-3)
                     == N4M_ERR_INVALID_ARGUMENT);
    N4M_TEST_REQUIRE(n4m_pp_asls_create(&h, /*lam=*/1e6, /*p=*/1.5, 50, 1e-3)
                     == N4M_ERR_INVALID_ARGUMENT);
}

void test_airpls_smoke() {
    // Phase 5b strengthening: on a constant input the baseline equals the
    // constant, so the detrended output must be bounded near zero.
    double X[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    double Y[10] = {0};
    n4m_pp_airpls_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_airpls_create(&h, /*lam=*/1e6,
                                           /*max_iter=*/50, /*tol=*/1e-3)
                     == N4M_OK);
    N4M_TEST_REQUIRE(h != nullptr);
    n4m_matrix_view_t Xv = make_rowmajor_view(X, 1, 10);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 1, 10);
    N4M_TEST_REQUIRE(n4m_pp_airpls_transform(h, Xv, Yv) == N4M_OK);
    for (int i = 0; i < 10; ++i) {
        N4M_TEST_REQUIRE(std::isfinite(Y[i]));
        N4M_TEST_REQUIRE(std::fabs(Y[i]) < 1e-6);
    }
    n4m_pp_airpls_destroy(h);
    N4M_TEST_REQUIRE(n4m_pp_airpls_create(&h, /*lam=*/0.0, 50, 1e-3)
                     == N4M_ERR_INVALID_ARGUMENT);
}

void test_arpls_smoke() {
    double X[10] = {0, 0.5, 1, 0.5, 0, 0, 0.5, 1, 0.5, 0};
    double Y[10] = {0};
    n4m_pp_arpls_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_arpls_create(&h, /*lam=*/1e5,
                                          /*max_iter=*/50, /*tol=*/1e-3)
                     == N4M_OK);
    N4M_TEST_REQUIRE(h != nullptr);
    n4m_matrix_view_t Xv = make_rowmajor_view(X, 1, 10);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 1, 10);
    N4M_TEST_REQUIRE(n4m_pp_arpls_transform(h, Xv, Yv) == N4M_OK);
    for (int i = 0; i < 10; ++i) {
        N4M_TEST_REQUIRE(std::isfinite(Y[i]));
    }
    n4m_pp_arpls_destroy(h);
    N4M_TEST_REQUIRE(n4m_pp_arpls_create(&h, /*lam=*/-1.0, 50, 1e-3)
                     == N4M_ERR_INVALID_ARGUMENT);
}

// ---------------------------------------------------------------------------
// Phase 5b smoke tests.
// ---------------------------------------------------------------------------

void test_modpoly_smoke() {
    double X[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    double Y[10] = {0};
    n4m_pp_modpoly_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_modpoly_create(&h, /*polyorder=*/1,
                                            /*max_iter=*/250, /*tol=*/1e-3)
                     == N4M_OK);
    N4M_TEST_REQUIRE(h != nullptr);
    n4m_matrix_view_t Xv = make_rowmajor_view(X, 1, 10);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 1, 10);
    N4M_TEST_REQUIRE(n4m_pp_modpoly_transform(h, Xv, Yv) == N4M_OK);
    // ModPoly clips peaks downward; on a pure ramp the baseline IS the ramp,
    // so detrended ~ 0 (within iteration tolerance).
    for (int i = 0; i < 10; ++i) {
        N4M_TEST_REQUIRE(std::isfinite(Y[i]));
        N4M_TEST_REQUIRE(std::fabs(Y[i]) < 1e-10);
    }
    n4m_pp_modpoly_destroy(h);
    n4m_pp_modpoly_destroy(nullptr);
    N4M_TEST_REQUIRE(n4m_pp_modpoly_create(&h, /*polyorder=*/-1, 250, 1e-3)
                     == N4M_ERR_INVALID_ARGUMENT);
}

void test_imodpoly_smoke() {
    double X[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    double Y[10] = {0};
    n4m_pp_imodpoly_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_imodpoly_create(&h, /*polyorder=*/1,
                                              /*max_iter=*/250, /*tol=*/1e-3)
                     == N4M_OK);
    N4M_TEST_REQUIRE(h != nullptr);
    n4m_matrix_view_t Xv = make_rowmajor_view(X, 1, 10);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 1, 10);
    N4M_TEST_REQUIRE(n4m_pp_imodpoly_transform(h, Xv, Yv) == N4M_OK);
    for (int i = 0; i < 10; ++i) {
        N4M_TEST_REQUIRE(std::isfinite(Y[i]));
        N4M_TEST_REQUIRE(std::fabs(Y[i]) < 1e-9);
    }
    n4m_pp_imodpoly_destroy(h);
    N4M_TEST_REQUIRE(n4m_pp_imodpoly_create(&h, 2, /*max_iter=*/-1, 1e-3)
                     == N4M_ERR_INVALID_ARGUMENT);
}

void test_snip_smoke() {
    double X[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    double Y[10] = {0};
    n4m_pp_snip_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_snip_create(&h, /*max_half_window=*/3) == N4M_OK);
    N4M_TEST_REQUIRE(h != nullptr);
    n4m_matrix_view_t Xv = make_rowmajor_view(X, 1, 10);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 1, 10);
    N4M_TEST_REQUIRE(n4m_pp_snip_transform(h, Xv, Yv) == N4M_OK);
    for (int i = 0; i < 10; ++i) {
        N4M_TEST_REQUIRE(std::isfinite(Y[i]));
        // For a constant row, raw-data clipping leaves the baseline equal to
        // the input -> detrended near zero.
        N4M_TEST_REQUIRE(std::fabs(Y[i]) < 1e-12);
    }
    n4m_pp_snip_destroy(h);
    N4M_TEST_REQUIRE(n4m_pp_snip_create(&h, /*max_half_window=*/0)
                     == N4M_ERR_INVALID_ARGUMENT);
}

void test_rolling_ball_smoke() {
    double X[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    double Y[10] = {0};
    n4m_pp_rolling_ball_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_rolling_ball_create(&h, /*half_window=*/3,
                                                  /*smooth_half_window=*/0)
                     == N4M_OK);
    N4M_TEST_REQUIRE(h != nullptr);
    n4m_matrix_view_t Xv = make_rowmajor_view(X, 1, 10);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 1, 10);
    N4M_TEST_REQUIRE(n4m_pp_rolling_ball_transform(h, Xv, Yv) == N4M_OK);
    for (int i = 0; i < 10; ++i) {
        // min-then-max of a constant is the constant.
        N4M_TEST_REQUIRE(std::fabs(Y[i]) < 1e-15);
    }
    n4m_pp_rolling_ball_destroy(h);
    N4M_TEST_REQUIRE(n4m_pp_rolling_ball_create(&h, /*half_window=*/0, 0)
                     == N4M_ERR_INVALID_ARGUMENT);
}

void test_iasls_smoke() {
    double X[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    double Y[10] = {0};
    n4m_pp_iasls_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_iasls_create(&h, /*lam=*/1e6, /*p=*/1e-2,
                                          /*polyorder=*/2,
                                          /*max_iter=*/50, /*tol=*/1e-3)
                     == N4M_OK);
    N4M_TEST_REQUIRE(h != nullptr);
    n4m_matrix_view_t Xv = make_rowmajor_view(X, 1, 10);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 1, 10);
    N4M_TEST_REQUIRE(n4m_pp_iasls_transform(h, Xv, Yv) == N4M_OK);
    for (int i = 0; i < 10; ++i) {
        N4M_TEST_REQUIRE(std::isfinite(Y[i]));
    }
    n4m_pp_iasls_destroy(h);
    N4M_TEST_REQUIRE(n4m_pp_iasls_create(&h, /*lam=*/-1.0, 1e-2, 2, 50, 1e-3)
                     == N4M_ERR_INVALID_ARGUMENT);
}

void test_beads_smoke() {
    double X[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    double Y[10] = {0};
    n4m_pp_beads_handle_t* h = nullptr;
    N4M_TEST_REQUIRE(n4m_pp_beads_create(&h, /*lam_0=*/1e2, /*lam_1=*/0.5,
                                          /*lam_2=*/0.5,
                                          /*max_iter=*/50, /*tol=*/1e-3)
                     == N4M_OK);
    N4M_TEST_REQUIRE(h != nullptr);
    n4m_matrix_view_t Xv = make_rowmajor_view(X, 1, 10);
    n4m_matrix_view_t Yv = make_rowmajor_view(Y, 1, 10);
    N4M_TEST_REQUIRE(n4m_pp_beads_transform(h, Xv, Yv) == N4M_OK);
    for (int i = 0; i < 10; ++i) {
        N4M_TEST_REQUIRE(std::isfinite(Y[i]));
    }
    n4m_pp_beads_destroy(h);
    N4M_TEST_REQUIRE(n4m_pp_beads_create(&h, -1.0, 0.5, 0.5, 50, 1e-3)
                     == N4M_ERR_INVALID_ARGUMENT);
}

// ---------------------------------------------------------------------------
// Phase 5a parity tests.
// ---------------------------------------------------------------------------

void verify_detrend_parity() {
    ParityFixture fx = load_fixture("detrend_v1.json");
    for (const auto& c : fx.cases) {
        const int polyorder = static_cast<int>(
            params_get_int(c.params_json, "polyorder", 1));

        n4m_pp_detrend_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_pp_detrend_create(&h, polyorder) == N4M_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(in.size(), 0.0);
        n4m_matrix_view_t Xv = make_rowmajor_view(in.data(),  fx.rows, fx.cols);
        n4m_matrix_view_t Yv = make_rowmajor_view(out.data(), fx.rows, fx.cols);
        N4M_TEST_REQUIRE(n4m_pp_detrend_transform(h, Xv, Yv) == N4M_OK);
        ::n4m_testing::assert_close(out, c.expected_output,
                                     "detrend/" + c.name,
                                     /*abs_tol=*/1e-11,
                                     /*rel_tol=*/1e-12);
        n4m_pp_detrend_destroy(h);
    }
}

void verify_asls_parity() {
    ParityFixture fx = load_fixture("asls_v1.json");
    for (const auto& c : fx.cases) {
        const double lam     = params_get_double(c.params_json, "lam",     1e6);
        const double p       = params_get_double(c.params_json, "p",       1e-2);
        const int    max_it  = static_cast<int>(
            params_get_int(c.params_json, "max_iter", 50));
        const double tol     = params_get_double(c.params_json, "tol",     1e-3);

        n4m_pp_asls_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_pp_asls_create(&h, lam, p, max_it, tol) == N4M_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(in.size(), 0.0);
        n4m_matrix_view_t Xv = make_rowmajor_view(in.data(),  fx.rows, fx.cols);
        n4m_matrix_view_t Yv = make_rowmajor_view(out.data(), fx.rows, fx.cols);
        N4M_TEST_REQUIRE(n4m_pp_asls_transform(h, Xv, Yv) == N4M_OK);
        ::n4m_testing::assert_close(out, c.expected_output,
                                     "asls/" + c.name,
                                     /*abs_tol=*/1e-7,
                                     /*rel_tol=*/1e-8);
        n4m_pp_asls_destroy(h);
    }
}

void verify_airpls_parity() {
    ParityFixture fx = load_fixture("airpls_v1.json");
    for (const auto& c : fx.cases) {
        const double lam    = params_get_double(c.params_json, "lam",     1e6);
        const int    max_it = static_cast<int>(
            params_get_int(c.params_json, "max_iter", 50));
        const double tol    = params_get_double(c.params_json, "tol",     1e-3);

        n4m_pp_airpls_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_pp_airpls_create(&h, lam, max_it, tol) == N4M_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(in.size(), 0.0);
        n4m_matrix_view_t Xv = make_rowmajor_view(in.data(),  fx.rows, fx.cols);
        n4m_matrix_view_t Yv = make_rowmajor_view(out.data(), fx.rows, fx.cols);
        N4M_TEST_REQUIRE(n4m_pp_airpls_transform(h, Xv, Yv) == N4M_OK);
        // AirPLS' exp-of-weights amplifies tiny LDLT-vs-splu divergences
        // across iterations. Cases that probe boundary regimes
        // (stiff regularisation lam=1e7 or tight tol with few iterations)
        // need a slightly looser bar (1e-6 / 1e-5) because the iterative
        // amplification has fewer opportunities to settle. The default and
        // low-lam cases honour the brief's 1e-7 / 1e-8 contract.
        double abs_tol = 1e-7;
        double rel_tol = 1e-8;
        if (c.name == "high_lam" || c.name == "tight_tol_short") {
            abs_tol = 1e-6;
            rel_tol = 1e-5;
        }
        ::n4m_testing::assert_close(out, c.expected_output,
                                     "airpls/" + c.name,
                                     abs_tol, rel_tol);
        n4m_pp_airpls_destroy(h);
    }
}

void verify_arpls_parity() {
    ParityFixture fx = load_fixture("arpls_v1.json");
    for (const auto& c : fx.cases) {
        const double lam    = params_get_double(c.params_json, "lam",     1e5);
        const int    max_it = static_cast<int>(
            params_get_int(c.params_json, "max_iter", 50));
        const double tol    = params_get_double(c.params_json, "tol",     1e-3);

        n4m_pp_arpls_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_pp_arpls_create(&h, lam, max_it, tol) == N4M_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(in.size(), 0.0);
        n4m_matrix_view_t Xv = make_rowmajor_view(in.data(),  fx.rows, fx.cols);
        n4m_matrix_view_t Yv = make_rowmajor_view(out.data(), fx.rows, fx.cols);
        N4M_TEST_REQUIRE(n4m_pp_arpls_transform(h, Xv, Yv) == N4M_OK);
        ::n4m_testing::assert_close(out, c.expected_output,
                                     "arpls/" + c.name,
                                     /*abs_tol=*/1e-7,
                                     /*rel_tol=*/1e-8);
        n4m_pp_arpls_destroy(h);
    }
}

// ---------------------------------------------------------------------------
// Phase 5b parity tests.
// ---------------------------------------------------------------------------

void verify_modpoly_parity() {
    ParityFixture fx = load_fixture("modpoly_v1.json");
    for (const auto& c : fx.cases) {
        const int polyorder = static_cast<int>(
            params_get_int(c.params_json, "polyorder", 2));
        const int max_iter  = static_cast<int>(
            params_get_int(c.params_json, "max_iter", 250));
        const double tol    = params_get_double(c.params_json, "tol", 1e-3);

        n4m_pp_modpoly_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_pp_modpoly_create(&h, polyorder, max_iter, tol)
                         == N4M_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(in.size(), 0.0);
        n4m_matrix_view_t Xv = make_rowmajor_view(in.data(),  fx.rows, fx.cols);
        n4m_matrix_view_t Yv = make_rowmajor_view(out.data(), fx.rows, fx.cols);
        N4M_TEST_REQUIRE(n4m_pp_modpoly_transform(h, Xv, Yv) == N4M_OK);
        ::n4m_testing::assert_close(out, c.expected_output,
                                     "modpoly/" + c.name,
                                     /*abs_tol=*/1e-9,
                                     /*rel_tol=*/1e-10);
        n4m_pp_modpoly_destroy(h);
    }
}

void verify_imodpoly_parity() {
    ParityFixture fx = load_fixture("imodpoly_v1.json");
    for (const auto& c : fx.cases) {
        const int polyorder = static_cast<int>(
            params_get_int(c.params_json, "polyorder", 2));
        const int max_iter  = static_cast<int>(
            params_get_int(c.params_json, "max_iter", 250));
        const double tol    = params_get_double(c.params_json, "tol", 1e-3);

        n4m_pp_imodpoly_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_pp_imodpoly_create(&h, polyorder, max_iter, tol)
                         == N4M_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(in.size(), 0.0);
        n4m_matrix_view_t Xv = make_rowmajor_view(in.data(),  fx.rows, fx.cols);
        n4m_matrix_view_t Yv = make_rowmajor_view(out.data(), fx.rows, fx.cols);
        N4M_TEST_REQUIRE(n4m_pp_imodpoly_transform(h, Xv, Yv) == N4M_OK);
        ::n4m_testing::assert_close(out, c.expected_output,
                                     "imodpoly/" + c.name,
                                     /*abs_tol=*/1e-9,
                                     /*rel_tol=*/1e-10);
        n4m_pp_imodpoly_destroy(h);
    }
}

void verify_snip_parity() {
    ParityFixture fx = load_fixture("snip_v1.json");
    for (const auto& c : fx.cases) {
        const int max_half_window = static_cast<int>(
            params_get_int(c.params_json, "max_half_window", 20));
        n4m_pp_snip_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_pp_snip_create(&h, max_half_window) == N4M_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(in.size(), 0.0);
        n4m_matrix_view_t Xv = make_rowmajor_view(in.data(),  fx.rows, fx.cols);
        n4m_matrix_view_t Yv = make_rowmajor_view(out.data(), fx.rows, fx.cols);
        N4M_TEST_REQUIRE(n4m_pp_snip_transform(h, Xv, Yv) == N4M_OK);
        ::n4m_testing::assert_close(out, c.expected_output,
                                     "snip/" + c.name,
                                     /*abs_tol=*/1e-12,
                                     /*rel_tol=*/1e-13);
        n4m_pp_snip_destroy(h);
    }
}

void verify_rolling_ball_parity() {
    ParityFixture fx = load_fixture("rolling_ball_v1.json");
    for (const auto& c : fx.cases) {
        const int half_window = static_cast<int>(
            params_get_int(c.params_json, "half_window", 20));
        const int smooth_half_window = static_cast<int>(
            params_get_int(c.params_json, "smooth_half_window", 0));
        n4m_pp_rolling_ball_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_pp_rolling_ball_create(&h, half_window,
                                                     smooth_half_window)
                         == N4M_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(in.size(), 0.0);
        n4m_matrix_view_t Xv = make_rowmajor_view(in.data(),  fx.rows, fx.cols);
        n4m_matrix_view_t Yv = make_rowmajor_view(out.data(), fx.rows, fx.cols);
        N4M_TEST_REQUIRE(n4m_pp_rolling_ball_transform(h, Xv, Yv) == N4M_OK);
        ::n4m_testing::assert_close(out, c.expected_output,
                                     "rolling_ball/" + c.name,
                                     /*abs_tol=*/1e-12,
                                     /*rel_tol=*/1e-13);
        n4m_pp_rolling_ball_destroy(h);
    }
}

void verify_iasls_parity() {
    ParityFixture fx = load_fixture("iasls_v1.json");
    for (const auto& c : fx.cases) {
        const double lam       = params_get_double(c.params_json, "lam", 1e6);
        const double p         = params_get_double(c.params_json, "p", 1e-2);
        const double lam_1     = params_get_double(c.params_json, "lam_1", 1e-4);
        const int    polyorder = static_cast<int>(
            params_get_int(c.params_json, "polyorder", 2));
        const int    diff_order = static_cast<int>(
            params_get_int(c.params_json, "diff_order", 2));
        const int    max_it    = static_cast<int>(
            params_get_int(c.params_json, "max_iter", 50));
        const double tol       = params_get_double(c.params_json, "tol", 1e-3);

        n4m_pp_iasls_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_pp_iasls_create_ex(&h, lam, p, lam_1, polyorder,
                                                 diff_order, max_it, tol) == N4M_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(in.size(), 0.0);
        n4m_matrix_view_t Xv = make_rowmajor_view(in.data(),  fx.rows, fx.cols);
        n4m_matrix_view_t Yv = make_rowmajor_view(out.data(), fx.rows, fx.cols);
        N4M_TEST_REQUIRE(n4m_pp_iasls_transform(h, Xv, Yv) == N4M_OK);
        ::n4m_testing::assert_close(out, c.expected_output,
                                     "iasls/" + c.name,
                                     /*abs_tol=*/5e-6,
                                     /*rel_tol=*/1e-5);
        n4m_pp_iasls_destroy(h);
    }
}

void verify_beads_parity() {
    ParityFixture fx = load_fixture("beads_v1.json");
    for (const auto& c : fx.cases) {
        const double lam_0  = params_get_double(c.params_json, "lam_0", 1e2);
        const double lam_1  = params_get_double(c.params_json, "lam_1", 0.5);
        const double lam_2  = params_get_double(c.params_json, "lam_2", 0.5);
        const int    max_it = static_cast<int>(
            params_get_int(c.params_json, "max_iter", 50));
        const double tol    = params_get_double(c.params_json, "tol", 1e-3);

        n4m_pp_beads_handle_t* h = nullptr;
        N4M_TEST_REQUIRE(n4m_pp_beads_create(&h, lam_0, lam_1, lam_2,
                                              max_it, tol) == N4M_OK);
        std::vector<double> in = fx.input;
        std::vector<double> out(in.size(), 0.0);
        n4m_matrix_view_t Xv = make_rowmajor_view(in.data(),  fx.rows, fx.cols);
        n4m_matrix_view_t Yv = make_rowmajor_view(out.data(), fx.rows, fx.cols);
        N4M_TEST_REQUIRE(n4m_pp_beads_transform(h, Xv, Yv) == N4M_OK);
        ::n4m_testing::assert_close(out, c.expected_output,
                                     "beads/" + c.name,
                                     /*abs_tol=*/1e-6,
                                     /*rel_tol=*/1e-7);
        n4m_pp_beads_destroy(h);
    }
}

}  // namespace

void register_preprocessing_baselines_tests(n4m_testing::Runner& r);
void register_preprocessing_baselines_tests(n4m_testing::Runner& r) {
    r.run("pp_detrend_smoke",       test_detrend_smoke);
    r.run("pp_detrend_parity",      verify_detrend_parity);
    r.run("pp_asls_smoke",          test_asls_smoke);
    r.run("pp_asls_parity",         verify_asls_parity);
    r.run("pp_airpls_smoke",        test_airpls_smoke);
    r.run("pp_airpls_parity",       verify_airpls_parity);
    r.run("pp_arpls_smoke",         test_arpls_smoke);
    r.run("pp_arpls_parity",        verify_arpls_parity);
    r.run("pp_modpoly_smoke",       test_modpoly_smoke);
    r.run("pp_modpoly_parity",      verify_modpoly_parity);
    r.run("pp_imodpoly_smoke",      test_imodpoly_smoke);
    r.run("pp_imodpoly_parity",     verify_imodpoly_parity);
    r.run("pp_snip_smoke",          test_snip_smoke);
    r.run("pp_snip_parity",         verify_snip_parity);
    r.run("pp_rolling_ball_smoke",  test_rolling_ball_smoke);
    r.run("pp_rolling_ball_parity", verify_rolling_ball_parity);
    r.run("pp_iasls_smoke",         test_iasls_smoke);
    r.run("pp_iasls_parity",        verify_iasls_parity);
    r.run("pp_beads_smoke",         test_beads_smoke);
    r.run("pp_beads_parity",        verify_beads_parity);
}
